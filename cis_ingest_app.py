import streamlit as st
import os
import shutil
import re
import gc
import pandas as pd
import webbrowser
import subprocess
import sys
import time
import threading
import traceback
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader as PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ================= CONFIG =================
VECTOR_DB_PATH = "./cis_vector_db"
COLLECTION_NAME = "cis_rules"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EXPORT_DIR = "./exports"

os.makedirs(EXPORT_DIR, exist_ok=True)

st.set_page_config(page_title="CIS Benchmark Ingestion", layout="wide")
st.title("📚 CIS Benchmark Ingestion Tool (Audit Ready)")

# ================= EMBEDDING =================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ================= PARSING =================
def normalize_text(pages):
    full_text = "\n".join([p.page_content for p in pages])
    
    # 1. Tìm vị trí của "Overview" xuất hiện lần thứ 2
    overview_keyword = "Overview"
    first_overview_idx = full_text.find(overview_keyword)
    
    start_idx = 0
    if first_overview_idx != -1:
        second_overview_idx = full_text.find(overview_keyword, first_overview_idx + len(overview_keyword))
        if second_overview_idx != -1:
            start_idx = second_overview_idx
            st.info(f"📍 Bắt đầu lấy nội dung từ 'Overview' lần thứ 2 (vị trí {start_idx})")
        else:
            # Fallback nếu chỉ có 1 Panorama
            start_idx = first_overview_idx
            st.warning("⚠️ Chỉ tìm thấy 1 từ khóa 'Overview', lấy từ vị trí đầu tiên.")
    
    # 2. Tìm vị trí của "Summary" xuất hiện lần cuối cùng
    summary_keyword = "Summary"
    last_summary_idx = full_text.rfind(summary_keyword)
    
    end_idx = len(full_text)
    if last_summary_idx != -1 and last_summary_idx > start_idx:
        end_idx = last_summary_idx
        st.info(f"🏁 Kết thúc lấy nội dung tại 'Summary' lần cuối cùng (vị trí {end_idx})")
    
    # 3. Cắt văn bản
    filtered_text = full_text[start_idx:end_idx]
    
    chars_removed_start = start_idx
    chars_removed_end = len(full_text) - end_idx
    total_removed = chars_removed_start + chars_removed_end
    
    if total_removed > 0:
        st.success(f"✂️ Đã lọc bỏ {total_removed} ký tự rác (Mục lục: {chars_removed_start}, Phần cuối: {chars_removed_end}).")
    
    return filtered_text

def extract_rule_details(rule_body):
    """
    Trích xuất Operator và Expected value từ nội dung Rule của CIS.
    Hỗ trợ: Enabled/Disabled, No One, các giá trị trong ngoặc kép, và số lượng (or more/fewer).
    """
    operator = "Unknown"
    expected = "Unknown"

    # Chuẩn hóa khoảng trắng và cắt bỏ xuống dòng để regex không bị đứt đoạn
    rule_body_clean = re.sub(r"\s+", " ", rule_body)
    body = rule_body_clean.lower()
    
    # 1. Map Enabled -> 1 / Disabled -> 0
    # \u2018-\u201d là các dải nháy thông minh
    if re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c]?enabled['\"’”?\u2019\u201d]?", body):
        return "==", "1"
    if re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c]?disabled['\"’”?\u2019\u201d]?", body):
        return "==", "0"
        
    # 2. Xử lý các mẫu số lượng (or more/fewer)
    m = re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c]?(\d+)\s+or more", body)
    if not m:
        m = re.search(r"(\d+)\s+or more", body)
    if m:
        return ">=", m.group(1)

    m = re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c]?(\d+)\s+or (?:fewer|less)", body)
    if not m:
        m = re.search(r"(\d+)\s+or (?:fewer|less)", body)
    if m:
        return "<=", m.group(1)

    # 3. Xử lý các chuỗi cụ thể nằm trong nháy (Ví dụ: 'No One', 'Administrators')
    # Chúng ta capture nội dung bên trong nháy nếu nó không phải chỉ là số
    quote_match = re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c](.*?)['\"’”?\u2019\u201d']", rule_body_clean)
    if quote_match:
        val = quote_match.group(1).strip()
        if not val.replace(".", "").isdigit(): # Nếu là chữ (như "No One")
            return "==", val
        
    # 4. Mẫu số đơn thuần: set to 'X' hoặc set to X
    m = re.search(r"(?:set\s+to|to\s+include)\s+['\"‘“\u2018\u201c]?(\d+)['\"’”?\u2019\u201d']?", body)
    if m:
        return "==", m.group(1)

    return operator, expected

def parse_cis_rules(full_text):
    # Cập nhật regex để bắt thêm nội dung sau tiêu đề đến khi gặp rule mới
    rule_pattern = re.compile(r"^(\d+(?:\.\d+)+)\s+\((L[12])\)\s+(.+)$", re.MULTILINE)
    matches = list(rule_pattern.finditer(full_text))
    documents = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        rule_content = full_text[start:end].strip()

        rule_id = m.group(1)
        level = m.group(2)
        
        # Cải thiện bóc tách Title: Lấy từ sau (L1)/(L2) cho đến khi gặp các section lớn hoặc xuống dòng kép
        # Regex này tìm đoạn text ngay sau Level indicator
        title_raw = m.group(3).strip()
        
        # Tìm xem title có kéo dài xuống các dòng tiếp theo không (trước khi gặp "Description" hoặc section khác)
        # Chúng ta sẽ tìm trong rule_content phần text bắt đầu từ title_raw
        title_search_area = rule_content.split(f"({level})", 1)[-1].strip()
        
        # Title thường kết thúc khi gặp "Description", "Profile Applicability", v.v. hoặc 2 dòng trống
        # Hoặc đơn giản là lấy phần đầu của rule_content đến khi gặp từ khóa mô tả
        title_end_match = re.search(r"(?:\n\s*\n|Description:|Profile Applicability:)", title_search_area, re.IGNORECASE)
        if title_end_match:
            title = title_search_area[:title_end_match.start()].strip()
        else:
            # Fallback nếu không thấy mốc kết thúc rõ ràng, lấy dòng đầu tiên hoặc title_raw ban đầu
            title = title_raw

        # Làm sạch title (xóa xuống dòng dư thừa)
        title = re.sub(r"\s+", " ", title)

        # Trích xuất Remediation từ nội dung rule
        remediation = "No remediation instructions found."
        rem_m = re.search(r"Remediation:\s*(.*?)(?:\n\d+\.|$)", rule_content, re.DOTALL | re.IGNORECASE)
        if rem_m:
            remediation = rem_m.group(1).strip()
        elif "Remediation" in rule_content:
            # Fallback nếu regex chặt chẽ không khớp
            parts = re.split(r"Remediation:", rule_content, flags=re.IGNORECASE)
            if len(parts) > 1:
                remediation = parts[1].split("\n\n")[0].strip()

        concept = re.sub(r"[^a-z0-9\s]", "", title.lower())
        operator, expected = extract_rule_details(rule_content)

        documents.append(
            Document(
                page_content=f"Security Concept:\n{concept}\n\nCIS Rule:\n{rule_content}",
                metadata={
                    "rule_id": rule_id,
                    "level": level,
                    "title": title,
                    "operator": operator,
                    "expected": expected,
                    "remediation": remediation,
                    "source": "CIS_PDF"
                }
            )
        )

    return documents

# ================= INGEST =================
def process_pdf(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyMuPDFLoader(temp_path)
    pages = loader.load()
    full_text = normalize_text(pages)
    docs = parse_cis_rules(full_text)

    db = Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(),
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Ép buộc đóng kết nối và giải phóng tài nguyên
    if hasattr(db, "_client") and hasattr(db._client, "close"):
        db._client.close()
    
    del db
    os.remove(temp_path)
    gc.collect()

    st.success(f"✅ Ingest thành công {len(docs)} CIS rules")

# ================= EXPORT DB → HTML =================
def export_db_to_html():
    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embedding_model(),
        collection_name=COLLECTION_NAME
    )

    raw = db._collection.get(include=["documents", "metadatas"])
    rows = []

    for i, (doc, meta) in enumerate(zip(raw["documents"], raw["metadatas"]), start=1):
        rows.append({
            "STT": i,
            "Rule ID": meta.get("rule_id"),
            "Level": meta.get("level"),
            "Title": meta.get("title"),
            "Operator": meta.get("operator"),
            "Expected": meta.get("expected"),
            "Remediation": meta.get("remediation", "N/A"),
            "Source": meta.get("source"),
            "Rule Content": doc.replace("\n", "<br>")
        })

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(EXPORT_DIR, f"cis_vector_db_{timestamp}.html")

    html = df.to_html(
        escape=False,
        index=False,
        border=0,
        classes="table table-striped"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CIS Vector Database</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
th {{ background-color: #f4f4f4; }}
</style>
</head>
<body>
<h2>CIS Vector Database Export</h2>
<p>Generated at: {timestamp}</p>
{html}
</body>
</html>
""")

    # Đảm bảo giải phóng database handle
    if hasattr(db, "_client") and hasattr(db._client, "close"):
        db._client.close()
        
    del db
    gc.collect()

    return output_path

def delete_db_completely(path: str):
    """Xóa database bằng kỹ thuật Rename-then-Delete để tránh lỗi Access Denied trên Windows"""
    try:
        if not os.path.exists(path):
            st.warning("Database không tồn tại.")
            return False

        # 1. Ép buộc giải phóng tài nguyên của process này
        gc.collect()
        time.sleep(1)

        # 2. Thực hiện Đổi tên (Rename) - Đây là bước quan trọng nhất trên Windows
        timestamp = datetime.now().strftime("%H%M%S")
        trash_path = f"{path}_TRASH_{timestamp}"
        
        try:
            os.rename(path, trash_path)
            st.info(f"📁 Đã di dời database cũ vào thư mục tạm: `{os.path.basename(trash_path)}`")
            st.success("✅ Đường dẫn gốc đã được giải phóng. Bạn có thể nạp dữ liệu mới ngay lập tức!")
        except Exception as e:
            st.error(f"❌ Không thể xóa/đổi tên thư mục: {str(e)}")
            st.warning("👉 **Cách bẻ khóa nhanh nhất trên Windows:**")
            st.info("1. Quay lại terminal đang chạy `cis_ingest_app.py`.\n2. Nhấn **Ctrl + C** để dừng ứng dụng.\n3. Chạy lại lệnh `streamlit run cis_ingest_app.py`.\n4. Nhấn xóa Database lại lần nữa.")
            return False

        # 3. Thử xóa thư mục tạm (sau khi đã đổi tên thành công)
        try:
            shutil.rmtree(trash_path, ignore_errors=True)
            if not os.path.exists(trash_path):
                st.write("✨ Đã dọn dẹp sạch sẽ dữ liệu cũ.")
            else:
                st.warning("⚠️ Thư mục tạm vẫn còn do có file đang bị khóa. Nó sẽ tự biến mất sau khi bạn tắt hoàn toàn terminal chạy ứng dụng này.")
        except Exception:
            pass 

        return True
            
    except Exception as e:
        st.error(f"❌ Lỗi hệ thống: {str(e)}")
        return False

# ================= UI =================
uploaded_file = st.file_uploader("Upload CIS Benchmark PDF", type="pdf")
if uploaded_file and st.button("🚀 Bắt đầu Ingest"):
    process_pdf(uploaded_file)

st.divider()

if os.path.exists(VECTOR_DB_PATH):
    col_exp, col_del = st.columns([1, 1])
    
    with col_exp:
        if st.button("📤 Export Vector DB → HTML", use_container_width=True):
            html_path = export_db_to_html()
            st.success("✅ Export thành công")

            with open(html_path, "rb") as f:
                st.download_button(
                    label="📥 Tải file HTML",
                    data=f,
                    file_name=os.path.basename(html_path),
                    mime="text/html",
                    use_container_width=True
                )
            st.info("👉 Mở file HTML bằng trình duyệt để xem Database")

    with col_del:
        # Sử dụng popover làm "Popup xác nhận"
        with st.popover("🗑️ Xóa toàn bộ CIS Database", use_container_width=True):
            st.warning("⚠️ Hành động này sẽ xóa vĩnh viễn toàn bộ dữ liệu đã ingest. Bạn có chắc chắn không?")
            if st.button("🔥 Xác nhận xóa vĩnh viễn", type="primary", use_container_width=True):
                if delete_db_completely(VECTOR_DB_PATH):
                    time.sleep(1)
                    st.rerun()


