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
    if re.search(r"set\s+to\s+['\"‘“\u2018\u201c]?enabled['\"’”?\u2019\u201d]?", body):
        return "==", "1"
    if re.search(r"set\s+to\s+['\"‘“\u2018\u201c]?disabled['\"’”?\u2019\u201d]?", body):
        return "==", "0"
    
    # New: Handle "to include" specifically for inclusion rules (like auditing)
    inc_m = re.search(r"to\s+include\s+['\"‘“\u2018\u201c](.*?)['\"’”?\u2019\u201d']", rule_body_clean)
    if inc_m:
        val = inc_m.group(1).strip()
        return "include", val
    
    # Fallback for "to include" without quotes for simple values
    if re.search(r"to\s+include\s+enabled", body): return "include", "1"
    if re.search(r"to\s+include\s+disabled", body): return "include", "0"

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
    quote_match = re.search(r"set\s+to\s+['\"‘“\u2018\u201c](.*?)['\"’”?\u2019\u201d']", rule_body_clean)
    if quote_match:
        val = quote_match.group(1).strip()
        if not val.replace(".", "").isdigit(): # Nếu là chữ (như "No One")
            return "==", val
        
    # 4. Mẫu số đơn thuần: set to 'X' hoặc set to X
    m = re.search(r"set\s+to\s+['\"‘“\u2018\u201c]?(\d+)['\"’”?\u2019\u201d']?", body)
    if m:
        return "==", m.group(1)

    # 5. Xử lý mẫu dải giá trị (Range): between X and Y
    range_m = re.search(r"between\s+(\d+)\s+and\s+(\d+)", body)
    if range_m:
        return "range", f"{range_m.group(1)}-{range_m.group(2)}"

    return operator, expected

def parse_cis_rules(full_text):
    # Cải thiện regex: Thêm \s* ở đầu để cho phép khoảng trắng, và dùng re.MULTILINE
    # Một số PDF có thể có khoảng trắng hoặc tab ở đầu dòng
    rule_pattern = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+\((L[12])\)\s+(.+)$", re.MULTILINE)
    matches = list(rule_pattern.finditer(full_text))
    documents = []
    
    if not matches:
        st.warning("⚠️ Không tìm thấy quy tắc nào khớp với định dạng 'X.X.X (L1/L2) Title'.")
        # Thử regex lỏng lẻo hơn nếu không thấy kết quả
        rule_pattern_loose = re.compile(r"(\d+(?:\.\d+)+)\s+\((L[12])\)\s+(.+)")
        matches = list(rule_pattern_loose.finditer(full_text))
        if matches:
            st.info(f"💡 Đã tìm thấy {len(matches)} quy tắc bằng phương pháp quét tự do.")

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

    st.info(f"🔄 Đang nạp {len(docs)} quy tắc vào Vector Database...")
    
    try:
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
        
    except Exception as e:
        error_msg = str(e)
        if "code: 14" in error_msg or "unable to open database file" in error_msg:
            st.error("❌ **DATABASE ĐANG BỊ KHÓA!**")
            st.warning("👉 Bạn đang chạy file `config_analyze_app.py` (ứng dụng Audit) hoặc một tiến trình khác đang sử dụng database này.")
            st.info("💡 **Cách khắc phục:**\n1. Tắt terminal đang chạy `config_analyze_app.py`.\n2. Thử nhấn nút **🚀 Bắt đầu Ingest** lại.\n3. Nếu vẫn không được, hãy nhấn nút **🗑️ Xóa/Đổi tên Database cũ** ở phía dưới trước khi nạp.")
        else:
            st.error(f"❌ Lỗi khi nạp vào Vector DB: {error_msg}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ================= EXPORT DB =================
def export_vector_db():
    """Xuất dữ liệu từ Vector DB ra các định dạng khác nhau (HTML, CSV)"""
    try:
        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=get_embedding_model(),
            collection_name=COLLECTION_NAME
        )

        raw = db._collection.get(include=["documents", "metadatas"])
        
        # Giải phóng lock ngay sau khi lấy dữ liệu
        if hasattr(db, "_client") and hasattr(db._client, "close"):
            db._client.close()
        del db
        gc.collect()

        if not raw["documents"]:
            return None, None

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
                "Rule Content": doc
            })

        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Tạo file HTML (với DataTables để tìm kiếm/phân trang)
        html_path = os.path.join(EXPORT_DIR, f"cis_export_{timestamp}.html")
        
        # HTML template với Bootstrap và DataTables
        html_content = df.copy()
        html_content["Rule Content"] = html_content["Rule Content"].str.replace("\n", "<br>")
        
        table_html = html_content.to_html(
            escape=False, index=False, border=0, 
            classes="table table-hover table-bordered display", 
            table_id="cisTable"
        )
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CIS Benchmark Export - {timestamp}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ background-color: #f8f9fa; padding-top: 2rem; }}
        .container {{ max-width: 95%; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h2 {{ color: #0d6efd; margin-bottom: 1.5rem; }}
        thead {{ background-color: #e9ecef; }}
        .rule-cell {{ font-size: 0.9rem; max-height: 200px; overflow-y: auto; display: block; }}
    </style>
</head>
<body>
    <div class="container mb-5">
        <h2>📚 CIS Benchmark Vector Database Map</h2>
        <p class="text-muted">Xuất bản lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr>
        <div class="table-responsive">
            {table_html}
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {{
            $('#cisTable').DataTable({{
                "language": {{
                    "lengthMenu": "Hiển thị _MENU_ dòng mỗi trang",
                    "zeroRecords": "Không tìm thấy dữ liệu",
                    "info": "Trang _PAGE_ / _PAGES_",
                    "infoEmpty": "Dữ liệu trống",
                    "infoFiltered": "(lọc từ _MAX_ dòng)",
                    "search": "Tìm kiếm:",
                    "paginate": {{
                        "first": "Đầu",
                        "last": "Cuối",
                        "next": "Sau",
                        "previous": "Trước"
                    }}
                }},
                "pageLength": 10,
                "order": [[0, "asc"]]
            }});
        }});
    </script>
</body>
</html>
""")

        # 2. Tạo file CSV
        csv_path = os.path.join(EXPORT_DIR, f"cis_export_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return html_path, csv_path

    except Exception as e:
        st.error(f"❌ Lỗi khi xuất dữ liệu: {str(e)}")
        return None, None

def delete_db_completely(path: str):
    """Xóa database bằng kỹ thuật Rename-then-Delete chuyên sâu cho Windows"""
    try:
        if not os.path.exists(path):
            st.warning("Database không tồn tại.")
            return False

        # 1. Giải phóng tài nguyên
        gc.collect()
        time.sleep(0.5)

        # 2. Đổi tên folder gốc sang thư mục Trash để giải phóng path ngay lập tức
        timestamp = datetime.now().strftime("%H%M%S")
        trash_path = f"{path}_TRASH_{timestamp}"
        
        try:
            os.rename(path, trash_path)
        except Exception as e:
            st.error(f"❌ File đang bị khóa bởi hệ thống: {str(e)}")
            st.info("💡 **Gợi ý:** Hãy đóng cửa sổ terminal đang chạy `config_analyze_app.py` và thử lại.")
            return False

        # 3. Thử xóa folder Trash
        try:
            shutil.rmtree(trash_path, ignore_errors=True)
            if not os.path.exists(trash_path):
                st.success("✨ Đã dọn dẹp sạch sẽ dữ liệu cũ.")
            else:
                st.info("📁 Folder cũ đã được đưa vào khu vực chờ xóa tự động.")
        except Exception:
            pass

        return True
    except Exception as e:
        st.error(f"❌ Lỗi hệ thống: {str(e)}")
        return False

# ================= UI =================
st.warning("⚠️ **Lưu ý:** Đảm bảo bạn đã **TẮT** ứng dụng Audit (`config_analyze_app.py`) trước khi nhấn Ingest để tránh lỗi tranh chấp file database.")

uploaded_file = st.file_uploader("Upload CIS Benchmark PDF", type="pdf")
if uploaded_file and st.button("🚀 Bắt đầu Ingest", use_container_width=True):
    process_pdf(uploaded_file)

st.divider()

if os.path.exists(VECTOR_DB_PATH):
    st.subheader("🛠️ Quản lý Cơ sở dữ liệu")
    col_exp, col_del = st.columns([1, 1])
    
    with col_exp:
        if st.button("📤 Xuất dữ liệu (Export)", use_container_width=True):
            html_p, csv_p = export_vector_db()
            if html_p and csv_p:
                st.success("✅ Đã tạo các tệp xuất dữ liệu.")
                
                # Tải HTML
                with open(html_p, "rb") as f:
                    st.download_button(
                        label="🌐 Tải HTML (Interactive)",
                        data=f,
                        file_name=os.path.basename(html_p),
                        mime="text/html",
                        use_container_width=True
                    )
                
                # Tải CSV
                with open(csv_p, "rb") as f:
                    st.download_button(
                        label="📄 Tải CSV (Excel Ready)",
                        data=f,
                        file_name=os.path.basename(csv_p),
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Không có dữ liệu để xuất.")

    with col_del:
        with st.popover("🗑️ Xóa Database", use_container_width=True):
            st.error("Hành động này không thể hoàn tác!")
            if st.button("🔥 Xác nhận xóa", type="primary", use_container_width=True):
                if delete_db_completely(VECTOR_DB_PATH):
                    st.rerun()


