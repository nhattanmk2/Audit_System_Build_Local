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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
    return "\n".join([p.page_content for p in pages])

def extract_rule_details(rule_body):
    operator = "Unknown"
    expected = "Unknown"

    body = rule_body.lower()
    if "set to 'enabled'" in body or "set to enabled" in body:
        operator, expected = "==", "Enabled"
    elif "set to 'disabled'" in body or "set to disabled" in body:
        operator, expected = "==", "Disabled"
    elif "or more" in body:
        operator = ">="
        m = re.search(r"(\d+)\s+or more", body)
        if m:
            expected = m.group(1)

    return operator, expected

def parse_cis_rules(full_text):
    rule_pattern = re.compile(r"^(\d+(?:\.\d+)+)\s+\((L[12])\)\s+(.+)$", re.MULTILINE)
    matches = list(rule_pattern.finditer(full_text))
    documents = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        rule_content = full_text[start:end].strip()

        rule_id = m.group(1)
        level = m.group(2)
        title = m.group(3).strip()

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

    Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(),
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

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

    del db
    gc.collect()

    return output_path

def delete_db_safely(path: str):
    def worker():
        try:
            if not os.path.exists(path):
                return

            renamed = path + "__deleting__"
            if os.path.exists(renamed):
                shutil.rmtree(renamed, ignore_errors=True)

            # 🔹 Rename trước (Windows cho phép)
            os.rename(path, renamed)

            # 🔹 Retry delete
            for _ in range(10):
                try:
                    shutil.rmtree(renamed)
                    return
                except Exception:
                    time.sleep(1)

        except Exception:
            with open("delete_db_error.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    threading.Thread(target=worker, daemon=True).start()

# ================= UI =================
uploaded_file = st.file_uploader("Upload CIS Benchmark PDF", type="pdf")
if uploaded_file and st.button("🚀 Bắt đầu Ingest"):
    process_pdf(uploaded_file)

st.divider()

if os.path.exists(VECTOR_DB_PATH):
    if st.button("📤 Export Vector DB → HTML"):
        html_path = export_db_to_html()
        st.success("✅ Export thành công")

        with open(html_path, "rb") as f:
            st.download_button(
                label="📥 Tải file HTML để xem Vector Database",
                data=f,
                file_name=os.path.basename(html_path),
                mime="text/html"
            )

        st.info("👉 Sau khi tải, mở file HTML bằng trình duyệt (Chrome/Edge/Firefox)")

        webbrowser.open(os.path.abspath(html_path))

    if st.button("🗑️ Xóa toàn bộ CIS Database", key="delete_db"):
        if os.path.exists(VECTOR_DB_PATH):
            delete_db_safely(VECTOR_DB_PATH)
            st.success("🧹 Đã yêu cầu xóa Database")
            st.info("ℹ️ Database sẽ được xóa hoàn toàn sau vài giây")
            st.info("🔄 Reload trang sau 3–5 giây để kiểm tra")
            st.stop()
        else:
            st.warning("Database không tồn tại")


