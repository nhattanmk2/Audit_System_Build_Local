import streamlit as st
import os
import chardet
import json
import io
import google.generativeai as genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # ⬅️ BẮT BUỘC

# print("DEBUG GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

# ===================== CONFIG =====================
CIS_DB_PATH = "./cis_vector_db"

# ===================== LLM CONFIG =====================
GEMINI_MODEL = "gemini-2.5-flash" 

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

DEBUG_MODE = True

st.set_page_config(page_title="Config Audit Agent", layout="wide")
st.title("🕵️ Config Audit AI Agent")

st.markdown("""
Hệ thống tự động so sánh file cấu hình (SecEdit, Registry dump...) với CIS Benchmark.
""")

# ===================== RESOURCES =====================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# @st.cache_resource
def get_llm():
    
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,   # ⬅️ BẮT BUỘC
        convert_system_message_to_human=True,
    )


# ===================== UTILS =====================
def detect_encoding(file_content: bytes):
    return chardet.detect(file_content).get("encoding")

def parse_secedit_config(content: str):
    docs = []
    section = "General"

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            docs.append(
                Document(
                    page_content=f"Section: {section}\nSetting: {key}\nValue: {value}",
                    metadata={
                        "section": section,
                        "key": key.strip(),
                        "value": value.strip()
                    }
                )
            )
    return docs

# ===================== NORMALIZATION =====================
def normalize_value(value):
    if value is None:
        return None

    if isinstance(value, (int, float, bool)):
        return value

    v = str(value).strip().lower()

    if v in ["1", "true", "yes", "enabled", "enable"]:
        return True
    if v in ["0", "false", "no", "disabled", "disable"]:
        return False
    if v.isdigit():
        return int(v)

    return v

# ===================== SEMANTIC NORMALIZATION =====================
def normalize_config_item(config_doc, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a security configuration normalizer.

TASK:
- Translate the CONFIG KEY into a human-readable security concept.
- Do NOT evaluate compliance.
- Do NOT reference CIS.
- Do NOT guess values not present.

Output STRICT JSON only.
        """),
        ("user", """
CONFIG:
Section: {section}
Key: {key}
Value: {value}
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "section": config_doc.metadata["section"],
            "key": config_doc.metadata["key"],
            "value": config_doc.metadata["value"]
        })

        if not raw or not raw.strip():
            raise ValueError("Empty LLM response")

        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        # Hard validation
        if "normalized_concept" not in data or "actual_value" not in data:
            raise ValueError("Missing required fields")

        return data

    except Exception:
        # 🔒 SAFE FALLBACK (KHÔNG CRASH)
        return {
            "normalized_concept": config_doc.metadata["key"],
            "value_type": "string",
            "actual_value": config_doc.metadata["value"]
        }

# ===================== EVALUATION =====================
def evaluate(actual, expected, operator):
    actual_n = normalize_value(actual)
    expected_n = normalize_value(expected)

    if actual_n is None or expected_n is None:
        return None, "Missing value"

    if type(actual_n) != type(expected_n):
        return None, f"Type mismatch: actual={type(actual_n).__name__}, expected={type(expected_n).__name__}"

    try:
        if operator == "==":
            return actual_n == expected_n, None
        if operator == "!=":
            return actual_n != expected_n, None
        if operator == ">=":
            return actual_n >= expected_n, None
        if operator == "<=":
            return actual_n <= expected_n, None
        if operator == ">":
            return actual_n > expected_n, None
        if operator == "<":
            return actual_n < expected_n, None
    except Exception as e:
        return None, str(e)

    return None, f"Unsupported operator {operator}"

# ===================== AUDIT CORE =====================
def audit_config_item(config_doc, cis_db, llm):
    normalized = normalize_config_item(config_doc, llm)

    concept = normalized["normalized_concept"]
    actual_value = normalized["actual_value"]

    retrieved_docs = cis_db.similarity_search(concept, k=5)

    cis_expectations = []
    for d in retrieved_docs:
        if d.metadata.get("expected") not in [None, "Unknown"]:
            cis_expectations.append({
                "rule_id": d.metadata.get("rule_id"),
                "title": d.metadata.get("title"),
                "expected": d.metadata.get("expected"),
                "operator": d.metadata.get("operator")
            })

    for rule in cis_expectations:
        result, err = evaluate(actual_value, rule["expected"], rule["operator"])
        if result is None:
            continue
        return {
            "rule_id": rule["rule_id"],
            "title": rule["title"],
            "actual": actual_value,
            "expected": rule["expected"],
            "compliance_status": "PASS" if result else "FAIL"
        }, retrieved_docs

    return {
        "rule_title": concept,
        "actual": actual_value,
        "compliance_status": "SKIP",
        "reason": "No comparable CIS expectation after type normalization"
    }, retrieved_docs

def generate_audit_pdf(pdf_rows):
    buffer = io.BytesIO()

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "HeiseiMin-W3"
    styles["Heading1"].fontName = "HeiseiMin-W3"

    elements = []

    # ===== TITLE =====
    elements.append(Paragraph(
        "CONFIGURATION AUDIT RESULT (PASS / FAIL)",
        styles["Heading1"]
    ))
    elements.append(Paragraph(
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 16))

    # ===== TABLE =====
    table_data = [
        ["No.", "Parameter Name", "Actual Value", "Result"]
    ]

    for idx, r in enumerate(pdf_rows, start=1):
        table_data.append([
            idx,
            r["param_name"],
            str(r["actual_value"]),
            r["result"]
        ])

    table = Table(
        table_data,
        colWidths=[40, 220, 160, 70]
    )

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("ALIGN", (-1,1), (-1,-1), "CENTER"),

        ("TEXTCOLOR", (-1,1), (-1,-1),
         lambda r, c, v: colors.green if v == "PASS" else colors.red)
    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return buffer


# ===================== UI =====================
uploaded_config = st.file_uploader(
    "Upload file Config (txt / inf / ini)",
    type=["txt", "inf", "ini"]
)

if uploaded_config and os.path.exists(CIS_DB_PATH):
    raw_bytes = uploaded_config.getvalue()
    encoding = detect_encoding(raw_bytes) or "utf-16"
    content = raw_bytes.decode(encoding, errors="ignore")

    config_docs = parse_secedit_config(content)

    if st.button("▶️ Bắt đầu Audit", type="primary"):
        cis_db = Chroma(
            persist_directory=CIS_DB_PATH,
            embedding_function=get_embedding_model(),
            collection_name="cis_rules"
        )
        llm = get_llm()

        st.subheader("📊 Kết quả Audit Realtime")
        metrics = {"PASS": 0, "FAIL": 0, "SKIP": 0}

        pdf_rows = []   # ===== DATA FOR PDF EXPORT =====

        for doc in config_docs:
            result, sources = audit_config_item(doc, cis_db, llm)
            status = result.get("compliance_status", "SKIP")
            metrics[status] += 1

            # ===== Collect PASS / FAIL only for PDF =====
            if status in ("PASS", "FAIL"):
                pdf_rows.append({
                    "param_name": doc.metadata.get("key"),
                    "actual_value": doc.metadata.get("value"),
                    "result": status
                })

            # ===== Realtime UI =====
            with st.expander(f"{status} — {doc.metadata.get('key')}"):
                st.json(result)
                if DEBUG_MODE:
                    for s in sources:
                        st.code(s.page_content)

        st.success("Hoàn tất kiểm tra!")

        col1, col2, col3 = st.columns(3)
        col1.metric("PASS", metrics["PASS"])
        col2.metric("FAIL", metrics["FAIL"])
        col3.metric("SKIP", metrics["SKIP"])

        # ===== EXPORT PDF =====
        if pdf_rows:
            pdf_buffer = generate_audit_pdf(pdf_rows)

            st.download_button(
                label="📥 Download Audit Result (PDF)",
                data=pdf_buffer,
                file_name=f"audit_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("No PASS / FAIL results available for PDF export.")


elif uploaded_config:
    st.error("❌ Không tìm thấy CIS Vector DB. Hãy chạy cis_ingest_app.py trước.")
