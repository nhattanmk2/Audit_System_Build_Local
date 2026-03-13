import streamlit as st
import os
import time
import chardet
import json
import io
import re
from typing import Dict, Any, List, Optional
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
from cmc_report_gen import generate_cmc_report

load_dotenv()  # ⬅️ BẮT BUỘC

# print("DEBUG GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

# ===================== CONFIG =====================
CIS_DB_PATH = "./cis_vector_db"

# ===================== LLM CONFIG =====================
GEMINI_MODEL = "gemini-flash-latest" 

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
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,   # ⬅️ BẮT BUỘC
        convert_system_message_to_human=True,
    )


# ===================== UTILS =====================
def invoke_chain_with_retry(chain, params, max_retries=6, base_delay=10):
    for attempt in range(max_retries):
        try:
            return chain.invoke(params)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "Quota" in str(e):
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (attempt + 1))
            else:
                raise

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
            section: str = line.removeprefix("[").removesuffix("]")
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            docs.append(
                Document(
                    page_content=f"Section: {section}\nSetting: {key}\nValue: {value}",
                    metadata={
                        "section": section,
                        "key": key.strip(),
                        "value": value.strip(),
                        "type": "secedit"
                    }
                )
            )
    return docs

def parse_registry_dump(content: str):
    """Phân tích tệp Registry (.reg hoặc .txt dump)"""
    docs = []
    current_key = "General"
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_key: str = line.removeprefix("[").removesuffix("]")
            continue
        if "=" in line:
            parts = line.split("=", 1)
            name = parts[0].strip().strip('"')
            val = parts[1].strip().strip('"')
            docs.append(
                Document(
                    page_content=f"Registry Key: {current_key}\nName: {name}\nValue: {val}",
                    metadata={
                        "section": current_key,
                        "key": name,
                        "value": val,
                        "type": "registry"
                    }
                )
            )
    return docs

def parse_os_patches(content: str):
    """Phân tích danh sách Hotfix (KB)"""
    docs = []
    # Giả định Source 4 cung cấp danh sách KB hoặc output từ 'systeminfo' / 'wmic qfe'
    # Tìm mã KB (ví dụ: KB5022507)
    kb_pattern = re.compile(r"KB\d+", re.IGNORECASE)
    matches = kb_pattern.findall(content)
    for kb in set(matches):
        docs.append(
            Document(
                page_content=f"OS Patch Installed: {kb}",
                metadata={
                    "section": "OS_Patch",
                    "key": "Hotfix",
                    "value": kb.upper(),
                    "type": "patch"
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
1. Translate the CONFIG KEY and SECTION into a highly specific security concept. 
   - ALWAYS include the service/section name if it's a service property (e.g., "TermService ImagePath" -> "Remote Desktop Service Executable Path").
2. Determine if this setting is typically relevant for security benchmarks (CIS, NIST, etc.).
   - Structural tags like "[Unicode]", "Version", "Signature" are NOT security-relevant.

Output STRICT JSON only:
{{
  "normalized_concept": "string (the specific security concept)",
  "is_security_relevant": boolean,
  "actual_value": "string (the original value)"
}}
        """),
        ("user", """
CONFIG TO NORMALIZE:
Section: {section}
Key: {key}
Value: {value}
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        raw = invoke_chain_with_retry(chain, {
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

    try:
        # Path normalization if both are strings
        if isinstance(actual_n, str) and isinstance(expected_n, str):
            if "\\" in actual_n or "/" in actual_n:
                actual_n = actual_n.replace("\\", "/").lower().strip()
                expected_n = expected_n.replace("\\", "/").lower().strip()

        # Cast for comparison
        if operator == "==":
            return actual_n == expected_n, None
        if operator == "!=":
            return actual_n != expected_n, None
        
        # For numeric comparisons, try float conversion
        if operator in [">=", "<=", ">", "<"]:
            try:
                a_f = float(str(actual_n))
                e_f = float(str(expected_n))
                if operator == ">=": return a_f >= e_f, None
                if operator == "<=": return a_f <= e_f, None
                if operator == ">": return a_f > e_f, None
                if operator == "<": return a_f < e_f, None
            except (ValueError, TypeError):
                # Fallback
                return actual_n >= expected_n, None # type: ignore
    except Exception as e:
        return None, str(e)

    return None, f"Unsupported operator {operator}"

def select_best_rule(candidates: List[Dict[str, Any]], key_name: str, section_name: str, llm: ChatGoogleGenerativeAI) -> Optional[str]:
    """Sử dụng LLM để chọn quy tắc khớp nhất trong danh sách các ứng viên."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["rule_id"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Bạn là một chuyên gia đối soát an ninh.

NHIỆM VỤ:
Trong danh sách các quy tắc CIS được cung cấp, hãy chọn ra duy nhất một Quy tắc (Rule ID) khớp chính xác nhất với tham số cấu hình.
- Chú ý sự khác biệt giữa "user account", "machine account", "domain member", "domain controller".
- Nếu không có quy tắc nào thực sự khớp 100%, hãy trả về "NONE".

ĐỊNH DẠNG TRẢ VỀ: Chỉ trả về duy nhất Rule ID hoặc "NONE". Không giải thích gì thêm.
        """),
        ("user", """
THÔNG TIN THAM SỐ:
- Tên tham số (Key): {key}
- Section: {section}

DANH SÁCH QUY TẮC ỨNG VIÊN:
{candidate_list}

HÃY CHỌN RULE ID KHỚP NHẤT:
        """)
    ])

    candidate_text = ""
    for c in candidates:
        candidate_text += f"- Rule ID: {c['rule_id']} | Title: {c['title']}\n"

    chain = prompt | llm | StrOutputParser()
    result = invoke_chain_with_retry(chain, {
        "key": key_name,
        "section": section_name,
        "candidate_list": candidate_text
    }).strip()

    # Kiểm tra xem kết quả có nằm trong danh sách ID không
    valid_ids = [c["rule_id"] for c in candidates]
    if result in valid_ids:
        return result
    return None

# ===================== AUDIT CORE =====================
def audit_config_item(config_doc, cis_db, llm):
    normalized = normalize_config_item(config_doc, llm)
    
    actual_value = normalized.get("actual_value")
    key_name = config_doc.metadata.get("key", "")
    section_name = config_doc.metadata.get("section", "")

    # Nếu AI đánh giá không phải tham số bảo mật (ví dụ: thẻ [Unicode])
    if not normalized.get("is_security_relevant", True):
        return {
            "rule_id": "N/A",
            "title": f"[Structural Tag] {key_name}",
            "actual": actual_value,
            "expected": "N/A",
            "compliance_status": "SKIP",
            "reason": "Tham số mang tính cấu trúc file hoặc không nằm trong phạm vi audit bảo mật."
        }, []

    concept = normalized["normalized_concept"]

    # Search for related rules
    retrieved_docs = cis_db.similarity_search(concept, k=5)

    # STRICT KEYWORD FILTERING
    # Chúng ta trích xuất các từ khóa quan trọng từ Key hoặc Concept
    # Ví dụ: "MinimumPasswordAge" -> ["minimum", "password", "age"]
    keywords = set(w.lower() for w in re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|$)', key_name))
    if not keywords or len(keywords) == 1:
        # Nếu chỉ có 1 từ (ví dụ do key viết thường hoàn toàn), ta dùng thêm Concept
        keywords.update(concept.lower().split())
    
    # Loại bỏ các từ quá ngắn hoặc quá chung chung
    stopwords = {"set", "ensure", "minimum", "maximum", "enable", "disable", "is", "to", "and", "the"}
    keywords = {w for w in keywords if len(w) > 2 and w not in stopwords}

    cis_candidates: List[Dict[str, Any]] = []
    for d in retrieved_docs:
        rule_content = d.page_content.lower()
        rule_title = d.metadata.get("title", "").lower()
        
        # 1. Kiểm tra từ khóa bắt buộc
        has_keyword_match = any(kw in rule_title or kw in rule_content for kw in keywords)
        
        # 2. Kiểm tra ngữ cảnh Service (đã có từ trước)
        is_relevant_service = True
        s_name_lower = section_name.lower()
        if "svc" in s_name_lower or s_name_lower in ["termservice", "wuauserv", "lanmanserver"]:
             if s_name_lower not in rule_content and s_name_lower not in rule_title:
                 base_name = s_name_lower.replace("svc", "")
                 if base_name not in rule_content and base_name not in rule_title:
                    is_relevant_service = False
        
        if has_keyword_match and is_relevant_service and d.metadata.get("expected") not in [None, "Unknown"]:
            cis_candidates.append({
                "rule_id": d.metadata.get("rule_id"),
                "title": d.metadata.get("title"),
                "expected": d.metadata.get("expected"),
                "operator": d.metadata.get("operator")
            })

    # DÙNG AI CHỌN QUY TẮC TỐT NHẤT (Nếu có nhiều hơn 1 ứng viên)
    best_rule_id = select_best_rule(cis_candidates, key_name, section_name, llm)
    
    if best_rule_id:
        # Tìm quy tắc được chọn trong danh sách ứng viên
        selected_rule = next((c for c in cis_candidates if c["rule_id"] == best_rule_id), None)
        if selected_rule:
            result, err = evaluate(actual_value, selected_rule["expected"], selected_rule["operator"])
            if result is not None:
                remediation = "N/A"
                for d in retrieved_docs:
                    if d.metadata.get("rule_id") == selected_rule["rule_id"]:
                        remediation = d.metadata.get("remediation", "Check CIS Benchmark for details.")
                        break

                return {
                    "rule_id": selected_rule["rule_id"],
                    "title": selected_rule["title"],
                    "actual": actual_value,
                    "expected": selected_rule["expected"],
                    "compliance_status": "PASS" if result else "FAIL",
                    "remediation": remediation if not result else ""
                }, retrieved_docs
            else:
                return {
                    "rule_id": selected_rule["rule_id"],
                    "title": selected_rule["title"],
                    "actual": actual_value,
                    "expected": selected_rule["expected"],
                    "compliance_status": "SKIP",
                    "reason": f"Lỗi trong quá trình so sánh giá trị: {err}"
                }, retrieved_docs
        else:
            return {
                "rule_id": "N/A",
                "title": f"Không tìm thấy quy tắc khớp 100% cho: {key_name}",
                "actual": actual_value,
                "expected": "N/A",
                "compliance_status": "SKIP",
                "reason": "AI re-ranker không chọn được quy tắc nào tối ưu nhất trong các ứng viên."
            }, retrieved_docs

    # Nếu không có ứng viên nào sau khi lọc
    skip_reason = f"Không tìm thấy quy tắc CIS nào khớp với từ khóa '{', '.join(keywords)}' trong ngữ cảnh {section_name}."
    if retrieved_docs:
        # Kiểm tra xem có phải do thiếu giá trị kỳ vọng (expected) không
        missing_expected = [d.metadata.get("rule_id") for d in retrieved_docs if d.metadata.get("expected") in [None, "Unknown"]]
        if missing_expected:
            skip_reason = f"Tìm thấy quy tắc ({', '.join(missing_expected)}) nhưng chưa trích xuất được giá trị kỳ vọng (expected) từ Benchmark."

    return {
        "rule_id": "N/A",
        "title": f"Không tìm thấy quy tắc cho: {key_name}",
        "actual": actual_value,
        "expected": "N/A",
        "compliance_status": "SKIP",
        "reason": skip_reason
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
            str(idx),
            str(r["param_name"]),
            str(r["actual_value"]),
            str(r["result"])
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

    config_docs = []
    content_lower = content.lower()
    
    # 1. Ưu tiên nhận diện theo nội dung (Content-based detection)
    if any(marker in content for marker in ["Windows Registry Editor", "[HKEY_", "REGEDIT4"]):
        config_docs = parse_registry_dump(content)
        st.info("Phát hiện định dạng tệp Registry")
    elif any(marker in content for marker in ["[Unicode]", "[Version]", "[System Access]", "[Privilege Rights]"]):
        config_docs = parse_secedit_config(content)
        st.info("Phát hiện định dạng tệp SecEdit (Inffile)")
    
    # 2. Nếu không có marker rõ ràng, thử theo extension hoặc tên file
    elif uploaded_config.name.endswith(".inf") or "secedit" in uploaded_config.name.lower():
        config_docs = parse_secedit_config(content)
    elif uploaded_config.name.endswith(".reg") or "registry" in uploaded_config.name.lower():
        config_docs = parse_registry_dump(content)
    
    # 3. Cuối cùng mới xét đến OS Patch (Hotfix) hoặc Fallback
    else:
        # Chỉ coi là Patch nếu có mẫu KB rõ ràng và không giống file config
        kb_pattern = re.compile(r"KB\d+", re.IGNORECASE)
        if kb_pattern.search(content) and "=" not in content:
            config_docs = parse_os_patches(content)
            st.info("Phát hiện danh sách OS Patch (Hotfix)")
        else:
            # Mặc định thử parse theo SecEdit nếu có dấu bằng (phổ biến nhất)
            if "=" in content:
                config_docs = parse_secedit_config(content)
            else:
                st.warning("Không thể xác định loại file. Thử xử lý như file cấu hình mặc định.")
                config_docs = parse_secedit_config(content)

    profile_ms = st.checkbox("Hệ thống là Member Server (MS)?", value=True)

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

            # ===== EXPORT CMC WORD REPORT =====
            cmc_data = []
            for doc_item in config_docs:
                # Tìm lại kết quả cho từng doc_item (vì PDF chỉ lấy PASS/FAIL)
                res, _ = audit_config_item(doc_item, cis_db, llm)
                cmc_data.append({
                    "param_name": doc_item.metadata.get("key"),
                    "actual_value": doc_item.metadata.get("value"),
                    "result": res.get("compliance_status", "SKIP"),
                    "remediation": res.get("remediation", "")
                })
            
            word_buffer = generate_cmc_report(cmc_data, target_server=uploaded_config.name)
            st.download_button(
                label="📘 Download CMC Report (Word)",
                data=word_buffer,
                file_name=f"CMC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.info("No PASS / FAIL results available for PDF export.")


elif uploaded_config:
    st.error("❌ Không tìm thấy CIS Vector DB. Hãy chạy cis_ingest_app.py trước.")
