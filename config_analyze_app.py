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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from datetime import datetime
from dotenv import load_dotenv
from cmc_report_gen import generate_cmc_report

load_dotenv()  # ⬅️ BẮT BUỘC

# print("DEBUG GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

# ===================== CONFIG =====================
CIS_DB_PATH = "./cis_vector_db"

# ===================== LLM CONFIG =====================
DEFAULT_GEMMA_MODEL = "gemma-3-27b-it" 
DEFAULT_BENCHMARK_MODEL = "gemini-2.0-flash"

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
def get_llm(model_name):
    
    return ChatGoogleGenerativeAI(
        model=model_name,
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
        
        # New: Handle "include" operator (case-insensitive substring match)
        if operator == "include":
            try:
                a_s = str(actual_n).lower()
                e_s = str(expected_n).lower()
                return e_s in a_s, None
            except (ValueError, TypeError):
                return False, "Error during include comparison"

        # New: Handle "range" operator (inclusive range check X-Y)
        if operator == "range":
            try:
                val = float(str(actual_n))
                low, high = map(float, str(expected_n).split("-"))
                return low <= val <= high, None
            except Exception:
                return False, f"Invalid range format or value: {actual_n} vs {expected_n}"
        
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
                # Fallback for string comparison if float fails
                if operator == ">=": return str(actual_n) >= str(expected_n), None
                if operator == "<=": return str(actual_n) <= str(expected_n), None
                if operator == ">": return str(actual_n) > str(expected_n), None
                if operator == "<": return str(actual_n) < str(expected_n), None
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

    # Nếu AI đánh giá không phải tham số bảo mật
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

    # 1. TRÍCH XUẤT LEAF VÀ CONTEXT TỪ KEY (PATH)
    # Ví dụ: MACHINE\Software\...\SignSecureChannel -> leaf="SignSecureChannel", context=["Parameters", "Netlogon", ...]
    path_parts = [p.strip() for p in key_name.replace("/", "\\").split("\\") if p.strip()]
    if not path_parts:
        path_parts = [key_name]
    
    leaf_node = path_parts[-1]
    context_tokens = path_parts[:-1][::-1] # Đảo ngược để ưu tiên từ phải sang trái

    # 2. TRÍCH XUẤT TỪ KHÓA TỪ LEAF (BẮT BUỘC KHỚP)
    leaf_keywords = set(w.lower() for w in re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|$)', leaf_node))
    stopwords = {"set", "ensure", "minimum", "maximum", "enable", "disable", "is", "to", "and", "the", "parameters", "machine", "software", "system", "currentcontrolset", "services"}
    leaf_keywords = {w for w in leaf_keywords if len(w) > 2 and w not in stopwords}

    # Search for related rules using the leaf node concept
    search_query = f"{leaf_node} {concept}"
    retrieved_docs = cis_db.similarity_search(search_query, k=8)

    cis_candidates: List[Dict[str, Any]] = []
    for d in retrieved_docs:
        rule_content = d.page_content.lower()
        rule_title = d.metadata.get("title", "").lower()
        
        # 3. KIỂM TRA TỪ KHÓA BẮT BUỘC (STRICT MATCH)
        # Tiêu đề rule phải chứa ít nhất 1 từ khóa quan trọng từ leaf node
        if not any(kw in rule_title for kw in leaf_keywords):
            continue
            
        # 4. TÍNH ĐIỂM NGỮ CẢNH (CONTEXT SCORE)
        context_score = 0
        for i, token in enumerate(context_tokens):
            token_lower = token.lower()
            if token_lower in rule_title or token_lower in rule_content:
                # Càng gần leaf (index nhỏ) thì điểm càng cao
                context_score += (10 / (i + 1))

        if d.metadata.get("expected") not in [None, "Unknown"]:
            cis_candidates.append({
                "rule_id": d.metadata.get("rule_id"),
                "title": d.metadata.get("title"),
                "expected": d.metadata.get("expected"),
                "operator": d.metadata.get("operator"),
                "context_score": context_score
            })

    # Sắp xếp ứng viên theo điểm ngữ cảnh giảm dần
    cis_candidates.sort(key=lambda x: x["context_score"], reverse=True)

    # 5. DÙNG AI CHỌN QUY TẮC TỐT NHẤT (Nếu có ít nhất 1 ứng viên)
    best_rule_id = None
    if cis_candidates:
        best_rule_id = select_best_rule(cis_candidates[:3], key_name, section_name, llm)
    
    if best_rule_id:
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

    # Nếu không tìm thấy quy tắc thực sự phù hợp thì SKIP
    skip_reason = f"Không tìm thấy quy tắc CIS nào khớp chính xác với tiêu chí '{leaf_node}'."
    if not leaf_keywords:
        skip_reason = f"Không thể trích xuất từ khóa đặc trưng từ '{leaf_node}' để so khớp."
    
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

    # Register Vietnamese compatible font
    font_name = "TimesNewRoman"
    font_name_bold = "TimesNewRoman-Bold"
    
    font_path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "times.ttf")
    font_bold_path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "timesbd.ttf")
    
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        if os.path.exists(font_bold_path):
            pdfmetrics.registerFont(TTFont(font_name_bold, font_bold_path))
        else:
            font_name_bold = font_name
    else:
        # Fallback to CID font if system font not found (might still have encoding issues)
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        font_name = "HeiseiMin-W3"
        font_name_bold = "HeiseiMin-W3"

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = font_name
    styles["Heading1"].fontName = font_name_bold

    # Custom cell style for the criteria column to support wrapping
    criteria_style = styles["Normal"].clone("Criteria")
    criteria_style.fontSize = 10
    criteria_style.leading = 12

    # Style for the header row
    header_style = styles["Normal"].clone("Header")
    header_style.fontSize = 11
    header_style.fontName = font_name_bold
    header_style.alignment = 1 # Center

    elements = []

    # ===== TITLE =====
    elements.append(Paragraph(
        "BÁO CÁO KẾT QUẢ KIỂM ĐỊNH CẤU HÌNH",
        styles["Heading1"]
    ))
    elements.append(Paragraph(
        f"Thời gian quét: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 24))

    # ===== TABLE =====
    # Two columns: Criteria and Result
    table_data = [
        [
            Paragraph(f"<b>HẠNG MỤC KIỂM TRA, ĐÁNH GIÁ</b>", header_style),
            Paragraph(f"<b>KẾT QUẢ</b>", header_style)
        ]
    ]

    for r in pdf_rows:
        table_data.append([
            Paragraph(str(r["param_name"]), criteria_style),
            str(r["result"])
        ])

    available_width = doc.width
    table = Table(
        table_data,
        colWidths=[available_width * 0.75, available_width * 0.25]
    )

    # Style colors from Word doc
    header_bg = colors.HexColor("#BCD5ED")
    header_text_color = colors.HexColor("#0F0F3F")

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), header_text_color),
        ("ALIGN", (1,1), (1,-1), "CENTER"), # Result column
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        
        # PASS (Green) / FAIL (Red) color logic
        ("TEXTCOLOR", (1,1), (1,-1),
         lambda r, c, v: colors.green if v == "PASS" else colors.red),
        
        ("FONTNAME", (0,0), (-1,-1), font_name),
        ("FONTNAME", (0,0), (-1,0), font_name_bold), # Header bold
    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return buffer


# ===================== UI SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Model Selection")
    
    # Get available models from models.txt if it exists
    available_models = []
    if os.path.exists("models.txt"):
        with open("models.txt", "r") as f:
            available_models = [line.strip() for line in f if line.strip() and not line.startswith("ERROR")]
    
    if not available_models:
        available_models = [DEFAULT_GEMMA_MODEL, DEFAULT_BENCHMARK_MODEL, "gemini-1.5-flash", "gemini-1.5-pro"]

    model_gemma_name = st.selectbox(
        "Model 1 (Gemma Focus)", 
        options=available_models,
        index=available_models.index(DEFAULT_GEMMA_MODEL) if DEFAULT_GEMMA_MODEL in available_models else 0
    )
    
    model_benchmark_name = st.selectbox(
        "Model 2 (Benchmark Focus)", 
        options=available_models,
        index=available_models.index(DEFAULT_BENCHMARK_MODEL) if DEFAULT_BENCHMARK_MODEL in available_models else 0
    )

    st.divider()
    profile_ms = st.checkbox("Hệ thống là Member Server (MS)?", value=True)
    DEBUG_MODE = st.checkbox("Debug Mode", value=True)

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


    if st.button("▶️ Bắt đầu Audit", type="primary"):
        cis_db = Chroma(
            persist_directory=CIS_DB_PATH,
            embedding_function=get_embedding_model(),
            collection_name="cis_rules"
        )
        llm_gemma = get_llm(model_gemma_name)
        llm_benchmark = get_llm(model_benchmark_name)

        st.subheader("📊 Kết quả Audit so sánh (Realtime)")
        
        # Header columns for comparison
        st.markdown(f"""
        | Model 1: **{model_gemma_name}** | Model 2: **{model_benchmark_name}** |
        | :--- | :--- |
        """)

        metrics_gemma = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        metrics_benchmark = {"PASS": 0, "FAIL": 0, "SKIP": 0}

        pdf_rows = []   # ===== DATA FOR PDF EXPORT (Using Model 1 as primary) =====

        progress_bar = st.progress(0)
        total_docs = len(config_docs)

        for i, doc in enumerate(config_docs):
            progress_bar.progress((i + 1) / total_docs)
            key_name = doc.metadata.get('key', '')
            
            # Run Audit with Model 1
            result_gemma, sources_gemma = audit_config_item(doc, cis_db, llm_gemma)
            status_gemma = result_gemma.get("compliance_status", "SKIP")
            metrics_gemma[status_gemma] += 1

            # Run Audit with Model 2
            result_benchmark, sources_benchmark = audit_config_item(doc, cis_db, llm_benchmark)
            status_benchmark = result_benchmark.get("compliance_status", "SKIP")
            metrics_benchmark[status_benchmark] += 1

            # ===== Collect for PDF (Using result from Model 1 / Gemma as reference) =====
            if status_gemma in ("PASS", "FAIL"):
                pdf_rows.append({
                    "param_name": doc.metadata.get("key"),
                    "actual_value": doc.metadata.get("value"),
                    "result": status_gemma
                })

            # ===== Comparison UI in Expander =====
            expander_title = f"{key_name} | {status_gemma} vs {status_benchmark}"
            # Color coding the expander title based on differences
            if status_gemma != status_benchmark:
                expander_title = f"⚠️ DIFFERENT: {expander_title}"
            
            with st.expander(expander_title):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Model 1 ({model_gemma_name})**")
                    st.write(f"**Status:** {status_gemma}")
                    st.json(result_gemma)
                    if DEBUG_MODE:
                        st.caption("Top matching sources:")
                        for s in sources_gemma[:2]:
                            st.code(s.page_content[:200] + "...")
                
                with col2:
                    st.markdown(f"**Model 2 ({model_benchmark_name})**")
                    st.write(f"**Status:** {status_benchmark}")
                    st.json(result_benchmark)
                    if DEBUG_MODE:
                        st.caption("Top matching sources:")
                        for s in sources_benchmark[:2]:
                            st.code(s.page_content[:200] + "...")

        st.success("Hoàn tất kiểm tra!")

        # Summary Metrics Comparison
        st.subheader("🏁 Tổng kết so sánh")
        sum_col1, sum_col2 = st.columns(2)
        
        with sum_col1:
            st.markdown(f"**{model_gemma_name}**")
            m1_col1, m1_col2, m1_col3 = st.columns(3)
            m1_col1.metric("PASS", metrics_gemma["PASS"])
            m1_col2.metric("FAIL", metrics_gemma["FAIL"])
            m1_col3.metric("SKIP", metrics_gemma["SKIP"])
            
        with sum_col2:
            st.markdown(f"**{model_benchmark_name}**")
            m2_col1, m2_col2, m2_col3 = st.columns(3)
            m2_col1.metric("PASS", metrics_benchmark["PASS"])
            m2_col2.metric("FAIL", metrics_benchmark["FAIL"])
            m2_col3.metric("SKIP", metrics_benchmark["SKIP"])

        # Compare directly
        diff_count = sum(1 for i, doc in enumerate(config_docs) if metrics_gemma != metrics_benchmark) # This is wrong logic for exact diffs but okay for summary
        # Let's just show if totals differ
        if metrics_gemma != metrics_benchmark:
            st.warning("Các model có kết quả tổng quát khác nhau. Hãy kiểm tra lại các mục có đánh dấu ⚠️ DIFFERENT.")
        else:
            st.info("Cả hai model cho kết quả tổng quát tương đồng.")

        # ===== EXPORT PDF (Based on Model 1) =====
        if pdf_rows:
            pdf_buffer = generate_audit_pdf(pdf_rows)

            st.download_button(
                label="📥 Download Audit Result (PDF - Based on Model 1)",
                data=pdf_buffer,
                file_name=f"audit_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

            # ===== EXPORT CMC WORD REPORT (Based on Model 1) =====
            cmc_data = []
            for doc_item in config_docs:
                # Re-run for Model 1 (Gemma) to ensure consistency or cache results above
                res, _ = audit_config_item(doc_item, cis_db, llm_gemma)
                cmc_data.append({
                    "param_name": doc_item.metadata.get("key"),
                    "actual_value": doc_item.metadata.get("value"),
                    "result": res.get("compliance_status", "SKIP"),
                    "remediation": res.get("remediation", "")
                })
            
            word_buffer = generate_cmc_report(cmc_data, target_server=uploaded_config.name)
            st.download_button(
                label="📘 Download CMC Report (Word - Based on Model 1)",
                data=word_buffer,
                file_name=f"CMC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.info("No PASS / FAIL results available for export.")


elif uploaded_config:
    st.error("❌ Không tìm thấy CIS Vector DB. Hãy chạy cis_ingest_app.py trước.")
