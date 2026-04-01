import streamlit as st
import os
import time
import chardet
import json
import io
import re
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import hashlib
from pathlib import Path

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
CACHE_DIR = Path("./.llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ===================== LLM CONFIG =====================
DEFAULT_MODEL = "gemma-3-27b-it" 

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

DEBUG_MODE = True

st.set_page_config(page_title="Config Audit Agent", layout="wide")
st.title("🕵️ Config Audit AI Agent")

# ===================== SESSION STATE =====================
if "audit_done" not in st.session_state:
    st.session_state.audit_done = False
if "audit_results" not in st.session_state:
    st.session_state.audit_results = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"PASS": 0, "FAIL": 0, "SKIP": 0}
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None
if "word_buffer" not in st.session_state:
    st.session_state.word_buffer = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = ""

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
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota" in err_msg:
                # Phân biệt giới hạn phút và giới hạn ngày
                if "daily" in err_msg.lower() or "limit: 20" in err_msg:
                    st.error("❌ Hết quota hàng ngày (Daily Quota Exceeded). Vui lòng thử lại sau hoặc đổi model/API key.")
                    raise Exception("QUOTA_EXHAUSTED_DAILY")
                
                if attempt == max_retries - 1:
                    raise
                wait_time = base_delay * (attempt + 1)
                st.warning(f"⚠️ Hit Rate Limit. Thử lại sau {wait_time}s... (Lần {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

def get_cache_path(key_data: str) -> Path:
    """Tạo path file cache dựa trên MD5 hash của key."""
    hash_key = hashlib.md5(key_data.encode()).hexdigest()
    return CACHE_DIR / f"{hash_key}.json"

def get_cached_response(cache_key: str) -> Optional[Any]:
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_response(cache_key: str, data: Any):
    cache_path = get_cache_path(cache_key)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

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

    # Caching key: section + key + value + model_name (from llm)
    model_name = getattr(llm, "model", "default")
    cache_key = f"norm_{config_doc.metadata['section']}_{config_doc.metadata['key']}_{config_doc.metadata['value']}_{model_name}"
    
    cached = get_cached_response(cache_key)
    if cached:
        return cached

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

        save_cached_response(cache_key, data)
        return data

    except Exception as e:
        if str(e) == "QUOTA_EXHAUSTED_DAILY":
            raise e
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
    candidate_ids = []
    for c in candidates:
        candidate_text += f"- Rule ID: {c['rule_id']} | Title: {c['title']}\n"
        candidate_ids.append(c['rule_id'])

    # Caching key
    model_name = getattr(llm, "model", "default")
    cache_key = f"select_{key_name}_{section_name}_{','.join(candidate_ids)}_{model_name}"
    
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    chain = prompt | llm | StrOutputParser()
    result = invoke_chain_with_retry(chain, {
        "key": key_name,
        "section": section_name,
        "candidate_list": candidate_text
    }).strip()

    # Kiểm tra xem kết quả có nằm trong danh sách ID không
    valid_ids = [c["rule_id"] for c in candidates]
    if result in valid_ids:
        save_cached_response(cache_key, result)
        return result
    
    if result == "NONE":
        save_cached_response(cache_key, "NONE")
        return "NONE"
    
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
    # Three columns: Item, Reference, and Result
    table_data = [
        [
            Paragraph(f"<b>HẠNG MỤC KIỂM TRA</b>", header_style),
            Paragraph(f"<b>THAM CHIẾU CIS RULE</b>", header_style),
            Paragraph(f"<b>KẾT QUẢ</b>", header_style)
        ]
    ]

    for r in pdf_rows:
        table_data.append([
            Paragraph(str(r["param_name"]), criteria_style),
            Paragraph(str(r["rule_ref"]), criteria_style),
            str(r["result"])
        ])

    available_width = doc.width
    table = Table(
        table_data,
        colWidths=[available_width * 0.40, available_width * 0.45, available_width * 0.15]
    )

    # Style colors from Word doc
    header_bg = colors.HexColor("#BCD5ED")
    header_text_color = colors.HexColor("#0F0F3F")

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), header_text_color),
        ("ALIGN", (2,1), (2,-1), "CENTER"), # Result column
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        
        # PASS (Green) / FAIL (Red) color logic
        ("TEXTCOLOR", (2,1), (2,-1),
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
        available_models = [DEFAULT_MODEL, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    model_name = st.selectbox(
        "Lựa chọn Model", 
        options=available_models,
        index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
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
        # Reset state for new audit
        st.session_state.audit_done = False
        st.session_state.audit_results = []
        st.session_state.metrics = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        st.session_state.pdf_buffer = None
        st.session_state.word_buffer = None
        st.session_state.last_file_name = uploaded_config.name

        cis_db = Chroma(
            persist_directory=CIS_DB_PATH,
            embedding_function=get_embedding_model(),
            collection_name="cis_rules"
        )
        llm = get_llm(model_name)

        st.subheader("📊 Đang thực hiện Audit...")
        
        pdf_rows = []
        progress_bar = st.progress(0)
        total_docs = len(config_docs)

        try:
            for i, doc in enumerate(config_docs):
                progress_bar.progress((i + 1) / total_docs)
                key_name = doc.metadata.get('key', '')
                
                # Run Audit
                result, sources = audit_config_item(doc, cis_db, llm)
                status = result.get("compliance_status", "SKIP")
                st.session_state.metrics[status] += 1

                # Save to session state
                st.session_state.audit_results.append({
                    "key": key_name,
                    "status": status,
                    "result": result,
                    "sources": sources
                })

                # ===== Collect for PDF =====
                if status in ("PASS", "FAIL"):
                    rule_id = result.get("rule_id", "N/A")
                    rule_title = result.get("title", "N/A")
                    pdf_rows.append({
                        "param_name": doc.metadata.get("key"),
                        "actual_value": doc.metadata.get("value"),
                        "result": status,
                        "rule_ref": f"{rule_id}: {rule_title}"
                    })

            st.session_state.audit_done = True
            
            # Generate PDF buffer once
            if pdf_rows:
                st.session_state.pdf_buffer = generate_audit_pdf(pdf_rows).getvalue()
                
                # Generate CMC Data
                cmc_data = []
                for res_item in st.session_state.audit_results:
                    # We can use the results already computed
                    res = res_item["result"]
                    cmc_data.append({
                        "param_name": res_item["key"],
                        "actual_value": res.get("actual", ""),
                        "result": res.get("compliance_status", "SKIP"),
                        "remediation": res.get("remediation", "")
                    })
                st.session_state.word_buffer = generate_cmc_report(cmc_data, target_server=uploaded_config.name).getvalue()

            st.rerun()

        except Exception as global_e:
            if str(global_e) == "QUOTA_EXHAUSTED_DAILY":
                st.error("🛑 Dừng quá trình Audit do hết quota hàng ngày.")
            else:
                st.error(f"❌ Có lỗi xảy ra: {global_e}")

    # ===================== DISPLAY RESULTS (Outside button block) =====================
    if st.session_state.audit_done:
        st.subheader(f"📊 Kết quả Audit: {st.session_state.last_file_name}")
        
        # Summary Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("PASS", st.session_state.metrics["PASS"])
        m_col2.metric("FAIL", st.session_state.metrics["FAIL"])
        m_col3.metric("SKIP", st.session_state.metrics["SKIP"])

        # Action Buttons
        dl_col1, dl_col2 = st.columns(2)
        if st.session_state.pdf_buffer:
            dl_col1.download_button(
                label="📥 Download Audit Result (PDF)",
                data=st.session_state.pdf_buffer,
                file_name=f"audit_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        if st.session_state.word_buffer:
            dl_col2.download_button(
                label="📘 Download CMC Report (Word)",
                data=st.session_state.word_buffer,
                file_name=f"CMC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        st.divider()

        # Detailed results list
        for res_item in st.session_state.audit_results:
            key_name = res_item["key"]
            status = res_item["status"]
            result = res_item["result"]
            sources = res_item["sources"]

            with st.expander(f"{key_name} | {status}"):
                st.write(f"**Trạng thái:** {status}")
                st.json(result)
                if DEBUG_MODE:
                    st.caption("Các nguồn (Rule) tham chiếu phù hợp:")
                    for s in sources[:2]:
                        # Check if s is a Document object (it should be)
                        content = s.page_content if hasattr(s, "page_content") else str(s)
                        st.code(content[:250] + "...")


elif uploaded_config:
    st.error("❌ Không tìm thấy CIS Vector DB. Hãy chạy cis_ingest_app.py trước.")
