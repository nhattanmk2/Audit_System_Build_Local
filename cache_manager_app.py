import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import hashlib
import time
import re

# Load environment variables
load_dotenv()

# langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Constants
CACHE_DIR = Path("./.llm_cache")
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Cache Optimization Manager", layout="wide")
st.title("🛡️ Cache Optimization & Deduplication Manager")

# ===================== PROVIDERS & MODELS =====================
def get_available_models():
    models = []
    if os.path.exists("models.txt"):
        with open("models.txt", "r") as f:
            models = [line.strip() for line in f if line.strip()]
    return models

def get_llm(model_selection):
    if ":" in model_selection:
        provider, model_name = model_selection.split(":", 1)
    else:
        provider, model_name = "groq", model_selection
    
    if provider == "groq":
        return ChatGroq(model_name=model_name, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

# ===================== CORE LOGIC =====================
def scan_cache() -> Dict[str, List[Dict]]:
    groups = {}
    if not CACHE_DIR.exists():
        return {}
    
    for file_path in CACHE_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict) or "metadata" not in data:
                    continue
                
                meta = data["metadata"]
                # Create a group key based on the parameter identifiers
                group_key = f"{meta.get('section')}|{meta.get('key')}|{meta.get('value')}"
                
                if group_key not in groups:
                    groups[group_key] = []
                
                data["file_path"] = str(file_path)
                groups[group_key].append(data)
        except Exception:
            continue
    return groups

def judge_winner(group_key: str, variants: List[Dict], llm) -> str:
    # prompt for judging
    section, key, value = group_key.split("|")
    
    prompt_text = f"""
    You are a senior security auditor. 
    Below are multiple AI evaluations for the SAME security configuration parameter.
    Your task is to identify which one is THE MOST ACCURATE and BEST FORMATTED.

    PARAMETER DETAILS:
    - Section: {section}
    - Key: {key}
    - Reported Value: {value}

    VARIANTS:
    """
    
    for i, v in enumerate(variants):
        prompt_text += f"\n--- VARIANT {i} (by Model: {v['metadata'].get('model', 'unknown')}) ---\n"
        prompt_text += json.dumps(v['response'], indent=2, ensure_ascii=False)
        prompt_text += "\n"

    prompt_text += """
    INSTRUCTIONS:
    1. Compare the 'normalized_concept'. It should be specific and technical.
    2. Check 'is_security_relevant'.
    3. Return ONLY a single integer representing the index of the best VARIANT (e.g., '0' or '1').
    Do not explain why. Just the number.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({}).strip()
        # Extract the first number found
        match = re.search(r'\d+', result)
        if match:
            return int(match.group())
    except Exception as e:
        st.error(f"Error during judging: {e}")
    return 0

# ===================== UI =====================
sidebar = st.sidebar
sidebar.header("⚙️ Settings")
model_options = get_available_models()
judge_model = sidebar.selectbox("Select Judge Model", options=model_options, index=0)

if sidebar.button("🔄 Rescan Cache"):
    st.rerun()

groups = scan_cache()
duplicates = {k: v for k, v in groups.items() if len(v) > 1}

col1, col2, col3 = st.columns(3)
col1.metric("Total Parameters", len(groups))
col2.metric("Duplicate Groups", len(duplicates))
col3.metric("Total Files in Cache", sum(len(g) for g in groups.values()))

st.divider()

if not duplicates:
    st.success("🎉 No duplicates found! Your cache is clean.")
else:
    st.subheader(f"🔍 Found {len(duplicates)} parameters with conflicting evaluations")
    
    if st.button("🚀 Auto-Clean All Duplicates (AI Judging)"):
        llm = get_llm(judge_model)
        progress = st.progress(0)
        status_text = st.empty()
        
        cleaned_count = 0
        for i, (group_key, variants) in enumerate(duplicates.items()):
            status_text.text(f"Judging: {group_key}...")
            winner_idx = judge_winner(group_key, variants, llm)
            
            # Keep winner, delete others
            for idx, v in enumerate(variants):
                if idx != winner_idx:
                    try:
                        os.remove(v["file_path"])
                    except:
                        pass
            
            cleaned_count += 1
            progress.progress((i + 1) / len(duplicates))
        
        st.success(f"✅ Cleaned {cleaned_count} groups! All duplicates resolved.")
        st.rerun()

    for group_key, variants in duplicates.items():
        with st.expander(f"📌 {group_key}"):
            cols = st.columns(len(variants))
            for i, v in enumerate(variants):
                with cols[i]:
                    st.markdown(f"**Model:** {v['metadata'].get('model', 'Unknown')}")
                    st.json(v["response"])
                    if st.button(f"Keep this (v{i})", key=f"keep_{v['file_path']}"):
                        # Manual pick logic
                        for other in variants:
                            if other != v:
                                os.remove(other["file_path"])
                        st.success("Cleaned!")
                        st.rerun()
