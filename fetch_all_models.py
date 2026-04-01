import os
import google.generativeai as genai
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def fetch_gemini_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found.")
        return []
    
    try:
        genai.configure(api_key=api_key)
        # Fetch models that support 'generateContent'
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_id = m.name.replace('models/', '')
                models.append(f"gemini:{model_id}")
        return models
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
        return []

def fetch_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found.")
        return []
    
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            return [f"groq:{m['id']}" for m in data['data']]
        else:
            print(f"Error fetching Groq models: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        print(f"Error fetching Groq models: {e}")
        return []

def main():
    print("Fetching models from providers...")
    gemini_models = fetch_gemini_models()
    groq_models = fetch_groq_models()
    
    all_models = sorted(list(set(gemini_models + groq_models)))
    
    if all_models:
        with open("models.txt", "w", encoding="utf-8") as f:
            for m in all_models:
                f.write(f"{m}\n")
        print(f"Successfully saved {len(all_models)} models to models.txt")
    else:
        print("No models found. models.txt was not updated.")

if __name__ == "__main__":
    main()
