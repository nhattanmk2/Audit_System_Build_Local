import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"API KEY (first 10): {api_key[:10] if api_key else 'None'}")
genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    with open("models.txt", "w") as f:
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                f.write(f"{m.name}\n")
    print("Models written to models.txt")
except Exception as e:
    with open("models.txt", "w") as f:
        f.write(f"ERROR: {str(e)}")
    print("Error written to models.txt")
