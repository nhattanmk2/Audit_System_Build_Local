import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv('E:/CIS_Audit_Project/.env')
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

results = []
models_to_test = ['gemini-flash-latest', 'gemini-3-flash-preview', 'gemini-2.5-flash-lite']

for m in models_to_test:
    successes = 0
    fail_reason = ""
    for i in range(25):
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content('Hi')
            successes += 1
        except Exception as e:
            fail_reason = f"Failed at {i} with {str(e)[:200]}"
            break
    results.append(f"{m}: {successes}/25 - {fail_reason}")

with open("E:/CIS_Audit_Project/quota_loop.txt", "w") as f:
    f.write("\n".join(results))
