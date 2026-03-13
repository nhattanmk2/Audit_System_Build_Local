import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv('E:/CIS_Audit_Project/.env')
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

test_models = [
    'gemini-flash-latest', 
    'gemini-2.5-flash', 
    'gemini-2.5-flash-lite', 
    'gemini-3-flash-preview', 
    'gemini-2.0-flash-lite',
    'gemini-pro-latest'
]

results = []
for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content('Hello')
        results.append(f'{model_name}: Success, returned {len(response.text)} chars')
    except Exception as e:
        results.append(f'{model_name}: Error {type(e).__name__} - {str(e)[:250]}')

with open("E:/CIS_Audit_Project/quota_results.txt", "w") as f:
    f.write("\n\n".join(results))
