import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv('E:/CIS_Audit_Project/.env')
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

print('Testing models...')
for model_name in ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro']:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content('Hello')
        print(f'{model_name}: Success, returned {len(response.text)} chars')
    except Exception as e:
        print(f'{model_name}: Error {type(e).__name__} - {str(e)}')
