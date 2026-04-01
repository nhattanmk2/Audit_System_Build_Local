import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

def test_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    results = ["# Groq API Test Results\n"]
    
    if not api_key:
        results.append("> [!CAUTION]\n> Error: GROQ_API_KEY not found in .env file.")
        with open("groq_test_results.md", "w", encoding="utf-8") as f:
            f.writelines(results)
        return

    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    for model_name in models:
        results.append(f"## Model: `{model_name}`")
        try:
            llm = ChatGroq(
                temperature=0,
                model_name=model_name,
                groq_api_key=api_key
            )
            
            response = llm.invoke("Hello, what is your model name and who created you? Answer in 1 short sentence.")
            results.append(f"**Response:** {response.content}\n")
        except Exception as e:
            results.append(f"> [!WARNING]\n> Error testing {model_name}: {str(e)}\n")

    with open("groq_test_results.md", "w", encoding="utf-8") as f:
        f.writelines([r + "\n" for r in results])

if __name__ == "__main__":
    test_groq_models()
