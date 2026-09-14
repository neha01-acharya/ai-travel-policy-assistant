import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )


# ============================================================
# CREATE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# ============================================================
# TEST GEMINI
# ============================================================

response = llm.invoke(
    "Explain RAG in one simple sentence."
)


# ============================================================
# DISPLAY RESPONSE
# ============================================================

print("\nGemini response:")
print(response.content)