from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import json

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-pro')

class Review(TypedDict):
    summary: str
    sentiment: str

# Manually prompt for structured JSON output
response = model.invoke("""
Provide a JSON response with the following structure:
{
    "summary": "<summary of the review>",
    "sentiment": "<positive, negative, or neutral>"
}

Review:
"The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can’t remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."
""")


# Parse the JSON response
try:
    result = json.loads(response)
    print(result)
except json.JSONDecodeError:
    print("Failed to parse JSON:", response)
