from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
chatmodel=GoogleGenerativeAI(model='gemini-1.5-pro')
result=chatmodel.invoke("what is the capital of pakistan")

print(result)