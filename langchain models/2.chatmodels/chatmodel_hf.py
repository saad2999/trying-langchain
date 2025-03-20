import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API Key
huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure API key is set correctly
if not huggingface_api_key:
    raise ValueError("Hugging Face API key not found. Check your .env file.")

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=huggingface_api_key  # Pass API key
)

# Use ChatHuggingFace
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of Pakistan?")

# Print Result
print(result.content)
