import os
from  langchain_community.embeddings import HuggingFaceHubEmbeddings
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()


hf_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure API key is available
if not hf_api_key:
    raise ValueError("Hugging Face API key not found. Please check your .env file.")

# Initialize Hugging Face Embeddings Model
embeddings = HuggingFaceHubEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Choose an embedding model
    huggingfacehub_api_token=hf_api_key
)

# Example Text to Embed
text = "What is the capital of Pakistan?"
vector = embeddings.embed_query(text)

# Print the Embedding Vector
print(vector)