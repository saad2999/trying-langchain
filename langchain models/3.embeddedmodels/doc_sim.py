import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load API key from .env
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure API key is available
if not hf_api_key:
    raise ValueError("Hugging Face API key not found. Please check your .env file.")

# Initialize Hugging Face Embeddings Model (Corrected)
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",  # Corrected parameter
    huggingfacehub_api_token=hf_api_key
)

documents = [
    "Wasim Akram – Known as the 'Sultan of Swing,' Wasim Akram was one of the greatest fast bowlers in cricket history and played a key role in Pakistan's 1992 World Cup victory.",
    "Javed Miandad – Famous for his last-ball six against India in the 1986 Austral-Asia Cup final, Javed Miandad was Pakistan's most dependable batsman in the 1980s.",
    "Imran Khan – The charismatic leader of the 1992 World Cup-winning team, Imran Khan was an exceptional all-rounder who later became Pakistan's Prime Minister.",
    "Inzamam-ul-Haq – A stylish middle-order batsman, Inzamam-ul-Haq was known for his calm demeanor and match-winning performances, especially in ODI cricket.",
    "Waqar Younis – A master of reverse swing, Waqar Younis formed a lethal bowling duo with Wasim Akram and was known for his toe-crushing yorkers."
]

query = "Wasim Akram was one of the greatest fast bowlers in cricket"

# Generate embeddings
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]


# Print the most relevant document  

print(f"The most relevant document to the given query is: {documents[index]}")
print(f'simierity score is {(score*100)}%')
