import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

load_dotenv()

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "accenture_10k"

# Global Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource(show_spinner="🧠 Loading AI Search Engine...")
def load_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_answer(query):
    try:
        # Load Cached Model
        model = load_embedding_model()
        
        # Connect to DB (Lazy load)
        client = QdrantClient(path=DB_PATH)
        
        # Search
        query_vector = list(model.embed([query]))[0]
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3 
        )
        client.close()
        
        # Context Extraction
        source_chunks = [hit.payload['document'] for hit in search_result.points]
        context_text = "\n\n".join(source_chunks)[:2000]

        # LLM Answer
        prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer as a legal expert:"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0,
        )
        
        return chat_completion.choices[0].message.content, source_chunks
    except Exception as e:
        return f"System Error: {str(e)}", []