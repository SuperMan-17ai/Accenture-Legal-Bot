import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "accenture_10k"

groq_client = Groq(api_key=GROQ_API_KEY)

def get_answer(query):
    try:
        client = QdrantClient(path=DB_PATH)
        
        # INCREASED LIMIT: From 3 to 5 to capture more context
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=5 
        )
        client.close()
        
        source_chunks = [hit.document for hit in search_result]
        context_text = "\n\n".join(source_chunks)[:3000] # Increased context window

        prompt = f"""
        You are a Legal AI Expert. Answer the question strictly using the context below.
        If the information is missing, say it's not in the document.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {query}
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0,
        )
        
        return chat_completion.choices[0].message.content, source_chunks
    except Exception as e:
        return f"System Error: {str(e)}", []