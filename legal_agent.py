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
        # Connect to DB
        client = QdrantClient(path=DB_PATH)
        
        # High-level query: automatically handles vectorizing the 'query_text'
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=3 
        )
        client.close()
        
        # Extract documents from hits
        source_chunks = [hit.document for hit in search_result]
        context_text = "\n\n".join(source_chunks)[:2000]

        # Call Groq
        prompt = f"""
        You are a legal expert. Use the following context from Accenture's 10-K to answer the question.
        If the answer isn't in the context, say you don't know.
        
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