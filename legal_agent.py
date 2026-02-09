import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Fallback if .env fails
if not GROQ_API_KEY:
    GROQ_API_KEY = "❌ Error: API Key missing."

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "accenture_10k"

# Initialize Global Clients (Stateless ones are fine)
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# NOTE: We removed qdrant_client from here to avoid file locking errors!

def get_answer(query):
    """
    1. Embeds query
    2. Searches Qdrant (Opens connection ONLY now)
    3. Sends to Groq Llama-3
    """
    print(f"🤖 Processing query: {query}")
    
    try:
        # 1. Initialize Qdrant Client JUST IN TIME
        # This prevents "Storage folder already accessed" errors
        client = QdrantClient(path=DB_PATH)
        
        # 2. Search
        query_vector = list(embedding_model.embed([query]))[0]
        
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3 
        )
        
        # Access the list of hits via .points
        hits = search_result.points
        
        # Close the client connection explicitly (Good practice)
        client.close()
        
        # 3. Context Building
        source_chunks = [hit.payload['text'] for hit in hits]
        safe_context = "\n\n".join(source_chunks)[:2500]

        # 4. Generation
        prompt = f"""
        You are a Legal AI Assistant. Answer based strictly on the provided context.
        
        CONTEXT:
        {safe_context}
        
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