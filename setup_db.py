import os
import shutil
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

# --- CONFIGURATION ---
PDF_URL = "https://www.accenture.com/content/dam/accenture/final/capabilities/corporate-functions/marketing-and-communications/marketing---communications/document/Accenture-2023-10-K.pdf"
PDF_FILE = "Accenture_FY23_10K.pdf"
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "accenture_10k"

def build_vector_db():
    print("🚀 STARTING PROFESSIONAL SETUP...")
    
    # 1. CLEANUP: Remove old DB to prevent corruption
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🗑️ Cleaned up old database.")

    # 2. DOWNLOAD PDF (If missing)
    if not os.path.exists(PDF_FILE):
        print(f"⬇️ Downloading {PDF_FILE}...")
        response = requests.get(PDF_URL)
        with open(PDF_FILE, 'wb') as f:
            f.write(response.content)
    
    # 3. PROCESS PDF
    print("📄 Processing PDF...")
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    # 4. EMBEDDING MODEL
    print("🧠 Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 5. CREATE QDRANT DB
    client = QdrantClient(path=DB_PATH)
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print("💾 Collection created.")

    # 6. INSERT DATA
    print("⚗️ Embedding and Indexing (This takes ~30s)...")
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    # Embed all texts
    embeddings = list(embedding_model.embed(texts))
    
    # Prepare for upload
    points = [
        models.PointStruct(id=idx, vector=emb, payload={"text": txt, "source": meta})
        for idx, (txt, emb, meta) in enumerate(zip(texts, embeddings, metadatas))
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ SUCCESS! Database ready at {DB_PATH}")

if __name__ == "__main__":
    build_vector_db()