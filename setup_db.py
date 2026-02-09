import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

def build_vector_db():
    pdf_url = "https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/Accenture-Plc-2023-10-K.pdf"
    pdf_path = "Accenture_FY23_10K.pdf"
    db_path = "./qdrant_db"
    collection_name = "accenture_10k"

    # FIX: Delete the file if it's tiny/broken so we can re-download
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) < 1000:
        os.remove(pdf_path)

    # 1. Download PDF with headers (to avoid being blocked)
    if not os.path.exists(pdf_path):
        print("⬇️ Downloading PDF...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers)
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    # 2. Load and Split
    print("📄 Processing PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        docs = loader.load_and_split(text_splitter)
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        if os.path.exists(pdf_path): os.remove(pdf_path) # Clean up broken file
        return

    # 3. Initialize Embeddings & Database
    print("🧠 Initializing Database...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(path=db_path)
    
    texts = [doc.page_content for doc in docs]
    client.add(collection_name=collection_name, documents=texts)
    
    print("✅ Database built successfully!")
    client.close()

if __name__ == "__main__":
    build_vector_db()