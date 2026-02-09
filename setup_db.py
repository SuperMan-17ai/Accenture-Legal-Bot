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

    # 1. Download PDF
    if not os.path.exists(pdf_path):
        print("⬇️ Downloading PDF...")
        response = requests.get(pdf_url)
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    # 2. Load and Split
    print("📄 Processing PDF...")
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)

    # 3. Initialize Embeddings
    print("🧠 Initializing FastEmbed...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 4. Create Qdrant Collection
    client = QdrantClient(path=db_path)
    
    texts = [doc.page_content for doc in docs]
    
    # FastEmbed 'add' method handles embedding and storage in one go
    client.add(
        collection_name=collection_name,
        documents=texts,
        metadata=[{"source": "Accenture 10-K FY23"} for _ in texts],
    )
    
    print("✅ Database built successfully!")
    client.close()

if __name__ == "__main__":
    build_vector_db()