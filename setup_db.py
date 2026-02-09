import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

def build_vector_db():
    pdf_path = "Accenture_FY23_10K.pdf"
    db_path = "./qdrant_db"
    collection_name = "accenture_10k"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found!")
        return

    # 1. Load and Split
    print("📄 Processing PDF...")
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter)
    texts = [doc.page_content for doc in docs]

    # 2. Initialize and Upload
    # client.add() handles the embedding model internally
    client = QdrantClient(path=db_path)
    
    print("🧠 Creating Vectors and Saving to Disk...")
    client.add(
        collection_name=collection_name,
        documents=texts,
        metadata=[{"source": "Accenture 10-K FY23"} for _ in texts]
    )
    
    print("✅ Database built successfully!")
    client.close()

if __name__ == "__main__":
    build_vector_db()