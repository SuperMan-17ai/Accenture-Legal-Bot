import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

def build_vector_db():
    pdf_path = "Accenture_FY23_10K.pdf"
    db_path = "./qdrant_db"
    collection_name = "accenture_10k"

    # 1. Check if file exists (it should now, since we uploaded it)
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found in the repository!")
        return

    # 2. Load and Split
    print("📄 Processing Local PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        docs = loader.load_and_split(text_splitter)
    except Exception as e:
        print(f"❌ PDF Reading Error: {e}")
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