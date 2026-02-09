import streamlit as st
import os
from qdrant_client import QdrantClient
from legal_agent import get_answer 
# We import build_vector_db inside the function to avoid circular import issues
import setup_db 

# Page Config
st.set_page_config(page_title="Accenture Legal Bot", layout="centered")

# --- ROBUST DATABASE CHECK ---
def ensure_database_ready():
    """
    Tries to get the collection. If it fails or is empty, it rebuilds.
    """
    db_path = "./qdrant_db"
    collection_name = "accenture_10k"
    
    # 1. Initialize Client
    client = QdrantClient(path=db_path)
    
    try:
        # Check if collection exists and has items
        client.get_collection(collection_name)
        # Verify it's not empty (optional but good)
        info = client.get_collection(collection_name)
        if info.points_count == 0:
            raise Exception("Collection empty")
            
    except Exception:
        # If ANY error happens (Not Found, Empty, etc.), we build.
        with st.spinner("🚀 Building Database... (This takes ~45 seconds)"):
            setup_db.build_vector_db()
            st.success("Database built! Reloading...")
            st.rerun() # Force a reload to use the new DB

# Run this check FIRST, before anything else
ensure_database_ready()

# --- APP INTERFACE ---
st.title("⚖️ Accenture 10-K Intelligence")
st.markdown("### AI-Powered Legal Analysis")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle User Input
if prompt := st.chat_input("Ask about risks, revenue, or strategy..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 10-K Document..."):
            try:
                response, sources = get_answer(prompt)
                st.markdown(response)
                
                # Source Expander
                with st.expander("📚 View Source Documents"):
                    for i, source in enumerate(sources):
                        st.info(f"**Reference {i+1}:** {source[:300]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"System Error: {str(e)}")