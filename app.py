import streamlit as st
import os
from legal_agent import get_answer
import setup_db 

st.set_page_config(page_title="Accenture Legal Bot", layout="centered")

# --- STARTUP CHECK ---
@st.cache_resource
def startup_logic():
    if not os.path.exists("./qdrant_db"):
        with st.spinner("🚀 Building Knowledge Base..."):
            setup_db.build_vector_db()
    return True

startup_logic()

# --- UI ---
st.title("⚖️ Accenture 10-K Intelligence")
st.markdown("### AI-Powered SEC Filing Analysis")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about Accenture..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            response, sources = get_answer(prompt)
            st.markdown(response)
            
            with st.expander("📚 Source Evidence"):
                for s in sources:
                    st.info(f"...{s[:400]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": response})