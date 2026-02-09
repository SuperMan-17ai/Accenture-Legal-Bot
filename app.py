import streamlit as st
import os
from legal_agent import get_answer
import setup_db 

st.set_page_config(page_title="Accenture Legal Bot", layout="centered")

# Ensure DB exists
@st.cache_resource
def startup_check():
    if not os.path.exists("./qdrant_db"):
        setup_db.build_vector_db()
    return True

startup_check()

st.title("⚖️ Accenture 10-K Intelligence")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching filing..."):
            response, sources = get_answer(prompt)
            st.markdown(response)
            
            with st.expander("📚 Source Evidence"):
                for s in sources:
                    st.info(f"{s[:500]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": response})