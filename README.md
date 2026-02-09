# ⚖️ Accenture Legal Intelligence Bot

A RAG (Retrieval-Augmented Generation) application that analyzes the Accenture FY23 10-K Annual Report to answer financial and strategic questions.

## 🛠️ Tech Stack
* **LLM:** Llama-3.1-8b (via Groq)
* **Vector DB:** Qdrant
* **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5)
* **Framework:** LangChain & Streamlit

## 🚀 Features
* **RAG Architecture:** Retrieves precise context from the 10-K PDF.
* **Source Citation:** Displays exact text chunks used for the answer.
* **Optimized Performance:** Handles API rate limits with smart context slicing.

## 💻 How to Run
1.  Clone the repo.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Add your Groq API key to a `.env` file.
4.  Run the app: `streamlit run app.py`