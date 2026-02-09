import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Modern import path
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# ... rest of the code stays the same ...