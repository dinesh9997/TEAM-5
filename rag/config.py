# rag/config.py
"""
RAG Configuration - Uses existing Ollama setup
"""

import os

# ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "speech_analysis_knowledge"

# Embedding model (using Ollama's embedding capability)
EMBEDDING_MODEL = "nomic-embed-text"  # Lightweight embedding model for Ollama

# Retrieval settings
TOP_K_RESULTS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
