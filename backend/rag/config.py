# rag/config.py
"""
RAG Configuration for Speech Analysis Pipeline.

Uses ChromaDB with its built-in sentence-transformers embeddings
(all-MiniLM-L6-v2) for fast local vector retrieval.
No external API needed for embeddings.
"""

import os

# ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "speech_analysis_knowledge"

# Retrieval settings
TOP_K_RESULTS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
