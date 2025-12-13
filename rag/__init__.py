# RAG Module - Retrieval Augmented Generation for Speech Analysis
# Uses ChromaDB for vector storage and Ollama for embeddings

from rag.knowledge_base import KnowledgeBase
from rag.retriever import RAGRetriever

__all__ = ["KnowledgeBase", "RAGRetriever"]
