"""
Retrieval module for Legal Assistant RAG Chatbot.

Provides vector search functionality with optional metadata filtering
and reranking capabilities. Returns documents with metadata and distances
for provenance tracking.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

from src.config import Config
from src.embeddings_ import embed_query


class LegalRetriever:
    """Handles retrieval of relevant legal documents from ChromaDB."""
    
    def __init__(self):
        """Initialize the retriever with ChromaDB client."""
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(
                name=Config.CHROMA_COLLECTION_NAME
            )
            print(f"Retriever initialized. Collection has {self.collection.count()} documents.")
        except Exception as e:
            print(f"ERROR: Could not load collection: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if top_k is None:
            top_k = Config.DEFAULT_TOP_K
        
        query_embedding = embed_query(query)
        
        where = None
        if filters:
            where = {k: v for k, v in filters.items() if v is not None}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
        
        return formatted_results


def get_retriever() -> LegalRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LegalRetriever()
    return _retriever_instance


_retriever_instance = None
