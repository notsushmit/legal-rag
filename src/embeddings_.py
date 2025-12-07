"""
Embeddings module for legal-rag-vibe.

Wrapper for sentence-transformers embeddings model.
Provides a simple interface to load the model and generate embeddings
for text chunks during ingestion and query time.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from src.config import Config


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to Config.EMBEDDING_MODEL.
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Embedding model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Convert numpy arrays to lists for ChromaDB compatibility
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
        
        Returns:
            Embedding vector as a list of floats
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


# Global instance for reuse
_embedding_model_instance = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the global embedding model instance.
    
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel()
    return _embedding_model_instance


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Convenience function to embed texts using the global model instance.
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding
    
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    return model.embed_texts(texts, batch_size=batch_size)


def embed_query(query: str) -> List[float]:
    """
    Convenience function to embed a query using the global model instance.
    
    Args:
        query: Query text to embed
    
    Returns:
        Embedding vector
    """
    model = get_embedding_model()
    return model.embed_query(query)
