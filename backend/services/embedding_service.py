"""
Embedding service for generating vector embeddings from text.
Uses multilingual models to support Arabic and English.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model (defaults to config)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        # The embedding model is multilingual so queries in Arabic and English map into the same vector space.
        
        # Load the embedding model (downloads on first use; cached by SentenceTransformers afterwards).
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Cache embedding dimension (needed to initialize FAISS and validate vector shapes).
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embedding vector
        """
        # `convert_to_numpy=True` is important because FAISS expects numpy float arrays.
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient).
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100  # Avoid noisy progress for small batches.
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 384, 768)
        """
        return self.embedding_dim

