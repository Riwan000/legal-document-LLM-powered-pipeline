"""
EmbeddingService
----------------

Pure embedding service for generating vector embeddings from text.

EmbeddingService guarantees:
- Multilingual embeddings in a shared vector space (Arabic + English)
- Unit-normalized output vectors (L2 norm = 1.0 for non-empty inputs)
- Deterministic embeddings (model in eval mode)
- FAISS-compatible shapes and dtypes (float32)
- No ranking, retrieval, or chunking logic

Important Notes:
- Text longer than the model's maximum token length is truncated by SentenceTransformers.
- Clause and document chunking MUST be handled upstream (e.g., ingestion/chunking services).
- This service does not perform chunking or splitting.
"""
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings


class EmbeddingService:
    """
    Service for generating deterministic, unit-normalized multilingual embeddings.

    Responsibilities:
    - Transform text → embeddings only.

    Non-responsibilities (must NOT be added here):
    - Ranking or retrieval logic
    - Query intent or legal heuristics
    - Chunking or splitting of long texts
    - Dynamic model switching
    """
    
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
        # Enforce eval mode for deterministic embeddings.
        self.model.eval()
        
        # Cache embedding dimension (needed to initialize FAISS and validate vector shapes).
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

        # Optional in-memory cache for repeated texts (best-effort, per-process only).
        # Cache key is the sanitized text. No eviction policy yet; can be extended with LRU if needed.
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text before embedding.

        - Ensures text is a string
        - Strips leading/trailing whitespace
        - Returns empty string for invalid input
        """
        if not isinstance(text, str):
            return ""
        return text.strip()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length (L2 norm).

        This is required for stable cosine similarity and FAISS scoring.
        Works for both 1D (single vector) and 2D (batch) inputs.
        """
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.clip(norms, 1e-10, None)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embedding vector (unit-normalized for non-empty text,
            zero vector for empty/invalid text)
        """
        # Sanitize text
        text = self._sanitize_text(text)

        # Empty or invalid text → zero vector (preserve shape, safe for FAISS)
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Optional per-process cache (best-effort)
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        # `convert_to_numpy=True` is important because FAISS expects numpy float arrays.
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Ensure float32 dtype
        embedding = embedding.astype(np.float32, copy=False)
        embedding = self._normalize(embedding)

        self._embedding_cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient).
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Sanitize inputs
        sanitized_texts = [self._sanitize_text(t) for t in texts]

        # If all texts are empty/invalid, return all-zero matrix
        if all(not t for t in sanitized_texts):
            return np.zeros((len(sanitized_texts), self.embedding_dim), dtype=np.float32)

        # Prepare indices for non-empty texts
        non_empty_indices = [i for i, t in enumerate(sanitized_texts) if t]
        non_empty_texts = [sanitized_texts[i] for i in non_empty_indices]

        # Encode only non-empty texts
        embeddings_non_empty = self.model.encode(
            non_empty_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(non_empty_texts) > 100  # Avoid noisy progress for small batches.
        )
        embeddings_non_empty = embeddings_non_empty.astype(np.float32, copy=False)
        embeddings_non_empty = self._normalize(embeddings_non_empty)

        # Allocate full result and fill with zeros by default
        result = np.zeros((len(sanitized_texts), self.embedding_dim), dtype=np.float32)

        # Scatter non-empty embeddings back to their original positions
        for idx, emb in zip(non_empty_indices, embeddings_non_empty):
            result[idx] = emb

        return result
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 384, 768)
        """
        return self.embedding_dim