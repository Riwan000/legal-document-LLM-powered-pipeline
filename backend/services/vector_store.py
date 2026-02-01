"""
VectorStore
-----------

FAISS-based vector store for storing and retrieving document embeddings.

VectorStore guarantees:
- Exact similarity search using FAISS IndexFlatL2
- Assumes unit-normalized float32 embeddings from EmbeddingService
- Supports weighted priority boosting for clause-aware ranking
- Never generates embeddings
- Never applies legal or business logic
- Never performs chunking or splitting
"""
import logging
import pickle
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from backend.config import settings
from backend.models.document import DocumentChunk

# Expected metadata schema for ranking (documentation)
REQUIRED_RANKING_METADATA = {
    "clause_types": list,
    "hierarchy_level": str,
}

class _NumpyIndex:
    """
    Minimal FAISS-like index for offline/unit tests.

    Stores normalized vectors and supports exact L2 search.
    Returns squared L2 distances (to match FAISS IndexFlatL2 behavior).
    """

    def __init__(self, d: int):
        self.d = int(d)
        self._vectors = np.empty((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"Vector dimension mismatch: expected (*, {self.d}), got {x.shape}")
        x = x.astype(np.float32, copy=False)
        self._vectors = np.vstack([self._vectors, x])

    def search(self, q: np.ndarray, k: int):
        if self.ntotal == 0:
            distances = np.empty((1, 0), dtype=np.float32)
            indices = np.empty((1, 0), dtype=np.int64)
            return distances, indices

        if q.ndim != 2 or q.shape[1] != self.d:
            raise ValueError(f"Query dimension mismatch: expected (1, {self.d}), got {q.shape}")
        q = q.astype(np.float32, copy=False)

        # Squared L2 distances for all vectors
        diffs = self._vectors - q[0]
        dists = np.sum(diffs * diffs, axis=1)
        k = min(int(k), self.ntotal)
        # argsort ascending (closest first)
        idx = np.argsort(dists)[:k]
        return dists[idx].reshape(1, -1), idx.astype(np.int64).reshape(1, -1)

    def dump_vectors(self) -> np.ndarray:
        return self._vectors

    def load_vectors(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.d:
            raise ValueError(f"Loaded vectors dimension mismatch: expected (*, {self.d}), got {vectors.shape}")
        self._vectors = vectors


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        
        # We use an exact FAISS index with L2 distance.
        # Important: we normalize vectors to unit length, so L2 distance can be converted to cosine similarity.
        # Prefer a pure-numpy index by default (reliable in offline/unit-test environments).
        # Opt into FAISS explicitly via USE_FAISS=1.
        self._faiss = None
        if os.environ.get("USE_FAISS", "").lower() in {"1", "true", "yes"}:
            import faiss as _faiss
            self._faiss = _faiss
            self.index = _faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = _NumpyIndex(embedding_dim)
        
        # Per-vector metadata aligned by index position (self.metadata[i] describes vector i).
        self.metadata: List[Dict[str, Any]] = []
        
        # Ensure vector store directory exists for persistence.
        settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    
    def add_chunks(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk],
        display_name: Optional[str] = None,
        document_hash: Optional[str] = None
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            chunks: List of DocumentChunk objects corresponding to embeddings
            display_name: User-friendly display name for citations (internal)
            document_hash: SHA-256 hash of document content (internal, never exposed)
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks")
        
        # Enforce float32 and validate embedding contract (unit-normalized from EmbeddingService)
        embeddings = embeddings.astype('float32', copy=False)
        
        # Validate embedding dimension
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        
        # Normalize embeddings to unit length (so L2 distances map to cosine similarity).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        
        # FAISS expects float32 arrays. Embeddings should already be normalized.
        self.index.add(embeddings)
        
        # Persist the fields we need to display citations and support filtering/reranking.
        for chunk in chunks:
            chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
            if chunk.clause_id:
                chunk_metadata['clause_id'] = chunk.clause_id
            
            # Store internal metadata (document_hash, display_name) for versioning and citations
            # These are used internally but never exposed in API responses
            metadata_entry = {
                'document_id': chunk.document_id,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'text': chunk.text,
                **chunk_metadata  # Include any additional metadata (hierarchy_level, clause_types, etc.)
            }
            
            # Add internal-only fields (never exposed to clients)
            if display_name:
                metadata_entry['display_name'] = display_name
            if document_hash:
                metadata_entry['document_hash'] = document_hash
            
            self.metadata.append(metadata_entry)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = -1.0  # Use -1.0 as sentinel for "use default"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return (defaults to config)
            document_id_filter: Optional filter by document ID
            similarity_threshold: Similarity threshold (defaults to config, None to bypass, -1.0 to use default)
            
        Returns:
            List of dicts with keys: text, page_number, document_id, chunk_index, score
        """
        if self.index.ntotal == 0:
            return []

        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        if top_k <= 0:
            return []
        
        # Threshold behavior:
        # - similarity_threshold == -1.0  => use config default
        # - similarity_threshold is None  => bypass threshold entirely
        # - otherwise                    => use provided value
        if similarity_threshold == -1.0:
            threshold = settings.SIMILARITY_THRESHOLD  # Use default
        elif similarity_threshold is None:
            threshold = None  # Bypass threshold
        else:
            threshold = similarity_threshold  # Use provided value
        
        # Enforce float32 and validate query embedding contract (unit-normalized from EmbeddingService)
        query_embedding = query_embedding.astype('float32', copy=False)
        
        # Validate query embedding dimension
        if query_embedding.shape[0] != self.embedding_dim:
            logging.warning(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[0]}"
            )
        
        # Normalize query embedding to unit length (to match stored vectors).
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []
        query_embedding = query_embedding / query_norm
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Exact search in FAISS (IndexFlatL2).
        # IMPORTANT: FAISS IndexFlatL2 returns *squared* L2 distances.
        # For unit-normalized vectors: cosine_similarity = 1 - (||x-y||^2 / 2)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert squared L2 distances to cosine similarities.
        # (Do NOT square again; that causes overflow and incorrect filtering.)
        similarities = 1 - (distances[0] / 2)
        
        results = []
        for idx, (distance, similarity) in enumerate(zip(distances[0], similarities)):
            metadata = self.metadata[indices[0][idx]]
            
            if document_id_filter and metadata['document_id'] != document_id_filter:
                continue
            
            # OCR text is often noisier, so we optionally allow a lower similarity threshold.
            is_ocr = metadata.get('is_ocr', False)
            
            # Backward-compat fallback: old indices may not have `is_ocr`; detect via common OCR artifacts.
            if not is_ocr:
                text = metadata.get('text', '')
                ocr_indicators = [
                    'Oy ether', 'l€rms', 'agreemeni', 'authort.es',  # Common OCR errors
                    'Nolce', 'terminateg', 'jne', 'Dy a',  # Notice period OCR errors
                ]
                if any(indicator in text for indicator in ocr_indicators):
                    is_ocr = True
            
            # Choose an effective threshold:
            # - default threshold can be lowered for OCR chunks
            # - the explicit MIN_SIMILARITY_THRESHOLD is already permissive and should not be lowered further
            if threshold is not None:
                effective_threshold = threshold
                if threshold == settings.SIMILARITY_THRESHOLD and is_ocr:
                    effective_threshold = settings.OCR_SIMILARITY_THRESHOLD
                elif threshold == settings.MIN_SIMILARITY_THRESHOLD:
                    effective_threshold = settings.MIN_SIMILARITY_THRESHOLD
            else:
                effective_threshold = None  # Bypass threshold
            
            if effective_threshold is not None and similarity < effective_threshold:
                continue
            
            # Build result dict - explicitly exclude internal-only fields
            result_entry = {
                'text': metadata['text'],
                'page_number': metadata['page_number'],
                'document_id': metadata['document_id'],
                'chunk_index': metadata['chunk_index'],
                'score': float(similarity),
                'distance': float(distance),
                'is_ocr': is_ocr,  # Include detected OCR flag
            }
            
            # Include display_name for citations (user-facing)
            if 'display_name' in metadata:
                result_entry['display_name'] = metadata['display_name']
            
            # Include other metadata except internal-only fields
            excluded_fields = {
                'text', 'page_number', 'document_id', 'chunk_index',
                'document_hash', 'chunk_id'  # Never expose hashes or internal chunk IDs
            }
            for k, v in metadata.items():
                if k not in excluded_fields:
                    result_entry[k] = v
            
            results.append(result_entry)
        
        return results
    
    def search_with_priority(
        self,
        query_embedding: np.ndarray,
        priority_weights: Dict[str, float] = None,
        # Backward-compat: older callers pass this keyword.
        priority_clause_types: Dict[str, float] = None,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = -1.0
    ) -> List[Dict[str, Any]]:
        """
        Search with weighted priority boosting for specific clause types.
        
        Uses weighted sum instead of binary matching: clauses matching multiple query types
        get cumulative weights, avoiding "everything is equally relevant" syndrome.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            priority_weights: Dict mapping clause_type → weight for weighted boosting
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            similarity_threshold: Similarity threshold
            
        Returns:
            List of results with boosted scores for priority clause types
        """
        if top_k is not None and top_k <= 0:
            return []

        # First get standard search results
        results = self.search(
            query_embedding=query_embedding,
            top_k=(top_k * 2) if top_k is not None else (settings.TOP_K_RESULTS * 2),  # Get more results for re-ranking
            document_id_filter=document_id_filter,
            similarity_threshold=similarity_threshold
        )
        
        if priority_weights is None and priority_clause_types is not None:
            priority_weights = priority_clause_types

        if not priority_weights:
            # No priority weights, return standard results
            return results[:top_k or settings.TOP_K_RESULTS]
        
        # Normalize keys to lowercase once (callers may use "Governing Law", etc.)
        normalized_priority_weights = {}
        for k, v in (priority_weights or {}).items():
            if not isinstance(k, str):
                continue
            try:
                normalized_priority_weights[k.lower()] = float(v)
            except Exception:
                continue

        # Boosting rules:
        # - priority_score is cumulative over matching clause types
        # - boost_factor scales the influence of priority over base similarity
        # - final score is capped at 1.0 to preserve valid similarity range
        boost_factor = 0.15  # Base boost factor
        
        for result in results:
            # Only boost reasonably good base matches (prevents weak matches from dominating)
            base_score = result.get("score", 0.0)
            if base_score < settings.MIN_BASE_SIMILARITY_FOR_BOOST:
                continue
            
            # Metadata is already spread into result dict, not nested
            # Harden metadata access (graceful degradation if missing/malformed)
            clause_types = result.get("clause_types") or []
            if not isinstance(clause_types, list):
                clause_types = []
            
            hierarchy_level = result.get("hierarchy_level", "contract")
            
            # Calculate priority score as sum of weights for matching clause types
            priority_score = sum(
                normalized_priority_weights.get(clause_type.lower(), 0.0)
                for clause_type in clause_types
            )
            
            # Also check hierarchy level for "Governing Law" type
            if "governing law" in normalized_priority_weights and hierarchy_level == "law":
                priority_score += normalized_priority_weights.get("governing law", 0.0)
            
            if priority_score > 0:
                # Apply weighted boost (clauses matching multiple types get higher boost)
                result['score'] = min(1.0, result['score'] + priority_score * boost_factor)
                result['priority_boosted'] = True
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k results
        return results[:top_k or settings.TOP_K_RESULTS]
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID to filter by
            
        Returns:
            List of chunk metadata dicts
        """
        return [
            metadata for metadata in self.metadata
            if metadata['document_id'] == document_id
        ]
        # Filter metadata by document_id
    
    def update_chunk_metadata(
        self,
        document_id: str,
        chunk_updates: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Update metadata for chunks (e.g., backfill chunk_id, chunk_type).
        
        Args:
            document_id: Document ID to filter by
            chunk_updates: Dict mapping chunk_id to dict of metadata updates
            
        Returns:
            Number of chunks updated
        """
        updated_count = 0
        for metadata in self.metadata:
            if metadata['document_id'] == document_id:
                chunk_id = metadata.get('chunk_id')
                if chunk_id and chunk_id in chunk_updates:
                    updates = chunk_updates[chunk_id]
                    metadata.update(updates)
                    updated_count += 1
        
        return updated_count
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Deletion is not safely supported with IndexFlatL2.
        In production, use ID-mapped indices (e.g., IndexIDMap) or rebuild the entire
        index with a controlled pipeline.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted (raises NotImplementedError)
        """
        raise NotImplementedError(
            "Vector deletion is not safely supported with IndexFlatL2. "
            "Use FAISS ID-mapped indices (IndexIDMap) or rebuild the index in a "
            "controlled pipeline."
        )
    
    def save(self, filepath: Path = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save index (defaults to config)
        """
        filepath = filepath or settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
        
        # Save index
        if self._faiss is not None:
            self._faiss.write_index(self.index, str(filepath))
        else:
            # Persist vectors for numpy index
            with open(filepath, "wb") as f:
                pickle.dump({"version": 1, "vectors": self.index.dump_vectors()}, f)
        
        # Save metadata with versioning for safe future migrations
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(
                {
                    "version": 1,
                    "metadata": self.metadata,
                },
                f
            )
    
    def load(self, filepath: Path = None) -> None:
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load index from (defaults to config)
        """
        filepath = filepath or settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vector store not found: {filepath}")
        
        # Load index
        if self._faiss is not None:
            self.index = self._faiss.read_index(str(filepath))
        else:
            with open(filepath, "rb") as f:
                payload = pickle.load(f)
            vectors = payload.get("vectors")
            self.index = _NumpyIndex(self.embedding_dim)
            self.index.load_vectors(vectors)
        
        # Load metadata with version checking
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            payload = pickle.load(f)
        
        # Check version for safe migrations
        version = payload.get("version", 0)
        if version != 1:
            error_msg = (
                f"Unsupported vector store metadata version: {version}. "
                "Implement a migration or rebuild the index."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        self.metadata = payload["metadata"]
        
        # Update embedding dimension from loaded index
        self.embedding_dim = self.index.d
        # FAISS index stores dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            # Number of vectors in index
            'embedding_dimension': self.embedding_dim,
            # Dimension of embeddings
            'unique_documents': len(set(m['document_id'] for m in self.metadata)),
            # Count unique document IDs
            'total_chunks': len(self.metadata)
            # Total number of chunks
        }

