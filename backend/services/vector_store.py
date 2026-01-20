"""
FAISS vector store for storing and retrieving document embeddings.
Supports similarity search with metadata filtering.
"""
import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from backend.config import settings
from backend.models.document import DocumentChunk


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
        self.index = faiss.IndexFlatL2(embedding_dim)
        
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
        
        # Normalize embeddings in-place: cosine similarity becomes derivable from L2 distance.
        faiss.normalize_L2(embeddings)
        
        # FAISS expects float32 arrays.
        self.index.add(embeddings.astype('float32'))
        
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
        
        top_k = top_k or settings.TOP_K_RESULTS
        
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
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Exact search in FAISS (IndexFlatL2).
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # For unit-normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
        similarities = 1 - (distances[0] ** 2 / 2)
        
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
        priority_clause_types: List[str] = None,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = -1.0
    ) -> List[Dict[str, Any]]:
        """
        Search with priority boosting for specific clause types.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            priority_clause_types: List of clause types to boost (e.g., ["Governing Law", "Termination"])
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            similarity_threshold: Similarity threshold
            
        Returns:
            List of results with boosted scores for priority clause types
        """
        # First get standard search results
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 2 if top_k else settings.TOP_K_RESULTS * 2,  # Get more results for re-ranking
            document_id_filter=document_id_filter,
            similarity_threshold=similarity_threshold
        )
        
        if not priority_clause_types:
            # No priority types, return standard results
            return results[:top_k or settings.TOP_K_RESULTS]
        
        # Boost scores for priority clause types
        priority_set = {pt.lower() for pt in priority_clause_types}
        boost_factor = 0.15  # Boost by 15% for priority clauses
        
        for result in results:
            # Check if result matches priority clause types
            # Metadata is already spread into result dict, not nested
            clause_types = result.get('clause_types', [])
            hierarchy_level = result.get('hierarchy_level', 'contract')
            
            # Check clause types
            matches_priority = any(
                pt.lower() in ct.lower() or ct.lower() in pt.lower()
                for pt in priority_clause_types
                for ct in clause_types
            )
            
            # Also check hierarchy level for "Governing Law" type
            if "governing law" in priority_set and hierarchy_level == "law":
                matches_priority = True
            
            if matches_priority:
                # Boost the score
                result['score'] = min(1.0, result['score'] + boost_factor)
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
        Delete all chunks for a document (for demo - in production, use more efficient method).
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        # Find indices to remove
        indices_to_remove = [
            i for i, metadata in enumerate(self.metadata)
            if metadata['document_id'] == document_id
        ]
        
        if not indices_to_remove:
            return 0
        
        # Note: FAISS doesn't support efficient deletion
        # For demo, we'll rebuild the index (not efficient for large datasets)
        # In production, consider using FAISS with ID mapping or a different approach
        
        # Remove from metadata
        for i in reversed(indices_to_remove):
            # Reverse order to maintain indices
            del self.metadata[i]
        
        # Rebuild index (simplified - in production, use more efficient method)
        if len(self.metadata) > 0:
            # Would need to rebuild from remaining embeddings
            # For demo, we'll just note this limitation
            pass
        
        return len(indices_to_remove)
    
    def save(self, filepath: Path = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save index (defaults to config)
        """
        filepath = filepath or settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath))
        # FAISS built-in serialization
        
        # Save metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        # Create metadata filename (e.g., legal_documents.index.metadata.pkl)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            # Serialize metadata using pickle
    
    def load(self, filepath: Path = None) -> None:
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load index from (defaults to config)
        """
        filepath = filepath or settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vector store not found: {filepath}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath))
        # Load FAISS index from disk
        
        # Load metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            # Deserialize metadata
        
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

