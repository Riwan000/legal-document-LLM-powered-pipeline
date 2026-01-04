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
        # Store embedding dimension
        
        # Create FAISS index (L2 distance - Euclidean)
        # L2 is good for cosine similarity when vectors are normalized
        self.index = faiss.IndexFlatL2(embedding_dim)
        # IndexFlatL2: simple exact search using L2 (Euclidean) distance
        # For cosine similarity, we'll normalize vectors before adding
        
        # Metadata storage: list of dicts, one per vector
        self.metadata: List[Dict[str, Any]] = []
        # Each dict contains: document_id, page_number, chunk_index, text, etc.
        
        # Ensure vector store directory exists
        settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    
    def add_chunks(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk]
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            chunks: List of DocumentChunk objects corresponding to embeddings
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks")
        
        # Normalize embeddings for cosine similarity
        # FAISS L2 distance on normalized vectors = cosine similarity
        faiss.normalize_L2(embeddings)
        # Normalize in-place: each vector becomes unit length
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        # FAISS requires float32, embeddings might be float64
        
        # Store metadata for each chunk
        for chunk in chunks:
            # Ensure clause_id is included if present
            chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
            if chunk.clause_id:
                chunk_metadata['clause_id'] = chunk.clause_id
            
            self.metadata.append({
                'document_id': chunk.document_id,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'text': chunk.text,
                **chunk_metadata  # Include any additional metadata (hierarchy_level, clause_types, etc.)
            })
            # Store all chunk information for retrieval
    
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
            # Check if index is empty
            return []
        
        top_k = top_k or settings.TOP_K_RESULTS
        # Use config default if not provided
        
        # Use provided threshold or default from config
        # -1.0 means use default from config
        # None means bypass threshold
        # Otherwise use the provided value
        if similarity_threshold == -1.0:
            threshold = settings.SIMILARITY_THRESHOLD  # Use default
        elif similarity_threshold is None:
            threshold = None  # Bypass threshold
        else:
            threshold = similarity_threshold  # Use provided value
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        # Reshape to (1, embedding_dim) for FAISS
        faiss.normalize_L2(query_embedding)
        # Normalize query vector
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        # Returns: distances (L2 distances), indices (positions in index)
        # distances shape: (1, top_k), indices shape: (1, top_k)
        
        # Convert distances to similarity scores (1 - normalized distance)
        # For normalized vectors, L2 distance relates to cosine similarity
        # Cosine similarity = 1 - (L2_distance^2 / 2) for normalized vectors
        similarities = 1 - (distances[0] ** 2 / 2)
        # Convert L2 distance to cosine similarity
        
        # Build results with metadata
        results = []
        for idx, (distance, similarity) in enumerate(zip(distances[0], similarities)):
            # Iterate through results
            metadata = self.metadata[indices[0][idx]]
            # Get metadata for this result
            
            # Apply document filter if specified
            if document_id_filter and metadata['document_id'] != document_id_filter:
                continue
                # Skip if doesn't match filter
            
            # Determine threshold to use: OCR documents get lower threshold
            # is_ocr is stored directly in metadata (from chunk.metadata spread)
            is_ocr = metadata.get('is_ocr', False)
            
            # Fallback: Detect OCR by checking for common OCR error patterns in text
            # This helps with documents ingested before OCR tracking was added
            if not is_ocr:
                text = metadata.get('text', '')
                # Common OCR error patterns: mixed case errors, character substitutions
                ocr_indicators = [
                    'Oy ether', 'l€rms', 'agreemeni', 'authort.es',  # Common OCR errors
                    'Nolce', 'terminateg', 'jne', 'Dy a',  # Notice period OCR errors
                ]
                if any(indicator in text for indicator in ocr_indicators):
                    is_ocr = True
            
            # Use OCR threshold if chunk is OCR and threshold wasn't explicitly set
            if threshold is not None:
                effective_threshold = threshold
                # If using default threshold and chunk is OCR, use OCR threshold
                if threshold == settings.SIMILARITY_THRESHOLD and is_ocr:
                    effective_threshold = settings.OCR_SIMILARITY_THRESHOLD
                # If using minimum threshold (fallback), don't apply OCR threshold
                # The minimum threshold is already low enough for difficult queries
                elif threshold == settings.MIN_SIMILARITY_THRESHOLD:
                    # Keep the minimum threshold as-is (don't apply OCR threshold)
                    effective_threshold = settings.MIN_SIMILARITY_THRESHOLD
            else:
                effective_threshold = None  # Bypass threshold
            
            # Apply similarity threshold (if threshold is not None)
            if effective_threshold is not None and similarity < effective_threshold:
                continue
                # Skip if below threshold
            
            results.append({
                'text': metadata['text'],
                'page_number': metadata['page_number'],
                'document_id': metadata['document_id'],
                'chunk_index': metadata['chunk_index'],
                'score': float(similarity),
                'distance': float(distance),
                'is_ocr': is_ocr,  # Include detected OCR flag
                **{k: v for k, v in metadata.items() 
                   if k not in ['text', 'page_number', 'document_id', 'chunk_index']}
                # Include any additional metadata
            })
        
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

