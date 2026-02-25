"""
Text chunking utilities with metadata preservation.
Splits documents into overlapping chunks for embedding.

Page invariant: every chunk belongs to exactly one page. page_number is immutable per chunk;
chunk_index resets per page; no chunk may span pages. Use chunk_pages() so chunk_text() is
called once per page only.
"""
from typing import List, Dict, Any, Optional
from backend.models.document import DocumentChunk
from backend.config import settings


class TextChunker:
    """Chunks text into overlapping segments with metadata. All chunks are page-scoped (one page per chunk)."""
    
    @staticmethod
    def chunk_text(
        text: str,
        page_number: int,
        document_id: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        is_ocr: bool = False
    ) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap and metadata.

        chunk_text MUST NOT be called on concatenated multi-page text.
        Call it once per page (e.g. via chunk_pages) so each chunk belongs to exactly one page.

        Args:
            text: Text to chunk (single page only)
            page_number: Page number where text appears
            document_id: Unique document identifier
            chunk_size: Maximum characters per chunk (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            is_ocr: Whether this text was extracted using OCR

        Returns:
            List of DocumentChunk objects
        """
        # Use config defaults if not provided
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Convert token-based size to character-based (rough estimate: 1 token ≈ 4 chars)
        char_chunk_size = chunk_size * 4
        char_overlap = chunk_overlap * 4
        
        chunks = []
        start = 0
        chunk_index = 0
        
        # Split text into sentences for better chunking
        sentences = TextChunker._split_into_sentences(text)
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > char_chunk_size and current_chunk:
                chunk_text = current_chunk.strip()
                if chunk_text:  # Only add non-empty chunks
                    # Generate stable chunk ID
                    chunk_id = f"c_p{page_number:04d}_i{chunk_index:04d}"
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        document_id=document_id,
                        metadata={
                            "char_length": len(chunk_text),
                            "is_ocr": is_ocr,
                            "chunk_id": chunk_id,
                            "chunk_type": "sentence",
                        }
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap (last N characters of previous chunk)
                if char_overlap > 0 and len(current_chunk) > char_overlap:
                    overlap_text = current_chunk[-char_overlap:]
                    current_chunk = overlap_text + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk += sentence
                current_length += sentence_length
        
        # Add final chunk if it exists (ensure we don't lose any text)
        if current_chunk.strip():
            chunk_id = f"c_p{page_number:04d}_i{chunk_index:04d}"
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                page_number=page_number,
                chunk_index=chunk_index,
                document_id=document_id,
                metadata={
                    "char_length": len(current_chunk),
                    "is_ocr": is_ocr,
                    "chunk_id": chunk_id,
                    "chunk_type": "sentence",
                }
            ))
        elif current_chunk:  # Even if only whitespace, preserve minimal content
            stripped = current_chunk.strip()
            if stripped or len(current_chunk) > 10:
                chunks.append(DocumentChunk(
                    text=stripped if stripped else current_chunk[:100],
                    page_number=page_number,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    metadata={
                        "char_length": len(current_chunk),
                        "is_ocr": is_ocr,
                        "minimal_content": True,
                        "chunk_type": "sentence",
                    }
                ))
        
        # Verification: Ensure all input text is accounted for
        total_chunked_length = sum(len(chunk.text) for chunk in chunks)
        input_length = len(text)
        
        # Allow some variance due to stripping and overlap, but warn if significant loss
        if input_length > 0:
            coverage_ratio = total_chunked_length / input_length
            if coverage_ratio < 0.8:  # Less than 80% coverage
                print(f"Warning: Low text coverage in chunking ({coverage_ratio:.1%}). "
                      f"Input: {input_length} chars, Chunked: {total_chunked_length} chars")
        
        return chunks
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences (simple approach).
        For production, consider using NLTK or spaCy.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on common delimiters
        import re
        # Split on sentence endings, preserving the delimiter
        sentences = re.split(r'([.!?]\s+)', text)
        
        # Recombine sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        
        # If odd number of parts, add last one
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        
        # Filter empty sentences
        return [s.strip() for s in result if s.strip()]
    
    @staticmethod
    def chunk_text_with_clauses(
        text: str,
        page_number: int,
        document_id: str,
        clauses: List[Dict[str, Any]] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        is_ocr: bool = False
    ) -> List[DocumentChunk]:
        """
        Chunk text while preserving clause boundaries.
        If clauses are provided, ensures clauses are not split across chunks.
        
        Args:
            text: Text to chunk
            page_number: Page number where text appears
            document_id: Unique document identifier
            clauses: Optional list of clause dicts with 'text' and 'start_index' keys
            chunk_size: Maximum characters per chunk (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            is_ocr: Whether this text was extracted using OCR
            
        Returns:
            List of DocumentChunk objects with clause metadata
        """
        # Use config defaults if not provided
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Convert token-based size to character-based
        char_chunk_size = chunk_size * 4
        char_overlap = chunk_overlap * 4
        
        # If no clauses provided, use standard chunking
        if not clauses:
            return TextChunker.chunk_text(text, page_number, document_id, chunk_size, chunk_overlap, is_ocr)
        
        # Build clause boundaries map
        clause_boundaries = []
        for clause in clauses:
            clause_text = clause.get('text', '').strip()
            if clause_text and clause_text in text:
                start_idx = text.find(clause_text)
                if start_idx != -1:
                    end_idx = start_idx + len(clause_text)
                    clause_boundaries.append({
                        'start': start_idx,
                        'end': end_idx,
                        'type': clause.get('type', 'Unknown'),
                        'text': clause_text,
                        'clause_id': clause.get('clause_id'),
                        'hierarchy_level': clause.get('hierarchy_level', 'contract'),
                        'legal_supremacy': clause.get('legal_supremacy', False),
                        'topics': clause.get('topics', []),
                        'jurisdiction': clause.get('jurisdiction')
                    })
        
        # Sort boundaries by start position
        clause_boundaries.sort(key=lambda x: x['start'])
        
        chunks = []
        chunk_index = 0
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # Find the end position for this chunk
            chunk_end = min(current_pos + char_chunk_size, text_length)
            
            # Check if we're in the middle of a clause
            # If so, extend chunk to include the entire clause
            for boundary in clause_boundaries:
                clause_start = boundary['start']
                clause_end = boundary['end']
                
                # If chunk boundary splits a clause, extend to end of clause
                if current_pos < clause_start < chunk_end < clause_end:
                    chunk_end = clause_end
                    break
                # If we're starting in the middle of a clause, start from clause start
                elif clause_start < current_pos < clause_end:
                    current_pos = clause_start
                    chunk_end = min(current_pos + char_chunk_size, text_length)
                    # Re-check if we need to extend further
                    if chunk_end < clause_end:
                        chunk_end = clause_end
                    break
            
            # Extract chunk text
            chunk_text = text[current_pos:chunk_end].strip()
            
            if chunk_text:
                # Find which clauses are in this chunk
                chunk_clauses = []
                chunk_clause_ids = []
                chunk_hierarchy_levels = []
                chunk_topics = []
                has_supremacy = False
                
                for boundary in clause_boundaries:
                    if (boundary['start'] >= current_pos and boundary['end'] <= chunk_end) or \
                       (current_pos <= boundary['start'] < chunk_end) or \
                       (current_pos < boundary['end'] <= chunk_end):
                        chunk_clauses.append(boundary['type'])
                        if boundary.get('clause_id'):
                            chunk_clause_ids.append(boundary['clause_id'])
                        if boundary.get('hierarchy_level'):
                            chunk_hierarchy_levels.append(boundary['hierarchy_level'])
                        if boundary.get('topics'):
                            chunk_topics.extend(boundary['topics'])
                        if boundary.get('legal_supremacy', False):
                            has_supremacy = True
                
                # Generate clause_id if clauses are present
                clause_id = None
                if chunk_clause_ids:
                    clause_id = chunk_clause_ids[0]  # Use first clause ID
                elif chunk_clauses:
                    clause_id = f"clause_{page_number}_{chunk_index}"
                
                # Determine hierarchy level (prefer law if any clause is law)
                hierarchy_level = 'contract'  # Default
                if 'law' in chunk_hierarchy_levels:
                    hierarchy_level = 'law'
                elif chunk_hierarchy_levels:
                    hierarchy_level = chunk_hierarchy_levels[0]
                
                # Generate stable chunk ID
                chunk_id = f"c_p{page_number:04d}_i{chunk_index:04d}"
                metadata = {
                    "char_length": len(chunk_text),
                    "clause_types": list(set(chunk_clauses)) if chunk_clauses else [],
                    "is_clause_complete": any(
                        current_pos <= b['start'] and b['end'] <= chunk_end
                        for b in clause_boundaries
                    ),
                    "is_ocr": is_ocr,
                    "clause_id": clause_id,
                    "hierarchy_level": hierarchy_level,
                    "legal_supremacy": has_supremacy,
                    "topics": list(set(chunk_topics)) if chunk_topics else [],
                    "chunk_id": chunk_id,
                    "chunk_type": "clause",
                }
                
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    metadata=metadata
                ))
                chunk_index += 1
            
            # Move to next chunk with overlap
            if chunk_end >= text_length:
                break
            
            # Calculate overlap start
            overlap_start = max(current_pos, chunk_end - char_overlap)
            current_pos = overlap_start
        
        return chunks
    
    @staticmethod
    def subchunk_clause(
        clause_id: str,
        verbatim_text: str,
        page_number: int,
        document_id: str,
        legal_category: Optional[str] = None,
        clause_number: Optional[str] = None,
        chunk_overlap: int = None,
        threshold: int = 2000,
    ) -> List[DocumentChunk]:
        """
        Split a long clause (> threshold chars) into overlapping sub-chunks.

        Each sub-chunk gets:
          parent_clause_id  = clause_id
          unit_type         = "clause_subchunk"
          legal_category    = inherited from parent
          clause_number     = inherited from parent

        Sub-chunk IDs: f"sc_{clause_id}_{sub_index:04d}"
        Overlap: chunk_overlap chars (defaults to settings.CHUNK_OVERLAP * 4).

        Returns empty list if text is short enough (caller should use the full clause chunk).
        """
        import re as _re
        if len(verbatim_text) <= threshold:
            return []

        char_overlap = (chunk_overlap or settings.CHUNK_OVERLAP) * 4
        sentences = TextChunker._split_into_sentences(verbatim_text)

        sub_chunks: List[DocumentChunk] = []
        current = ""
        sub_index = 0

        for sentence in sentences:
            if len(current) + len(sentence) > threshold and current:
                sc_id = f"sc_{clause_id}_{sub_index:04d}"
                chunk = DocumentChunk(
                    text=current.strip(),
                    page_number=page_number,
                    chunk_index=sub_index,
                    document_id=document_id,
                    clause_id=clause_id,
                    parent_clause_id=clause_id,
                    unit_type="clause_subchunk",
                    legal_category=legal_category,
                    clause_number=clause_number,
                    metadata={
                        "chunk_id": sc_id,
                        "chunk_type": "clause_subchunk",
                        "char_length": len(current.strip()),
                    },
                )
                sub_chunks.append(chunk)
                sub_index += 1
                # Overlap: keep last char_overlap chars
                current = (current[-char_overlap:] if char_overlap and len(current) > char_overlap else "") + sentence
            else:
                current += sentence

        if current.strip():
            sc_id = f"sc_{clause_id}_{sub_index:04d}"
            chunk = DocumentChunk(
                text=current.strip(),
                page_number=page_number,
                chunk_index=sub_index,
                document_id=document_id,
                clause_id=clause_id,
                parent_clause_id=clause_id,
                unit_type="clause_subchunk",
                legal_category=legal_category,
                clause_number=clause_number,
                metadata={
                    "chunk_id": sc_id,
                    "chunk_type": "clause_subchunk",
                    "char_length": len(current.strip()),
                },
            )
            sub_chunks.append(chunk)

        return sub_chunks

    @staticmethod
    def chunk_pages(
        pages: List[tuple],
        document_id: str,
        clauses_by_page: Dict[int, List[Dict[str, Any]]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk multiple pages of text, optionally preserving clause boundaries.
        
        Args:
            pages: List of (text, page_number), (text, page_number, is_ocr), or PageInfo-like (has .text, .page_number, .is_ocr)
            document_id: Unique document identifier
            clauses_by_page: Optional dict mapping page_number to list of clauses
            
        Returns:
            List of all DocumentChunk objects from all pages
        """
        all_chunks = []
        clauses_by_page = clauses_by_page or {}
        
        for page_data in pages:
            # Support PageInfo and tuple formats
            if hasattr(page_data, "text"):
                text = page_data.text
                page_number = page_data.page_number
                is_ocr = getattr(page_data, "is_ocr", False)
            elif len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                text, page_number = page_data
                is_ocr = False
            
            # Get clauses for this page if available
            page_clauses = clauses_by_page.get(page_number, None)
            
            if page_clauses:
                # Use clause-aware chunking
                page_chunks = TextChunker.chunk_text_with_clauses(
                    text=text,
                    page_number=page_number,
                    document_id=document_id,
                    clauses=page_clauses,
                    is_ocr=is_ocr
                )
            else:
                # Use standard chunking
                page_chunks = TextChunker.chunk_text(
                    text=text,
                    page_number=page_number,
                    document_id=document_id,
                    is_ocr=is_ocr
                )
            all_chunks.extend(page_chunks)
        
        return all_chunks

