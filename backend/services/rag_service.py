"""
RAG (Retrieval-Augmented Generation) service.
Retrieves relevant document chunks and generates responses using Ollama LLM.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import ollama
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.query_classifier import QueryClassifier
from backend.services.legal_reasoning_service import LegalReasoningService
from backend.services.legal_hierarchy_service import LegalHierarchyService
from backend.services.clause_store import ClauseStore
from backend.models.document import AnswerResponse, LegalHierarchyLevel
from backend.config import settings

if TYPE_CHECKING:
    from backend.services.translation_service import TranslationService


class RAGService:
    """Service for RAG-based query answering with citations."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        clause_store: Optional[ClauseStore] = None,
        translation_service: Optional["TranslationService"] = None
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            clause_store: Optional clause store for structured clause retrieval
            translation_service: Optional translation service for query translation
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.clause_store = clause_store
        self.translation_service = translation_service
        # Core dependencies:
        # - Embeddings: convert query text into a dense vector representation.
        # - Vector store: similarity search over document chunks.
        # - Clause store (optional): structured clause retrieval for contract analysis.
        
        # Rule-guided / policy-layer services:
        # - Classify query intent and required safety constraints.
        # - Apply lightweight “legal hierarchy” heuristics (law > contract > policy).
        self.query_classifier = QueryClassifier()
        self.legal_reasoning = LegalReasoningService()
        self.hierarchy_service = LegalHierarchyService()
        
        # Ollama client used for local LLM generation (no external API calls).
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def search(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        priority_clause_types: Any = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks with clause-aware priority boosting.
        Implements two-pass search with query translation if languages don't match.
        
        Args:
            query: Search query (can be Arabic or English)
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            priority_clause_types: List of clause types to boost (e.g., ["Governing Law", "Termination"])
            
        Returns:
            List of relevant chunks with metadata, ranked by authority
        """
        if top_k is not None and top_k <= 0:
            return []

        # 1) Classify the query and choose “priority” clause types (boosting).
        # `priority_clause_types` is a backward-compat parameter name; in this codebase it
        # is typically a weighted mapping: {clause_type: weight}.
        if priority_clause_types is None:
            classification = self.query_classifier.classify_query(query)
            priority_clause_types = self._get_priority_clause_types(classification)
        
        # Normalize priority config to a weights dict.
        priority_weights: Optional[Dict[str, float]] = None
        if isinstance(priority_clause_types, dict):
            priority_weights = priority_clause_types
        elif isinstance(priority_clause_types, list):
            # Treat list as "equal weight" boosting
            priority_weights = {str(t): 1.0 for t in priority_clause_types}

        # 2) Encode query into an embedding vector (multilingual model supports Arabic/English).
        query_embedding = self.embedding_service.embed_text(query)
        
        # 3) Retrieve top-k chunks. If we have priority clause types, do a “boosted” re-ranking.
        if priority_weights:
            results = self.vector_store.search_with_priority(
                query_embedding=query_embedding,
                priority_weights=priority_weights,
                top_k=top_k,
                document_id_filter=document_id_filter,
                similarity_threshold=similarity_threshold,
            )
        else:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id_filter=document_id_filter,
                similarity_threshold=similarity_threshold if similarity_threshold is not None else -1.0,
            )
        
        # 4) If semantic results are empty, lower the similarity threshold to be more permissive.
        # This mainly helps OCR-derived text, where embeddings can be noisier.
        if not results:
            effective_top_k = top_k or settings.TOP_K_RESULTS
            if priority_weights:
                results = self.vector_store.search_with_priority(
                    query_embedding=query_embedding,
                    priority_weights=priority_weights,
                    top_k=effective_top_k,
                    document_id_filter=document_id_filter,
                    similarity_threshold=None,
                )
            else:
                results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=effective_top_k,
                    document_id_filter=document_id_filter,
                    similarity_threshold=None,
                )
        
        # 5) Final fallback: keyword match inside stored chunk text for a given document.
        # We only do this if a document filter is provided (avoid scanning entire corpus).
        if not results and document_id_filter:
            results = self._keyword_fallback_search(
                query=query,
                document_id=document_id_filter,
                top_k=top_k
            )
        
        # 6) Rank results by legal authority (law > contract > policy) to prefer governing rules.
        if results:
            results = self.hierarchy_service.rank_by_authority(results)
        
        return results
    
    def _search_without_translation(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        priority_clause_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Internal search method without translation logic (original implementation).
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            priority_clause_types: List of clause types to boost
            
        Returns:
            List of relevant chunks with metadata, ranked by authority
        """
        # FIX 2: Bilingual parity - check if translation is enabled
        if settings.ENABLE_QUERY_TRANSLATION and self.translation_service:
            # Detect query language
            query_language = self.translation_service.detect_language(query)
            
            # First pass: Search with original query
            initial_top_k = max(top_k or settings.TOP_K_RESULTS, 10) if top_k else 15
            original_results = self._search_without_translation(
                query, initial_top_k, document_id_filter, priority_clause_types
            )
            
            # Track if original search found grounded text
            original_found_grounded = len(original_results) > 0
            
            # If no results, return early (fail closed)
            if not original_results:
                return []
            
            # Detect document language from retrieved chunks
            document_language = self._detect_document_language(original_results, document_id_filter)
            
            # If languages match, return original results
            if query_language == document_language:
                return original_results[:top_k] if top_k else original_results
            
            # Languages don't match - translate and re-search
            try:
                translated_query = self.translation_service.translate_text(
                    query, query_language, document_language
                )
                
                # Second pass: Search with translated query
                translated_results = self._search_without_translation(
                    translated_query, top_k or settings.TOP_K_RESULTS, document_id_filter, priority_clause_types
                )
                
                # FIX 2: Bilingual parity enforcement
                # If original found nothing, we already returned []
                # If translated found nothing but original found something, use original (grounded text exists)
                if not translated_results and original_found_grounded:
                    return original_results[:top_k] if top_k else original_results
                elif not translated_results:
                    # Both found nothing - fail closed
                    return []
                elif not original_found_grounded and translated_results:
                    # Original found nothing but translated found something - violates parity
                    # Fail closed: if original language fails, all languages must fail
                    return []
                
                # Both found results - merge
                merged_results = self._merge_search_results(original_results, translated_results, top_k)
                return merged_results
                
            except Exception as e:
                print(f"Warning: Query translation failed: {str(e)}. Using original results.")
                if settings.TRANSLATION_FALLBACK_TO_ORIGINAL:
                    # FIX 2: If translation fails and original found nothing, fail closed
                    if not original_found_grounded:
                        return []
                    return original_results[:top_k] if top_k else original_results
                raise
        
        # Fallback to original search without translation
        return self._search_without_translation(query, top_k, document_id_filter, priority_clause_types)
    
    def _detect_document_language(
        self,
        chunks: List[Dict[str, Any]],
        document_id_filter: Optional[str] = None
    ) -> str:
        """
        Detect primary language of document from chunks or stored metadata.
        
        Args:
            chunks: List of retrieved chunks
            document_id_filter: Optional document ID to check stored language
            
        Returns:
            Language code ('ar' for Arabic, 'en' for English)
        """
        # If document_id_filter provided, try to get stored language from metadata
        if document_id_filter and self.translation_service:
            # Check if any chunks have language metadata
            for chunk in chunks:
                if chunk.get('language'):
                    return chunk['language']
            
            # Try to get from vector store metadata
            if hasattr(self.vector_store, 'metadata'):
                for metadata in self.vector_store.metadata:
                    if metadata.get('document_id') == document_id_filter and metadata.get('language'):
                        return metadata['language']
        
        # Fallback: detect from chunk text samples
        if not chunks:
            return 'en'  # Default to English
        
        # Sample text from top chunks
        sample_text = " ".join([chunk.get('text', '')[:500] for chunk in chunks[:5]])
        
        if not sample_text.strip():
            return 'en'
        
        # Use translation service's language detection
        if self.translation_service:
            return self.translation_service.detect_language(sample_text)
        
        # Fallback to simple detection
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        text_chars = set(sample_text.replace(' ', ''))
        arabic_ratio = len(text_chars.intersection(arabic_chars)) / max(len(text_chars), 1)
        
        return 'ar' if arabic_ratio > settings.LANGUAGE_DETECTION_THRESHOLD else 'en'
    
    def _detect_document_type(self, chunks: List[Dict[str, Any]], document_id_filter: Optional[str] = None) -> str:
        """
        Detect primary document type (statute/law vs contract).
        
        Args:
            chunks: List of retrieved chunks
            document_id_filter: Optional document ID to filter by
            
        Returns:
            'statute' if document is primarily a law/statute, 'contract' otherwise
        """
        if not chunks:
            return 'contract'  # Default to contract
        
        # Count hierarchy levels in chunks
        law_count = 0
        contract_count = 0
        
        for chunk in chunks:
            hierarchy_level = chunk.get('hierarchy_level', 'contract')
            if hierarchy_level == 'law':
                law_count += 1
            elif hierarchy_level == 'contract':
                contract_count += 1
        
        # If majority of chunks are law-level, document is a statute
        total = law_count + contract_count
        if total > 0 and law_count / total > 0.5:
            return 'statute'
        
        return 'contract'
    
    def _get_not_specified_message(self, document_type: str) -> str:
        """
        Get appropriate "not specified" message based on document type.
        
        Args:
            document_type: 'statute' or 'contract'
            
        Returns:
            Appropriate message string
        """
        if document_type == 'statute':
            return "The law does not expressly state this."
        else:
            # Tests expect a "couldn't find" phrasing for no-results cases.
            return "I couldn't find this information in the provided contract."
    
    def _validate_citation_support(self, answer: str, cited_chunks: List[Dict[str, Any]], query: str) -> bool:
        """
        Validate that cited chunks directly support the answer claim.
        Citations must directly support the claim, not just be thematically related.
        
        Args:
            answer: Generated answer text
            cited_chunks: List of chunks that were cited in the answer
            query: Original query
            
        Returns:
            True if citations directly support the answer, False otherwise
        """
        if not cited_chunks or not answer:
            return True  # No citations to validate
        
        # Extract cited source numbers from answer
        import re
        cited_sources = set(re.findall(r'\[Source\s+(\d+)\]', answer))
        if not cited_sources:
            return True  # No citations found in answer
        
        # Get the actual chunks that were cited
        cited_chunk_indices = [int(src) - 1 for src in cited_sources if src.isdigit()]
        actual_cited_chunks = [cited_chunks[i] for i in cited_chunk_indices if 0 <= i < len(cited_chunks)]
        
        if not actual_cited_chunks:
            return True  # Couldn't map citations, assume valid
        
        # Extract key terms from query and answer
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'source'}
        query_terms = {t for t in query_terms if t not in stop_words and len(t) > 3}
        answer_terms = {t for t in answer_terms if t not in stop_words and len(t) > 3}
        
        # Key terms that should appear in cited chunks
        key_terms = query_terms.union(answer_terms)
        
        # Check if cited chunks contain key terms
        for chunk in actual_cited_chunks:
            chunk_text = chunk.get('text', '').lower()
            chunk_terms = set(re.findall(r'\b\w+\b', chunk_text))
            
            # Check for direct term overlap
            term_overlap = key_terms.intersection(chunk_terms)
            
            # If less than 2 key terms overlap, citation may not directly support
            if len(term_overlap) < 2 and len(key_terms) > 2:
                # Additional check: semantic similarity via embedding
                # For now, we'll be lenient - if chunk has any key terms, consider it valid
                if len(term_overlap) == 0:
                    return False  # No term overlap at all
        
        return True  # Citations appear to support the answer
    
    def _merge_search_results(
        self,
        original_results: List[Dict[str, Any]],
        translated_results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Merge results from original and translated query searches.
        Deduplicates by chunk_index and prefers higher scores.
        
        Args:
            original_results: Results from original query
            translated_results: Results from translated query
            top_k: Maximum number of results to return
            
        Returns:
            Merged and deduplicated list of results
        """
        # Create a map of chunk_index -> best result (highest score)
        result_map = {}
        
        # Add original results
        for result in original_results:
            chunk_idx = result.get('chunk_index', -1)
            if chunk_idx not in result_map or result.get('score', 0.0) > result_map[chunk_idx].get('score', 0.0):
                result_map[chunk_idx] = result
        
        # Add translated results (prefer if higher score)
        for result in translated_results:
            chunk_idx = result.get('chunk_index', -1)
            if chunk_idx not in result_map or result.get('score', 0.0) > result_map[chunk_idx].get('score', 0.0):
                result_map[chunk_idx] = result
        
        # Convert back to list and sort by score
        merged = list(result_map.values())
        merged.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Return top_k if specified
        return merged[:top_k] if top_k else merged
    
    def _keyword_fallback_search(
        self,
        query: str,
        document_id: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search when semantic search fails.
        Useful for finding content that exists but has low semantic similarity.
        
        Args:
            query: Search query
            document_id: Document ID to search in
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks found via keyword matching
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Extract “keywords” from query by removing stop-words and short tokens.
        # This is intentionally simple: it’s a last-resort fallback when embeddings yield nothing.
        import re
        
        # Detect if query contains Arabic characters
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        query_chars = set(query.replace(' ', ''))
        has_arabic = len(query_chars.intersection(arabic_chars)) > 0
        
        # English stop words
        english_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'this', 'that', 'these', 'those'}
        
        # Arabic stop words (common Arabic function words)
        arabic_stop_words = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'الذين', 'اللاتي',
            'كان', 'كانت', 'يكون', 'تكون', 'يكونون', 'يكون', 'كانوا', 'كانت', 'كانتا', 'كانوا',
            'له', 'لها', 'لهم', 'لهن', 'لهما', 'لي', 'لك', 'لكم', 'لكن', 'لنا',
            'ال', 'و', 'ف', 'ثم', 'أو', 'لكن', 'بل', 'إلا', 'إن', 'أن', 'ما', 'ماذا', 'من', 'متى', 'أين', 'كيف', 'لماذا',
            'هو', 'هي', 'هم', 'هن', 'هما', 'أنا', 'أنت', 'أنتم', 'أنتن', 'نحن'
        }
        
        # Combine stop words based on query language
        stop_words = english_stop_words
        if has_arabic:
            stop_words = stop_words.union(arabic_stop_words)
        
        # Extract words: handle both English (word boundaries) and Arabic (space-separated)
        if has_arabic:
            # For Arabic, split by spaces and Arabic punctuation
            words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', query)
        else:
            # For English, use word boundaries
            words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short tokens
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        if not keywords:
            return []
        
        # Search in vector store metadata for matching chunks within this document.
        # Note: we search stored chunk text; we do not hit FAISS here.
        matching_chunks = []
        for metadata in self.vector_store.metadata:
            if metadata.get('document_id') != document_id:
                continue
            
            chunk_text = metadata.get('text', '')
            chunk_text_lower = chunk_text.lower()
            
            # Count keyword matches (simple containment check).
            # For Arabic, do case-insensitive matching; for English, use lowercase
            if has_arabic:
                # For mixed queries, check both original and lowercase
                matches = sum(1 for keyword in keywords if keyword in chunk_text or keyword.lower() in chunk_text_lower)
            else:
                matches = sum(1 for keyword in keywords if keyword.lower() in chunk_text_lower)
            
            if matches > 0:
                # Relevance is normalized by the number of keywords so results across queries are comparable.
                relevance = matches / len(keywords)
                matching_chunks.append({
                    'text': metadata.get('text', ''),
                    'page_number': metadata.get('page_number', 0),
                    'document_id': metadata.get('document_id', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'score': relevance,  # Keyword-based relevance score (not a true embedding similarity).
                    'distance': 1.0 - relevance,  # Inverse for compatibility with “distance”-expecting callers.
                    'is_ocr': metadata.get('is_ocr', False),
                    'keyword_match': True,  # Flag to indicate keyword-based match
                    **{k: v for k, v in metadata.items() 
                       if k not in ['text', 'page_number', 'document_id', 'chunk_index', 'is_ocr']}
                })
        
        # Sort by keyword relevance (highest first).
        matching_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return matching_chunks[:top_k]
    
    def _get_priority_clause_types(self, classification) -> Dict[str, float]:
        """
        Determine priority clause types with weights based on query classification.
        
        Returns weighted mapping: clause_type → weight
        Clauses matching multiple query types get cumulative weights.
        """
        priority_weights = {}
        
        # If requires legal hierarchy, boost governing law clauses
        if classification.requires_legal_hierarchy:
            priority_weights["Governing Law"] = 1.0
        
        # Map query_types (intent) to clause types with weights
        # query_types represent user intent - what they want to know
        for query_type in classification.query_types:
            if query_type == "termination":
                priority_weights["Termination"] = priority_weights.get("Termination", 0.0) + 1.0
            elif query_type == "benefits":
                priority_weights["Payment Terms"] = priority_weights.get("Payment Terms", 0.0) + 0.6
                priority_weights["Benefits"] = priority_weights.get("Benefits", 0.0) + 1.0
            elif query_type == "compensation":
                priority_weights["Payment Terms"] = priority_weights.get("Payment Terms", 0.0) + 0.8
                priority_weights["Compensation"] = priority_weights.get("Compensation", 0.0) + 1.0
            elif query_type == "legality":
                priority_weights["Governing Law"] = priority_weights.get("Governing Law", 0.0) + 1.0
            elif query_type == "notice":
                priority_weights["Notice"] = priority_weights.get("Notice", 0.0) + 1.0
            elif query_type == "probation":
                priority_weights["Probation"] = priority_weights.get("Probation", 0.0) + 1.0
        
        return priority_weights
    
    def query(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        generate_response: bool = True,
        response_language: Optional[str] = None,
        chunks_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a query using rule-guided RAG with legal hierarchy and citation enforcement.
        
        Args:
            query: User query (can be Arabic or English)
            top_k: Number of chunks to retrieve
            document_id_filter: Optional filter by document ID
            generate_response: Whether to generate LLM response (True) or just return chunks (False)
            response_language: Optional response language ('ar', 'en', or None for auto)
            
        Returns:
            Dict with AnswerResponse structure:
                - status: explicitly_stated, governed_by_law, not_specified, refused
                - answer: Direct answer
                - citation: Citation reference
                - confidence: high, medium, low
                - refusal_reason: Optional refusal reason
                - hierarchy_analysis: Optional hierarchy analysis
                - sources: List of source chunks
        """
        # Edge case: explicitly request zero retrieval. Return a safe "not specified"
        # response with no sources.
        if top_k == 0:
            return {
                "status": "not_specified",
                "answer": self._get_not_specified_message("contract"),
                "citation": None,
                "confidence": "low",
                "refusal_reason": None,
                "hierarchy_analysis": {},
                "sources": [],
                "query": query,
            }

        # Step 1: Classify query
        classification = self.query_classifier.classify_query(query)
        
        # Step 2: Check scope and refuse if out of scope
        if self.query_classifier.is_out_of_scope(query):
            return {
                'status': 'refused',
                'answer': 'This topic is not covered in the provided documents.',
                'citation': None,
                'confidence': 'low',
                'refusal_reason': 'out_of_scope',
                'hierarchy_analysis': None,
                'sources': [],
                'query': query
            }
        
        structured_clauses: List[Any] = []
        if chunks_override is None:
            # Step 3: Try to retrieve structured clauses first (if clause store available)
            if self.clause_store and document_id_filter:
                try:
                    # RAG retrieval uses ranking-ready candidate clauses from ClauseStore.
                    # ClauseStore remains ranking-agnostic: it only returns data + normalized metadata.
                    candidate_dicts = self.clause_store.get_candidate_clauses(
                        document_ids=[document_id_filter]
                    )

                    # If we have specific query_types, filter candidates by normalized clause_type.
                    filtered_candidates = candidate_dicts
                    if classification.query_types and classification.query_types[0] != "general":
                        primary_type = classification.query_types[0].lower()
                        filtered_candidates = [
                            c for c in candidate_dicts
                            if c.get("clause_type") == primary_type
                        ]

                    # Use the underlying StructuredClause objects for hierarchy analysis.
                    structured_clauses = [c["clause"] for c in filtered_candidates]
                except Exception as e:
                    print(f"Error querying clause store: {str(e)}")
                    structured_clauses = []

            # Step 4: Retrieve relevant chunks with priority boosting (weighted)
            priority_weights = self._get_priority_clause_types(classification)
            chunks = self.search(
                query,
                top_k=top_k,
                document_id_filter=document_id_filter,
                priority_clause_types=priority_weights,
            )
        else:
            # Use provided chunks and skip retrieval/structured clause lookup
            chunks = chunks_override
        
        # Step 5: Analyze legal hierarchy (use structured clauses if available)
        if structured_clauses:
            # Use structured clauses for hierarchy analysis
            hierarchy_analysis = self._analyze_hierarchy_from_clauses(structured_clauses)
        else:
            hierarchy_analysis = self.legal_reasoning.analyze_legal_hierarchy(query, chunks) if chunks else {}
        
        # Step 6: Detect document type for appropriate messaging
        document_type = self._detect_document_type(chunks, document_id_filter) if chunks else 'contract'
        
        # Step 7: Check if topic is not specified
        # Only return "not_specified" if we truly have no chunks AND no structured clauses
        # AND keyword fallback also found nothing
        if not chunks and not structured_clauses:
            # Try one more time with keyword search if we have document_id
            if document_id_filter:
                keyword_results = self._keyword_fallback_search(
                    query=query,
                    document_id=document_id_filter,
                    top_k=top_k or settings.TOP_K_RESULTS
                )
                if keyword_results:
                    chunks = keyword_results
                    # Update hierarchy analysis and document type with keyword results
                    hierarchy_analysis = self.legal_reasoning.analyze_legal_hierarchy(query, chunks) if chunks else {}
                    document_type = self._detect_document_type(chunks, document_id_filter)
                else:
                    # Truly no results found - use document-type aware message
                    not_specified_msg = self._get_not_specified_message(document_type)
                    return {
                        'status': 'not_specified',
                        'answer': not_specified_msg,
                        'citation': None,
                        'confidence': 'low',
                        'refusal_reason': None,
                        'hierarchy_analysis': hierarchy_analysis,
                        'sources': [],
                        'query': query
                    }
            else:
                # No document_id, can't do keyword search
                not_specified_msg = self._get_not_specified_message(document_type)
                return {
                    'status': 'not_specified',
                    'answer': not_specified_msg,
                    'citation': None,
                    'confidence': 'low',
                    'refusal_reason': None,
                    'hierarchy_analysis': hierarchy_analysis,
                    'sources': [],
                    'query': query
                }
        
        # Only check detect_not_specified if we have chunks but they might not be relevant
        # FIX 1: Disable contract fallback logic for statutes
        if chunks and not structured_clauses:
            # For statutes, don't use detect_not_specified (contract fallback logic)
            if document_type == 'statute':
                # For statutes, if we have chunks, use them (no fallback to "not specified")
                pass  # Continue with chunks
            else:
                # For contracts, use existing detect_not_specified logic
                # Check if we have keyword matches - if so, content exists
                has_keyword_matches = any(c.get('keyword_match', False) for c in chunks)
                if not has_keyword_matches:
                    # Only check detect_not_specified if no keyword matches
                    if self.legal_reasoning.detect_not_specified(chunks, query, classification):
                        not_specified_msg = self._get_not_specified_message(document_type)
                        return {
                            'status': 'not_specified',
                            'answer': not_specified_msg,
                            'citation': None,
                            'confidence': 'low',
                            'refusal_reason': None,
                            'hierarchy_analysis': hierarchy_analysis,
                            'sources': [],
                            'query': query
                        }
        
        # Step 8: Enforce citation requirement for legal queries
        citation_required = self.legal_reasoning.check_citation_requirement(classification)
        if citation_required and not chunks and not structured_clauses:
            return {
                'status': 'refused',
                'answer': '⚠️ Cannot provide a legal answer without a cited clause.',
                'citation': None,
                'confidence': 'low',
                'refusal_reason': 'missing_citation',
                'hierarchy_analysis': hierarchy_analysis,
                'sources': [],
                'query': query
            }
        
        # Step 9: Determine if explicit clause found
        has_explicit_clause = (len(structured_clauses) > 0) or (len(chunks) > 0 and chunks[0].get('score', 0.0) >= 0.6)
        
        # Step 10: Calculate confidence
        confidence = self.legal_reasoning.calculate_confidence(chunks, hierarchy_analysis, has_explicit_clause)
        
        # Step 11: Determine status
        if hierarchy_analysis.get('has_governing_law', False) and hierarchy_analysis.get('has_potential_conflict', False):
            status = 'governed_by_law'
        elif has_explicit_clause:
            status = 'explicitly_stated'
        else:
            status = 'governed_by_law' if hierarchy_analysis.get('has_governing_law', False) else 'explicitly_stated'
        
        if not generate_response:
            # Return chunks/clauses only (no LLM generation)
            # Use structured clauses if available, otherwise chunks
            sources = self._format_clauses_as_sources(structured_clauses) if structured_clauses else chunks
            citation = None
            if structured_clauses and structured_clauses[0].evidence:
                citation = self._format_clause_citation(structured_clauses[0])
            elif chunks:
                citation = self.legal_reasoning.format_citation(chunks[0])
            
            return {
                'status': status,
                'answer': None,
                'citation': citation,
                'confidence': confidence,
                'refusal_reason': None,
                'hierarchy_analysis': hierarchy_analysis,
                'sources': sources,
                'query': query
            }
        
        # Step 12: Build context from retrieved chunks/clauses
        if structured_clauses:
            context = self._build_context_from_clauses(structured_clauses)
        else:
            context = self._build_context(chunks)
        # Combine chunks into a single context string
        
        # Step 13: Generate prompt for LLM with legal-safe template (document-type aware)
        prompt = self._build_legal_prompt(query, context, classification, hierarchy_analysis, response_language, document_type)
        # Create prompt with legal hierarchy awareness and citation requirements
        
        # Step 14: Generate response using Ollama
        try:
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.1  # Low temperature for deterministic legal answers
                }
            )
            # Call Ollama API to generate response
            # Uses local LLM (no data sent externally)
            
            # Extract generated text from response - handle both dict and object formats
            if isinstance(response, dict):
                # Old format: {'response': '...'}
                answer = response.get('response', '')
            elif hasattr(response, 'response'):
                # New format: response object with .response attribute
                answer = response.response
            else:
                # Fallback: try to get text directly
                answer = str(response)
            
            # If still empty, try alternative keys (for dict format)
            if not answer and isinstance(response, dict):
                answer = response.get('text', response.get('content', ''))
            
            # Post-process: Remove "not specified" phrase if answer contains citations
            # This handles cases where LLM incorrectly adds the phrase despite finding relevant information
            if answer and '[Source' in answer and 'not specified' in answer.lower():
                # Check if "not specified" appears at the end (common pattern)
                answer_clean = answer.strip()
                # Check for both contract and statute messages
                contract_phrase = "This is not specified in the provided contract."
                statute_phrase = "The law does not expressly state this."
                if answer_clean.endswith(contract_phrase):
                    answer = answer_clean[:-len(contract_phrase)].strip()
                elif answer_clean.endswith(statute_phrase):
                    answer = answer_clean[:-len(statute_phrase)].strip()
                # Also remove if it appears as a separate sentence at the end
                elif answer_clean.endswith('.' + contract_phrase):
                    answer = answer_clean[:-len('.' + contract_phrase)].strip()
                elif answer_clean.endswith('.' + statute_phrase):
                    answer = answer_clean[:-len('.' + statute_phrase)].strip()
                elif answer_clean.endswith('\n' + contract_phrase):
                    answer = answer_clean[:-len('\n' + contract_phrase)].strip()
                elif answer_clean.endswith('\n' + statute_phrase):
                    answer = answer_clean[:-len('\n' + statute_phrase)].strip()
            
            # FIX 3: Validate citation-answer semantic support
            if answer and chunks:
                # Extract cited chunks from answer
                cited_chunks = chunks  # All chunks in context are potentially cited
                citation_valid = self._validate_citation_support(answer, cited_chunks, query)
                if not citation_valid:
                    # Citations don't directly support the answer - refuse
                    print("Warning: Citations do not directly support the answer claim. Refusing answer.")
                    return {
                        'status': 'refused',
                        'answer': 'Cannot provide answer: citations do not directly support the claim. Citations must directly support the answer, not just be thematically related.',
                        'citation': None,
                        'confidence': 'low',
                        'refusal_reason': 'invalid_citation',
                        'hierarchy_analysis': hierarchy_analysis,
                        'sources': chunks[:3] if chunks else [],  # Include some sources for context
                        'query': query
                    }
            
            # Post-process: Detect and handle translation error messages in answer
            translation_error_indicators = [
                "I'm sorry, but I cannot provide a translation",
                "cannot provide a translation for the user's question",
                "Can I help you with something else"
            ]
            if answer and any(indicator.lower() in answer.lower() for indicator in translation_error_indicators):
                print(f"Warning: LLM answer contains translation error message. This suggests translation may have failed.")
                # If we have chunks, try to generate answer without translation
                if chunks:
                    # Rebuild prompt with original query (no translation)
                    prompt_original = self._build_legal_prompt(query, context, classification, hierarchy_analysis, response_language)
                    try:
                        response_original = self.ollama_client.generate(
                            model=settings.OLLAMA_MODEL,
                            prompt=prompt_original,
                            options={'temperature': 0.1}
                        )
                        if isinstance(response_original, dict):
                            answer = response_original.get('response', answer)
                        elif hasattr(response_original, 'response'):
                            answer = response_original.response
                        else:
                            answer = str(response_original)
                    except:
                        pass  # Keep original answer if retry fails
            
        except Exception as e:
            # Handle Ollama errors (server not running, model not found, etc.)
            answer = f"Error generating response: {str(e)}. Please ensure Ollama is running and the model is available."
        
        # Step 15: Format sources with enhanced citations
        if structured_clauses:
            sources = self._format_clauses_as_sources(structured_clauses)
            citation = self._format_clause_citation(structured_clauses[0]) if structured_clauses else None
        else:
            sources = self._format_sources_enhanced(chunks)
            citation = self.legal_reasoning.format_citation(chunks[0]) if chunks else None
        
        return {
            'status': status,
            'answer': answer,
            'citation': citation,
            'confidence': confidence,
            'refusal_reason': None,
            'hierarchy_analysis': hierarchy_analysis,
            'sources': sources,
            'query': query
        }
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Use display_name if available, otherwise fall back to document_id
            doc_name = chunk.get('display_name', chunk.get('document_id', 'Unknown'))
            # Format each chunk with citation info (never expose document_hash)
            context_parts.append(
                f"[Source {i} - Document: {doc_name}, Page: {chunk['page_number']}]\n"
                f"{chunk['text']}\n"
            )
            # Include source metadata and text
            # Format: [Source N - Document: DisplayName, Page: X]\nText\n
        
        return "\n".join(context_parts)
        # Join all chunks with newlines
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM with context and query (legacy method, kept for compatibility).
        
        Args:
            query: User query
            context: Retrieved document context
            
        Returns:
            Formatted prompt string
        """
        return self._build_legal_prompt(query, context, None, None)
    
    def _build_legal_prompt(
        self,
        query: str,
        context: str,
        classification = None,
        hierarchy_analysis: Dict[str, Any] = None,
        response_language: Optional[str] = None,
        document_type: str = 'contract'
    ) -> str:
        """
        Build legal-safe prompt with hierarchy awareness and citation requirements.
        
        Args:
            query: User query
            context: Retrieved document context
            classification: QueryClassification object
            hierarchy_analysis: Legal hierarchy analysis results
            response_language: Optional response language ('ar', 'en', or None for auto)
            
        Returns:
            Formatted prompt string
        """
        # Build hierarchy context if available
        hierarchy_context = ""
        if hierarchy_analysis:
            if hierarchy_analysis.get('has_governing_law', False):
                hierarchy_context += "\nIMPORTANT: Governing law clauses are present. Law takes precedence over contract clauses.\n"
            if hierarchy_analysis.get('has_potential_conflict', False):
                hierarchy_context += "\nCONFLICT DETECTED: Law clauses override conflicting contract clauses.\n"
        
        # Build citation requirement (based on risk level)
        citation_instruction = ""
        if classification and classification.is_legal_query:
            if classification.risk_level == "high":
                citation_instruction = "\nCRITICAL: You MUST cite the specific clause using [Source N] format. Legal answers without citations are invalid.\n"
            elif classification.risk_level == "medium":
                citation_instruction = "\nIMPORTANT: Cite sources using [Source N] format when referencing information.\n"
        
        # Build clarification instruction (conditional answering with follow-up)
        clarification_instruction = ""
        if classification and classification.requires_clarification:
            missing_ctx = ", ".join(classification.missing_context)
            clarification_instruction = f"\nCLARIFICATION NEEDED: The query lacks critical context ({missing_ctx}). Provide a conditional answer based on available information and ask a follow-up question. Example: 'Under Saudi labor law, termination rules depend on contract type. Could you confirm which contract type applies?' Do NOT refuse the query - answer conditionally with available information.\n"
        
        # Build bilingual instruction
        bilingual_instruction = ""
        if response_language:
            if response_language == "ar":
                bilingual_instruction = "\nLANGUAGE REQUIREMENT: Provide your answer in Arabic. Include citations in Arabic format: [المصدر N] or [Source N]. If the source text is in English, translate relevant parts to Arabic in your answer while maintaining accuracy and keeping citations clear."
            elif response_language == "en":
                bilingual_instruction = "\nLANGUAGE REQUIREMENT: Provide your answer in English. Include citations in English format: [Source N]."
        
        # FIX 4: Document-type aware "not specified" message
        not_specified_msg = self._get_not_specified_message(document_type)
        document_type_instruction = ""
        if document_type == 'statute':
            document_type_instruction = "\n- For statutes/laws: If information is not explicitly stated, use: 'The law does not expressly state this.' Do not summarize or infer."
        
        prompt = f"""You are a legal document assistant. Answer the user's question based ONLY on the provided document context.

STRICT RULES:
- Only use information from the provided context
- ALWAYS cite sources using [Source N] format when referencing information
- Citations MUST directly support your claim, not just be thematically related
- If you find relevant information in the context that answers the question, provide that answer with citations. DO NOT say "not specified" if you have found and cited relevant information.
- ONLY state "{not_specified_msg}" if you truly cannot find ANY relevant information in the context that addresses the question
- Do not use phrases like "it seems", "probably", "might be" - be direct and factual
- Do not provide legal advice or interpretation beyond what is explicitly stated
- Present information exactly as it appears in the documents{document_type_instruction}{bilingual_instruction}{hierarchy_context}{citation_instruction}{clarification_instruction}

Context from documents:
{context}

User Question: {query}

Answer (be direct, factual, and always cite sources):"""
        
        return prompt
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sources with citation information (legacy method).
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of formatted source dictionaries
        """
        return self._format_sources_enhanced(chunks)
    
    def _format_sources_enhanced(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sources with enhanced citation information including hierarchy.
        Never exposes document_hash or internal chunk IDs.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of formatted source dictionaries with enhanced metadata
        """
        sources = []
        
        for chunk in chunks:
            # Use display_name for citation (never expose document_hash)
            display_name = chunk.get('display_name', chunk.get('document_id', 'Unknown'))
            
            # Format citation using legal reasoning service (will use display_name)
            citation = self.legal_reasoning.format_citation(chunk)
            
            sources.append({
                'text': chunk.get('text', ''),
                # Original chunk text
                'document_id': chunk.get('document_id', ''),
                # Source document ID (DOC-####)
                'display_name': display_name,
                # User-friendly display name for citations
                'page_number': chunk.get('page_number', 0),
                # Page number for citation
                'chunk_index': chunk.get('chunk_index', 0),
                # Position in document
                'score': chunk.get('score', 0.0),
                # Similarity score
                'citation': citation,
                # Enhanced citation format (uses display_name)
                'hierarchy_level': chunk.get('hierarchy_level', 'contract'),
                # Legal hierarchy level
                'legal_supremacy': chunk.get('legal_supremacy', False),
                # Supremacy indicator
                'clause_types': chunk.get('clause_types', []),
                # Clause types
                'topics': chunk.get('topics', [])
                # Topics/keywords
                # Note: clause_id and document_hash are intentionally excluded
            })
        
        return sources
    
    def query_multilingual(
        self,
        query: str,
        response_language: Optional[str] = None,
        top_k: int = None,
        document_id_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query with explicit language handling.
        Arabic queries can retrieve English documents and vice versa.
        
        Args:
            query: User query (Arabic or English)
            response_language: Desired response language ('ar', 'en', or None for auto)
            top_k: Number of chunks to retrieve
            document_id_filter: Optional filter by document ID
            
        Returns:
            Dict with answer, sources, and query
        """
        # Retrieve chunks and generate answer with language support
        # Pass response_language directly to query() so it's used in the initial prompt
        result = self.query(
            query=query,
            top_k=top_k,
            document_id_filter=document_id_filter,
            generate_response=True,
            response_language=response_language
        )
        
        return result

