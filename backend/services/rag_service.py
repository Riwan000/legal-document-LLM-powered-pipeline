"""
RAG (Retrieval-Augmented Generation) service.
Retrieves relevant document chunks and generates responses using Ollama LLM.
"""
from typing import List, Dict, Any, Optional
import ollama
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.query_classifier import QueryClassifier
from backend.services.legal_reasoning_service import LegalReasoningService
from backend.services.legal_hierarchy_service import LegalHierarchyService
from backend.services.clause_store import ClauseStore
from backend.models.document import AnswerResponse, LegalHierarchyLevel
from backend.config import settings


class RAGService:
    """Service for RAG-based query answering with citations."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        clause_store: Optional[ClauseStore] = None
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            clause_store: Optional clause store for structured clause retrieval
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.clause_store = clause_store
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
        priority_clause_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks with clause-aware priority boosting.
        
        Args:
            query: Search query (can be Arabic or English)
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            priority_clause_types: List of clause types to boost (e.g., ["Governing Law", "Termination"])
            
        Returns:
            List of relevant chunks with metadata, ranked by authority
        """
        # 1) Classify the query and choose “priority” clause types (boosting).
        if priority_clause_types is None:
            classification = self.query_classifier.classify_query(query)
            priority_clause_types = self._get_priority_clause_types(classification)
        
        # 2) Encode query into an embedding vector (multilingual model supports Arabic/English).
        query_embedding = self.embedding_service.embed_text(query)
        
        # 3) Retrieve top-k chunks. If we have priority clause types, do a “boosted” re-ranking.
        if priority_clause_types:
            results = self.vector_store.search_with_priority(
                query_embedding=query_embedding,
                priority_clause_types=priority_clause_types,
                top_k=top_k,
                document_id_filter=document_id_filter
            )
        else:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id_filter=document_id_filter
            )
        
        # 4) If semantic results are empty, lower the similarity threshold to be more permissive.
        # This mainly helps OCR-derived text, where embeddings can be noisier.
        if not results and top_k:
            if priority_clause_types:
                results = self.vector_store.search_with_priority(
                    query_embedding=query_embedding,
                    priority_clause_types=priority_clause_types,
                    top_k=top_k,
                    document_id_filter=document_id_filter,
                    similarity_threshold=settings.MIN_SIMILARITY_THRESHOLD
                )
            else:
                results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    document_id_filter=document_id_filter,
                    similarity_threshold=settings.MIN_SIMILARITY_THRESHOLD
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
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'this', 'that', 'these', 'those'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            return []
        
        # Search in vector store metadata for matching chunks within this document.
        # Note: we search stored chunk text; we do not hit FAISS here.
        matching_chunks = []
        for metadata in self.vector_store.metadata:
            if metadata.get('document_id') != document_id:
                continue
            
            chunk_text = metadata.get('text', '').lower()
            
            # Count keyword matches (simple containment check).
            matches = sum(1 for keyword in keywords if keyword in chunk_text)
            
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
    
    def _get_priority_clause_types(self, classification) -> List[str]:
        """Determine priority clause types based on query classification."""
        priority_types = []
        
        # If requires legal hierarchy, boost governing law clauses
        if classification.requires_legal_hierarchy:
            priority_types.append("Governing Law")
        
        # Boost clause types based on query type
        query_type = classification.query_type
        if query_type == "termination":
            priority_types.append("Termination")
        elif query_type == "benefits":
            priority_types.append("Payment Terms")
            priority_types.append("Benefits")
        elif query_type == "compensation":
            priority_types.append("Payment Terms")
            priority_types.append("Compensation")
        elif query_type == "legality":
            priority_types.append("Governing Law")
        
        return priority_types
    
    def query(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using rule-guided RAG with legal hierarchy and citation enforcement.
        
        Args:
            query: User query (can be Arabic or English)
            top_k: Number of chunks to retrieve
            document_id_filter: Optional filter by document ID
            generate_response: Whether to generate LLM response (True) or just return chunks (False)
            
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
        
        # Step 3: Try to retrieve structured clauses first (if clause store available)
        structured_clauses: List[Any] = []
        if self.clause_store and document_id_filter:
            try:
                # Query clause store for relevant clauses
                structured_clauses = self.clause_store.query_clauses(
                    document_id=document_id_filter,
                    clause_type=classification.query_type if classification.query_type != "general" else None
                )
            except Exception as e:
                print(f"Error querying clause store: {str(e)}")
                structured_clauses = []
        
        # Step 4: Retrieve relevant chunks with priority boosting
        chunks = self.search(
            query, 
            top_k=top_k, 
            document_id_filter=document_id_filter,
            priority_clause_types=self._get_priority_clause_types(classification)
        )
        
        # Step 5: Analyze legal hierarchy (use structured clauses if available)
        if structured_clauses:
            # Use structured clauses for hierarchy analysis
            hierarchy_analysis = self._analyze_hierarchy_from_clauses(structured_clauses)
        else:
            hierarchy_analysis = self.legal_reasoning.analyze_legal_hierarchy(query, chunks) if chunks else {}
        
        # Step 6: Check if topic is not specified
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
                    # Update hierarchy analysis with keyword results
                    hierarchy_analysis = self.legal_reasoning.analyze_legal_hierarchy(query, chunks) if chunks else {}
                else:
                    # Truly no results found
                    return {
                        'status': 'not_specified',
                        'answer': 'This is not specified in the provided contract.',
                        'citation': None,
                        'confidence': 'low',
                        'refusal_reason': None,
                        'hierarchy_analysis': hierarchy_analysis,
                        'sources': [],
                        'query': query
                    }
            else:
                # No document_id, can't do keyword search
                return {
                    'status': 'not_specified',
                    'answer': 'This is not specified in the provided contract.',
                    'citation': None,
                    'confidence': 'low',
                    'refusal_reason': None,
                    'hierarchy_analysis': hierarchy_analysis,
                    'sources': [],
                    'query': query
                }
        
        # Only check detect_not_specified if we have chunks but they might not be relevant
        # Be more lenient - if keyword matches exist, don't mark as not specified
        if chunks and not structured_clauses:
            # Check if we have keyword matches - if so, content exists
            has_keyword_matches = any(c.get('keyword_match', False) for c in chunks)
            if not has_keyword_matches:
                # Only check detect_not_specified if no keyword matches
                if self.legal_reasoning.detect_not_specified(chunks, query, classification):
                    return {
                        'status': 'not_specified',
                        'answer': 'This is not specified in the provided contract.',
                        'citation': None,
                        'confidence': 'low',
                        'refusal_reason': None,
                        'hierarchy_analysis': hierarchy_analysis,
                        'sources': [],
                        'query': query
                    }
        
        # Step 7: Enforce citation requirement for legal queries
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
        
        # Step 8: Determine if explicit clause found
        has_explicit_clause = (len(structured_clauses) > 0) or (len(chunks) > 0 and chunks[0].get('score', 0.0) >= 0.6)
        
        # Step 9: Calculate confidence
        confidence = self.legal_reasoning.calculate_confidence(chunks, hierarchy_analysis, has_explicit_clause)
        
        # Step 10: Determine status
        if hierarchy_analysis.get('has_governing_law', False) and hierarchy_analysis.get('has_conflict', False):
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
        
        # Step 11: Build context from retrieved chunks/clauses
        if structured_clauses:
            context = self._build_context_from_clauses(structured_clauses)
        else:
            context = self._build_context(chunks)
        # Combine chunks into a single context string
        
        # Step 12: Generate prompt for LLM with legal-safe template
        prompt = self._build_legal_prompt(query, context, classification, hierarchy_analysis)
        # Create prompt with legal hierarchy awareness and citation requirements
        
        # Step 13: Generate response using Ollama
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
                not_specified_phrase = "This is not specified in the provided contract."
                if answer_clean.endswith(not_specified_phrase):
                    answer = answer_clean[:-len(not_specified_phrase)].strip()
                # Also remove if it appears as a separate sentence at the end
                elif answer_clean.endswith('.' + not_specified_phrase):
                    answer = answer_clean[:-len('.' + not_specified_phrase)].strip()
                elif answer_clean.endswith('\n' + not_specified_phrase):
                    answer = answer_clean[:-len('\n' + not_specified_phrase)].strip()
            
        except Exception as e:
            # Handle Ollama errors (server not running, model not found, etc.)
            answer = f"Error generating response: {str(e)}. Please ensure Ollama is running and the model is available."
        
        # Step 14: Format sources with enhanced citations
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
        hierarchy_analysis: Dict[str, Any] = None
    ) -> str:
        """
        Build legal-safe prompt with hierarchy awareness and citation requirements.
        
        Args:
            query: User query
            context: Retrieved document context
            classification: QueryClassification object
            hierarchy_analysis: Legal hierarchy analysis results
            
        Returns:
            Formatted prompt string
        """
        # Build hierarchy context if available
        hierarchy_context = ""
        if hierarchy_analysis:
            if hierarchy_analysis.get('has_governing_law', False):
                hierarchy_context += "\nIMPORTANT: Governing law clauses are present. Law takes precedence over contract clauses.\n"
            if hierarchy_analysis.get('has_conflict', False):
                hierarchy_context += "\nCONFLICT DETECTED: Law clauses override conflicting contract clauses.\n"
        
        # Build citation requirement
        citation_instruction = ""
        if classification and classification.is_legal_query:
            citation_instruction = "\nCRITICAL: You MUST cite the specific clause using [Source N] format. Legal answers without citations are invalid.\n"
        
        prompt = f"""You are a legal document assistant. Answer the user's question based ONLY on the provided document context.

STRICT RULES:
- Only use information from the provided context
- ALWAYS cite sources using [Source N] format when referencing information
- If you find relevant information in the context that answers the question, provide that answer with citations. DO NOT say "not specified" if you have found and cited relevant information.
- ONLY state "This is not specified in the provided contract." if you truly cannot find ANY relevant information in the context that addresses the question
- Do not use phrases like "it seems", "probably", "might be" - be direct and factual
- Do not provide legal advice or interpretation beyond what is explicitly stated
- Present information exactly as it appears in the documents{hierarchy_context}{citation_instruction}

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
        # Retrieve chunks (multilingual embeddings handle cross-language)
        result = self.query(
            query=query,
            top_k=top_k,
            document_id_filter=document_id_filter,
            generate_response=True
        )
        
        # If response language is specified, regenerate answer in requested language
        if response_language and result.get('status') not in ['refused', 'not_specified']:
            # Modify prompt to request specific language
            chunks = result.get('sources', [])
            if chunks:
                context = self._build_context(chunks)
                classification = self.query_classifier.classify_query(query)
                hierarchy_analysis = result.get('hierarchy_analysis', {})
                
                language_instruction = "ar" if response_language == "ar" else "en"
                prompt = self._build_legal_prompt(query, context, classification, hierarchy_analysis)
                prompt += f"\n\nPlease respond in {language_instruction}."
                
                try:
                    response = self.ollama_client.generate(
                        model=settings.OLLAMA_MODEL,
                        prompt=prompt,
                        options={'temperature': 0.1}
                    )
                    answer = response['response'] if isinstance(response, dict) else str(response)
                    result['answer'] = answer
                except Exception as e:
                    result['answer'] = f"Error: {str(e)}"
        
        return result

