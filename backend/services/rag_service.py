"""
RAG (Retrieval-Augmented Generation) service.
Retrieves relevant document chunks and generates responses using Ollama LLM.
"""
from typing import List, Dict, Any, Optional
import ollama
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.config import settings


class RAGService:
    """Service for RAG-based query answering with citations."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
        """
        self.embedding_service = embedding_service
        # Embedding service for query encoding
        self.vector_store = vector_store
        # Vector store for retrieving relevant chunks
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        # Create Ollama client pointing to local server
    
    def search(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks (without LLM generation).
        
        Args:
            query: Search query (can be Arabic or English)
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        # Convert query text to embedding vector
        # Multilingual model handles Arabic and English queries
        
        # Search vector store with default threshold
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id_filter=document_id_filter
        )
        # Retrieve top-k most similar chunks
        
        # If no results found, try with lower threshold (for OCR documents or difficult queries)
        if not results and top_k:
            # Try with minimum threshold to catch OCR documents or semantically related content
            # This helps with queries that have lower semantic similarity due to OCR errors or terminology differences
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id_filter=document_id_filter,
                similarity_threshold=settings.MIN_SIMILARITY_THRESHOLD
            )
            # Note: MIN_SIMILARITY_THRESHOLD is applied, but OCR chunks will still use OCR threshold if detected
        
        return results
    
    def query(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG: retrieve relevant chunks and generate response.
        
        Args:
            query: User query (can be Arabic or English)
            top_k: Number of chunks to retrieve
            document_id_filter: Optional filter by document ID
            generate_response: Whether to generate LLM response (True) or just return chunks (False)
            
        Returns:
            Dict with:
                - answer: LLM-generated response (if generate_response=True)
                - sources: List of source chunks with citations
                - query: Original query
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.search(query, top_k=top_k, document_id_filter=document_id_filter)
        
        if not chunks:
            # No relevant chunks found
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'query': query
            }
        
        if not generate_response:
            # Return chunks only (no LLM generation)
            return {
                'answer': None,
                'sources': chunks,
                'query': query
            }
        
        # Step 2: Build context from retrieved chunks
        context = self._build_context(chunks)
        # Combine chunks into a single context string
        
        # Step 3: Generate prompt for LLM
        prompt = self._build_prompt(query, context)
        # Create prompt that includes context and query
        
        # Step 4: Generate response using Ollama
        try:
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt
            )
            # Call Ollama API to generate response
            # Uses local LLM (no data sent externally)
            
            answer = response['response']
            # Extract generated text from response
            
        except Exception as e:
            # Handle Ollama errors (server not running, model not found, etc.)
            answer = f"Error generating response: {str(e)}. Please ensure Ollama is running and the model is available."
        
        # Step 5: Format sources with citations
        sources = self._format_sources(chunks)
        # Add citation format (document, page number)
        
        return {
            'answer': answer,
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
            # Format each chunk with citation info
            context_parts.append(
                f"[Source {i} - Document: {chunk['document_id']}, Page: {chunk['page_number']}]\n"
                f"{chunk['text']}\n"
            )
            # Include source metadata and text
            # Format: [Source N - Document: ID, Page: X]\nText\n
        
        return "\n".join(context_parts)
        # Join all chunks with newlines
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM with context and query.
        
        Args:
            query: User query
            context: Retrieved document context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a legal document assistant. Answer the user's question based ONLY on the provided document context.

IMPORTANT RULES:
- Only use information from the provided context
- If the answer is not in the context, say so
- Cite sources using [Source N] format when referencing information
- Do not provide legal advice or interpretation
- Simply extract and present information from the documents

Context from documents:
{context}

User Question: {query}

Answer:"""
        # Prompt engineering:
        # - Clear role definition
        # - Instructions to only use context
        # - Citation format
        # - Legal disclaimer (no advice)
        
        return prompt
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sources with citation information.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for chunk in chunks:
            sources.append({
                'text': chunk['text'],
                # Original chunk text
                'document_id': chunk['document_id'],
                # Source document ID
                'page_number': chunk['page_number'],
                # Page number for citation
                'chunk_index': chunk.get('chunk_index', 0),
                # Position in document
                'score': chunk.get('score', 0.0),
                # Similarity score
                'citation': f"Document {chunk['document_id'][:8]}..., Page {chunk['page_number']}"
                # Human-readable citation format
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
        
        # If response language is specified, add instruction to prompt
        if response_language:
            # Modify prompt to request specific language
            chunks = self.search(query, top_k=top_k, document_id_filter=document_id_filter)
            context = self._build_context(chunks)
            
            language_instruction = "ar" if response_language == "ar" else "en"
            prompt = self._build_prompt(query, context)
            prompt += f"\n\nPlease respond in {language_instruction}."
            # Add language instruction to prompt
            
            try:
                response = self.ollama_client.generate(
                    model=settings.OLLAMA_MODEL,
                    prompt=prompt
                )
                result['answer'] = response['response']
            except Exception as e:
                result['answer'] = f"Error: {str(e)}"
        
        return result

