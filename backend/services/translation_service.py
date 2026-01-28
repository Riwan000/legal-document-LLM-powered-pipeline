"""
Translation service for bilingual support.
Handles Arabic-English cross-language document search and response generation.
"""
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.rag_service import RAGService

from backend.config import settings

# Optional translation imports
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


class TranslationService:
    """Service for handling bilingual queries and responses."""
    
    def __init__(self, rag_service: "RAGService"):
        """
        Initialize the translation service.
        
        Args:
            rag_service: RAG service for cross-language search
        """
        self.rag_service = rag_service
        # RAG service already uses multilingual embeddings
        
        # Initialize translator if available
        self.translator = None
        if TRANSLATION_AVAILABLE:
            try:
                self.translator = GoogleTranslator
            except Exception as e:
                print(f"Warning: Translation service not available: {str(e)}")
                self.translator = None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code ('ar' or 'en')
            target_lang: Target language code ('ar' or 'en')
            
        Returns:
            Translated text, or original text if translation fails
        """
        # If languages are the same, no translation needed
        if source_lang == target_lang:
            return text
        
        # If translation not available, return original
        if not TRANSLATION_AVAILABLE or not self.translator:
            if settings.TRANSLATION_FALLBACK_TO_ORIGINAL:
                return text
            raise RuntimeError("Translation service not available")
        
        try:
            # Map language codes to GoogleTranslator format
            lang_map = {'ar': 'ar', 'en': 'en'}
            source = lang_map.get(source_lang, 'en')
            target = lang_map.get(target_lang, 'en')
            
            # Translate using GoogleTranslator
            translator_instance = self.translator(source=source, target=target)
            translated = translator_instance.translate(text)
            
            # Check if GoogleTranslator returned an error message instead of translation
            error_indicators = [
                "I'm sorry",
                "cannot provide a translation",
                "Can I help you with something else",
                "translation failed",
                "unable to translate"
            ]
            if any(indicator.lower() in translated.lower() for indicator in error_indicators):
                print(f"Warning: Translation returned error message: {translated}")
                if settings.TRANSLATION_FALLBACK_TO_ORIGINAL:
                    return text
                raise RuntimeError(f"Translation failed: GoogleTranslator returned error message: {translated}")
            
            return translated
        except Exception as e:
            print(f"Warning: Translation failed: {str(e)}")
            if settings.TRANSLATION_FALLBACK_TO_ORIGINAL:
                return text
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text (simple heuristic).
        For production, use a proper language detection library.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('ar' for Arabic, 'en' for English)
        """
        # Simple heuristic: check for Arabic characters
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        # Arabic Unicode character range
        
        text_chars = set(text.replace(' ', ''))
        # Remove spaces and get unique characters
        
        arabic_ratio = len(text_chars.intersection(arabic_chars)) / max(len(text_chars), 1)
        # Calculate ratio of Arabic characters
        
        if arabic_ratio > 0.1:  # More than 10% Arabic characters
            return 'ar'
        else:
            return 'en'
    
    def query_bilingual(
        self,
        query: str,
        response_language: Optional[str] = None,
        top_k: int = None,
        document_id_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query with automatic language detection and cross-language search.
        
        Args:
            query: User query (Arabic or English)
            response_language: Desired response language ('ar', 'en', or None for auto)
            top_k: Number of chunks to retrieve
            document_id_filter: Optional filter by document ID
            
        Returns:
            Dict with answer, sources, query, and detected language
        """
        # Detect query language
        query_language = self.detect_language(query)
        # Determine if query is Arabic or English
        
        # Auto-set response language if not specified
        if response_language is None:
            response_language = query_language
            # Respond in same language as query
        
        # Use RAG service's multilingual query method
        result = self.rag_service.query_multilingual(
            query=query,
            response_language=response_language,
            top_k=top_k,
            document_id_filter=document_id_filter
        )
        # RAG service handles:
        # - Multilingual embeddings (Arabic query → English docs)
        # - Response language control
        
        # Add language metadata
        result['query_language'] = query_language
        result['response_language'] = response_language
        
        return result
    
    def search_bilingual(
        self,
        query: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search documents with bilingual support (no LLM generation).
        
        Args:
            query: Search query (Arabic or English)
            top_k: Number of results to return
            document_id_filter: Optional filter by document ID
            
        Returns:
            Dict with search results and language info
        """
        query_language = self.detect_language(query)
        # Detect query language
        
        # Search using RAG service (multilingual embeddings handle cross-language)
        results = self.rag_service.search(
            query=query,
            top_k=top_k,
            document_id_filter=document_id_filter
        )
        # Multilingual embeddings enable:
        # - Arabic query → English document chunks
        # - English query → Arabic document chunks
        
        return {
            'results': results,
            'query_language': query_language,
            'total_results': len(results)
        }
    
    def explain_bilingual_capability(self) -> str:
        """
        Generate explanation of how bilingual search works.
        
        Returns:
            Explanation text
        """
        return """
## How Bilingual Search Works

### Multilingual Embeddings
The system uses a multilingual embedding model (paraphrase-multilingual-MiniLM-L12-v2) that:
- Maps Arabic and English text to the same embedding space
- Enables semantic similarity across languages
- No translation needed - direct semantic matching

### Cross-Language Retrieval
- **Arabic queries** can retrieve relevant **English documents**
- **English queries** can retrieve relevant **Arabic documents**
- Works because similar meanings map to similar vectors, regardless of language

### Response Language
- Query language is automatically detected
- Response language can be controlled (Arabic or English)
- LLM generates response in requested language

### Example
- Query (Arabic): "ما هي شروط الدفع؟" (What are the payment terms?)
- System retrieves: English contract documents with payment terms
- Response: Generated in Arabic (or English, if requested)

### Technical Details
- Embeddings are language-agnostic (same space for all languages)
- Semantic similarity works across languages
- No translation step required for retrieval
- Translation only needed for response generation (if different from query language)
"""

