"""
OCR Cleanup & Text Normalization utilities.
Handles language-aware spell correction, de-hyphenation, and OCR artifact removal.
"""
import re
from typing import Dict, List, Tuple, Optional


class OCRCleanupService:
    """Service for cleaning and normalizing OCR text while preserving raw text."""
    
    def __init__(self):
        """Initialize OCR cleanup service with legal vocabulary patterns."""
        # Common OCR error patterns for legal text
        self.ocr_corrections = {
            # Common OCR errors
            'shali': 'shall',
            'const.tule': 'constitute',
            'agreemeni': 'agreement',
            'authort.es': 'authorities',
            'Nolce': 'Notice',
            'terminateg': 'terminate',
            'jne': 'the',
            'Dy a': 'by a',
            'Oy ether': 'either',
            'l€rms': 'terms',
            # Legal terms
            'employmeni': 'employment',
            'compen.sation': 'compensation',
            'termina.tion': 'termination',
            'obliga.tion': 'obligation',
            'jurisdic.tion': 'jurisdiction',
            'arbitra.tion': 'arbitration',
            'confiden.tial': 'confidential',
            'liabili.ty': 'liability',
        }
        
        # Arabic OCR patterns (common errors)
        self.arabic_ocr_corrections = {
            # Add Arabic-specific corrections if needed
        }
    
    def normalize_text(
        self,
        raw_text: str,
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Normalize OCR text while preserving raw text.
        
        Args:
            raw_text: Original OCR output
            language: Language code ('ar', 'en', or None for auto-detect)
            
        Returns:
            Dict with 'raw_text' and 'clean_text'
        """
        if not raw_text:
            return {'raw_text': raw_text, 'clean_text': raw_text}
        
        # Detect language if not provided
        if language is None:
            language = self._detect_language(raw_text)
        
        clean_text = raw_text
        
        # Step 1: De-hyphenation (join words split across lines)
        clean_text = self._dehyphenate(clean_text)
        
        # Step 2: Remove OCR artifacts
        clean_text = self._remove_ocr_artifacts(clean_text)
        
        # Step 3: Language-aware spell correction
        if language == 'en':
            clean_text = self._correct_english_ocr_errors(clean_text)
        elif language == 'ar':
            clean_text = self._correct_arabic_ocr_errors(clean_text)
        else:
            # Try both if uncertain
            clean_text = self._correct_english_ocr_errors(clean_text)
        
        # Step 4: Normalize whitespace
        clean_text = self._normalize_whitespace(clean_text)
        
        return {
            'raw_text': raw_text,
            'clean_text': clean_text
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        text_chars = set(text.replace(' ', ''))
        arabic_ratio = len(text_chars.intersection(arabic_chars)) / max(len(text_chars), 1)
        return 'ar' if arabic_ratio > 0.1 else 'en'
    
    def _dehyphenate(self, text: str) -> str:
        """Join words split across lines by hyphens."""
        # Pattern: word-\nword -> word word
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Pattern: word-\n -> word
        text = re.sub(r'(\w+)-\s*\n', r'\1 ', text)
        return text
    
    def _remove_ocr_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts."""
        # Remove stray punctuation that's clearly OCR errors
        text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1\2', text)  # Remove dots between letters
        text = re.sub(r'([a-zA-Z])\s+\.\s+([a-zA-Z])', r'\1 \2', text)  # Remove isolated dots
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _correct_english_ocr_errors(self, text: str) -> str:
        """Apply English OCR error corrections."""
        for error, correction in self.ocr_corrections.items():
            # Case-insensitive replacement
            text = re.sub(re.escape(error), correction, text, flags=re.IGNORECASE)
        return text
    
    def _correct_arabic_ocr_errors(self, text: str) -> str:
        """Apply Arabic OCR error corrections."""
        for error, correction in self.arabic_ocr_corrections.items():
            text = text.replace(error, correction)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Trim
        text = text.strip()
        return text
    
    def separate_languages(self, text: str) -> Dict[str, str]:
        """
        Separate Arabic and English text.
        
        Args:
            text: Mixed-language text
            
        Returns:
            Dict with 'arabic' and 'english' keys
        """
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        
        arabic_lines = []
        english_lines = []
        
        for line in text.split('\n'):
            line_chars = set(line.replace(' ', ''))
            arabic_ratio = len(line_chars.intersection(arabic_chars)) / max(len(line_chars), 1)
            
            if arabic_ratio > 0.3:  # More than 30% Arabic characters
                arabic_lines.append(line)
            else:
                english_lines.append(line)
        
        return {
            'arabic': '\n'.join(arabic_lines),
            'english': '\n'.join(english_lines)
        }

