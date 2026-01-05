"""
Deterministic chunk classification for case files.
Classifies chunks into legal document sections.
"""
from typing import List, Dict, Any, Optional
import ollama
import re
import json
from backend.config import settings


class CaseChunkClassifier:
    """Deterministic classifier for case file chunks."""
    
    # Allowed chunk types per PRD
    ALLOWED_TYPES = [
        "background",
        "procedural_history",
        "issue_framing",
        "argument_claimant",
        "argument_defendant",
        "holding",
        "reasoning",
        "citation"
    ]
    
    def __init__(self):
        """Initialize the classifier."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def classify_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        page_number: int,
        chunk_index: int,
        total_chunks: int,
        document_id: str
    ) -> str:
        """
        Classify a chunk into one of the allowed types.
        Uses heuristics first, then LLM fallback.
        
        Args:
            chunk_text: Text content of chunk
            chunk_id: Stable chunk identifier
            page_number: Page number
            chunk_index: Chunk index within document
            total_chunks: Total number of chunks
            document_id: Document ID
            
        Returns:
            One of the allowed chunk types
        """
        # Try heuristic classification first
        heuristic_type = self._classify_by_heuristics(
            chunk_text, page_number, chunk_index, total_chunks
        )
        
        if heuristic_type:
            return heuristic_type
        
        # Fallback to LLM classification (deterministic)
        return self._classify_with_llm(chunk_text, chunk_id)
    
    def _classify_by_heuristics(
        self,
        text: str,
        page_number: int,
        chunk_index: int,
        total_chunks: int
    ) -> Optional[str]:
        """
        Heuristic classification based on keywords, position, and patterns.
        
        Returns:
            Chunk type or None if ambiguous
        """
        text_lower = text.lower()
        
        # Procedural history indicators
        procedural_keywords = [
            "filed", "complaint", "motion", "order", "judgment", "appeal",
            "hearing", "trial", "deposition", "discovery", "subpoena",
            "petition", "response", "reply", "brief", "memorandum",
            "on [date]", "in [year]", "the court", "the judge"
        ]
        if any(kw in text_lower for kw in procedural_keywords):
            # Check for date patterns
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
                r'\d{4}'
            ]
            if any(re.search(pattern, text_lower) for pattern in date_patterns):
                return "procedural_history"
        
        # Issue framing indicators
        issue_keywords = [
            "issue", "question", "whether", "the court must determine",
            "the central question", "the key issue", "the main issue",
            "this case presents", "the dispute concerns"
        ]
        if any(kw in text_lower for kw in issue_keywords):
            return "issue_framing"
        
        # Argument indicators (claimant/plaintiff)
        claimant_keywords = [
            "plaintiff", "claimant", "appellant", "petitioner",
            "plaintiff argues", "claimant contends", "appellant claims",
            "we submit", "we argue", "our position"
        ]
        if any(kw in text_lower for kw in claimant_keywords):
            return "argument_claimant"
        
        # Argument indicators (defendant/respondent)
        defendant_keywords = [
            "defendant", "respondent", "appellee",
            "defendant argues", "respondent contends", "appellee claims",
            "defendant submits", "respondent maintains"
        ]
        if any(kw in text_lower for kw in defendant_keywords):
            return "argument_defendant"
        
        # Holding indicators
        holding_keywords = [
            "the court holds", "we hold", "it is held", "the court finds",
            "the court concludes", "the court rules", "judgment is",
            "the court grants", "the court denies", "the court affirms",
            "the court reverses"
        ]
        if any(kw in text_lower for kw in holding_keywords):
            return "holding"
        
        # Reasoning indicators
        reasoning_keywords = [
            "because", "therefore", "thus", "consequently", "accordingly",
            "the reason", "the rationale", "the basis", "in support",
            "this conclusion", "this finding"
        ]
        if any(kw in text_lower for kw in reasoning_keywords):
            return "reasoning"
        
        # Citation indicators
        citation_patterns = [
            r'\d+\s+[A-Z]\.\d+',  # Case citations like "123 F.3d"
            r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+',  # Case names like "Smith v. Jones"
            r'\d+\s+U\.S\.C\.',  # USC citations
            r'[A-Z]{2,}\s+\d+',  # Abbreviated citations
        ]
        if any(re.search(pattern, text) for pattern in citation_patterns):
            return "citation"
        
        # Background indicators (often early in document)
        if chunk_index < total_chunks * 0.2:  # First 20% of chunks
            background_keywords = [
                "background", "facts", "the parties", "this case involves",
                "the dispute arose", "the following facts", "the record shows"
            ]
            if any(kw in text_lower for kw in background_keywords):
                return "background"
        
        # Default to background if early in document and no other match
        if chunk_index < total_chunks * 0.1:
            return "background"
        
        return None  # Ambiguous, needs LLM
    
    def _classify_with_llm(self, chunk_text: str, chunk_id: str) -> str:
        """
        Classify chunk using LLM with deterministic settings.
        
        Args:
            chunk_text: Text content
            chunk_id: Chunk identifier
            
        Returns:
            One of the allowed chunk types
        """
        prompt = f"""Classify the following legal document chunk into exactly one of these types:
- background: Factual background, case context, parties
- procedural_history: Court proceedings, filings, dates, orders
- issue_framing: Legal questions, issues to be decided
- argument_claimant: Arguments by plaintiff/claimant/appellant
- argument_defendant: Arguments by defendant/respondent/appellee
- holding: Court's decision, judgment, ruling
- reasoning: Legal reasoning, analysis, justification
- citation: Case citations, statutory references

Return ONLY a JSON object with this exact structure:
{{"chunk_id": "{chunk_id}", "chunk_type": "<one of the types above>"}}

Chunk text:
{chunk_text[:2000]}  # Limit to avoid token limits

JSON response:"""
        
        try:
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': settings.CASE_SUMMARY_TEMPERATURE,
                    'seed': settings.CASE_SUMMARY_SEED,
                    'format': 'json'
                }
            )
            
            response_text = response.get('response', '')
            # Extract JSON
            json_text = self._extract_json(response_text)
            
            if json_text:
                result = json.loads(json_text)
                chunk_type = result.get('chunk_type', '').lower()
                
                # Validate it's an allowed type
                if chunk_type in self.ALLOWED_TYPES:
                    return chunk_type
            
            # Fallback to background if LLM fails
            return "background"
            
        except Exception as e:
            print(f"Error in LLM classification for {chunk_id}: {e}")
            # Safe fallback
            return "background"
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response."""
        # Try to find JSON object
        start_brace = text.find('{')
        end_brace = text.rfind('}')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            return text[start_brace:end_brace + 1]
        
        return None
    
    def classify_chunks_batch(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple chunks, adding chunk_type to each.
        
        Args:
            chunks: List of chunk dicts with text, chunk_id, page_number, etc.
            
        Returns:
            Same chunks with chunk_type added
        """
        total_chunks = len(chunks)
        
        for chunk in chunks:
            if 'chunk_type' not in chunk or not chunk.get('chunk_type'):
                chunk_type = self.classify_chunk(
                    chunk_text=chunk.get('text', ''),
                    chunk_id=chunk.get('chunk_id', ''),
                    page_number=chunk.get('page_number', 0),
                    chunk_index=chunk.get('chunk_index', 0),
                    total_chunks=total_chunks,
                    document_id=chunk.get('document_id', '')
                )
                chunk['chunk_type'] = chunk_type
        
        return chunks

