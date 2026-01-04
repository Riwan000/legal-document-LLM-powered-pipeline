"""
Clause extraction service.
Uses LLM to identify and extract contract clauses verbatim with page references.
No legal interpretation - pure extraction only.
"""
from typing import List, Dict, Any, Optional
import ollama
import json
from backend.config import settings
from backend.services.document_ingestion import DocumentIngestionService


class ClauseExtractionService:
    """Service for extracting clauses from contracts."""
    
    def __init__(self):
        """Initialize the clause extraction service."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        # Connect to local Ollama server
        self.ingestion_service = DocumentIngestionService()
        # For parsing documents
    
    def extract_clauses(
        self,
        file_path: str,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract clauses from a contract document.
        
        Args:
            file_path: Path to the contract file
            document_id: Unique document identifier
            
        Returns:
            List of clause dictionaries with: type, text, page_number
        """
        from pathlib import Path
        
        file_path = Path(file_path)
        
        # Parse document to get pages with text
        pages = self.ingestion_service.parser.parse_file(file_path)
        # Get list of (text, page_number) tuples
        
        if not pages:
            return []
        
        # Extract clauses from each page
        all_clauses = []
        
        for page_data in pages:
            # Handle both old format (text, page_number) and new format (text, page_number, is_ocr)
            if len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                # Backward compatibility: assume not OCR if not specified
                text, page_number = page_data
                is_ocr = False
            
            # Process each page
            page_clauses = self._extract_clauses_from_page(text, page_number, document_id)
            # Extract clauses from this page
            all_clauses.extend(page_clauses)
            # Add to total list
        
        # Deduplicate clauses (same text might appear on multiple pages)
        unique_clauses = self._deduplicate_clauses(all_clauses)
        
        return unique_clauses
    
    def _extract_clauses_from_page(
        self,
        text: str,
        page_number: int,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract clauses from a single page of text.
        
        Args:
            text: Page text content
            page_number: Page number
            document_id: Document identifier
            
        Returns:
            List of clause dictionaries
        """
        # Build prompt for LLM
        prompt = self._build_extraction_prompt(text)
        
        try:
            # Call Ollama to extract clauses
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistent extraction
                    'format': 'json'  # Request JSON output
                }
            )
            
            # Parse JSON response - handle both old and new API formats
            if isinstance(response, dict):
                # Old format: {'response': '...'}
                response_text = response.get('response', '')
            elif hasattr(response, 'response'):
                # New format: response object with .response attribute
                response_text = response.response
            else:
                # Try to get text directly
                response_text = str(response)
            
            # If still empty, try to extract from full response
            if not response_text and isinstance(response, dict):
                # Try alternative keys
                response_text = response.get('text', response.get('content', ''))
            
            # Try to extract JSON from response
            clauses = self._parse_clause_response(response_text, page_number, document_id)
            # Parse and validate clauses
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error extracting clauses from page {page_number}: {str(e)}")
            clauses = []
        
        return clauses
    
    def _build_extraction_prompt(self, text: str) -> str:
        """
        Build prompt for clause extraction.
        
        Args:
            text: Text content to extract clauses from
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a document extraction tool. Your task is to identify and extract contract clauses from the following text.

IMPORTANT RULES:
- Extract clauses VERBATIM (exact text, do not paraphrase)
- Identify common clause types: Payment Terms, Termination, Liability, Confidentiality, Force Majeure, Dispute Resolution, Governing Law, etc.
- Return ONLY the clauses found in the text
- If no clauses are found, return empty array
- Do NOT interpret, summarize, or modify the text
- Do NOT provide legal advice

Return your response as a JSON array of objects with this exact format:
[
  {{
    "type": "Payment Terms",
    "text": "Exact clause text from document",
    "start_index": 0
  }},
  {{
    "type": "Termination",
    "text": "Exact clause text from document",
    "start_index": 150
  }}
]

Document text:
{text}

JSON response:"""
        # Prompt engineering:
        # - Clear role (extraction tool, not legal advisor)
        # - Emphasize verbatim extraction
        # - List common clause types (helps LLM identify them)
        # - Explicit JSON format requirement
        # - No interpretation rule
        
        return prompt
    
    def _parse_clause_response(
        self,
        response_text: str,
        page_number: int,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Parse LLM response and extract clause information.
        
        Args:
            response_text: LLM response text (should be JSON)
            page_number: Page number for citations
            document_id: Document identifier
            
        Returns:
            List of clause dictionaries
        """
        clauses = []
        
        # Try to extract JSON from response
        # LLM might wrap JSON in markdown or add explanations
        json_text = self._extract_json_from_response(response_text)
        
        if not json_text:
            return clauses
        
        try:
            # Parse JSON
            parsed_clauses = json.loads(json_text)
            
            # Validate and format
            if isinstance(parsed_clauses, list):
                for clause in parsed_clauses:
                    if isinstance(clause, dict) and 'text' in clause:
                        # Validate clause has required fields
                        clauses.append({
                            'type': clause.get('type', 'Unknown'),
                            # Clause type (Payment Terms, etc.)
                            'text': clause['text'],
                            # Verbatim clause text
                            'page_number': page_number,
                            # Page where clause appears
                            'document_id': document_id,
                            # Source document
                            'start_index': clause.get('start_index', 0)
                            # Character position in page (if provided)
                        })
            
        except json.JSONDecodeError as e:
            # JSON parsing failed
            print(f"Failed to parse JSON response: {str(e)}")
            # Could implement fallback parsing here
        
        return clauses
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract JSON from LLM response (might be wrapped in markdown or have explanations).
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            JSON string or None
        """
        # Remove markdown code blocks if present
        if '```json' in response_text:
            # Extract content between ```json and ```
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        elif '```' in response_text:
            # Generic code block
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        
        # Try to find JSON array
        start_brace = response_text.find('[')
        end_brace = response_text.rfind(']')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            return response_text[start_brace:end_brace + 1]
        
        # Return as-is if no markers found
        return response_text.strip()
    
    def _deduplicate_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate clauses (same text appearing on multiple pages).
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Deduplicated list
        """
        seen_texts = set()
        unique_clauses = []
        
        for clause in clauses:
            # Normalize text for comparison (lowercase, strip whitespace)
            normalized_text = clause['text'].lower().strip()
            
            if normalized_text not in seen_texts:
                # First occurrence of this clause
                seen_texts.add(normalized_text)
                unique_clauses.append(clause)
            else:
                # Duplicate - update page number to include both pages
                # Find existing clause and update page reference
                for existing in unique_clauses:
                    if existing['text'].lower().strip() == normalized_text:
                        # If page numbers differ, note both
                        if existing['page_number'] != clause['page_number']:
                            existing['page_number'] = f"{existing['page_number']}, {clause['page_number']}"
                        break
        
        return unique_clauses
    
    def extract_clauses_by_type(
        self,
        file_path: str,
        document_id: str,
        clause_types: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract clauses filtered by type.
        
        Args:
            file_path: Path to contract file
            document_id: Document identifier
            clause_types: List of clause types to extract (e.g., ["Payment Terms", "Termination"])
            
        Returns:
            Dictionary mapping clause type to list of clauses
        """
        all_clauses = self.extract_clauses(file_path, document_id)
        # Extract all clauses
        
        # Filter by type
        filtered = {clause_type: [] for clause_type in clause_types}
        # Initialize dict with empty lists for each type
        
        for clause in all_clauses:
            clause_type = clause.get('type', '')
            # Get clause type
            
            # Case-insensitive matching
            for target_type in clause_types:
                if target_type.lower() in clause_type.lower() or clause_type.lower() in target_type.lower():
                    # Match if types overlap (handles variations)
                    filtered[target_type].append(clause)
                    break
        
        return filtered

