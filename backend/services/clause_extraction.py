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
from backend.services.legal_hierarchy_service import LegalHierarchyService
from backend.services.structured_clause_extraction import StructuredClauseExtractionService


class ClauseExtractionService:
    """Service for extracting clauses from contracts."""
    
    def __init__(self):
        """Initialize the clause extraction service."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        # Connect to local Ollama server
        self.ingestion_service = DocumentIngestionService()
        # For parsing documents
        self.hierarchy_service = LegalHierarchyService()
        # For detecting legal hierarchy and extracting metadata
        self.structured_extractor = StructuredClauseExtractionService()
        # For structured clause extraction with authority modeling
    
    def extract_clauses(
        self,
        file_path: str,
        document_id: str,
        use_structured: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract clauses from a contract document.
        
        Args:
            file_path: Path to the contract file
            document_id: Unique document identifier
            use_structured: If True, use structured extraction engine (default: True)
            
        Returns:
            List of clause dictionaries with: type, text, page_number
            If use_structured=True, returns structured clauses with authority and evidence
        """
        from pathlib import Path
        
        file_path = Path(file_path)
        
        # Use structured extraction if requested (default)
        if use_structured:
            try:
                structured_clauses = self.structured_extractor.extract_structured_clauses(
                    str(file_path), document_id
                )
                # Convert to dict format for backward compatibility
                return [self._structured_to_dict(clause) for clause in structured_clauses]
            except Exception as e:
                print(f"Structured extraction failed, falling back to legacy: {str(e)}")
                # Fall through to legacy extraction
        
        # Legacy extraction method
        pages = self.ingestion_service.parser.parse_file(file_path)
        if not pages:
            return []
        
        all_clauses = []
        for page_data in pages:
            if len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                text, page_number = page_data
                is_ocr = False
            
            page_clauses = self._extract_clauses_from_page(text, page_number, document_id)
            all_clauses.extend(page_clauses)
        
        unique_clauses = self._deduplicate_clauses(all_clauses)
        return unique_clauses
    
    def _structured_to_dict(self, clause) -> Dict[str, Any]:
        """Convert StructuredClause to dict format for backward compatibility."""
        evidence = clause.evidence[0] if clause.evidence else None
        return {
            'clause_id': clause.clause_id,
            'type': clause.type.value if hasattr(clause.type, 'value') else str(clause.type),
            'title': clause.title,
            'text': evidence.clean_text if evidence else '',
            'raw_text': evidence.raw_text if evidence else '',
            'page_number': evidence.page if evidence else 0,
            'paragraph': evidence.paragraph if evidence else None,
            'line_start': evidence.line_start if evidence else None,
            'line_end': evidence.line_end if evidence else None,
            'document_id': clause.clause_id.split('_')[1] if '_' in clause.clause_id else '',
            'authority_level': clause.authority_level.value if hasattr(clause.authority_level, 'value') else str(clause.authority_level),
            'can_override_contract': clause.can_override_contract,
            'overrides': clause.overrides,
            'jurisdiction': clause.jurisdiction,
            'subtype': clause.subtype,
            'explicitly_provided': clause.explicitly_provided,
            'linked_clause_id': clause.linked_clause_id,
            'consistency_flag': clause.consistency_flag,
            'language': clause.language,
            'topics': clause.metadata.get('topics', []),
            'hierarchy_level': 'law' if clause.authority_level.value == 'supreme' else 'contract',
            'legal_supremacy': clause.can_override_contract,
            'start_index': 0  # For backward compatibility
        }
    
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
                        clause_text = clause['text']
                        
                        # Detect hierarchy level
                        hierarchy_level = self.hierarchy_service.detect_hierarchy_level(clause_text)
                        
                        # Detect supremacy clause
                        is_supremacy = self.hierarchy_service.is_supremacy_clause(clause_text)
                        
                        # Extract jurisdiction
                        jurisdiction = self.hierarchy_service.extract_jurisdiction(clause_text)
                        
                        # Extract topics
                        topics = self.hierarchy_service.extract_topics(clause_text)
                        
                        # Generate clause ID
                        clause_id = f"clause_{page_number}_{len(clauses) + 1}"
                        
                        # Validate clause has required fields
                        clauses.append({
                            'type': clause.get('type', 'Unknown'),
                            # Clause type (Payment Terms, etc.)
                            'text': clause_text,
                            # Verbatim clause text
                            'page_number': page_number,
                            # Page where clause appears
                            'document_id': document_id,
                            # Source document
                            'start_index': clause.get('start_index', 0),
                            # Character position in page (if provided)
                            'clause_id': clause_id,
                            # Unique clause identifier
                            'hierarchy_level': hierarchy_level.value,
                            # Legal hierarchy level
                            'legal_supremacy': is_supremacy,
                            # Supremacy indicator
                            'jurisdiction': jurisdiction,
                            # Jurisdiction
                            'topics': topics
                            # Topics/keywords
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
        import re

        # Remove markdown code blocks if present
        if '```json' in response_text:
            # Extract content between ```json and ```
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        elif '```' in response_text:
            # Generic code block
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        else:
            json_text = response_text.strip()
        
        # Try to find JSON array within extracted text
        start_brace = json_text.find('[')
        end_brace = json_text.rfind(']')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_text = json_text[start_brace:end_brace + 1]

        # Remove invisible Unicode format characters (BOM, zero-width spaces, directional marks)
        json_text = re.sub(r'[\ufeff\u200b\u200c\u200d\u2060\u200e\u200f\u202a-\u202e]', '', json_text)

        # If the model returned literal "\n" / "\t" / "\r" tokens as STRUCTURAL whitespace,
        # convert them to real whitespace OUTSIDE string literals so json.loads works.
        if '\\n' in json_text or '\\t' in json_text or '\\r' in json_text:
            result = []
            in_string = False
            escape_next = False
            i = 0
            while i < len(json_text):
                ch = json_text[i]
                if escape_next:
                    if ch in ('n', 't', 'r'):
                        if in_string:
                            # Preserve escapes inside JSON strings
                            result.append('\\' + ch)
                        else:
                            # Convert structural escapes to real whitespace
                            result.append({'n': '\n', 't': '\t', 'r': '\r'}[ch])
                    else:
                        # Keep unknown escapes as-is
                        result.append('\\' + ch)
                    escape_next = False
                elif ch == '\\':
                    escape_next = True
                elif ch == '"':
                    # Toggle string state unless the quote itself is escaped
                    in_string = not in_string
                    result.append(ch)
                else:
                    result.append(ch)
                i += 1
            if escape_next:
                result.append('\\')
            json_text = ''.join(result)

        return json_text.strip()
    
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

