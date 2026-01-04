"""
Structured Legal Clause Extraction Engine.
Transforms unstructured documents into clean, authoritative, machine-readable legal clauses.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import ollama
import json
import uuid
import re

from backend.config import settings
from backend.services.document_ingestion import DocumentIngestionService
from backend.utils.ocr_cleanup import OCRCleanupService
from backend.services.authority_classifier import AuthorityClassifier
from backend.services.clause_taxonomy import ClauseTaxonomyService
from backend.models.clause import (
    StructuredClause, EvidenceBlock, AuthorityLevel, ClauseType, TerminationSubtype
)
from backend.services.clause_validator import ClauseValidator


class StructuredClauseExtractionService:
    """Service for extracting structured legal clauses with authority modeling."""
    
    def __init__(self):
        """Initialize structured clause extraction service."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.ingestion_service = DocumentIngestionService()
        self.ocr_cleanup = OCRCleanupService()
        self.authority_classifier = AuthorityClassifier()
        self.taxonomy_service = ClauseTaxonomyService()
        self.validator = ClauseValidator()
    
    def extract_structured_clauses(
        self,
        file_path: str,
        document_id: str
    ) -> List[StructuredClause]:
        """
        Extract structured legal clauses from document.
        
        Args:
            file_path: Path to document file
            document_id: Unique document identifier
            
        Returns:
            List of StructuredClause objects
        """
        file_path = Path(file_path)
        
        # Step 1: Parse document pages
        pages = self.ingestion_service.parser.parse_file(file_path)
        if not pages:
            return []
        
        # Step 2: Extract raw clauses with OCR cleanup
        raw_clauses = []
        for page_data in pages:
            if len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                text, page_number = page_data
                is_ocr = False
            
            # Clean OCR text if needed
            if is_ocr or self._needs_ocr_cleanup(text):
                cleaned = self.ocr_cleanup.normalize_text(text)
                raw_text = cleaned['raw_text']
                clean_text = cleaned['clean_text']
            else:
                raw_text = text
                clean_text = text
            
            # Extract clauses from page
            page_clauses = self._extract_clauses_from_page(
                clean_text, raw_text, page_number, document_id
            )
            raw_clauses.extend(page_clauses)
        
        # Step 3: Structure clauses with authority and taxonomy
        structured_clauses = []
        for raw_clause in raw_clauses:
            structured = self._structure_clause(raw_clause, document_id)
            if structured:
                structured_clauses.append(structured)
        
        # Step 4: Merge evidence blocks for same clauses
        merged_clauses = self._merge_clause_evidence(structured_clauses)
        
        # Step 5: Link bilingual clauses
        linked_clauses = self._link_bilingual_clauses(merged_clauses)
        
        # Step 6: Validate clauses
        validated_clauses = []
        for clause in linked_clauses:
            validation = self.validator.validate_clause(clause)
            if validation['valid']:
                validated_clauses.append(clause)
            else:
                print(f"Warning: Clause {clause.clause_id} failed validation: {validation['errors']}")
                # Still include invalid clauses but log warnings
                validated_clauses.append(clause)
        
        return validated_clauses
    
    def _needs_ocr_cleanup(self, text: str) -> bool:
        """Check if text needs OCR cleanup."""
        # Check for common OCR error patterns
        ocr_indicators = [
            'shali', 'const.tule', 'agreemeni', 'authort.es',
            'Nolce', 'terminateg', 'jne', 'Dy a'
        ]
        return any(indicator in text.lower() for indicator in ocr_indicators)
    
    def _extract_clauses_from_page(
        self,
        clean_text: str,
        raw_text: str,
        page_number: int,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract clauses from a page using LLM.
        
        Args:
            clean_text: Normalized text
            raw_text: Original OCR text
            page_number: Page number
            document_id: Document ID
            
        Returns:
            List of raw clause dictionaries
        """
        prompt = self._build_extraction_prompt(clean_text)
        
        try:
            # Limit response length and add timeout handling
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'format': 'json',
                    'num_predict': 2000  # Limit response length to avoid very long responses
                }
            )
            
            response_text = response.get('response', '') if isinstance(response, dict) else str(response)
            clauses = self._parse_clause_response(response_text, raw_text, clean_text, page_number)
            
            return clauses
        except Exception as e:
            print(f"Error extracting clauses from page {page_number}: {str(e)}")
            return []
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for clause extraction."""
        return f"""You are a legal clause extraction system. Extract legal clauses from the following text.

IMPORTANT RULES:
- Extract complete legal clauses, not text fragments
- One clause may span multiple sentences
- Clauses are defined by legal intent, not formatting
- Return ONLY clauses found in the text
- Do NOT create or fabricate clauses

Return your response as a JSON array of objects with this exact format:
[
  {{
    "title": "Clause title or heading",
    "text": "Complete clause text verbatim",
    "start_index": 0
  }},
  {{
    "title": "Another clause",
    "text": "Complete clause text",
    "start_index": 150
  }}
]

Document text:
{text}

JSON response:"""
    
    def _parse_clause_response(
        self,
        response_text: str,
        raw_text: str,
        clean_text: str,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """Parse LLM response and extract clause information."""
        clauses = []
        
        # Extract JSON from response
        json_text = self._extract_json_from_response(response_text)
        if not json_text:
            return clauses
        
        try:
            # Try to parse JSON - handle both string and actual JSON
            if isinstance(json_text, str):
                parsed_clauses = json.loads(json_text)
            else:
                parsed_clauses = json_text  # Already parsed
            
            if isinstance(parsed_clauses, list):
                for clause in parsed_clauses:
                    if isinstance(clause, dict) and 'text' in clause:
                        # Find paragraph and line information
                        clause_text = clause['text']
                        start_index = clause.get('start_index', 0)
                        
                        # Find paragraph number
                        paragraph = self._find_paragraph_number(clean_text, start_index)
                        
                        # Find line range
                        line_start, line_end = self._find_line_range(clean_text, start_index, len(clause_text))
                        
                        clauses.append({
                            'title': clause.get('title', 'Untitled Clause'),
                            'text': clause_text,
                            'raw_text': raw_text[start_index:start_index + len(clause_text)] if start_index < len(raw_text) else clause_text,
                            'page_number': page_number,
                            'paragraph': paragraph,
                            'line_start': line_start,
                            'line_end': line_end,
                            'start_index': start_index
                        })
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            print(f"JSON text (first 500 chars): {json_text[:500]}")
            # Try to extract clauses manually using regex as fallback
            try:
                clauses = self._fallback_extract_clauses(json_text, clean_text, raw_text, page_number)
                if clauses:
                    print(f"Fallback extraction succeeded: found {len(clauses)} clauses")
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {str(fallback_error)}")
        
        return clauses
    
    def _find_paragraph_number(self, text: str, start_index: int) -> Optional[int]:
        """Find paragraph number for given index."""
        # Count paragraphs before start_index
        text_before = text[:start_index]
        paragraphs = text_before.split('\n\n')
        return len(paragraphs) if paragraphs else None
    
    def _find_line_range(self, text: str, start_index: int, length: int) -> Tuple[Optional[int], Optional[int]]:
        """Find line range for clause."""
        text_before = text[:start_index]
        text_clause = text[start_index:start_index + length]
        
        line_start = text_before.count('\n') + 1
        line_end = line_start + text_clause.count('\n')
        
        return (line_start, line_end)
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response with robust error handling."""
        import re
        
        # Remove markdown code blocks
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        else:
            json_text = response_text.strip()
        
        # Try to find JSON array
        if not json_text.startswith('['):
            start_brace = json_text.find('[')
            if start_brace != -1:
                end_brace = json_text.rfind(']')
                if end_brace != -1 and end_brace > start_brace:
                    json_text = json_text[start_brace:end_brace + 1]
        
        # Handle literal \n characters (backslash-n) - convert to actual newlines
        # But preserve escaped \n inside string values
        # Check if we have literal \n (not actual newlines) in the structure
        if '\\n' in json_text:
            # Try to decode if it's a Python string representation
            try:
                import ast
                # If it looks like a string representation, try to decode it
                if json_text.startswith('"') or json_text.startswith("'"):
                    decoded = ast.literal_eval(json_text)
                    if isinstance(decoded, str):
                        json_text = decoded
                else:
                    # It might be a raw string with literal \n - convert them carefully
                    # Only convert \n that's outside of string values
                    # Simple approach: replace literal \n with actual newline, but be careful with escaped ones in strings
                    # We'll use a state machine to track if we're inside a string
                    result = []
                    in_string = False
                    escape_next = False
                    i = 0
                    while i < len(json_text):
                        char = json_text[i]
                        if escape_next:
                            if char == 'n':
                                if in_string:
                                    # Keep escaped \n inside string values
                                    result.append('\\n')
                                else:
                                    # Convert literal \n to actual newline outside strings
                                    result.append('\n')
                            else:
                                result.append('\\' + char)
                            escape_next = False
                        elif char == '\\':
                            escape_next = True
                            # Don't append yet, wait to see if it's \n
                        elif char == '"':
                            # Toggle string state
                            if i > 0 and json_text[i-1] == '\\' and not escape_next:
                                # This quote is escaped, don't toggle
                                result.append(char)
                            else:
                                in_string = not in_string
                                result.append(char)
                        else:
                            if escape_next:
                                # We had a backslash but it wasn't \n, add it
                                result.append('\\')
                                escape_next = False
                            result.append(char)
                        i += 1
                    # Handle trailing backslash
                    if escape_next:
                        result.append('\\')
                    json_text = ''.join(result)
            except Exception as e:
                # If decoding fails, try simple replacement for structure newlines only
                # Replace \n that appears after [ or { or , and before " (indicating structure)
                json_text = re.sub(r'([\[{,\s])\\n(\s*[{"])', r'\1\n\2', json_text)
                json_text = re.sub(r'([}\]\s])\\n(\s*[}\]])', r'\1\n\2', json_text)
                # Last resort: if we still have literal \n and no actual newlines, replace all
                if '\\n' in json_text and '\n' not in json_text[:200]:
                    # This is risky but better than failing - replace all literal \n
                    json_text = json_text.replace('\\n', '\n')
        
        # Clean up common JSON issues more carefully
        # First, try to fix single quotes to double quotes (but be careful with apostrophes)
        # Only replace single quotes that are clearly string delimiters (not inside words)
        json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)  # Fix keys with single quotes
        json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)  # Fix string values with single quotes
        
        # Fix unquoted property names (but not if already quoted or if it's a number/boolean)
        # Match property names at the start of object or after comma
        json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)
        
        # Remove trailing commas before closing brackets/braces
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
        
        # Fix escaped quotes inside strings - handle cases like: "text": ""quoted text""
        # Replace double quotes at start/end of string values with escaped quotes
        json_text = re.sub(r':\s*""([^"]+)""', r': "\1"', json_text)  # Fix double-double quotes
        json_text = re.sub(r':\s*"([^"]*)"([^,}\]]*)"', lambda m: f': "{m.group(1)}{m.group(2)}"', json_text)
        
        # Fix control characters (newlines, tabs) - escape them properly
        json_text = re.sub(r'[\x00-\x1f]', lambda m: '\\u{:04x}'.format(ord(m.group(0))), json_text)
        # But preserve intentional \n and \t
        json_text = json_text.replace('\\u000a', '\\n').replace('\\u0009', '\\t').replace('\\u000d', '\\r')
        
        return json_text
    
    def _fallback_extract_clauses(
        self,
        json_text: str,
        clean_text: str,
        raw_text: str,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """Fallback method to extract clauses using regex when JSON parsing fails."""
        clauses = []
        
        # Try to find clause-like patterns in the text
        # Look for patterns like: "title": "...", "text": "..." or 'title': '...', 'text': '...'
        # Handle both double and single quotes
        patterns = [
            r'"title"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"',
            r"'title'\s*:\s*'([^']+)'\s*,\s*'text'\s*:\s*'([^']+)'",
            r'title["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*text["\']\s*:\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, json_text, re.DOTALL)
            for match in matches:
                title = match.group(1)
                text = match.group(2)
                
                # Clean up escaped characters
                title = title.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
                text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
                
                # Find start index in clean text
                # Try to find a unique substring from the text
                search_text = text[:100] if len(text) > 100 else text
                start_index = clean_text.find(search_text)
                if start_index == -1:
                    start_index = 0
                
                paragraph = self._find_paragraph_number(clean_text, start_index)
                line_start, line_end = self._find_line_range(clean_text, start_index, len(text))
                
                clauses.append({
                    'title': title,
                    'text': text,
                    'raw_text': raw_text[start_index:start_index + len(text)] if start_index < len(raw_text) else text,
                    'page_number': page_number,
                    'paragraph': paragraph,
                    'line_start': line_start,
                    'line_end': line_end,
                    'start_index': start_index
                })
        
        return clauses
    
    def _structure_clause(
        self,
        raw_clause: Dict[str, Any],
        document_id: str
    ) -> Optional[StructuredClause]:
        """
        Structure a raw clause with authority, taxonomy, and evidence.
        
        Args:
            raw_clause: Raw clause dictionary
            document_id: Document ID
            
        Returns:
            StructuredClause or None if invalid
        """
        clause_text = raw_clause['text']
        
        # Step 1: Classify clause type
        clause_type = self.taxonomy_service.classify_clause_type(clause_text)
        
        # Step 2: Classify authority level
        jurisdiction = self.authority_classifier.extract_jurisdiction(clause_text)
        # Get clause type value (handle both enum and string)
        clause_type_str = clause_type.value if hasattr(clause_type, 'value') else str(clause_type)
        authority_level = self.authority_classifier.classify_authority(
            clause_text, clause_type_str, jurisdiction
        )
        
        # Step 3: Determine override capability
        override_info = self.authority_classifier.determine_override_capability(
            authority_level, clause_text
        )
        
        # Step 4: Classify termination subtype if applicable
        subtype = None
        # Compare enum values properly - handle both enum and string
        clause_type_enum = clause_type if isinstance(clause_type, ClauseType) else ClauseType(clause_type) if isinstance(clause_type, str) else clause_type
        if clause_type_enum == ClauseType.TERMINATION:
            subtype_obj = self.taxonomy_service.classify_termination_subtype(clause_text)
            if subtype_obj:
                subtype = subtype_obj.value if hasattr(subtype_obj, 'value') else str(subtype_obj)
        
        # Step 5: Separate payment categories
        payment_info = self.taxonomy_service.separate_payment_categories(clause_text)
        
        # Step 6: Detect language
        language = self.ocr_cleanup._detect_language(clause_text)
        
        # Step 7: Generate clause ID
        clause_id = self._generate_clause_id(
            clause_type, raw_clause['page_number'], document_id
        )
        
        # Step 8: Create evidence block
        evidence = EvidenceBlock(
            page=raw_clause['page_number'],
            paragraph=raw_clause.get('paragraph'),
            line_start=raw_clause.get('line_start'),
            line_end=raw_clause.get('line_end'),
            raw_text=raw_clause.get('raw_text', clause_text),
            clean_text=clause_text
        )
        
        # Step 9: Create structured clause
        structured_clause = StructuredClause(
            clause_id=clause_id,
            title=raw_clause.get('title', 'Untitled Clause'),
            type=clause_type,
            subtype=subtype,
            authority_level=authority_level,
            jurisdiction=jurisdiction,
            can_override_contract=override_info['can_override_contract'],
            overrides=override_info['overrides'],
            evidence=[evidence],
            explicitly_provided=payment_info.get('explicitly_provided', True),
            language=language,
            metadata={
                'is_salary': payment_info.get('is_salary', False),
                'is_allowance': payment_info.get('is_allowance', False),
                'is_employer_cost': payment_info.get('is_employer_cost', False)
            }
        )
        
        return structured_clause
    
    def _generate_clause_id(
        self,
        clause_type: ClauseType,
        page_number: int,
        document_id: str
    ) -> str:
        """Generate unique clause ID."""
        # Handle both enum and string
        if hasattr(clause_type, 'value'):
            type_slug = clause_type.value.replace('_', '-')
        else:
            type_slug = str(clause_type).replace('_', '-')
        return f"{type_slug}_{document_id[:8]}_p{page_number}_{uuid.uuid4().hex[:6]}"
    
    def _merge_clause_evidence(
        self,
        clauses: List[StructuredClause]
    ) -> List[StructuredClause]:
        """
        Merge evidence blocks for clauses that appear multiple times.
        Multiple evidences may belong to one clause.
        """
        # Group clauses by semantic similarity (same title and similar text)
        clause_groups = {}
        
        for clause in clauses:
            # Create key from title and first 50 chars of text
            clean_text = clause.evidence[0].clean_text if clause.evidence else ""
            key = f"{clause.title}_{clean_text[:50]}"
            
            if key not in clause_groups:
                clause_groups[key] = []
            clause_groups[key].append(clause)
        
        # Merge evidence blocks
        merged_clauses = []
        for key, group in clause_groups.items():
            if len(group) == 1:
                merged_clauses.append(group[0])
            else:
                # Merge: take first clause, combine all evidence
                primary = group[0]
                all_evidence = []
                for clause in group:
                    all_evidence.extend(clause.evidence)
                
                # Update primary clause with all evidence
                primary.evidence = all_evidence
                merged_clauses.append(primary)
        
        return merged_clauses
    
    def _link_bilingual_clauses(
        self,
        clauses: List[StructuredClause]
    ) -> List[StructuredClause]:
        """
        Link Arabic and English versions of the same clause.
        """
        # Separate by language
        arabic_clauses = [c for c in clauses if c.language == 'ar']
        english_clauses = [c for c in clauses if c.language == 'en']
        
        # Try to match clauses by type and similar content
        for ar_clause in arabic_clauses:
            for en_clause in english_clauses:
                if (ar_clause.type == en_clause.type and
                    ar_clause.authority_level == en_clause.authority_level):
                    # Link them
                    ar_clause.linked_clause_id = en_clause.clause_id
                    en_clause.linked_clause_id = ar_clause.clause_id
                    
                    # Check for consistency (simplified - would need translation in production)
                    # For now, mark for review if types match but content differs significantly
                    if len(ar_clause.evidence[0].clean_text) != len(en_clause.evidence[0].clean_text):
                        ar_clause.consistency_flag = "review_required"
                        en_clause.consistency_flag = "review_required"
        
        return clauses

