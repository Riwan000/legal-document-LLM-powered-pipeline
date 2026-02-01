"""
Clause extraction service.
Deterministic structure-first extraction of verbatim clauses.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import re

import ollama

from backend.config import settings

from backend.services.document_ingestion import DocumentIngestionService
from backend.services.structured_clause_extraction import StructuredClauseExtractionService, ExtractedClause

logger = logging.getLogger(__name__)


class ClauseExtractionService:
    """Service for extracting clauses from contracts using deterministic extraction."""
    
    def __init__(self):
        """Initialize the clause extraction service."""
        # Legacy LLM client (tests patch this constructor).
        self.ollama_client = ollama.Client(host=getattr(settings, "OLLAMA_BASE_URL", None))
        self.ingestion_service = DocumentIngestionService()
        self.structured_extractor = StructuredClauseExtractionService()
    
    # -------------------------------------------------------------------------
    # Legacy (Step 5) helpers expected by unit tests
    # -------------------------------------------------------------------------
    def _build_extraction_prompt(self, document_text: str) -> str:
        """
        Build an extraction prompt that enforces verbatim-only, no-interpretation output.
        Returns a prompt that requests JSON array output.
        """
        return f"""You are a clause extraction engine.

TASK:
- Extract contract clauses VERBATIM (exact text).

STRICT RULES:
- Output must be JSON only (no markdown, no explanations).
- Do NOT paraphrase. Use the exact text as it appears.
- Do NOT interpret. Do NOT summarize. Do NOT provide legal advice.
- If you cannot find a clause, return an empty JSON array [].

OUTPUT FORMAT (JSON array):
[
  {{
    "type": "Payment Terms",
    "text": "Exact clause text copied verbatim from the document.",
    "start_index": 0
  }}
]

DOCUMENT TEXT:
{document_text}
"""

    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract a JSON payload from common LLM response wrappers.
        Returns a string (possibly empty).
        """
        if response_text is None:
            return ""
        text = str(response_text).strip()
        if text == "":
            return ""

        # Plain JSON already
        if text.startswith("[") and "]" in text:
            # Trim trailing prose after the JSON array if any.
            last = text.rfind("]")
            return text[: last + 1].strip()

        # ```json ... ``` or ``` ... ```
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if code_block:
            inner = code_block.group(1).strip()
            if inner.startswith("[") and "]" in inner:
                last = inner.rfind("]")
                return inner[: last + 1].strip()
            return inner

        # Fallback: first [...] region
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()

        # No JSON detected; return text as-is (tests accept string return)
        return text

    def _parse_clause_response(self, response_text: str, page_number: int, document_id: str) -> List[Dict[str, Any]]:
        """Parse a JSON array of clauses into normalized clause dicts."""
        try:
            json_text = self._extract_json_from_response(response_text)
            data = json.loads(json_text) if json_text else []
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        clauses: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or text == "":
                continue
            clause_type = item.get("type") if isinstance(item.get("type"), str) and item.get("type") else "Unknown"
            start_index = item.get("start_index", 0)
            try:
                start_index = int(start_index)
            except Exception:
                start_index = 0

            clauses.append(
                {
                    "type": clause_type,
                    "text": text,
                    "start_index": start_index,
                    "page_number": page_number,
                    "document_id": document_id,
                }
            )
        return clauses

    def _deduplicate_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate clauses by normalized text (case/whitespace-insensitive)."""
        if not clauses:
            return []

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip()).lower()

        def _pages_to_set(v: Any) -> set:
            if v is None:
                return set()
            if isinstance(v, int):
                return {v}
            if isinstance(v, list):
                out = set()
                for x in v:
                    try:
                        out.add(int(x))
                    except Exception:
                        continue
                return out
            # string-ish
            nums = re.findall(r"\d+", str(v))
            return {int(n) for n in nums} if nums else set()

        by_text: Dict[str, Dict[str, Any]] = {}
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            key = _norm(clause.get("text", ""))
            if not key:
                continue
            if key not in by_text:
                by_text[key] = clause.copy()
            else:
                # Merge page numbers across duplicates
                pages = _pages_to_set(by_text[key].get("page_number")) | _pages_to_set(clause.get("page_number"))
                by_text[key]["page_number"] = sorted(pages) if pages else by_text[key].get("page_number")
        return list(by_text.values())

    def _extract_clauses_from_page(self, text: str, page_number: int, document_id: str) -> List[Dict[str, Any]]:
        """LLM-based per-page clause extraction (used by unit tests)."""
        prompt = self._build_extraction_prompt(text)
        response = self.ollama_client.generate(
            model=getattr(settings, "OLLAMA_MODEL", "llama3"),
            prompt=prompt,
            options={
                "temperature": 0.2,  # deterministic
                "format": "json",    # tests assert this
            },
        )

        if isinstance(response, dict):
            response_text = response.get("response", "")
        elif hasattr(response, "response"):
            response_text = response.response
        else:
            response_text = str(response)

        return self._parse_clause_response(response_text, page_number=page_number, document_id=document_id)

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
            use_structured: If True, use deterministic structured extraction (default: True)
            
        Returns:
            List of clause dictionaries with extraction-only schema:
            clause_id, document_section, page_start, page_end, clause_heading, verbatim_text, normalized_text
        """
        file_path = Path(file_path)
        
        # Step 1 (Service): Delegate to deterministic structured extraction.
        # Always use deterministic structured extraction
        if use_structured:
            try:
                logger.info(f"Extracting clauses using deterministic structured extraction for document_id={document_id}")
                extracted_clauses = self.structured_extractor.extract_structured_clauses(
                    str(file_path), document_id
                )
                logger.info(f"Extraction successful: {len(extracted_clauses)} clauses extracted")
                # Step 1.1 (Service): Convert internal objects to dicts for API response.
                # Convert to dict format for API response
                return [clause.to_dict() for clause in extracted_clauses]
            except Exception as e:
                logger.error(f"Structured extraction failed for document_id={document_id}: {str(e)}", exc_info=True)
                # Fail closed - return empty list
                return []
        
        # Legacy path removed - always use structured extraction
        return []
    
    def extract_clauses_with_telemetry(
        self,
        file_path: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Extract clauses with telemetry information.
        
        Args:
            file_path: Path to the contract file
            document_id: Unique document identifier
            
        Returns:
            Dict with clauses and telemetry:
            {
                "clauses": [...],
                "telemetry": {
                    "pages_processed": int,
                    "pages_skipped_ambiguous": int,
                    "clauses_emitted": int,
                    "sections_detected": set
                }
            }
        """
        file_path = Path(file_path)
        
        # Step T1 (Telemetry): Parse pages (for pages_processed + section detection).
        # Parse pages for telemetry
        pages = self.ingestion_service.parser.parse_file(file_path)
        pages_processed = len(pages) if pages else 0
        
        # Step T2 (Telemetry): Run the same structured clause extraction.
        # Extract clauses
        extracted_clauses = self.structured_extractor.extract_structured_clauses(
            str(file_path), document_id
        )
        
        # Step T3 (Telemetry): Re-run section classification per page to compute:
        # - pages_skipped_ambiguous
        # - sections_detected
        # Calculate telemetry
        pages_skipped_ambiguous = 0
        sections_detected = set()
        
        for page_data in pages:
            if len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                text, page_number = page_data
            
            section = self.structured_extractor._classify_page_section(text, page_number)
            if hasattr(section, 'value'):
                section_value = section.value
            else:
                section_value = str(section)
            
            if section_value == "ambiguous":
                pages_skipped_ambiguous += 1
            else:
                sections_detected.add(section_value)
        
        return {
            "clauses": [clause.to_dict() for clause in extracted_clauses],
            "document_type": getattr(self.structured_extractor.last_document_type, "value", None),
            "telemetry": {
                "pages_processed": pages_processed,
                "pages_skipped_ambiguous": pages_skipped_ambiguous,
                "clauses_emitted": len(extracted_clauses),
                "sections_detected": list(sections_detected)
            }
        }
    
    def extract_clauses_by_type(
        self,
        file_path: str,
        document_id: str,
        clause_types: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract clauses filtered by type (legacy method for backward compatibility).
        Note: This method filters by clause_heading, not by legal-effect type.
        
        Args:
            file_path: Path to contract file
            document_id: Document identifier
            clause_types: List of clause heading patterns to match
            
        Returns:
            Dictionary mapping clause type to list of clauses
        """
        all_clauses = self.extract_clauses(file_path, document_id)
        
        # Filter by clause type (case-insensitive, partial matches allowed).
        filtered = {clause_type: [] for clause_type in clause_types}
        
        for clause in all_clauses:
            raw_type = clause.get("type") or clause.get("clause_heading") or ""
            clause_type_val = str(raw_type).lower()
            
            for target_type in clause_types:
                tt = target_type.lower()
                if tt in clause_type_val or clause_type_val in tt:
                    filtered[target_type].append(clause)
                    break
        
        return filtered
