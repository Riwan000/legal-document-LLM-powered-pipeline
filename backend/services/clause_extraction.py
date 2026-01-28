"""
Clause extraction service.
Deterministic structure-first extraction of verbatim clauses.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from backend.services.document_ingestion import DocumentIngestionService
from backend.services.structured_clause_extraction import StructuredClauseExtractionService, ExtractedClause

logger = logging.getLogger(__name__)


class ClauseExtractionService:
    """Service for extracting clauses from contracts using deterministic extraction."""
    
    def __init__(self):
        """Initialize the clause extraction service."""
        self.ingestion_service = DocumentIngestionService()
        self.structured_extractor = StructuredClauseExtractionService()
    
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
        
        # Always use deterministic structured extraction
        if use_structured:
            try:
                logger.info(f"Extracting clauses using deterministic structured extraction for document_id={document_id}")
                extracted_clauses = self.structured_extractor.extract_structured_clauses(
                    str(file_path), document_id
                )
                logger.info(f"Extraction successful: {len(extracted_clauses)} clauses extracted")
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
        
        # Parse pages for telemetry
        pages = self.ingestion_service.parser.parse_file(file_path)
        pages_processed = len(pages) if pages else 0
        
        # Extract clauses
        extracted_clauses = self.structured_extractor.extract_structured_clauses(
            str(file_path), document_id
        )
        
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
        
        # Filter by heading pattern (case-insensitive)
        filtered = {clause_type: [] for clause_type in clause_types}
        
        for clause in all_clauses:
            clause_heading = clause.get('clause_heading', '').lower()
            
            for target_type in clause_types:
                if target_type.lower() in clause_heading or clause_heading in target_type.lower():
                    filtered[target_type].append(clause)
                    break
        
        return filtered
