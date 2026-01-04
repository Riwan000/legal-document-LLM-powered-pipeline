"""
Clause validation service.
Ensures clauses meet PRD requirements for legal-grade quality.
"""
from typing import List, Dict, Any, Optional
from backend.models.clause import StructuredClause, ClauseType, AuthorityLevel
from backend.services.clause_taxonomy import ClauseTaxonomyService


class ClauseValidator:
    """Service for validating structured clauses against PRD requirements."""
    
    def __init__(self):
        """Initialize clause validator."""
        self.taxonomy_service = ClauseTaxonomyService()
    
    def validate_clause(self, clause: StructuredClause) -> Dict[str, Any]:
        """
        Validate a clause against all PRD requirements.
        
        Args:
            clause: StructuredClause to validate
            
        Returns:
            Dict with 'valid' (bool) and 'errors' (list of error messages)
        """
        errors = []
        
        # Check 1: Required fields
        if not clause.clause_id:
            errors.append("Missing clause_id")
        if not clause.title:
            errors.append("Missing title")
        if not clause.evidence:
            errors.append("Missing evidence blocks")
        
        # Check 2: Raw text preservation
        for evidence in clause.evidence:
            if not evidence.raw_text:
                errors.append(f"Evidence block missing raw_text (required for audit trail)")
            if not evidence.clean_text:
                errors.append(f"Evidence block missing clean_text")
            # Ensure raw_text is not overwritten (should be different from clean_text if OCR was used)
            if evidence.raw_text == evidence.clean_text and 'OCR' in str(clause.metadata.get('is_ocr', '')):
                # This is OK - no OCR was needed
                pass
        
        # Check 3: Precise location metadata
        for evidence in clause.evidence:
            if evidence.page is None or evidence.page < 1:
                errors.append(f"Evidence block missing valid page number")
            # Paragraph and line range are optional but should be present if available
        
        # Check 4: Incompatible category check
        category_errors = self.check_incompatible_categories(clause)
        errors.extend(category_errors)
        
        # Check 5: Evidence block validation
        evidence_errors = self.verify_evidence_blocks(clause)
        errors.extend(evidence_errors)
        
        # Check 6: Authority level consistency
        if clause.authority_level == AuthorityLevel.SUPREME and not clause.can_override_contract:
            errors.append("Supreme authority clauses should be able to override contract")
        
        # Check 7: Termination subtype requirement
        if clause.type == ClauseType.TERMINATION and not clause.subtype:
            errors.append("Termination clauses must have a subtype")
        
        # Check 8: Override list consistency
        if clause.can_override_contract and not clause.overrides:
            # This is a warning, not an error - override list might be empty by design
            pass
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': self._generate_warnings(clause)
        }
    
    def check_incompatible_categories(self, clause: StructuredClause) -> List[str]:
        """
        Check for incompatible category assignments.
        
        Args:
            clause: StructuredClause to check
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check incompatible pairs
        incompatible_pairs = [
            (ClauseType.CONDUCT_DISCIPLINE, ClauseType.CONFIDENTIALITY),
            (ClauseType.TERMINATION, ClauseType.CONFIDENTIALITY),
        ]
        
        # For now, we only check the primary type
        # In future, could check if clause has multiple incompatible types in metadata
        clause_type = clause.type
        
        # Check if type is in any incompatible pair
        for type1, type2 in incompatible_pairs:
            if clause_type == type1:
                # Check metadata for secondary types
                secondary_types = clause.metadata.get('secondary_types', [])
                if type2.value in secondary_types:
                    errors.append(f"Incompatible categories: {type1.value} and {type2.value}")
        
        return errors
    
    def verify_evidence_blocks(self, clause: StructuredClause) -> List[str]:
        """
        Verify evidence blocks have precise location metadata.
        
        Args:
            clause: StructuredClause to verify
            
        Returns:
            List of error messages
        """
        errors = []
        
        if not clause.evidence:
            errors.append("Clause must have at least one evidence block")
            return errors
        
        for i, evidence in enumerate(clause.evidence):
            # Page number is mandatory
            if evidence.page is None or evidence.page < 1:
                errors.append(f"Evidence block {i+1}: Missing or invalid page number")
            
            # Paragraph and line range are optional but recommended
            if evidence.paragraph is None:
                # Warning only, not error
                pass
            
            # Raw text must be preserved
            if not evidence.raw_text:
                errors.append(f"Evidence block {i+1}: Missing raw_text (required for audit trail)")
            
            # Clean text must exist
            if not evidence.clean_text:
                errors.append(f"Evidence block {i+1}: Missing clean_text")
        
        return errors
    
    def _generate_warnings(self, clause: StructuredClause) -> List[str]:
        """Generate warnings (non-blocking issues) for a clause."""
        warnings = []
        
        # Warning: Missing paragraph index
        for evidence in clause.evidence:
            if evidence.paragraph is None:
                warnings.append(f"Evidence block on page {evidence.page}: Missing paragraph index")
            if evidence.line_start is None or evidence.line_end is None:
                warnings.append(f"Evidence block on page {evidence.page}: Missing line range")
        
        # Warning: No jurisdiction for supreme/regulatory clauses
        if clause.authority_level in [AuthorityLevel.SUPREME, AuthorityLevel.REGULATORY]:
            if not clause.jurisdiction:
                warnings.append("Supreme or regulatory clause missing jurisdiction")
        
        # Warning: Bilingual clause without link
        if clause.language and not clause.linked_clause_id:
            warnings.append("Bilingual clause detected but no linked_clause_id")
        
        return warnings
    
    def validate_clauses_batch(self, clauses: List[StructuredClause]) -> Dict[str, Any]:
        """
        Validate a batch of clauses.
        
        Args:
            clauses: List of StructuredClause objects
            
        Returns:
            Dict with validation summary
        """
        results = []
        total_errors = 0
        total_warnings = 0
        
        for clause in clauses:
            validation = self.validate_clause(clause)
            results.append({
                'clause_id': clause.clause_id,
                'valid': validation['valid'],
                'errors': validation['errors'],
                'warnings': validation['warnings']
            })
            
            if not validation['valid']:
                total_errors += len(validation['errors'])
            total_warnings += len(validation['warnings'])
        
        valid_count = sum(1 for r in results if r['valid'])
        
        return {
            'total_clauses': len(clauses),
            'valid_clauses': valid_count,
            'invalid_clauses': len(clauses) - valid_count,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'results': results
        }

