"""
Contract comparison service.
Compares uploaded contracts against firm templates using embedding similarity and text diff.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import difflib
from backend.services.clause_extraction import ClauseExtractionService
from backend.services.embedding_service import EmbeddingService


class ComparisonService:
    """Service for comparing contracts against templates."""
    
    def __init__(self):
        """Initialize the comparison service."""
        self.clause_extractor = ClauseExtractionService()
        # For extracting clauses from contracts
        self.embedding_service = EmbeddingService()
        # For semantic similarity comparison
    
    def compare_contracts(
        self,
        contract_path: str,
        template_path: str,
        contract_id: str,
        template_id: str
    ) -> Dict[str, Any]:
        """
        Compare a contract against a template.
        
        Args:
            contract_path: Path to uploaded contract
            template_path: Path to firm template
            contract_id: Contract document ID
            template_id: Template document ID
            
        Returns:
            Dictionary with comparison results:
                - matched_clauses: Clauses that match
                - modified_clauses: Clauses with differences
                - missing_clauses: Clauses in template but not in contract
                - extra_clauses: Clauses in contract but not in template
        """
        # Extract clauses from both documents
        contract_clauses = self.clause_extractor.extract_clauses(contract_path, contract_id)
        # Get all clauses from uploaded contract
        template_clauses = self.clause_extractor.extract_clauses(template_path, template_id)
        # Get all clauses from firm template
        
        # Compare clauses
        comparison = self._compare_clauses(contract_clauses, template_clauses)
        # Perform semantic and text-based comparison
        
        return comparison
    
    def _compare_clauses(
        self,
        contract_clauses: List[Dict[str, Any]],
        template_clauses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare two sets of clauses.
        
        Args:
            contract_clauses: Clauses from uploaded contract
            template_clauses: Clauses from template
            
        Returns:
            Comparison results dictionary
        """
        # Generate embeddings for all clauses
        contract_embeddings = self._embed_clauses(contract_clauses)
        # Embed contract clauses for semantic comparison
        template_embeddings = self._embed_clauses(template_clauses)
        # Embed template clauses
        
        # Find matches and differences
        matched = []
        # Clauses that match (semantically similar)
        modified = []
        # Clauses with differences
        missing = []
        # Clauses in template but not in contract
        extra = []
        # Clauses in contract but not in template
        
        # Compare each template clause against contract clauses
        template_matched = set()
        # Track which template clauses have been matched
        
        for i, template_clause in enumerate(template_clauses):
            # For each template clause
            template_emb = template_embeddings[i]
            # Get its embedding
            
            # Find best matching contract clause
            best_match_idx = None
            best_similarity = 0.0
            
            for j, contract_clause in enumerate(contract_clauses):
                # Compare against each contract clause
                contract_emb = contract_embeddings[j]
                # Get contract clause embedding
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(template_emb, contract_emb)
                # Semantic similarity score
                
                if similarity > best_similarity:
                    # Found better match
                    best_similarity = similarity
                    best_match_idx = j
            
            # Determine if match is good enough
            similarity_threshold = 0.85
            # High threshold for "match" (85% similarity)
            
            if best_similarity >= similarity_threshold and best_match_idx is not None:
                # Good semantic match - check text differences
                contract_clause = contract_clauses[best_match_idx]
                # Get the matched contract clause
                
                # Compare text for exact differences
                text_diff = self._compare_text(
                    template_clause['text'],
                    contract_clause['text']
                )
                # Get detailed text differences
                
                if text_diff['is_identical']:
                    # Texts are identical
                    matched.append({
                        'template_clause': template_clause,
                        'contract_clause': contract_clause,
                        'similarity': best_similarity
                    })
                else:
                    # Texts differ (semantically similar but textually different)
                    modified.append({
                        'template_clause': template_clause,
                        'contract_clause': contract_clause,
                        'similarity': best_similarity,
                        'differences': text_diff
                    })
                
                template_matched.add(i)
                # Mark template clause as matched
            else:
                # No good match found - clause is missing from contract
                missing.append({
                    'template_clause': template_clause,
                    'similarity': best_similarity if best_match_idx is not None else 0.0
                })
        
        # Find extra clauses (in contract but not in template)
        contract_matched = set()
        # Track which contract clauses have been matched
        
        for i, contract_clause in enumerate(contract_clauses):
            # Check if this contract clause matches any template clause
            contract_emb = contract_embeddings[i]
            
            best_similarity = 0.0
            for j, template_clause in enumerate(template_clauses):
                template_emb = template_embeddings[j]
                similarity = self._cosine_similarity(contract_emb, template_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
            
            if best_similarity < 0.85:
                # No good match in template - this is an extra clause
                extra.append({
                    'contract_clause': contract_clause,
                    'best_similarity': best_similarity
                })
        
        return {
            'matched_clauses': matched,
            # Clauses that match exactly
            'modified_clauses': modified,
            # Clauses with differences
            'missing_clauses': missing,
            # Clauses missing from contract
            'extra_clauses': extra,
            # Extra clauses in contract
            'summary': {
                'total_template_clauses': len(template_clauses),
                'total_contract_clauses': len(contract_clauses),
                'matched_count': len(matched),
                'modified_count': len(modified),
                'missing_count': len(missing),
                'extra_count': len(extra)
            }
        }
    
    def _embed_clauses(self, clauses: List[Dict[str, Any]]) -> List:
        """
        Generate embeddings for a list of clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List of embedding vectors
        """
        texts = [clause['text'] for clause in clauses]
        # Extract clause texts
        embeddings = self.embedding_service.embed_batch(texts)
        # Batch embed for efficiency
        return embeddings
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        # Add small epsilon to avoid division by zero
        
        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        # Returns value between -1 and 1 (typically 0-1 for embeddings)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def _compare_text(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, Any]:
        """
        Compare two texts and find differences.
        
        Args:
            text1: First text (template)
            text2: Second text (contract)
            
        Returns:
            Dictionary with difference information
        """
        # Check if identical
        if text1.strip() == text2.strip():
            return {
                'is_identical': True,
                'differences': []
            }
        
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            lineterm='',
            n=0  # Context lines (0 = no context)
        ))
        # unified_diff shows additions and deletions
        
        # Parse diff into structured format
        differences = []
        current_diff = None
        
        for line in diff:
            if line.startswith('---') or line.startswith('+++'):
                # Header lines, skip
                continue
            elif line.startswith('@@'):
                # Hunk header, skip
                continue
            elif line.startswith('-'):
                # Line removed from template (present in template, not in contract)
                differences.append({
                    'type': 'removed',
                    'text': line[1:].rstrip(),
                    'line_number': None  # Could track line numbers if needed
                })
            elif line.startswith('+'):
                # Line added in contract (not in template)
                differences.append({
                    'type': 'added',
                    'text': line[1:].rstrip(),
                    'line_number': None
                })
        
        return {
            'is_identical': False,
            'differences': differences,
            'raw_diff': diff
        }
    
    def generate_comparison_report(
        self,
        comparison: Dict[str, Any],
        format: str = 'text'
    ) -> str:
        """
        Generate a human-readable comparison report.
        
        Args:
            comparison: Comparison results dictionary
            format: Output format ('text' or 'markdown')
            
        Returns:
            Formatted report string
        """
        summary = comparison['summary']
        
        if format == 'markdown':
            report = f"""# Contract Comparison Report

## Summary

- **Template Clauses**: {summary['total_template_clauses']}
- **Contract Clauses**: {summary['total_contract_clauses']}
- **Matched**: {summary['matched_count']}
- **Modified**: {summary['modified_count']}
- **Missing**: {summary['missing_count']}
- **Extra**: {summary['extra_count']}

"""
            
            # Matched clauses
            if comparison['matched_clauses']:
                report += "## ✅ Matched Clauses\n\n"
                for match in comparison['matched_clauses']:
                    clause = match['template_clause']
                    report += f"- **{clause['type']}** (Page {clause['page_number']})\n"
                    report += f"  - Similarity: {match['similarity']:.2%}\n\n"
            
            # Modified clauses
            if comparison['modified_clauses']:
                report += "## ⚠️ Modified Clauses\n\n"
                for mod in comparison['modified_clauses']:
                    template_clause = mod['template_clause']
                    contract_clause = mod['contract_clause']
                    report += f"### {template_clause['type']}\n\n"
                    report += f"**Template (Page {template_clause['page_number']}):**\n"
                    report += f"```\n{template_clause['text']}\n```\n\n"
                    report += f"**Contract (Page {contract_clause['page_number']}):**\n"
                    report += f"```\n{contract_clause['text']}\n```\n\n"
                    report += f"**Similarity**: {mod['similarity']:.2%}\n\n"
            
            # Missing clauses
            if comparison['missing_clauses']:
                report += "## ❌ Missing Clauses\n\n"
                for missing in comparison['missing_clauses']:
                    clause = missing['template_clause']
                    report += f"- **{clause['type']}** (Page {clause['page_number']})\n"
                    report += f"  ```\n{clause['text']}\n  ```\n\n"
            
            # Extra clauses
            if comparison['extra_clauses']:
                report += "## ➕ Extra Clauses\n\n"
                for extra in comparison['extra_clauses']:
                    clause = extra['contract_clause']
                    report += f"- **{clause['type']}** (Page {clause['page_number']})\n"
                    report += f"  ```\n{clause['text']}\n  ```\n\n"
            
            return report
        
        else:  # text format
            report = f"""Contract Comparison Report
{'=' * 50}

Summary:
  Template Clauses: {summary['total_template_clauses']}
  Contract Clauses: {summary['total_contract_clauses']}
  Matched: {summary['matched_count']}
  Modified: {summary['modified_count']}
  Missing: {summary['missing_count']}
  Extra: {summary['extra_count']}

"""
            
            # Add detailed sections (similar to markdown but plain text)
            if comparison['modified_clauses']:
                report += "\nModified Clauses:\n" + "-" * 50 + "\n"
                for mod in comparison['modified_clauses']:
                    report += f"\n{mod['template_clause']['type']}:\n"
                    report += f"Template: {mod['template_clause']['text'][:100]}...\n"
                    report += f"Contract: {mod['contract_clause']['text'][:100]}...\n"
            
            if comparison['missing_clauses']:
                report += "\nMissing Clauses:\n" + "-" * 50 + "\n"
                for missing in comparison['missing_clauses']:
                    report += f"\n{missing['template_clause']['type']}:\n"
                    report += f"{missing['template_clause']['text'][:200]}...\n"
            
            return report

