"""
Case spine builder - creates foundational case structure.
Mandatory before any summarization.
"""
from typing import List, Dict, Any, Optional
import ollama
import json
from backend.config import settings
from backend.models.case_summary import CaseSpine


class CaseSpineBuilder:
    """Builds case spine from background, holding, and issue_framing chunks."""
    
    def __init__(self):
        """Initialize the spine builder."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def build_case_spine(
        self,
        background_chunks: List[Dict[str, Any]],
        holding_chunks: List[Dict[str, Any]],
        issue_chunks: List[Dict[str, Any]]
    ) -> CaseSpine:
        """
        Build case spine from classified chunks.
        
        Args:
            background_chunks: Chunks classified as background
            holding_chunks: Chunks classified as holding
            issue_chunks: Chunks classified as issue_framing
            
        Returns:
            Validated CaseSpine object
            
        Raises:
            ValueError: If spine cannot be built or validated
        """
        # Combine all relevant chunks
        all_chunks = background_chunks + holding_chunks + issue_chunks
        
        if not all_chunks:
            raise ValueError("Insufficient chunks to build case spine. Need background, holding, or issue_framing chunks.")
        
        # Build context
        context = self._build_context(all_chunks)
        
        # Generate spine with retry
        spine_data = None
        for attempt in range(2):  # Retry once
            try:
                spine_data = self._generate_spine(context, attempt)
                if spine_data:
                    break
            except Exception as e:
                if attempt == 1:  # Last attempt
                    raise ValueError(f"Failed to build case spine after retry: {str(e)}")
                continue
        
        if not spine_data:
            raise ValueError("Unable to construct case spine reliably.")
        
        # Validate and return
        try:
            return CaseSpine(**spine_data)
        except Exception as e:
            raise ValueError(f"Case spine validation failed: {str(e)}")
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Chunk {i} - Page {chunk.get('page_number', 0)}, Type: {chunk.get('chunk_type', 'unknown')}]\n"
                f"{chunk.get('text', '')}\n"
            )
        return "\n".join(context_parts)
    
    def _generate_spine(
        self,
        context: str,
        attempt: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate case spine using LLM.
        
        Args:
            context: Combined context from chunks
            attempt: Retry attempt number
            
        Returns:
            Spine data dict or None
        """
        prompt = f"""Extract the case spine from the following legal document chunks.

Return a JSON object with this exact structure:
{{
  "case_name": "Name of the case (e.g., 'Smith v. Jones')",
  "court": "Court name (e.g., 'Supreme Court of X')",
  "date": "Case date (YYYY-MM-DD or YYYY format)",
  "parties": ["Party 1", "Party 2"],
  "procedural_posture": "Current procedural status (e.g., 'On appeal from...')",
  "core_issues": ["Issue 1", "Issue 2"]
}}

CRITICAL RULES:
- Extract ONLY information present in the provided context
- Do NOT invent or guess information
- If information is missing, use empty string or empty array
- Be precise and factual

Context:
{context[:4000]}  # Limit context to avoid token limits

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
            json_text = self._extract_json(response_text)
            
            if json_text:
                spine_data = json.loads(json_text)
                
                # Ensure all required fields exist
                required_fields = ['case_name', 'court', 'date', 'parties', 'procedural_posture', 'core_issues']
                for field in required_fields:
                    if field not in spine_data:
                        if field == 'parties':
                            spine_data[field] = []
                        elif field == 'core_issues':
                            spine_data[field] = []
                        else:
                            spine_data[field] = ""
                
                return spine_data
            
            return None
            
        except Exception as e:
            print(f"Error generating case spine (attempt {attempt}): {e}")
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response."""
        # Remove markdown code blocks
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()
        elif '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()
        
        # Try to find JSON object
        start_brace = text.find('{')
        end_brace = text.rfind('}')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            return text[start_brace:end_brace + 1]
        
        return None

