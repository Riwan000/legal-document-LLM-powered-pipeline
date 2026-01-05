"""
Section-wise summarizers for case files.
Each section is generated independently with strict citation enforcement.
"""
from typing import List, Dict, Any, Optional
import ollama
import json
from backend.config import settings
from backend.models.case_summary import (
    CaseSpine, ExecutiveSummaryItem, TimelineEvent,
    ArgumentItem, OpenIssue, SourceCitation
)


class CaseSectionSummarizers:
    """Generates individual sections of case summary."""
    
    def __init__(self):
        """Initialize summarizers."""
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def generate_executive_summary(
        self,
        case_spine: CaseSpine,
        chunks: List[Dict[str, Any]]
    ) -> List[ExecutiveSummaryItem]:
        """
        Generate executive summary from background and holding chunks.
        
        Args:
            case_spine: Validated case spine
            chunks: Pre-filtered chunks (background + holding)
            
        Returns:
            List of executive summary items with citations
        """
        if not chunks:
            return []
        
        context = self._build_context_with_citations(chunks)
        
        prompt = f"""Generate an executive summary for this legal case.

Case Spine:
- Case: {case_spine.case_name}
- Court: {case_spine.court}
- Date: {case_spine.date}
- Parties: {', '.join(case_spine.parties)}
- Core Issues: {', '.join(case_spine.core_issues)}

Return a JSON array of summary items. Each item must:
1. Be a factual statement about the case
2. Include a source citation pointing to the chunk used
3. Cover: nature of dispute, parties, legal context, what court decided

Return this exact structure:
[
  {{
    "text": "Summary statement",
    "source": {{
      "document": "<document_id>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }}
  }}
]

CRITICAL RULES:
- Each item MUST cite a source chunk_id from the context
- Do NOT invent facts not in the context
- Base statements ONLY on provided chunks
- Maximum 5-7 items

Context:
{context[:4000]}

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
            items_data = self._extract_json_array(response_text)
            
            if not items_data:
                return []
            
            # Validate and convert to models
            summary_items = []
            chunk_map = {c.get('chunk_id'): c for c in chunks}
            
            for item_data in items_data:
                try:
                    source_data = item_data.get('source', {})
                    chunk_id = source_data.get('chunk_id', '')
                    
                    # Validate chunk_id exists
                    if chunk_id not in chunk_map:
                        continue  # Skip invalid citations
                    
                    source_chunk = chunk_map[chunk_id]
                    source = SourceCitation(
                        document=source_chunk.get('document_id', ''),
                        page=source_chunk.get('page_number', 0),
                        chunk_id=chunk_id
                    )
                    
                    summary_items.append(ExecutiveSummaryItem(
                        text=item_data.get('text', ''),
                        source=source
                    ))
                except Exception as e:
                    print(f"Error processing executive summary item: {e}")
                    continue
            
            return summary_items
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return []
    
    def generate_timeline(
        self,
        case_spine: CaseSpine,
        chunks: List[Dict[str, Any]]
    ) -> List[TimelineEvent]:
        """
        Generate timeline from procedural_history chunks.
        
        Args:
            case_spine: Validated case spine
            chunks: Pre-filtered procedural_history chunks
            
        Returns:
            List of timeline events with citations
        """
        if not chunks:
            return []
        
        context = self._build_context_with_citations(chunks)
        
        prompt = f"""Generate a chronological timeline of events for this legal case.

Case: {case_spine.case_name}

Return a JSON array of timeline events. Each event must:
1. Have a date (or relative ordering if date missing)
2. Describe a procedural event
3. Cite the source chunk

Return this exact structure:
[
  {{
    "date": "YYYY-MM or YYYY-MM-DD or 'Before [event]'",
    "event": "Description of event",
    "source": {{
      "document": "<document_id>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }}
  }}
]

CRITICAL RULES:
- Events must be chronological
- Each event MUST cite a source chunk_id
- Use dates from context, or relative ordering if dates missing
- Maximum 15-20 events

Context:
{context[:4000]}

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
            events_data = self._extract_json_array(response_text)
            
            if not events_data:
                return []
            
            # Validate and convert
            timeline_events = []
            chunk_map = {c.get('chunk_id'): c for c in chunks}
            
            for event_data in events_data:
                try:
                    source_data = event_data.get('source', {})
                    chunk_id = source_data.get('chunk_id', '')
                    
                    if chunk_id not in chunk_map:
                        continue
                    
                    source_chunk = chunk_map[chunk_id]
                    source = SourceCitation(
                        document=source_chunk.get('document_id', ''),
                        page=source_chunk.get('page_number', 0),
                        chunk_id=chunk_id
                    )
                    
                    timeline_events.append(TimelineEvent(
                        date=event_data.get('date', ''),
                        event=event_data.get('event', ''),
                        source=source
                    ))
                except Exception as e:
                    print(f"Error processing timeline event: {e}")
                    continue
            
            # Sort by date if possible
            timeline_events.sort(key=lambda e: self._parse_date_for_sort(e.date))
            
            return timeline_events
            
        except Exception as e:
            print(f"Error generating timeline: {e}")
            return []
    
    def generate_claimant_arguments(
        self,
        case_spine: CaseSpine,
        chunks: List[Dict[str, Any]]
    ) -> List[ArgumentItem]:
        """
        Generate claimant arguments from argument_claimant chunks.
        
        Args:
            case_spine: Validated case spine
            chunks: Pre-filtered argument_claimant chunks
            
        Returns:
            List of argument items with citations
        """
        if not chunks:
            return []
        
        context = self._build_context_with_citations(chunks)
        
        prompt = f"""Extract key arguments made by the claimant/plaintiff/appellant.

Case: {case_spine.case_name}
Parties: {', '.join(case_spine.parties)}

Return a JSON array of arguments. Each argument must:
1. Paraphrase (not quote) the argument
2. Cite the source chunk

Return this exact structure:
[
  {{
    "text": "Paraphrased argument",
    "source": {{
      "document": "<document_id>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }}
  }}
]

CRITICAL RULES:
- Paraphrase, don't quote verbatim
- Each argument MUST cite a source chunk_id
- Maximum 10-15 arguments

Context:
{context[:4000]}

JSON response:"""
        
        return self._generate_arguments(prompt, context, chunks)
    
    def generate_defendant_arguments(
        self,
        case_spine: CaseSpine,
        chunks: List[Dict[str, Any]]
    ) -> List[ArgumentItem]:
        """
        Generate defendant arguments from argument_defendant chunks.
        
        Args:
            case_spine: Validated case spine
            chunks: Pre-filtered argument_defendant chunks
            
        Returns:
            List of argument items with citations
        """
        if not chunks:
            return []
        
        context = self._build_context_with_citations(chunks)
        
        prompt = f"""Extract key arguments made by the defendant/respondent/appellee.

Case: {case_spine.case_name}
Parties: {', '.join(case_spine.parties)}

Return a JSON array of arguments. Each argument must:
1. Paraphrase (not quote) the argument
2. Cite the source chunk

Return this exact structure:
[
  {{
    "text": "Paraphrased argument",
    "source": {{
      "document": "<document_id>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }}
  }}
]

CRITICAL RULES:
- Paraphrase, don't quote verbatim
- Each argument MUST cite a source chunk_id
- Maximum 10-15 arguments

Context:
{context[:4000]}

JSON response:"""
        
        return self._generate_arguments(prompt, context, chunks)
    
    def generate_open_issues(
        self,
        case_spine: CaseSpine,
        chunks: List[Dict[str, Any]]
    ) -> List[OpenIssue]:
        """
        Generate open issues from issue_framing chunks.
        
        Args:
            case_spine: Validated case spine
            chunks: Pre-filtered issue_framing chunks
            
        Returns:
            List of open issues with citations
        """
        if not chunks:
            return []
        
        context = self._build_context_with_citations(chunks)
        
        prompt = f"""Identify open/unresolved issues in this case.

Case: {case_spine.case_name}
Core Issues: {', '.join(case_spine.core_issues)}

Return a JSON array of open issues. Each issue must:
1. Be an issue framed by the court or pending adjudication
2. NOT predict outcomes
3. Cite the source chunk

Return this exact structure:
[
  {{
    "text": "Description of unresolved issue",
    "source": {{
      "document": "<document_id>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }}
  }}
]

CRITICAL RULES:
- Only include issues NOT finally determined
- Do NOT predict outcomes
- Do NOT invent unresolved points
- Each issue MUST cite a source chunk_id

Context:
{context[:4000]}

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
            issues_data = self._extract_json_array(response_text)
            
            if not issues_data:
                return []
            
            # Validate and convert
            open_issues = []
            chunk_map = {c.get('chunk_id'): c for c in chunks}
            
            for issue_data in issues_data:
                try:
                    source_data = issue_data.get('source', {})
                    chunk_id = source_data.get('chunk_id', '')
                    
                    if chunk_id not in chunk_map:
                        continue
                    
                    source_chunk = chunk_map[chunk_id]
                    source = SourceCitation(
                        document=source_chunk.get('document_id', ''),
                        page=source_chunk.get('page_number', 0),
                        chunk_id=chunk_id
                    )
                    
                    open_issues.append(OpenIssue(
                        text=issue_data.get('text', ''),
                        source=source
                    ))
                except Exception as e:
                    print(f"Error processing open issue: {e}")
                    continue
            
            return open_issues
            
        except Exception as e:
            print(f"Error generating open issues: {e}")
            return []
    
    def _generate_arguments(
        self,
        prompt: str,
        context: str,
        chunks: List[Dict[str, Any]]
    ) -> List[ArgumentItem]:
        """Helper to generate arguments (used by both claimant and defendant)."""
        try:
            # Prompt already includes context
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
            args_data = self._extract_json_array(response_text)
            
            if not args_data:
                return []
            
            # Validate and convert
            arguments = []
            chunk_map = {c.get('chunk_id'): c for c in chunks}
            
            for arg_data in args_data:
                try:
                    source_data = arg_data.get('source', {})
                    chunk_id = source_data.get('chunk_id', '')
                    
                    if chunk_id not in chunk_map:
                        continue
                    
                    source_chunk = chunk_map[chunk_id]
                    source = SourceCitation(
                        document=source_chunk.get('document_id', ''),
                        page=source_chunk.get('page_number', 0),
                        chunk_id=chunk_id
                    )
                    
                    arguments.append(ArgumentItem(
                        text=arg_data.get('text', ''),
                        source=source
                    ))
                except Exception as e:
                    print(f"Error processing argument: {e}")
                    continue
            
            return arguments
            
        except Exception as e:
            print(f"Error generating arguments: {e}")
            return []
    
    def _build_context_with_citations(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string with chunk citations."""
        context_parts = []
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            page = chunk.get('page_number', 0)
            doc_id = chunk.get('document_id', '')
            text = chunk.get('text', '')
            
            context_parts.append(
                f"[Chunk ID: {chunk_id}, Document: {doc_id}, Page: {page}]\n"
                f"{text}\n"
            )
        return "\n".join(context_parts)
    
    def _extract_json_array(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response."""
        # Remove markdown code blocks
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        
        # Try to find JSON array
        start_bracket = text.find('[')
        end_bracket = text.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            json_text = text[start_bracket:end_bracket + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _parse_date_for_sort(self, date_str: str) -> str:
        """Parse date string for sorting (returns sortable string)."""
        # Try to extract year-month-day for sorting
        import re
        # Match YYYY-MM-DD
        match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
        if match:
            return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
        
        # Match YYYY-MM
        match = re.search(r'(\d{4})-(\d{1,2})', date_str)
        if match:
            return f"{match.group(1)}-{match.group(2).zfill(2)}-00"
        
        # Match YYYY
        match = re.search(r'(\d{4})', date_str)
        if match:
            return f"{match.group(1)}-00-00"
        
        # Default: return as-is for relative dates
        return date_str

