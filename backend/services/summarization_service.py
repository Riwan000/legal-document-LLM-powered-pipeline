"""
Case file summarization service.
PRD-compliant deterministic section-aware case summarization.
"""
from typing import List, Dict, Any, Optional
import ollama
from backend.services.rag_service import RAGService
from backend.services.case_chunk_classifier import CaseChunkClassifier
from backend.services.case_spine_builder import CaseSpineBuilder
from backend.services.case_section_summarizers import CaseSectionSummarizers
from backend.models.case_summary import (
    CaseSummary, CaseSummaryError, CitationMetadata, KeyArguments
)
from backend.config import settings


class SummarizationService:
    """Service for summarizing case files with citations."""
    
    def __init__(self, rag_service: RAGService):
        """
        Initialize the summarization service.
        
        Args:
            rag_service: RAG service for retrieving relevant document sections
        """
        self.rag_service = rag_service
        # Use RAG service to retrieve relevant chunks
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        # Direct Ollama client for structured generation
        
        # PRD-compliant components
        self.chunk_classifier = CaseChunkClassifier()
        self.spine_builder = CaseSpineBuilder()
        self.section_summarizers = CaseSectionSummarizers()
    
    def summarize_case_file(
        self,
        document_id: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary of a case file (PRD-compliant).
        
        Args:
            document_id: Document ID of the case file
            top_k: Number of chunks to retrieve for context (legacy, uses config defaults)
            
        Returns:
            CaseSummary model or error dict
        """
        try:
            # Step 1: Retrieve all chunks for the document
            all_chunks = self._get_all_document_chunks(document_id)
            
            if not all_chunks:
                return {
                    "error": {
                        "code": "NO_CHUNKS_FOUND",
                        "message": "No chunks found for document.",
                        "details": {"document_id": document_id}
                    }
                }
            
            # Step 2: Backfill chunk IDs and classify chunks
            all_chunks = self._backfill_and_classify_chunks(all_chunks, document_id)
            
            # Step 3: Group chunks by type
            chunks_by_type = self._group_chunks_by_type(all_chunks)
            
            # Step 4: Build case spine (MANDATORY)
            try:
                case_spine = self.spine_builder.build_case_spine(
                    background_chunks=chunks_by_type.get('background', []),
                    holding_chunks=chunks_by_type.get('holding', []),
                    issue_chunks=chunks_by_type.get('issue_framing', [])
                )
            except ValueError as e:
                return {
                    "error": {
                        "code": "CASE_SPINE_FAILED",
                        "message": str(e),
                        "details": {
                            "background_chunks": len(chunks_by_type.get('background', [])),
                            "holding_chunks": len(chunks_by_type.get('holding', [])),
                            "issue_chunks": len(chunks_by_type.get('issue_framing', []))
                        }
                    }
                }
            
            # Step 5: Hard-gated chunk selection per section
            exec_chunks = self._gate_chunks_for_section(
                chunks_by_type, ['background', 'holding'], 'executive_summary'
            )
            timeline_chunks = self._gate_chunks_for_section(
                chunks_by_type, ['procedural_history'], 'timeline'
            )
            claimant_chunks = self._gate_chunks_for_section(
                chunks_by_type, ['argument_claimant'], 'claimant_arguments'
            )
            defendant_chunks = self._gate_chunks_for_section(
                chunks_by_type, ['argument_defendant'], 'defendant_arguments'
            )
            issues_chunks = self._gate_chunks_for_section(
                chunks_by_type, ['issue_framing'], 'open_issues'
            )
            
            # Step 6: Generate sections independently
            executive_summary = self.section_summarizers.generate_executive_summary(
                case_spine, exec_chunks
            )
            timeline = self.section_summarizers.generate_timeline(
                case_spine, timeline_chunks
            )
            claimant_args = self.section_summarizers.generate_claimant_arguments(
                case_spine, claimant_chunks
            )
            defendant_args = self.section_summarizers.generate_defendant_arguments(
                case_spine, defendant_chunks
            )
            open_issues = self.section_summarizers.generate_open_issues(
                case_spine, issues_chunks
            )
            
            # Step 7: Collect all citations
            all_citations = self._collect_citations(
                executive_summary + timeline + claimant_args + defendant_args + open_issues,
                all_chunks
            )
            
            # Step 8: Assemble and validate
            case_summary = CaseSummary(
                case_spine=case_spine,
                executive_summary=executive_summary,
                timeline=timeline,
                key_arguments=KeyArguments(
                    claimant=claimant_args,
                    defendant=defendant_args
                ),
                open_issues=open_issues,
                citations=all_citations
            )
            
            # Return as dict for API compatibility
            return case_summary.model_dump()
            
        except Exception as e:
            return {
                "error": {
                    "code": "SUMMARIZATION_ERROR",
                    "message": f"Error generating summary: {str(e)}",
                    "details": {}
                }
            }
    
    def _get_all_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document from vector store."""
        # Use RAG service's vector store to get all chunks
        if hasattr(self.rag_service, 'vector_store'):
            return self.rag_service.vector_store.get_chunks_by_document(document_id)
        return []
    
    def _backfill_and_classify_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Backfill missing chunk_id and classify chunks."""
        total_chunks = len(chunks)
        needs_update = False
        chunk_updates = {}
        
        for i, chunk in enumerate(chunks):
            original_chunk_id = chunk.get('chunk_id')
            
            # Backfill chunk_id if missing
            if 'chunk_id' not in chunk or not chunk.get('chunk_id'):
                page = chunk.get('page_number', 0)
                chunk_idx = chunk.get('chunk_index', i)
                chunk['chunk_id'] = f"c_p{page:04d}_i{chunk_idx:04d}"
                needs_update = True
            
            # Ensure document_id is set
            chunk['document_id'] = chunk.get('document_id', document_id)
            
            # Classify if missing
            if 'chunk_type' not in chunk or not chunk.get('chunk_type'):
                chunk_type = self.chunk_classifier.classify_chunk(
                    chunk_text=chunk.get('text', ''),
                    chunk_id=chunk.get('chunk_id', ''),
                    page_number=chunk.get('page_number', 0),
                    chunk_index=chunk.get('chunk_index', i),
                    total_chunks=total_chunks,
                    document_id=document_id
                )
                chunk['chunk_type'] = chunk_type
                needs_update = True
            
            # Track updates for vector store
            if needs_update:
                chunk_id = chunk.get('chunk_id')
                chunk_updates[chunk_id] = {
                    'chunk_id': chunk_id,
                    'chunk_type': chunk.get('chunk_type')
                }
        
        # Update vector store if needed
        if needs_update and hasattr(self.rag_service, 'vector_store'):
            self.rag_service.vector_store.update_chunk_metadata(document_id, chunk_updates)
            # Save to persist updates
            self.rag_service.vector_store.save()
        
        return chunks
    
    def _group_chunks_by_type(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their chunk_type."""
        grouped = {}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'background')
            if chunk_type not in grouped:
                grouped[chunk_type] = []
            grouped[chunk_type].append(chunk)
        return grouped
    
    def _gate_chunks_for_section(
        self,
        chunks_by_type: Dict[str, List[Dict[str, Any]]],
        required_types: List[str],
        section_name: str
    ) -> List[Dict[str, Any]]:
        """
        Hard gate: return chunks only if required types are present.
        
        Raises:
            ValueError: If required chunk types are missing
        """
        result = []
        for chunk_type in required_types:
            if chunk_type in chunks_by_type:
                result.extend(chunks_by_type[chunk_type])
            else:
                # Check if we have any chunks at all
                if not result:
                    # This will be caught by caller and return safe error
                    pass
        
        return result
    
    def _collect_citations(
        self,
        items_with_sources: List,
        all_chunks: List[Dict[str, Any]]
    ) -> List[CitationMetadata]:
        """Collect all unique citations from summary items."""
        chunk_map = {c.get('chunk_id'): c for c in all_chunks}
        seen_citations = set()
        citations = []
        
        for item in items_with_sources:
            if hasattr(item, 'source'):
                source = item.source
                citation_key = (source.document, source.page, source.chunk_id)
                
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    chunk = chunk_map.get(source.chunk_id, {})
                    citations.append(CitationMetadata(
                        document=source.document,
                        page=source.page,
                        chunk_id=source.chunk_id,
                        chunk_type=chunk.get('chunk_type', 'unknown')
                    ))
        
        return citations
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks based on text content.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Deduplicated list
        """
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Normalize text for comparison
            normalized_text = chunk['text'].lower().strip()[:100]
            # Use first 100 chars as key (handles minor variations)
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from chunks with citations.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i} - Document: {chunk['document_id']}, Page: {chunk['page_number']}]\n"
                f"{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_structured_summary(
        self,
        context: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate structured summary using LLM.
        
        Args:
            context: Combined context from all chunks
            chunks: List of chunks for citations
            
        Returns:
            Structured summary dictionary
        """
        # Build prompt for structured summary
        prompt = self._build_summary_prompt(context)
        
        try:
            # Generate summary
            response = self.ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Slightly creative but still factual
                    'format': 'json'  # Request structured JSON output
                }
            )
            
            # Parse response
            summary_data = self._parse_summary_response(response.get('response', ''))
            
            # Add citations
            summary_data['citations'] = self._format_citations(chunks)
            
            return summary_data
            
        except Exception as e:
            # Fallback if LLM generation fails
            return {
                'executive_summary': f"Error generating summary: {str(e)}",
                'timeline': [],
                'key_arguments': [],
                'open_issues': [],
                'citations': self._format_citations(chunks)
            }
    
    def _build_summary_prompt(self, context: str) -> str:
        """
        Build prompt for case file summarization.
        
        Args:
            context: Document context from RAG retrieval
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a legal case file analyst. Analyze the following case file content and generate a structured summary.

IMPORTANT RULES:
- Base your analysis ONLY on the provided context
- Cite sources using [Source N] format when referencing information
- Do NOT provide legal advice or interpretation
- Extract factual information only
- If information is not in the context, do not make it up

Return your response as a JSON object with this exact structure:
{{
  "executive_summary": "High-level overview of the case (2-3 paragraphs)",
  "timeline": [
    {{
      "date": "Date or time period",
      "event": "Description of event",
      "source": "Source N"
    }}
  ],
  "key_arguments": [
    {{
      "argument": "Description of argument",
      "source": "Source N"
    }}
  ],
  "open_issues": [
    {{
      "issue": "Description of unresolved issue",
      "source": "Source N"
    }}
  ]
}}

Case file content:
{context}

JSON response:"""
        
        return prompt
    
    def _parse_summary_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract summary information.
        
        Args:
            response_text: LLM response (should be JSON)
            
        Returns:
            Parsed summary dictionary
        """
        import json
        
        # Extract JSON from response (might be wrapped in markdown)
        json_text = self._extract_json_from_response(response_text)
        
        if not json_text:
            return {
                'executive_summary': "Failed to parse summary response.",
                'timeline': [],
                'key_arguments': [],
                'open_issues': []
            }
        
        try:
            summary = json.loads(json_text)
            
            # Validate structure
            return {
                'executive_summary': summary.get('executive_summary', ''),
                'timeline': summary.get('timeline', []),
                'key_arguments': summary.get('key_arguments', []),
                'open_issues': summary.get('open_issues', [])
            }
            
        except json.JSONDecodeError:
            return {
                'executive_summary': "Failed to parse JSON response.",
                'timeline': [],
                'key_arguments': [],
                'open_issues': []
            }
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract JSON from LLM response (might be wrapped in markdown).
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            JSON string or None
        """
        # Remove markdown code blocks
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        
        # Try to find JSON object
        start_brace = response_text.find('{')
        end_brace = response_text.rfind('}')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            return response_text[start_brace:end_brace + 1]
        
        return response_text.strip()
    
    def _format_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format citations from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of formatted citation dictionaries
        """
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            citations.append({
                'source_number': i,
                'document_id': chunk['document_id'],
                'page_number': chunk['page_number'],
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'citation': f"[Source {i}] Document {chunk['document_id'][:8]}..., Page {chunk['page_number']}"
            })
        
        return citations
    
    def generate_summary_report(
        self,
        summary: Dict[str, Any],
        format: str = 'markdown'
    ) -> str:
        """
        Generate a human-readable summary report.
        Handles both old and new summary formats.
        
        Args:
            summary: Summary dictionary (new format with case_spine, or old format)
            format: Output format ('markdown' or 'text')
            
        Returns:
            Formatted report string
        """
        # Detect format: new format has case_spine, old format has executive_summary as string
        is_new_format = 'case_spine' in summary
        
        if format == 'markdown':
            report = "# Case File Summary\n\n"
            
            # Case Spine (new format only)
            if is_new_format and summary.get('case_spine'):
                spine = summary['case_spine']
                report += "## Case Spine\n\n"
                report += f"- **Case:** {spine.get('case_name', 'N/A')}\n"
                report += f"- **Court:** {spine.get('court', 'N/A')}\n"
                report += f"- **Date:** {spine.get('date', 'N/A')}\n"
                report += f"- **Parties:** {', '.join(spine.get('parties', []))}\n"
                report += f"- **Procedural Posture:** {spine.get('procedural_posture', 'N/A')}\n"
                if spine.get('core_issues'):
                    report += "- **Core Issues:**\n"
                    for issue in spine['core_issues']:
                        report += f"  - {issue}\n"
                report += "\n"
            
            # Executive Summary
            report += "## Executive Summary\n\n"
            if is_new_format:
                # New format: array of items
                exec_items = summary.get('executive_summary', [])
                if exec_items:
                    for item in exec_items:
                        source = item.get('source', {})
                        report += f"{item.get('text', '')}\n"
                        report += f"*Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}*\n\n"
                else:
                    report += "No executive summary available.\n\n"
            else:
                # Old format: string
                report += f"{summary.get('executive_summary', 'No summary available')}\n\n"
            
            # Timeline
            if summary.get('timeline'):
                report += "## Timeline of Events\n\n"
                for event in summary['timeline']:
                    if is_new_format:
                        source = event.get('source', {})
                        report += f"- **{event.get('date', 'N/A')}**: {event.get('event', '')}\n"
                        report += f"  *Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}*\n"
                    else:
                        report += f"- **{event.get('date', 'N/A')}**: {event.get('event', '')} {event.get('source', '')}\n"
                report += "\n"
            
            # Key Arguments
            if summary.get('key_arguments'):
                report += "## Key Arguments\n\n"
                if is_new_format:
                    args = summary['key_arguments']
                    if args.get('claimant'):
                        report += "### Claimant/Plaintiff Arguments\n\n"
                        for arg in args['claimant']:
                            source = arg.get('source', {})
                            report += f"- {arg.get('text', '')}\n"
                            report += f"  *Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}*\n"
                        report += "\n"
                    if args.get('defendant'):
                        report += "### Defendant/Respondent Arguments\n\n"
                        for arg in args['defendant']:
                            source = arg.get('source', {})
                            report += f"- {arg.get('text', '')}\n"
                            report += f"  *Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}*\n"
                        report += "\n"
                else:
                    # Old format
                    for arg in summary['key_arguments']:
                        report += f"- {arg.get('argument', '')} {arg.get('source', '')}\n"
                    report += "\n"
            
            # Open Issues
            if summary.get('open_issues'):
                report += "## Open Issues\n\n"
                for issue in summary['open_issues']:
                    if is_new_format:
                        source = issue.get('source', {})
                        report += f"- {issue.get('text', '')}\n"
                        report += f"  *Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}*\n"
                    else:
                        report += f"- {issue.get('issue', '')} {issue.get('source', '')}\n"
                report += "\n"
            
            # Citations
            if summary.get('citations'):
                report += "## Source Citations\n\n"
                for citation in summary['citations']:
                    if is_new_format:
                        report += f"- **{citation.get('chunk_id', '')}**: Page {citation.get('page', 0)}, Type: {citation.get('chunk_type', 'unknown')}\n"
                    else:
                        report += f"**{citation.get('citation', '')}**\n"
                        report += f"> {citation.get('text', '')}\n\n"
            
            return report
        
        else:  # text format
            report = f"""Case File Summary
{'=' * 50}

"""
            if is_new_format and summary.get('case_spine'):
                spine = summary['case_spine']
                report += f"Case: {spine.get('case_name', 'N/A')}\n"
                report += f"Court: {spine.get('court', 'N/A')}\n"
                report += f"Date: {spine.get('date', 'N/A')}\n\n"
            
            report += "Executive Summary:\n" + "-" * 50 + "\n"
            if is_new_format:
                for item in summary.get('executive_summary', []):
                    report += f"{item.get('text', '')}\n"
            else:
                report += f"{summary.get('executive_summary', '')}\n"
            
            if summary.get('timeline'):
                report += "\nTimeline:\n" + "-" * 50 + "\n"
                for event in summary['timeline']:
                    report += f"{event.get('date', 'N/A')}: {event.get('event', '')}\n"
            
            if summary.get('key_arguments'):
                report += "\nKey Arguments:\n" + "-" * 50 + "\n"
                if is_new_format:
                    args = summary['key_arguments']
                    if args.get('claimant'):
                        report += "Claimant Arguments:\n"
                        for arg in args['claimant']:
                            report += f"- {arg.get('text', '')}\n"
                    if args.get('defendant'):
                        report += "Defendant Arguments:\n"
                        for arg in args['defendant']:
                            report += f"- {arg.get('text', '')}\n"
                else:
                    for arg in summary['key_arguments']:
                        report += f"- {arg.get('argument', '')}\n"
            
            if summary.get('open_issues'):
                report += "\nOpen Issues:\n" + "-" * 50 + "\n"
                for issue in summary['open_issues']:
                    if is_new_format:
                        report += f"- {issue.get('text', '')}\n"
                    else:
                        report += f"- {issue.get('issue', '')}\n"
            
            return report

