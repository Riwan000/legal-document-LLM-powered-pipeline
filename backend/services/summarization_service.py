"""
Case file summarization service.
Uses RAG to retrieve relevant sections and generates structured summaries.
"""
from typing import List, Dict, Any, Optional
import ollama
from backend.services.rag_service import RAGService
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
    
    def summarize_case_file(
        self,
        document_id: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary of a case file.
        
        Args:
            document_id: Document ID of the case file
            top_k: Number of chunks to retrieve for context
            
        Returns:
            Dictionary with:
                - executive_summary: High-level overview
                - timeline: Chronological events
                - key_arguments: Main arguments
                - open_issues: Unresolved questions
                - citations: Source citations
        """
        # Step 1: Retrieve relevant chunks using broad queries
        all_chunks = []
        # Accumulator for all relevant chunks
        
        # Query for different aspects of the case
        queries = [
            "What is this case about?",
            "What are the key events and timeline?",
            "What are the main arguments?",
            "What are the unresolved issues?",
            "Who are the parties involved?",
            "What are the key facts?"
        ]
        # Broad queries to capture different aspects
        
        for query in queries:
            chunks = self.rag_service.search(
                query=query,
                top_k=top_k,
                document_id_filter=document_id
            )
            # Retrieve relevant chunks for each query
            all_chunks.extend(chunks)
            # Add to total collection
        
        # Deduplicate chunks (same chunk might match multiple queries)
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        if not unique_chunks:
            return {
                'executive_summary': "No relevant information found in the case file.",
                'timeline': [],
                'key_arguments': [],
                'open_issues': [],
                'citations': []
            }
        
        # Step 2: Build context from all chunks
        context = self._build_context(unique_chunks)
        
        # Step 3: Generate structured summary
        summary = self._generate_structured_summary(context, unique_chunks)
        
        return summary
    
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
        
        Args:
            summary: Summary dictionary
            format: Output format ('markdown' or 'text')
            
        Returns:
            Formatted report string
        """
        if format == 'markdown':
            report = f"""# Case File Summary

## Executive Summary

{summary['executive_summary']}

"""
            
            # Timeline
            if summary['timeline']:
                report += "## Timeline of Events\n\n"
                for event in summary['timeline']:
                    report += f"- **{event.get('date', 'N/A')}**: {event.get('event', '')} {event.get('source', '')}\n"
                report += "\n"
            
            # Key Arguments
            if summary['key_arguments']:
                report += "## Key Arguments\n\n"
                for arg in summary['key_arguments']:
                    report += f"- {arg.get('argument', '')} {arg.get('source', '')}\n"
                report += "\n"
            
            # Open Issues
            if summary['open_issues']:
                report += "## Open Issues\n\n"
                for issue in summary['open_issues']:
                    report += f"- {issue.get('issue', '')} {issue.get('source', '')}\n"
                report += "\n"
            
            # Citations
            if summary['citations']:
                report += "## Source Citations\n\n"
                for citation in summary['citations']:
                    report += f"**{citation['citation']}**\n"
                    report += f"> {citation['text']}\n\n"
            
            return report
        
        else:  # text format
            report = f"""Case File Summary
{'=' * 50}

Executive Summary:
{summary['executive_summary']}

"""
            
            if summary['timeline']:
                report += "\nTimeline:\n" + "-" * 50 + "\n"
                for event in summary['timeline']:
                    report += f"{event.get('date', 'N/A')}: {event.get('event', '')}\n"
            
            if summary['key_arguments']:
                report += "\nKey Arguments:\n" + "-" * 50 + "\n"
                for arg in summary['key_arguments']:
                    report += f"- {arg.get('argument', '')}\n"
            
            if summary['open_issues']:
                report += "\nOpen Issues:\n" + "-" * 50 + "\n"
                for issue in summary['open_issues']:
                    report += f"- {issue.get('issue', '')}\n"
            
            return report

