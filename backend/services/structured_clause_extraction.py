"""
Deterministic Structure-Aware Legal Clause Extraction Engine.
Converts unstructured legal documents into verbatim, auditable clauses without interpretation.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass

from backend.config import settings
from backend.services.document_ingestion import DocumentIngestionService
from backend.services.clause_taxonomy import ClauseTaxonomyService
from backend.utils.log_safety import sanitize_for_log

# Configure logger for clause extraction
logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Represents a page with page number and lines."""
    page_number: int
    lines: List[str]
    is_ocr: bool = False
    
    def get_text(self) -> str:
        """Get full text by joining lines (for section classification)."""
        return '\n'.join(self.lines)


@dataclass
class _ClauseBufferState:
    """Mutable accumulator for the in-progress clause during section traversal."""
    buffer: Optional[Dict[str, Any]] = None
    section: Optional["DocumentSection"] = None
    page_start: Optional[int] = None

    def reset(self) -> None:
        self.buffer = None
        self.section = None
        self.page_start = None


class DocumentSection(str, Enum):
    """Document section types."""
    ADMINISTRATIVE_MATERIAL = "administrative_material"
    CONTRACTUAL_TERMS = "contractual_terms"
    JUDICIAL_REASONING = "judicial_reasoning"
    STATUTORY_TEXT = "statutory_text"
    ANNEXURES_SCHEDULES = "annexures_schedules"
    AMBIGUOUS = "ambiguous"  # Page cannot be confidently classified
    UNKNOWN = "unknown"      # Section unknown but still eligible for extraction


class DocumentType(str, Enum):
    """Document type for section gating."""
    CONTRACT = "contract"
    JUDGMENT = "judgment"
    STATUTE = "statute"
    UNKNOWN = "unknown"


class ExtractedClause:
    """Represents an extracted clause with verbatim text."""
    
    def __init__(
        self,
        clause_id: str,
        document_section: str,
        page_start: int,
        page_end: int,
        clause_heading: str,
        verbatim_text: str,
        normalized_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.clause_id = clause_id
        self.document_section = document_section
        self.page_start = page_start
        self.page_end = page_end
        self.clause_heading = clause_heading
        self.verbatim_text = verbatim_text
        self.normalized_text = normalized_text
        self.metadata = metadata or {}
        # Phase 2 — RAG 5-Layer Upgrade
        self.clause_number: Optional[str] = None
        self.clause_title: Optional[str] = None   # same as clause_heading but explicitly named
        self.legal_category: Optional[str] = None
        self.unit_type: str = "clause"            # "clause" | "definition" | "page_chunk"
        self.is_definition: bool = False
        self.parent_clause_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response (strict JSON schema)."""
        result = {
            "clause_id": self.clause_id,
            "document_section": self.document_section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "clause_heading": self.clause_heading if self.clause_heading else None,
            "verbatim_text": self.verbatim_text,
            # Phase 2 fields
            "clause_number": getattr(self, "clause_number", None),
            "clause_title": getattr(self, "clause_title", None),
            "legal_category": getattr(self, "legal_category", None),
            "unit_type": getattr(self, "unit_type", "clause"),
            "is_definition": getattr(self, "is_definition", False),
            "parent_clause_id": getattr(self, "parent_clause_id", None),
        }
        if self.normalized_text:
            result["normalized_text"] = self.normalized_text
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class StructuredClauseExtractionService:
    """Deterministic service for extracting legal clauses with structure-first approach."""
    
    # Hard condition: Minimum character count for valid clause
    MIN_CLAUSE_LENGTH = 50
    MIN_VERB_LINES_FOR_CONTRACT = 2
    MAX_HEADING_LENGTH = 80
    MAX_DOC_TYPE_SAMPLE_PAGES = 3
    
    # Conservative keyword map for post-extraction section labeling.
    SECTION_KEYWORDS = {
        "termination": [
            "terminate",
            "termination",
            "probation",
            "notice period",
            "notice",
            "expiry",
        ],
        "payment": [
            "payment",
            "wages",
            "salary",
            "compensation",
            "fees",
            "remuneration",
            "blood money",
        ],
        "liability": [
            "liable",
            "liability",
            "responsible",
            "indemnify",
            "indemnity",
        ],
    }
    
    CONFIDENTIALITY_PATTERNS = [
        r"shall\s+keep\s+confidential",
        r"must\s+keep\s+confidential",
        r"undertakes?\s+to\s+keep\s+confidential",
        r"shall\s+not\s+disclose",
        r"non[-\s]?disclosure",
    ]
    
    CONFIDENCE_MAP = {
        "keyword_match": "medium",
        "multiple_section_matches": "low",
        "no_keywords_found": "low",
        "document_type_gate": "high",
        "case_citation_pattern": "medium",
    }
    
    def __init__(self):
        """Initialize structured clause extraction service."""
        self.ingestion_service = DocumentIngestionService()
        self.taxonomy_service = ClauseTaxonomyService()
        self.last_document_type: Optional[DocumentType] = None
        
        # Section detection patterns (rule-based)
        self.section_patterns = {
            DocumentSection.ADMINISTRATIVE_MATERIAL: [
                r'\b(cover\s+letter|dear\s+sir|dear\s+madam|yours\s+sincerely|regards|signature|date:|dated:)',
                r'\b(page\s+\d+|page\s+of\s+\d+)',
                r'\b(private\s+and\s+confidential|internal\s+use\s+only|confidential\s+document)',
            ],
            DocumentSection.CONTRACTUAL_TERMS: [
                r'\b(contract|agreement|terms\s+and\s+conditions|clause|article|section)',
                r'\b(party|parties|employer|employee|contractor)',
                r'\b(hereby|whereas|now\s+therefore)',
            ],
            DocumentSection.JUDICIAL_REASONING: [
                r'\b(court|judge|judgment|order|ruling|decision)',
                r'\b(plaintiff|defendant|petitioner|respondent)',
                r'\b(whereas\s+the\s+court|it\s+is\s+ordered|the\s+court\s+finds)',
            ],
            DocumentSection.STATUTORY_TEXT: [
                r'\b(act|statute|regulation|law|legislation)',
                r'\b(section\s+\d+|article\s+\d+|clause\s+\d+)',
                r'\b(shall\s+be|must\s+comply|required\s+by\s+law)',
            ],
            DocumentSection.ANNEXURES_SCHEDULES: [
                r'\b(annexure|annex|schedule|appendix|attachment)',
                r'\b(schedule\s+[a-z]|annexure\s+\d+)',
            ],
        }
        
        # Operative sections that can produce clauses.
        # ANNEXURES_SCHEDULES must be included so that SLA/IP clauses defined
        # in Schedule-A or Annexure sections are not silently dropped.
        self.operative_sections = {
            DocumentSection.CONTRACTUAL_TERMS,
            DocumentSection.JUDICIAL_REASONING,
            DocumentSection.STATUTORY_TEXT,
            DocumentSection.ANNEXURES_SCHEDULES,
            DocumentSection.UNKNOWN,
        }
        
        # Contract body entry gate patterns (must match at least two on the same page).
        # Covers employment contracts, MSAs, NDAs, and generic service agreements.
        self.contract_entry_patterns = [
            r"\bemployment\s+agreement\b",
            r"\b(first|1st)\s+party\b",
            r"\b(second|2nd)\s+party\b",
            r"\bthis\s+agreement\b",
            r"\bthe\s+first\s+party\s+shall\b",
            r"\bthe\s+second\s+party\s+shall\b",
            # MSA / service contract patterns
            r"\bmaster\s+service\s+agreement\b",
            r"\bservice\s+provider\b",
            r"\bhereinafter\s+(referred\s+to\s+as|called)\b",
            r"\bthis\s+master\s+service\s+agreement\b",
            r"\bclient\s+shall\b",
            r"\bvendor\s+shall\b",
            # NDA-specific patterns
            r"\bnon.?disclosure\s+agreement\b",
            r"\bconfidentiality\s+agreement\b",
            r"\bdisclosing\s+party\b",
            r"\breceiving\s+party\b",
            r"\brecipient\s+shall\b",
            r"\bconfidential\s+information\s+shall\b",
            r"\bunauthorized\s+disclosure\b",
            r"\bobligation\s+of\s+confidentiality\b",
            # General contract patterns
            r"\bservice\s+agreement\b",
            r"\bthe\s+parties\s+agree\b",
            r"\bhereby\s+agree[sd]?\b",
            r"\bin\s+witness\s+whereof\b",
            r"\bnow[\s,]+therefore\b",
        ]
        
        # Administrative override markers (force administrative_material)
        self.administrative_markers = [
            r"\bministry\b",
            r"\bminister\b",
            r"\bcircular\b",
            r"\bdear\s+sir\b",
            r"\bsir\b",
            r"\bemail\b",
            r"\btelephone\b",
            r"\bphone\b",
            r"\bministry\s+no\.\s*\d+\b",
            r"\bcircular\s+no\.\s*\d+\b",
        ]
        
        # Verb list for substantive clause detection
        self.verb_markers = [
            "shall", "may", "will", "must", "acknowledge", "agree", "agrees",
            "undertake", "undertakes", "require", "requires", "provide", "provides",
            "terminate", "terminates", "pay", "pays", "comply", "complies",
            "entitled", "liable", "obligated", "obliges", "warrant", "warrants"
        ]
        
        # Label/field keywords to reject as clauses
        self.label_markers = [
            "visa no", "visa number", "origin", "telephone", "phone", "email",
            "nationality", "passport", "address", "date of birth", "employee no"
        ]
    
    def extract_structured_clauses(
        self,
        file_path: str,
        document_id: str
    ) -> List[ExtractedClause]:
        """
        Extract clauses using deterministic structure-first approach.
        
        Args:
            file_path: Path to document file
            document_id: Unique document identifier
            
        Returns:
            List of ExtractedClause objects
        """
        file_path = Path(file_path)
        logger.info(f"Starting clause extraction for document_id={sanitize_for_log(document_id)}, file_path={sanitize_for_log(file_path)}")

        # Step 1: Parse + fail-closed validation of pages.
        parsed = self._load_and_validate_pages(file_path)
        if parsed is None:
            return []
        pages, pages_text = parsed

        document_type = self.detect_document_type(pages_text)
        self.last_document_type = document_type

        # Step 2: Classify each page into exactly one section (rule-based; fail-closed on error).
        page_sections = self._classify_pages_into_sections(pages)
        if page_sections is None:
            return []

        # Step 3: Global fail-closed check: if no operative sections exist, emit zero clauses.
        has_operative_section = any(
            page['section'] in self.operative_sections
            for page in page_sections
        )
        if not has_operative_section:
            logger.warning(f"No operative sections detected in document {sanitize_for_log(document_id)}. Fail-closed: returning empty clauses.")
            return []

        logger.info("Operative sections detected. Proceeding with clause extraction.")

        # Step 4: Clause extraction (structure-first traversal of sections).
        all_clauses = self._extract_clauses_from_sections(page_sections, document_id, document_type)

        # Steps 5 & 6: Deduplicate then deterministically order.
        return self._deduplicate_and_order(all_clauses, document_id)

    def _load_and_validate_pages(
        self,
        file_path: Path,
    ) -> Optional[Tuple[List[Page], List[str]]]:
        """
        Parse the document and convert to ``List[Page]`` with verbatim ``lines[]``.

        Returns ``(pages, pages_text)`` or ``None`` to signal a fail-closed
        condition (parse error, no pages, invalid page number, non-textual
        content). Callers must emit zero clauses on ``None``.
        """
        # Step 1: Parse document pages.
        try:
            raw_pages = self.ingestion_service.parser.parse_file(file_path)
        except Exception:
            logger.exception(f"Error parsing file {sanitize_for_log(file_path)}")
            return None

        if not raw_pages:
            logger.warning(f"No pages extracted from {sanitize_for_log(file_path)}")
            return None

        logger.info(f"Parsed {len(raw_pages)} pages from document")

        # Convert parser output to List[Page] with lines[]
        # EXECUTION GUARANTEE: Validate page-level input, fail if invalid
        pages: List[Page] = []
        pages_text: List[str] = []

        for page_data in raw_pages:
            converted = self._to_validated_page(page_data)
            if converted is None:
                return None
            page, text = converted
            pages.append(page)
            pages_text.append(text)

        return pages, pages_text

    def _to_validated_page(self, page_data) -> Optional[Tuple[Page, str]]:
        """
        Convert one parser page tuple to a validated ``(Page, text)`` pair.

        Returns ``None`` (fail-closed) when the page number is invalid or the
        content appears non-textual/encoded.
        """
        if len(page_data) == 3:
            text, page_number, is_ocr = page_data
        else:
            text, page_number = page_data
            is_ocr = False

        # FAIL-CLOSED: Reject invalid page numbers (must be 1-indexed, never 0)
        if page_number is None or page_number <= 0:
            logger.error(f"Invalid page_number detected: {page_number}. Must be 1-indexed. Fail-closed.")
            return None

        # FAIL-CLOSED: Reject binary/encoded content (alpha ratio < 10%)
        _alpha_chars = sum(1 for c in (text or "") if c.isalpha())
        if text and len(text) > 50 and _alpha_chars / len(text) < 0.10:
            logger.error("Input appears to be non-textual/encoded content. Fail-closed.")
            return None

        # Split text into lines (preserve verbatim line structure)
        lines = text.split('\n') if text else []

        page = Page(page_number=page_number, lines=lines, is_ocr=is_ocr)
        logger.debug(f"Page {page_number}: {len(lines)} lines, is_ocr={is_ocr}, text_length={len(text)}")
        return page, (text or "")

    def _classify_pages_into_sections(
        self,
        pages: List[Page],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Classify each page into exactly one section (fail-closed on error).

        Returns a list of ``{'page', 'section'}`` dicts, or ``None`` if any page
        raised during classification (caller must emit zero clauses).
        """
        page_sections: List[Dict[str, Any]] = []
        section_counts: Dict[str, int] = {}

        for page in pages:
            # Use joined text for section classification (pattern matching)
            text = page.get_text()

            try:
                section = self._classify_page_section(text, page.page_number)
            except Exception:
                logger.exception(f"Error classifying section for page {page.page_number}")
                # Fail closed on classification error
                return None

            # Ambiguous pages still allow extraction, but are labeled as unknown.
            if section == DocumentSection.AMBIGUOUS:
                logger.debug(f"Page {page.page_number}: classified as AMBIGUOUS -> UNKNOWN for extraction")
                section = DocumentSection.UNKNOWN

            page_sections.append({'page': page, 'section': section})
            section_counts[section.value] = section_counts.get(section.value, 0) + 1
            logger.debug(f"Page {page.page_number}: classified as section={section.value}")

        logger.info(f"Section classification complete: {section_counts}")
        return page_sections

    def _extract_clauses_from_sections(
        self,
        page_sections: List[Dict[str, Any]],
        document_id: str,
        document_type: DocumentType,
    ) -> List[ExtractedClause]:
        """
        Traverse classified pages and emit clauses (Step 4 of extraction).

        Enforces the contract-body entry gate, skips non-operative sections,
        detects clause starts, and buffers verbatim lines until a boundary.
        """
        all_clauses: List[ExtractedClause] = []
        state = _ClauseBufferState()
        contract_body_active = False

        for i, page_info in enumerate(page_sections):
            section = page_info['section']
            page = page_info['page']
            page_number = page.page_number
            lines = page.lines  # Use lines[] directly from Page object

            # Step 4.1: Contract body entry gate (applies only to contractual_terms).
            proceed, contract_body_active = self._apply_entry_gate(
                section, lines, contract_body_active, state, page_number
            )
            if not proceed:
                continue

            # Step 4.2: Skip non-operative sections (terminate any spanning clause first).
            if section not in self.operative_sections:
                self._flush_clause(state, page_number - 1, all_clauses, document_id, document_type)
                continue

            # Step 4.3: Terminate current clause on operative section boundary change.
            if state.buffer and state.section != section:
                self._flush_clause(state, page_number - 1, all_clauses, document_id, document_type)

            # Step 4.4: Detect clause starts on this page (headings / numbering heuristics).
            clause_starts = self._clause_starts_for_page(page, lines, document_type)
            logger.debug(f"Page {page_number} ({section.value}): detected {len(clause_starts)} clause starts")

            # Step 4.5: Buffer verbatim lines into clauses.
            self._accumulate_page_lines(
                lines, clause_starts, section, page_number, state,
                all_clauses, document_id, document_type,
            )

            # Step 4.6: Finalize clause buffers at end-of-section / end-of-document boundaries.
            if self._is_section_boundary(page_sections, i, section):
                self._flush_clause(state, page_number, all_clauses, document_id, document_type)

        return all_clauses

    def _apply_entry_gate(
        self,
        section: DocumentSection,
        lines: List[str],
        contract_body_active: bool,
        state: _ClauseBufferState,
        page_number: int,
    ) -> Tuple[bool, bool]:
        """
        Apply the contract-body entry gate (Step 4.1).

        Returns ``(proceed, contract_body_active)``. When the gate has not yet
        been crossed for ``contractual_terms`` pages, resets the buffer and
        signals ``proceed=False`` so the caller skips the page.
        """
        if section != DocumentSection.CONTRACTUAL_TERMS or contract_body_active:
            return True, contract_body_active
        if self._meets_contract_entry_gate(lines):
            logger.info(f"Contract body entry gate crossed on page {page_number}.")
            return True, True
        # Do not emit clauses or start buffers until gate is crossed
        state.reset()
        return False, contract_body_active

    def _clause_starts_for_page(
        self,
        page: Page,
        lines: List[str],
        document_type: DocumentType,
    ) -> List[Dict[str, Any]]:
        """Detect clause starts (Step 4.4) with the judgment case-citation fallback."""
        clause_starts = self._detect_clause_starts(lines)
        if (
            not clause_starts
            and document_type == DocumentType.JUDGMENT
            and self.has_case_citation(page.get_text())
        ):
            first_line = next((line.strip() for line in lines if line.strip()), "")
            clause_starts = [{
                "line": 0,
                "heading": first_line,
                "priority": 0,
            }] if first_line else []
        return clause_starts

    def _is_section_boundary(
        self,
        page_sections: List[Dict[str, Any]],
        i: int,
        section: DocumentSection,
    ) -> bool:
        """Return True at end-of-document or when the next page changes section (Step 4.6)."""
        is_last_page = (i == len(page_sections) - 1)
        if is_last_page:
            return True
        next_section = page_sections[i + 1]['section']
        return bool(next_section and next_section != section)

    def _accumulate_page_lines(
        self,
        lines: List[str],
        clause_starts: List[Dict[str, Any]],
        section: DocumentSection,
        page_number: int,
        state: _ClauseBufferState,
        all_clauses: List[ExtractedClause],
        document_id: str,
        document_type: DocumentType,
    ) -> None:
        """Buffer verbatim lines into the current clause (Step 4.5)."""
        starts_by_line = {start['line']: start for start in clause_starts}

        for line_idx, line in enumerate(lines):
            clause_start_info = starts_by_line.get(line_idx)
            if clause_start_info is not None:
                # Step 4.5a: Finalize previous clause before starting new one.
                self._flush_clause(state, page_number, all_clauses, document_id, document_type)

                # Step 4.5b: Start new clause buffer at this heading.
                logger.debug(f"Starting new clause at page {page_number}, line {line_idx}: heading='{clause_start_info['heading']}'")
                state.buffer = {
                    'heading': clause_start_info['heading'],
                    'text_lines': [line],
                    'start_line': line_idx,
                    'clause_number': clause_start_info.get('clause_number'),
                }
                state.section = section
                state.page_start = page_number
            elif state.buffer:
                # Step 4.5c: Continue current clause buffer (may span pages).
                state.buffer['text_lines'].append(line)

    def _flush_clause(
        self,
        state: _ClauseBufferState,
        page_end: int,
        all_clauses: List[ExtractedClause],
        document_id: str,
        document_type: DocumentType,
    ) -> None:
        """Finalize the buffered clause (if any) into ``all_clauses`` and reset state."""
        if state.buffer:
            clause = self._finalize_clause(
                state.buffer,
                state.section,
                state.page_start,
                page_end,
                document_id,
                document_type,
            )
            if clause:
                all_clauses.append(clause)
                logger.debug(f"Finalized clause: {clause.clause_id} (pages {clause.page_start}-{clause.page_end})")
        state.reset()

    def _deduplicate_and_order(
        self,
        all_clauses: List[ExtractedClause],
        document_id: str,
    ) -> List[ExtractedClause]:
        """Deduplicate (Step 5) then apply deterministic ordering (Step 6)."""
        logger.info(f"Before deduplication: {len(all_clauses)} clauses")
        deduplicated = self._deduplicate_clauses(all_clauses)
        logger.info(f"After deduplication: {len(deduplicated)} clauses")

        section_order = {
            DocumentSection.CONTRACTUAL_TERMS.value: 0,
            DocumentSection.ANNEXURES_SCHEDULES.value: 1,
            DocumentSection.JUDICIAL_REASONING.value: 2,
            DocumentSection.STATUTORY_TEXT.value: 3,
            DocumentSection.UNKNOWN.value: 4,
        }
        deduplicated.sort(key=lambda c: (
            c.page_start,
            section_order.get(c.document_section, 999),
            c.clause_heading
        ))

        logger.info(f"Clause extraction complete for {sanitize_for_log(document_id)}: {len(deduplicated)} clauses extracted")
        return deduplicated

    def _classify_page_section(self, text: str, page_number: int) -> DocumentSection:
        """
        Classify a page into exactly one section.
        Returns AMBIGUOUS if classification is uncertain.
        """
        text_lower = text.lower()
        section_scores = {}
        
        # Administrative override: force administrative_material
        if self._is_administrative_page(text_lower):
            logger.debug(f"Page {page_number}: Administrative override matched -> ADMINISTRATIVE_MATERIAL")
            return DocumentSection.ADMINISTRATIVE_MATERIAL
        
        # Score each section based on pattern matches
        for section, patterns in self.section_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            section_scores[section] = score
        
        # Find highest scoring section
        if not section_scores or max(section_scores.values()) == 0:
            # No patterns matched - check if page is too fragmented
            if len(text.strip()) < 50:  # Very short page
                logger.debug(f"Page {page_number}: No patterns matched, very short page -> ADMINISTRATIVE_MATERIAL")
                return DocumentSection.ADMINISTRATIVE_MATERIAL
            # Default to ambiguous if we can't classify
            logger.debug(f"Page {page_number}: No patterns matched -> AMBIGUOUS")
            return DocumentSection.AMBIGUOUS
        
        max_score = max(section_scores.values())
        top_sections = [s for s, score in section_scores.items() if score == max_score]
        
        # If multiple sections tie, mark as ambiguous
        if len(top_sections) > 1:
            logger.debug(f"Page {page_number}: Multiple sections tied (score={max_score}): {[s.value for s in top_sections]} -> AMBIGUOUS")
            return DocumentSection.AMBIGUOUS
        
        result = top_sections[0] if top_sections else DocumentSection.AMBIGUOUS
        
        # Tighten contractual_terms: require party definitions OR sustained modal language
        if result == DocumentSection.CONTRACTUAL_TERMS:
            if not self._has_party_definitions(text_lower) and not self._has_sustained_modal_language(text):
                logger.debug(f"Page {page_number}: contractual_terms missing party definitions/modal language -> AMBIGUOUS")
                return DocumentSection.AMBIGUOUS
        
        logger.debug(f"Page {page_number}: Classified as {result.value} (score={max_score})")
        return result
    
    def _detect_clause_starts(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Detect clause start positions with explicit priority order.
        
        Args:
            lines: List of lines from Page.lines[]
            
        Returns:
            List of dicts with 'line' and 'heading' keys.
        """
        clause_starts = []

        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            matched = self._match_clause_heading(line, line_stripped, line_idx, lines)
            if matched:
                heading, priority, clause_number = matched
                clause_starts.append({
                    'line': line_idx,
                    'heading': heading,
                    'priority': priority,
                    'clause_number': clause_number,
                })

        # Sort by line number (maintain document order)
        clause_starts.sort(key=lambda x: x['line'])

        return self._filter_conflicting_starts(clause_starts)

    def _match_clause_heading(
        self,
        line: str,
        line_stripped: str,
        line_idx: int,
        lines: List[str],
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """
        Match a single line against the clause-heading priority cascade.

        Returns ``(heading, priority, clause_number)`` for the first matching
        rule, or ``None`` if the line is not a clause heading. Rules are tried
        in strict priority order (lower priority number wins).
        """
        return (
            self._match_numbered_heading(line_stripped)
            or self._match_caps_or_title_heading(line_stripped, line_idx, lines)
            or self._match_indent_heading(line, line_idx, lines)
        )

    @staticmethod
    def _match_numbered_heading(
        line_stripped: str,
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """Priorities 0.5–4: numbered / roman / arabic clause headings."""
        # Priority 0.5: Sub-clauses (^\d+\.\d+) — before integer-only numbering
        match = re.match(r'^(\d+\.\d+)\s*(.+)', line_stripped)
        if match:
            return match.group(2).strip(), 1, match.group(1)

        # Priority 1: Explicit numbering (^\d+\.)
        match = re.match(r'^(\d+)\.\s*(.+)', line_stripped)
        if match:
            return match.group(2).strip(), 1, match.group(1)

        # Priority 2: Explicit numbering with parenthesis (^\d+\))
        match = re.match(r'^(\d+)\)\s*(.+)', line_stripped)
        if match:
            return match.group(2).strip(), 2, match.group(1)

        # Priority 3: Roman numerals (^I+\.)
        match = re.match(r'^(I{1,4}|IV|IX|V|VI{0,3}|X{1,3})\.\s*(.+)', line_stripped, re.IGNORECASE)
        if match:
            return match.group(2).strip(), 3, match.group(1).upper()

        # Priority 4: Arabic numerals (^[٠-٩]+)
        match = re.match(r'^([٠-٩]+)[\.\):]\s*(.+)', line_stripped)
        if match:
            return match.group(2).strip(), 4, match.group(1)

        return None

    @staticmethod
    def _match_caps_or_title_heading(
        line_stripped: str,
        line_idx: int,
        lines: List[str],
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """Priority 5 (ALL CAPS) and Priority 6 (Title Case isolated by blank lines)."""
        # Priority 5: ALL CAPS headings
        if line_stripped.isupper() and len(line_stripped) > 5:
            return line_stripped, 5, None

        # Priority 6: Title Case headings (isolated by line breaks)
        if (line_idx == 0 or not lines[line_idx - 1].strip()) and \
           (line_idx == len(lines) - 1 or not lines[line_idx + 1].strip()):
            words = line_stripped.split()
            if words and all(w[0].isupper() for w in words if w):
                return line_stripped, 6, None

        return None

    @staticmethod
    def _match_indent_heading(
        line: str,
        line_idx: int,
        lines: List[str],
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """Priority 7: indentation heuristic (lowest priority)."""
        if (line.startswith('    ') or line.startswith('\t')) and line_idx > 0:
            prev_line = lines[line_idx - 1]
            if not prev_line.startswith('    ') and not prev_line.startswith('\t'):
                return line.strip(), 7, None
        return None

    def _filter_conflicting_starts(
        self,
        clause_starts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop lower-priority starts within 3 lines of a higher-priority one."""
        filtered_starts = []
        for start in clause_starts:
            conflict = any(
                other['priority'] < start['priority']
                and abs(other['line'] - start['line']) <= 3
                for other in clause_starts
            )
            if not conflict:
                filtered_starts.append(start)
        return filtered_starts

    def _meets_contract_entry_gate(self, lines: List[str]) -> bool:
        """Return True if at least two contract entry patterns are present on the page."""
        text_lower = "\n".join(lines).lower()
        hits = 0
        for pattern in self.contract_entry_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                hits += 1
        return hits >= 2

    def _is_administrative_page(self, text_lower: str) -> bool:
        """Detect administrative material overrides (force administrative_material)."""
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.administrative_markers)

    def _has_party_definitions(self, text_lower: str) -> bool:
        """Detect party definitions across Employment, NDA, and MSA document types."""
        employment = "first party" in text_lower and "second party" in text_lower
        msa = any(t in text_lower for t in [
            "service provider", "the client", "the vendor", "licensor", "licensee",
            "purchaser", "the buyer", "the seller", "the supplier",
            "hereinafter referred to as", "hereinafter called",
        ])
        nda = any(t in text_lower for t in [
            "disclosing party", "receiving party", "the recipient",
            "the disclosing party", "the receiving party",
        ])
        return employment or msa or nda

    def _has_sustained_modal_language(self, text: str) -> bool:
        """Require sustained 'shall/may/will' across multiple lines."""
        lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
        modal_lines = 0
        for line in lines:
            if any(modal in line for modal in ["shall", "may", "will"]):
                modal_lines += 1
        return modal_lines >= self.MIN_VERB_LINES_FOR_CONTRACT

    def _is_all_caps(self, text: str) -> bool:
        """Return True if text is all caps (ignoring non-alpha)."""
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    def _is_label_or_field(self, lines: List[str]) -> bool:
        """Detect labels/field names (e.g., 'Visa No', 'Origin', 'Telephone No')."""
        if not lines:
            return True
        first_line = lines[0].strip().lower()
        if any(marker in first_line for marker in self.label_markers):
            return True
        # Label patterns: short noun phrase, optional colon, no verbs
        if len(first_line.split()) <= 5 and re.match(r"^[A-Za-z][A-Za-z\s\-/]*:?$", first_line, re.IGNORECASE):
            if not self._contains_verb(first_line):
                return True
        return False

    def _contains_verb(self, text: str) -> bool:
        """Check for at least one verb marker in text (tolerant to OCR noise)."""
        lower = text.lower()
        # Standard verb markers
        if any(re.search(rf"\b{verb}\b", lower) for verb in self.verb_markers):
            return True
        # OCR-tolerant patterns for 'shall'/'may'/'will'
        if re.search(r"\bsh[a1]ll\b", lower):
            return True
        if re.search(r"\bma[yv]\b", lower):
            return True
        if re.search(r"\bwi[l1]{2}\b", lower):
            return True
        return False

    def _is_substantive_clause(self, verbatim_text: str, lines: List[str]) -> bool:
        """Check substantive clause rules."""
        if self._is_all_caps(verbatim_text):
            return False
        if self._is_label_or_field(lines):
            return False
        if not self._contains_verb(verbatim_text):
            return False
        # Reject very short noun-phrase-only buffers
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if len(non_empty_lines) <= 2 and sum(len(l.split()) for l in non_empty_lines) <= 8:
            return False
        return True
    
    def _finalize_clause(
        self,
        clause_buffer: Dict[str, Any],
        section: DocumentSection,
        page_start: int,
        page_end: int,
        document_id: str,
        document_type: DocumentType
    ) -> Optional[ExtractedClause]:
        """
        Finalize a clause from buffer and create ExtractedClause object.
        
        Hard Condition: Do not emit a clause unless:
        A. A valid clause start was detected (enforced by clause_buffer creation)
        B. At least MIN_CLAUSE_LENGTH characters of verbatim text exist
        
        Returns None if conditions are not met.
        """
        if not clause_buffer or not clause_buffer['text_lines']:
            return None

        # Join text lines (preserve verbatim formatting)
        verbatim_text = '\n'.join(clause_buffer['text_lines']).strip()

        # FAIL-CLOSED: Drop empty clauses
        if verbatim_text == "":
            logger.debug("Empty clause detected - discarding (fail-closed)")
            return None

        # Hard condition B: Minimum character count
        if len(verbatim_text) < self.MIN_CLAUSE_LENGTH:
            logger.debug(f"Clause too short ({len(verbatim_text)} chars < {self.MIN_CLAUSE_LENGTH}) - discarding (fail-closed)")
            return None

        # Heading-only rule: explicit heading cannot be a clause by itself
        raw_heading = (clause_buffer.get('heading') or "").strip()
        if raw_heading and self._is_heading_only(clause_buffer):
            logger.debug("Heading-only buffer detected - discarding clause (fail-closed)")
            return None

        # Derive a readable clause heading from verbatim text.
        heading = self.extract_clause_heading(verbatim_text)

        # Post-extraction section classification (keyword-based, conservative).
        classified_section, section_reason = self.classify_clause_section_with_reason(
            verbatim_text,
            document_type
        )

        # Substantive text gating: must contain verb and not be label/ALL CAPS.
        if self._should_reject_non_substantive(
            verbatim_text, clause_buffer['text_lines'], document_type, section_reason
        ):
            logger.debug("Non-substantive clause text detected - discarding clause (fail-closed)")
            return None

        return self._build_extracted_clause(
            clause_buffer, heading, verbatim_text,
            classified_section, section_reason,
            page_start, page_end, document_id, document_type,
        )

    @staticmethod
    def _is_heading_only(clause_buffer: Dict[str, Any]) -> bool:
        """Return True when the buffer has a heading line but no body content."""
        body_lines = clause_buffer['text_lines'][1:]
        return not any(line.strip() for line in body_lines)

    def _should_reject_non_substantive(
        self,
        verbatim_text: str,
        text_lines: List[str],
        document_type: DocumentType,
        section_reason: str,
    ) -> bool:
        """
        Decide whether to drop a clause for lacking substantive language.

        Exception: legal reasoning in judgments (case-citation pattern) may
        lack obligation language and is kept.
        """
        if self._is_substantive_clause(verbatim_text, text_lines):
            return False
        return not (document_type == DocumentType.JUDGMENT and section_reason == "case_citation_pattern")

    def _build_extracted_clause(
        self,
        clause_buffer: Dict[str, Any],
        heading: str,
        verbatim_text: str,
        classified_section: str,
        section_reason: str,
        page_start: int,
        page_end: int,
        document_id: str,
        document_type: DocumentType,
    ) -> ExtractedClause:
        """Construct the ExtractedClause object and set its derived fields."""
        section_confidence = self.CONFIDENCE_MAP.get(section_reason, "low")

        # Generate deterministic clause_id (heading may be empty string, will be null in output)
        clause_id = self._generate_clause_id(
            document_id,
            classified_section,
            heading or "",  # Use empty string for hash, but output will be null
            verbatim_text
        )

        # Normalization disabled - preserve verbatim text exactly
        normalized_text = self._normalize_text(verbatim_text)

        extracted = ExtractedClause(
            clause_id=clause_id,
            document_section=classified_section,
            page_start=page_start,
            page_end=page_end,
            clause_heading=heading,
            verbatim_text=verbatim_text,
            normalized_text=normalized_text,
            metadata={
                "section_reason": section_reason,
                "section_confidence": section_confidence,
                "document_type": document_type.value,
            }
        )

        # Phase 2: set additional fields
        extracted.clause_number = clause_buffer.get('clause_number')
        extracted.clause_title = heading

        # Definition detection
        if self._detect_definition_block(heading or "", verbatim_text):
            extracted.is_definition = True
            extracted.unit_type = "definition"
            extracted.legal_category = None
        else:
            extracted.unit_type = "clause"
            extracted.legal_category = self.taxonomy_service.classify_legal_category(
                verbatim_text, heading or ""
            )

        return extracted
    
    def _generate_clause_id(
        self,
        document_id: str,
        document_section: str,
        clause_heading: str,
        verbatim_text: str
    ) -> str:
        """
        Generate deterministic clause_id using SHA-256 hash.
        Hash includes: document_id + document_section + clause_heading + verbatim_text
        """
        # Create hash input (no derived fields like page_end or index)
        hash_input = f"{document_id}|{document_section}|{clause_heading}|{verbatim_text}"
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
        
        # Create readable ID
        section_slug = document_section.replace('_', '-')[:10]
        heading_slug = re.sub(r'[^a-zA-Z0-9]', '', clause_heading)[:20]
        
        return f"{section_slug}_{heading_slug}_{hash_hex}"

    def detect_document_type(self, pages: List[str]) -> DocumentType:
        """Detect document type using a deterministic keyword heuristic."""
        sample = " ".join(pages[: self.MAX_DOC_TYPE_SAMPLE_PAGES]).lower()
        
        if "judgment" in sample or "lord" in sample or "court of appeal" in sample:
            return DocumentType.JUDGMENT
        if "this agreement" in sample or ("party" in sample and "agreement" in sample):
            return DocumentType.CONTRACT
        if "an act to" in sample:
            return DocumentType.STATUTE
        return DocumentType.UNKNOWN

    def has_case_citation(self, text: str) -> bool:
        return bool(re.search(r"\bv\b|\[[0-9]{4}\]", text))

    def has_obligation_language(self, text: str) -> bool:
        return bool(re.search(r"\b(shall|must|agree|undertakes)\b", text, re.IGNORECASE))

    def classify_clause_section_with_reason(
        self,
        verbatim_text: str,
        document_type: DocumentType
    ) -> Tuple[str, str]:
        """Classify clause section using keyword presence on verbatim text."""
        text = (verbatim_text or "").lower()
        
        if document_type == DocumentType.JUDGMENT:
            if self.has_case_citation(text) and not self.has_obligation_language(text):
                return "legal_reasoning", "case_citation_pattern"
            return "unknown", "document_type_gate"
        
        if document_type != DocumentType.CONTRACT:
            return "unknown", "document_type_gate"
        
        matches: List[str] = []
        
        for section, keywords in self.SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    matches.append(section)
                    break
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.CONFIDENTIALITY_PATTERNS):
            matches.append("confidentiality")
        
        if len(matches) == 1:
            return matches[0], "keyword_match"
        if len(matches) > 1:
            return "unknown", "multiple_section_matches"
        return "unknown", "no_keywords_found"

    def extract_clause_heading(self, verbatim_text: str) -> str:
        """Derive a readable clause heading from verbatim text."""
        lines = [line.strip() for line in (verbatim_text or "").splitlines() if line.strip()]
        if not lines:
            return "Untitled Clause"
        
        first_line = lines[0]
        first_line = re.sub(r"^\d+\.\s*", "", first_line)
        first_line = re.sub(r"\s+", " ", first_line).strip()
        return first_line[: self.MAX_HEADING_LENGTH]
    
    def _normalize_text(self, verbatim_text: str) -> Optional[str]:
        """
        Generate normalized text (optional, for downstream use only).
        
        ABSOLUTE PROHIBITION: Must NOT modify verbatim text.
        Forbidden: spell correction, hyphen joining, word reconstruction, whitespace cleanup.
        
        Returns None (normalization disabled to preserve verbatim text exactly).
        """
        # STRICT RULE: Do not normalize OCR text - preserve verbatim exactly
        # All normalization is FORBIDDEN per specification:
        # - Hyphen joining: FORBIDDEN
        # - Whitespace cleanup: FORBIDDEN  
        # - Spell correction: FORBIDDEN
        # - Word reconstruction: FORBIDDEN
        
        # Return None to omit normalized_text field
        # Downstream systems should use verbatim_text only
        return None
    
    def _detect_definition_block(self, clause_heading: str, verbatim_text: str) -> bool:
        """
        Detect whether a clause is a definitions block.

        Rules:
        1. heading matches r'^definitions?\\b' (case-insensitive)
        2. verbatim_text contains >= 3 occurrences of ' means ' or ' shall mean '
        3. Quoted-term pattern found >= 2 times
        """
        if re.match(r'^definitions?\b', clause_heading.strip(), re.IGNORECASE):
            return True
        text_lower = verbatim_text.lower()
        means_count = text_lower.count(" means ") + text_lower.count(" shall mean ")
        if means_count >= 3:
            return True
        quoted_pattern = re.compile(
            r'["\u201c]([A-Z][^"\u201d]+)["\u201d]\s+means\s', re.IGNORECASE
        )
        if len(quoted_pattern.findall(verbatim_text)) >= 2:
            return True
        return False

    def extract_defined_terms(self, clauses: List["ExtractedClause"]) -> Dict[str, str]:
        """
        Build a {term.lower(): definition_text} dict from definition clauses.

        Scans only clauses where is_definition=True. Applies two regex patterns
        per line to extract term → definition text pairs.
        """
        terms: Dict[str, str] = {}
        for clause in clauses:
            if not getattr(clause, 'is_definition', False):
                continue
            text = clause.verbatim_text
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Pattern 1: "Term" means definition text.
                m = re.match(
                    r'["\u201c]([A-Z][^"\u201d]+)["\u201d]\s+(?:means|shall mean)\s+(.+)',
                    line, re.IGNORECASE
                )
                if m:
                    term = m.group(1).strip()
                    definition = m.group(2).strip()
                    terms[term.lower()] = definition
                    continue
                # Pattern 2: Term means definition text (unquoted, capitalized)
                m = re.match(
                    r'^([A-Z][A-Za-z\s]+)\s+(?:means|shall mean)\s+(.+)',
                    line
                )
                if m:
                    term = m.group(1).strip()
                    definition = m.group(2).strip()
                    terms[term.lower()] = definition
        return terms

    def _deduplicate_clauses(
        self,
        clauses: List[ExtractedClause]
    ) -> List[ExtractedClause]:
        """
        Deduplicate clauses using exact match on verbatim_text.
        Only deduplicates within same document and same section.
        """
        seen = {}  # (section, verbatim_text) -> clause
        unique_clauses = []
        duplicates_count = 0
        
        for clause in clauses:
            key = (clause.document_section, clause.verbatim_text)
            
            if key not in seen:
                seen[key] = clause
                unique_clauses.append(clause)
            else:
                duplicates_count += 1
                logger.debug(f"Duplicate clause detected: {clause.clause_id} (same as {seen[key].clause_id})")
        
        if duplicates_count > 0:
            logger.info(f"Removed {duplicates_count} duplicate clauses")
        
        return unique_clauses
