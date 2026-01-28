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


class DocumentSection(str, Enum):
    """Document section types."""
    ADMINISTRATIVE_MATERIAL = "administrative_material"
    CONTRACTUAL_TERMS = "contractual_terms"
    JUDICIAL_REASONING = "judicial_reasoning"
    STATUTORY_TEXT = "statutory_text"
    ANNEXURES_SCHEDULES = "annexures_schedules"
    AMBIGUOUS = "ambiguous"  # Page cannot be confidently classified


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
        normalized_text: Optional[str] = None
    ):
        self.clause_id = clause_id
        self.document_section = document_section
        self.page_start = page_start
        self.page_end = page_end
        self.clause_heading = clause_heading
        self.verbatim_text = verbatim_text
        self.normalized_text = normalized_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response (strict JSON schema)."""
        result = {
            "clause_id": self.clause_id,
            "document_section": self.document_section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "clause_heading": self.clause_heading if self.clause_heading else None,  # null if not present
            "verbatim_text": self.verbatim_text
        }
        if self.normalized_text:
            result["normalized_text"] = self.normalized_text
        return result


class StructuredClauseExtractionService:
    """Deterministic service for extracting legal clauses with structure-first approach."""
    
    # Hard condition: Minimum character count for valid clause
    MIN_CLAUSE_LENGTH = 50
    MIN_VERB_LINES_FOR_CONTRACT = 2
    
    def __init__(self):
        """Initialize structured clause extraction service."""
        self.ingestion_service = DocumentIngestionService()
        
        # Section detection patterns (rule-based)
        self.section_patterns = {
            DocumentSection.ADMINISTRATIVE_MATERIAL: [
                r'\b(cover\s+letter|dear\s+sir|dear\s+madam|yours\s+sincerely|regards|signature|date:|dated:)',
                r'\b(page\s+\d+|page\s+of\s+\d+)',
                r'\b(confidential|private|internal\s+use\s+only)',
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
        
        # Operative sections that can produce clauses
        self.operative_sections = {
            DocumentSection.CONTRACTUAL_TERMS,
            DocumentSection.JUDICIAL_REASONING,
            DocumentSection.STATUTORY_TEXT
        }
        
        # Contract body entry gate patterns (must match at least two on the same page)
        self.contract_entry_patterns = [
            r"\bemployment\s+agreement\b",
            r"\b(first|1st)\s+party\b",
            r"\b(second|2nd)\s+party\b",
            r"\bthis\s+agreement\b",
            r"\bthe\s+first\s+party\s+shall\b",
            r"\bthe\s+second\s+party\s+shall\b",
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
            r"@\w+\.\w+",
            r"\bno\.\s*\d+\b"
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
        logger.info(f"Starting clause extraction for document_id={document_id}, file_path={file_path}")
        
        # Step 1: Parse document pages and convert to List[Page]
        try:
            raw_pages = self.ingestion_service.parser.parse_file(file_path)
        except Exception as parse_error:
            logger.error(f"Error parsing file {file_path}: {str(parse_error)}", exc_info=True)
            return []
        
        if not raw_pages:
            logger.warning(f"No pages extracted from {file_path}")
            return []
        
        logger.info(f"Parsed {len(raw_pages)} pages from document")
        
        # Convert parser output to List[Page] with lines[]
        # EXECUTION GUARANTEE: Validate page-level input, fail if invalid
        pages: List[Page] = []
        
        for page_data in raw_pages:
            if len(page_data) == 3:
                text, page_number, is_ocr = page_data
            else:
                text, page_number = page_data
                is_ocr = False
            
            # FAIL-CLOSED: Reject invalid page numbers (must be 1-indexed, never 0)
            if page_number is None or page_number <= 0:
                logger.error(f"Invalid page_number detected: {page_number}. Must be 1-indexed. Fail-closed.")
                return []
            
            # FAIL-CLOSED: Reject chunk-like input (check for embeddings/tokens indicators)
            if text and ('embedding' in text.lower() or 'chunk' in text.lower() or 'token' in text.lower()):
                # This might be chunk metadata, not page text - fail closed
                logger.error("Input appears to be chunks/embeddings, not page-level text. Fail-closed.")
                return []
            
            # Split text into lines (preserve verbatim line structure)
            lines = text.split('\n') if text else []
            
            # Validate we have lines[] structure
            if not isinstance(lines, list):
                logger.error(f"Invalid input: lines is not a list. Fail-closed.")
                return []
            
            pages.append(Page(
                page_number=page_number,
                lines=lines,
                is_ocr=is_ocr
            ))
            logger.debug(f"Page {page_number}: {len(lines)} lines, is_ocr={is_ocr}, text_length={len(text)}")
        
        # Step 2: Classify sections (one section per page, fail-closed per page)
        page_sections = []
        section_counts = {}
        
        for page in pages:
            # Use joined text for section classification (pattern matching)
            text = page.get_text()
            
            try:
                section = self._classify_page_section(text, page.page_number)
            except Exception as section_error:
                logger.error(f"Error classifying section for page {page.page_number}: {str(section_error)}", exc_info=True)
                # Fail closed on classification error
                return []
            
            page_sections.append({
                'page': page,
                'section': section
            })
            section_counts[section.value] = section_counts.get(section.value, 0) + 1
            logger.debug(f"Page {page.page_number}: classified as section={section.value}")
        
        logger.info(f"Section classification complete: {section_counts}")
        
        # Step 3: Check if any operative section exists (global fail-closed check)
        has_operative_section = any(
            page['section'] in self.operative_sections
            for page in page_sections
        )
        
        if not has_operative_section:
            # No operative sections detected - fail closed
            logger.warning(f"No operative sections detected in document {document_id}. Fail-closed: returning empty clauses.")
            return []
        
        logger.info("Operative sections detected. Proceeding with clause extraction.")
        
        # Step 4: Extract clauses from operative sections only
        all_clauses = []
        current_clause_buffer = None
        current_section = None
        current_page_start = None
        contract_body_active = False
        
        for i, page_info in enumerate(page_sections):
            section = page_info['section']
            page = page_info['page']
            page_number = page.page_number
            lines = page.lines  # Use lines[] directly from Page object
            
            # Contract body entry gate (applies only to contractual_terms)
            if section == DocumentSection.CONTRACTUAL_TERMS and not contract_body_active:
                if self._meets_contract_entry_gate(lines):
                    contract_body_active = True
                    logger.info(f"Contract body entry gate crossed on page {page_number}.")
                else:
                    # Do not emit clauses or start buffers until gate is crossed
                    if current_clause_buffer:
                        current_clause_buffer = None
                        current_section = None
                        current_page_start = None
                    continue
            
            # FAIL-CLOSED: Skip ambiguous pages BEFORE any clause operations
            # Do not create clause buffers, do not emit clauses from ambiguous pages
            if section == DocumentSection.AMBIGUOUS:
                logger.debug(f"Page {page_number}: AMBIGUOUS section - skipping (fail-closed)")
                # Terminate any clause that was spanning into this ambiguous page
                if current_clause_buffer:
                    logger.debug(f"Terminating clause buffer spanning into ambiguous page {page_number}")
                    clause = self._finalize_clause(
                        current_clause_buffer,
                        current_section,
                        current_page_start,
                        page_number - 1,
                        document_id
                    )
                    if clause:
                        all_clauses.append(clause)
                        logger.debug(f"Finalized clause from previous page: {clause.clause_id}")
                    current_clause_buffer = None
                    current_section = None
                # Skip this page entirely - do not process for clause extraction
                continue
            
            # Skip non-operative sections (but not ambiguous - already handled above)
            if section not in self.operative_sections:
                # Terminate any clause that was spanning into this non-operative section
                if current_clause_buffer:
                    clause = self._finalize_clause(
                        current_clause_buffer,
                        current_section,
                        current_page_start,
                        page_number - 1,
                        document_id
                    )
                    if clause:
                        all_clauses.append(clause)
                    current_clause_buffer = None
                    current_section = None
                continue
            
            # Check if we need to terminate current clause due to section boundary change
            if current_clause_buffer and current_section != section:
                # Section changed (but both are operative) - finalize current clause
                clause = self._finalize_clause(
                    current_clause_buffer,
                    current_section,
                    current_page_start,
                    page_number - 1,
                    document_id
                )
                if clause:
                    all_clauses.append(clause)
                current_clause_buffer = None
                current_section = None
            
            # Extract clause boundaries from this page's lines
            clause_starts = self._detect_clause_starts(lines)
            logger.debug(f"Page {page_number} ({section.value}): detected {len(clause_starts)} clause starts")
            
            # Process lines directly from Page object
            for line_idx, line in enumerate(lines):
                # Check if this line starts a new clause
                is_clause_start = any(
                    start['line'] == line_idx for start in clause_starts
                )
                
                if is_clause_start:
                    # Finalize previous clause if exists
                    if current_clause_buffer:
                        logger.debug(f"Finalizing previous clause before starting new one at page {page_number}, line {line_idx}")
                        clause = self._finalize_clause(
                            current_clause_buffer,
                            current_section,
                            current_page_start,
                            page_number,
                            document_id
                        )
                        if clause:
                            all_clauses.append(clause)
                            logger.debug(f"Finalized clause: {clause.clause_id} (pages {clause.page_start}-{clause.page_end})")
                    
                    # Start new clause
                    clause_start_info = next(
                        s for s in clause_starts if s['line'] == line_idx
                    )
                    logger.debug(f"Starting new clause at page {page_number}, line {line_idx}: heading='{clause_start_info['heading']}'")
                    current_clause_buffer = {
                        'heading': clause_start_info['heading'],
                        'text_lines': [line],
                        'start_line': line_idx
                    }
                    current_section = section
                    current_page_start = page_number
                elif current_clause_buffer:
                    # Continue current clause (may span pages if section is same)
                    current_clause_buffer['text_lines'].append(line)
                elif not current_clause_buffer and section in self.operative_sections:
                    # No current clause but we're in an operative section
                    # This might be continuation text - check if next page continues
                    # For now, we'll only start clauses on explicit starts
                    pass
            
            # Check if we're at document end or section boundary
            is_last_page = (i == len(page_sections) - 1)
            next_section = page_sections[i + 1]['section'] if not is_last_page else None
            
            if is_last_page or (next_section and next_section != section):
                # Finalize current clause at end of section
                if current_clause_buffer:
                    logger.debug(f"Finalizing clause at end of section (page {page_number}, is_last={is_last_page})")
                    clause = self._finalize_clause(
                        current_clause_buffer,
                        current_section,
                        current_page_start,
                        page_number,
                        document_id
                    )
                    if clause:
                        all_clauses.append(clause)
                        logger.debug(f"Finalized clause: {clause.clause_id} (pages {clause.page_start}-{clause.page_end})")
                    current_clause_buffer = None
                    current_section = None
        
        # Step 5: Deduplicate clauses (exact match on verbatim_text, same section)
        logger.info(f"Before deduplication: {len(all_clauses)} clauses")
        deduplicated = self._deduplicate_clauses(all_clauses)
        logger.info(f"After deduplication: {len(deduplicated)} clauses")
        
        # Step 6: Sort by page_start, section order, appearance
        section_order = {
            DocumentSection.CONTRACTUAL_TERMS.value: 0,
            DocumentSection.JUDICIAL_REASONING.value: 1,
            DocumentSection.STATUTORY_TEXT.value: 2,
        }
        deduplicated.sort(key=lambda c: (
            c.page_start,
            section_order.get(c.document_section, 999),
            c.clause_heading
        ))
        
        logger.info(f"Clause extraction complete for {document_id}: {len(deduplicated)} clauses extracted")
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
            
            heading = None
            priority = None
            
            # Priority 1: Explicit numbering (^\\d+\\.)
            match = re.match(r'^(\d+)\.\s*(.+)', line_stripped)
            if match:
                heading = match.group(2).strip()
                priority = 1
            
            # Priority 2: Explicit numbering with parenthesis (^\\d+\\))
            if not heading:
                match = re.match(r'^(\d+)\)\s*(.+)', line_stripped)
                if match:
                    heading = match.group(2).strip()
                    priority = 2
            
            # Priority 3: Roman numerals (^I+\\.)
            if not heading:
                match = re.match(r'^(I{1,4}|IV|IX|V|VI{0,3}|X{1,3})\.\s*(.+)', line_stripped, re.IGNORECASE)
                if match:
                    heading = match.group(2).strip()
                    priority = 3
            
            # Priority 4: Arabic numerals (^[٠-٩]+)
            if not heading:
                match = re.match(r'^([٠-٩]+)[\.\)]\s*(.+)', line_stripped)
                if match:
                    heading = match.group(2).strip()
                    priority = 4
            
            # Priority 5: ALL CAPS headings
            if not heading and line_stripped.isupper() and len(line_stripped) > 5:
                heading = line_stripped
                priority = 5
            
            # Priority 6: Title Case headings (isolated by line breaks)
            if not heading:
                # Check if line is Title Case and isolated
                if (line_idx == 0 or not lines[line_idx - 1].strip()) and \
                   (line_idx == len(lines) - 1 or not lines[line_idx + 1].strip()):
                    words = line_stripped.split()
                    if words and all(w[0].isupper() for w in words if w):
                        heading = line_stripped
                        priority = 6
            
            # Priority 7: Indentation heuristics (lowest priority)
            if not heading:
                # Check for significant indentation (at least 4 spaces or tab)
                if line.startswith('    ') or line.startswith('\t'):
                    # Check if previous line was less indented
                    if line_idx > 0:
                        prev_line = lines[line_idx - 1]
                        if not prev_line.startswith('    ') and not prev_line.startswith('\t'):
                            heading = line_stripped
                            priority = 7
            
            if heading:
                clause_starts.append({
                    'line': line_idx,
                    'heading': heading,
                    'priority': priority
                })
        
        # Sort by line number (maintain document order)
        clause_starts.sort(key=lambda x: x['line'])
        
        # Remove lower-priority starts that conflict with higher-priority ones
        filtered_starts = []
        for start in clause_starts:
            # Check if there's a higher-priority start nearby (within 3 lines)
            conflict = False
            for other in clause_starts:
                if other['priority'] < start['priority'] and \
                   abs(other['line'] - start['line']) <= 3:
                    conflict = True
                    break
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
        """Detect party definitions in a page."""
        return ("first party" in text_lower and "second party" in text_lower)

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
        if any(re.search(rf"\\b{verb}\\b", lower) for verb in self.verb_markers):
            return True
        # OCR-tolerant patterns for 'shall'/'may'/'will'
        if re.search(r"\\bsh[a1]ll\\b", lower):
            return True
        if re.search(r"\\bma[yv]\\b", lower):
            return True
        if re.search(r"\\bwi[l1]{2}\\b", lower):
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
        document_id: str
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
            logger.debug(f"Empty clause detected - discarding (fail-closed)")
            return None
        
        # Hard condition B: Minimum character count
        if len(verbatim_text) < self.MIN_CLAUSE_LENGTH:
            logger.debug(f"Clause too short ({len(verbatim_text)} chars < {self.MIN_CLAUSE_LENGTH}) - discarding (fail-closed)")
            return None
        
        # Clause heading may be absent; keep as None in output
        heading = clause_buffer.get('heading')
        if heading:
            heading = heading.strip()
        
        # Heading-only rule: heading cannot be a clause by itself
        if heading:
            body_lines = clause_buffer['text_lines'][1:]
            has_body = any(line.strip() for line in body_lines)
            if not has_body:
                logger.debug("Heading-only buffer detected - discarding clause (fail-closed)")
                return None
        
        # Substantive text gating: must contain verb and not be label/ALL CAPS
        if not self._is_substantive_clause(verbatim_text, clause_buffer['text_lines']):
            logger.debug("Non-substantive clause text detected - discarding clause (fail-closed)")
            return None
        
        # Generate deterministic clause_id (heading may be empty string, will be null in output)
        clause_id = self._generate_clause_id(
            document_id,
            section.value,
            heading or "",  # Use empty string for hash, but output will be null
            verbatim_text
        )
        
        # Normalization disabled - preserve verbatim text exactly
        normalized_text = self._normalize_text(verbatim_text)
        
        return ExtractedClause(
            clause_id=clause_id,
            document_section=section.value,
            page_start=page_start,
            page_end=page_end,
            clause_heading=heading,
            verbatim_text=verbatim_text,
            normalized_text=normalized_text
        )
    
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
