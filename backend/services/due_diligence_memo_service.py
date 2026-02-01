"""
Due Diligence Memo workflow service (v0.2).

Provides:
- Document readiness + document_type classification
- Section-level generation with independent failures/skips
- Guardrails + determinism-friendly outputs
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple

from backend.config import settings
from backend.models.case_summary import (
    CaseSpine,
    KeyArguments,
    CitationMetadata,
)
from backend.models.workflow import WorkflowContext
from backend.services.case_chunk_classifier import CaseChunkClassifier
from backend.services.case_spine_builder import CaseSpineBuilder
from backend.services.case_section_summarizers import CaseSectionSummarizers
from backend.services.guardrails import GuardrailViolation, enforce_non_prescriptive_language


STEP_NAME = "due_diligence"


class DueDiligenceMemoService:
    """Orchestrates Due Diligence Memo workflow with explicit status semantics."""

    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.chunk_classifier = CaseChunkClassifier()
        self.spine_builder = CaseSpineBuilder()
        self.section_summarizers = CaseSectionSummarizers()

    def run(self, ctx: WorkflowContext, document_id: str) -> WorkflowContext:
        """Run Due Diligence Memo workflow and populate WorkflowContext."""
        if ctx.status == "failed":
            return ctx

        if not document_id:
            ctx.fail(
                code="MISSING_INPUT",
                message="Due Diligence Memo requires a document_id.",
                step=STEP_NAME,
                details={},
            )
            ctx.final_output = None
            return ctx

        # Layer 1: readiness and document type
        ctx.current_step = f"{STEP_NAME}.readiness"
        chunks = self._get_document_chunks(document_id)
        readiness = self._classify_readiness(chunks)
        ctx.add_result(
            f"{STEP_NAME}.readiness",
            {"status": "completed", "output": readiness, "error": None, "warnings": []},
        )

        allowed_mode = readiness["allowed_mode"]
        if allowed_mode == "fail":
            ctx.fail(
                code="DOCUMENT_NOT_READY",
                message="Document is not suitable for Due Diligence Memo generation.",
                step=STEP_NAME,
                details=readiness,
            )
            self._clear_section_outputs(ctx)
            ctx.final_output = None
            return ctx

        # Layer 2: section orchestration
        section_results: Dict[str, Dict[str, Any]] = {}
        mandatory_sections = ["case_spine", "executive_summary", "timeline"]

        try:
            case_spine_result = self._run_case_spine(document_id, allowed_mode)
            section_results["case_spine"] = case_spine_result
            ctx.add_result(f"{STEP_NAME}.case_spine", case_spine_result)

            # Use minimal spine in partial mode when case spine failed
            case_spine = case_spine_result.get("output") if case_spine_result["status"] == "completed" else None
            if case_spine is None:
                case_spine = CaseSpine(
                    case_name="",
                    court="",
                    date="",
                    parties=[],
                    procedural_posture="",
                    core_issues=[],
                )

            exec_result = self._run_executive_summary(document_id, case_spine, allowed_mode)
            section_results["executive_summary"] = exec_result
            ctx.add_result(f"{STEP_NAME}.executive_summary", exec_result)

            timeline_result = self._run_timeline(document_id, case_spine, allowed_mode)
            section_results["timeline"] = timeline_result
            ctx.add_result(f"{STEP_NAME}.timeline", timeline_result)

            args_result = self._run_key_arguments(document_id, case_spine)
            section_results["key_arguments"] = args_result
            ctx.add_result(f"{STEP_NAME}.key_arguments", args_result)

            issues_result = self._run_open_issues(document_id, case_spine)
            section_results["open_issues"] = issues_result
            ctx.add_result(f"{STEP_NAME}.open_issues", issues_result)

            # Collect citations from completed sections
            citations = self._collect_citations_from_sections(section_results)
            section_results["citations"] = {
                "status": "completed",
                "output": citations,
                "error": None,
                "warnings": [],
            }
            ctx.add_result(f"{STEP_NAME}.citations", section_results["citations"])

        except GuardrailViolation as e:
            ctx.fail(
                code=e.code,
                message=e.message,
                step=e.step,
                details=e.details,
            )
            self._clear_section_outputs(ctx)
            ctx.final_output = None
            return ctx

        # Layer 3: status semantics + final output
        ctx.status = self._derive_status(allowed_mode, section_results, mandatory_sections)

        if ctx.status == "failed":
            ctx.final_output = None
            self._clear_section_outputs(ctx)
            return ctx

        ctx.final_output = self._build_final_output(section_results)
        return ctx

    # --------------------------
    # Readiness + classification
    # --------------------------
    def _get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        if hasattr(self.rag_service, "vector_store"):
            return self.rag_service.vector_store.get_chunks_by_document(document_id) or []
        return []

    def _classify_readiness(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = "\n".join([(c.get("text") or "") for c in chunks]).lower()
        has_background = self._has_any(text, ["background", "facts", "the parties", "this case involves", "the dispute"])
        has_issue_framing = self._has_any(text, ["issue", "question", "whether", "central question", "key issue"])
        has_party_positions = self._has_any(text, ["argues", "contends", "submits", "claims", "maintains", "position"])
        has_dates = bool(re.search(r"\b(19|20)\d{2}\b", text)) or bool(
            re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        )

        document_type = self._classify_document_type(text, has_background, has_issue_framing, has_party_positions)
        allowed_mode = self._allowed_mode_for_type(document_type, has_background, has_issue_framing, has_party_positions)

        return {
            "document_type": document_type,
            "has_background": has_background,
            "has_issue_framing": has_issue_framing,
            "has_party_positions": has_party_positions,
            "has_dates": has_dates,
            "allowed_mode": allowed_mode,
        }

    def _classify_document_type(
        self,
        text: str,
        has_background: bool,
        has_issue_framing: bool,
        has_party_positions: bool,
    ) -> str:
        if self._has_any(text, ["procedural order", "scheduling order", "case management order"]):
            return "procedural_order"
        if self._has_any(text, ["request for arbitration", "notice of arbitration"]):
            return "arbitration_request"
        if self._has_any(text, ["judgment", "the court held", "the court finds", "the court concludes", "opinion"]):
            return "court_judgment"
        if self._has_any(text, ["statement of facts"]):
            return "statement_of_facts"
        if self._has_any(text, ["agreement", "contract", "party hereto", "terms and conditions"]):
            return "contract"
        if self._has_any(text, ["law review", "journal article", "abstract", "doi"]):
            return "legal_article"
        if has_background and has_issue_framing and has_party_positions:
            return "case_bundle"
        return "unknown"

    def _allowed_mode_for_type(
        self,
        document_type: str,
        has_background: bool,
        has_issue_framing: bool,
        has_party_positions: bool,
    ) -> str:
        if document_type in {"procedural_order", "contract", "legal_article"}:
            return "fail"
        if document_type in {"court_judgment", "arbitration_request"}:
            return "partial"
        if document_type in {"case_bundle", "statement_of_facts"}:
            return "full"
        # Ambiguous documents: default to fail or partial, never full
        if has_background or has_issue_framing or has_party_positions:
            return "partial"
        return "fail"

    def _has_any(self, text: str, terms: List[str]) -> bool:
        return any(term in text for term in terms)

    # --------------------------
    # Section runners
    # --------------------------
    def _run_case_spine(self, document_id: str, allowed_mode: str) -> Dict[str, Any]:
        chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["background", "holding", "issue_framing"],
            top_k=settings.CASE_SUMMARY_SPINE_MAX_CHUNKS,
            query="case name court parties procedural posture core legal issues",
        )
        if not chunks or not self._has_any_required_types(chunks, ["background", "issue_framing"]):
            return self._section_failure(
                "CASE_SPINE_FAILED",
                "Insufficient evidence to establish factual background",
                required_types=["background", "holding", "issue_framing"],
                allowed_mode=allowed_mode,
                mandatory=True,
                force_status="failed",
            )
        try:
            spine_by_type = self._group_chunks_by_type(chunks)
            spine = self.spine_builder.build_case_spine(
                background_chunks=spine_by_type.get("background", []),
                holding_chunks=spine_by_type.get("holding", []),
                issue_chunks=spine_by_type.get("issue_framing", []),
            )
            # Guardrails on generated fields
            spine.case_name = enforce_non_prescriptive_language(spine.case_name, step=f"{STEP_NAME}.case_spine.case_name")
            spine.court = enforce_non_prescriptive_language(spine.court, step=f"{STEP_NAME}.case_spine.court")
            spine.date = enforce_non_prescriptive_language(spine.date, step=f"{STEP_NAME}.case_spine.date")
            spine.procedural_posture = enforce_non_prescriptive_language(
                spine.procedural_posture, step=f"{STEP_NAME}.case_spine.procedural_posture"
            )
            spine.parties = [enforce_non_prescriptive_language(p, step=f"{STEP_NAME}.case_spine.parties") for p in (spine.parties or [])]
            spine.core_issues = [
                enforce_non_prescriptive_language(i, step=f"{STEP_NAME}.case_spine.core_issues") for i in (spine.core_issues or [])
            ]
            return {"status": "completed", "output": spine, "error": None, "warnings": []}
        except GuardrailViolation:
            raise
        except Exception as e:
            return self._section_failure(
                "CASE_SPINE_FAILED",
                str(e),
                required_types=["background", "holding", "issue_framing"],
                allowed_mode=allowed_mode,
                mandatory=True,
                force_status="failed",
            )

    def _run_executive_summary(self, document_id: str, case_spine: CaseSpine, allowed_mode: str) -> Dict[str, Any]:
        chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["background", "holding"],
            top_k=settings.CASE_SUMMARY_EXEC_MAX_CHUNKS,
            query="case background facts parties dispute court decision holding",
            case_spine=case_spine,
        )
        if not chunks:
            return self._section_failure(
                "INSUFFICIENT_CONTEXT",
                "Insufficient background or holding evidence for executive summary.",
                required_types=["background", "holding"],
                allowed_mode=allowed_mode,
                mandatory=True,
            )
        if not self._has_any_required_types(chunks, ["background"]):
            return self._section_failure(
                "INSUFFICIENT_CONTEXT",
                "Executive summary requires background evidence.",
                required_types=["background"],
                allowed_mode=allowed_mode,
                mandatory=True,
            )
        items = self.section_summarizers.generate_executive_summary(case_spine, chunks)
        for item in items:
            item.text = enforce_non_prescriptive_language(item.text, step=f"{STEP_NAME}.executive_summary")
        items = self._filter_items_by_chunk_ids(items, chunks)
        return {"status": "completed", "output": items, "error": None, "warnings": []}

    def _run_timeline(self, document_id: str, case_spine: CaseSpine, allowed_mode: str) -> Dict[str, Any]:
        chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["procedural_history"],
            top_k=settings.CASE_SUMMARY_TIMELINE_MAX_CHUNKS,
            query="procedural history dates events filings orders hearings timeline chronology",
            case_spine=case_spine,
        )
        if not chunks:
            return self._section_failure(
                "INSUFFICIENT_CONTEXT",
                "No procedural history found for timeline generation.",
                required_types=["procedural_history"],
                allowed_mode=allowed_mode,
                mandatory=True,
            )
        events = self.section_summarizers.generate_timeline(case_spine, chunks)
        for ev in events:
            ev.date = enforce_non_prescriptive_language(ev.date, step=f"{STEP_NAME}.timeline.date")
            ev.event = enforce_non_prescriptive_language(ev.event, step=f"{STEP_NAME}.timeline.event")
        events = self._filter_items_by_chunk_ids(events, chunks)
        return {"status": "completed", "output": events, "error": None, "warnings": []}

    def _run_key_arguments(self, document_id: str, case_spine: CaseSpine) -> Dict[str, Any]:
        claimant_chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["argument_claimant"],
            top_k=settings.CASE_SUMMARY_ARGUMENTS_MAX_CHUNKS,
            query="plaintiff claimant appellant arguments contentions claims position",
            case_spine=case_spine,
        )
        defendant_chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["argument_defendant"],
            top_k=settings.CASE_SUMMARY_ARGUMENTS_MAX_CHUNKS,
            query="defendant respondent appellee arguments contentions defense position",
            case_spine=case_spine,
        )
        claimant_args = self.section_summarizers.generate_claimant_arguments(case_spine, claimant_chunks) if claimant_chunks else []
        defendant_args = self.section_summarizers.generate_defendant_arguments(case_spine, defendant_chunks) if defendant_chunks else []
        for arg in claimant_args:
            arg.text = enforce_non_prescriptive_language(arg.text, step=f"{STEP_NAME}.arguments.claimant")
        for arg in defendant_args:
            arg.text = enforce_non_prescriptive_language(arg.text, step=f"{STEP_NAME}.arguments.defendant")
        claimant_args = self._filter_items_by_chunk_ids(claimant_args, claimant_chunks)
        defendant_args = self._filter_items_by_chunk_ids(defendant_args, defendant_chunks)
        key_args = KeyArguments(claimant=claimant_args, defendant=defendant_args)
        return {"status": "completed", "output": key_args, "error": None, "warnings": []}

    def _run_open_issues(self, document_id: str, case_spine: CaseSpine) -> Dict[str, Any]:
        chunks = self._select_chunks_for_section(
            document_id=document_id,
            chunk_types=["issue_framing"],
            top_k=settings.CASE_SUMMARY_ISSUES_MAX_CHUNKS,
            query="unresolved issues questions pending adjudication open issues",
            case_spine=case_spine,
        )
        if not chunks:
            return {"status": "completed", "output": [], "error": None, "warnings": []}
        issues = self.section_summarizers.generate_open_issues(case_spine, chunks)
        for issue in issues:
            issue.text = enforce_non_prescriptive_language(issue.text, step=f"{STEP_NAME}.open_issues")
        issues = self._filter_items_by_chunk_ids(issues, chunks)
        return {"status": "completed", "output": issues, "error": None, "warnings": []}

    # --------------------------
    # Helpers
    # --------------------------
    def _select_chunks_for_section(
        self,
        *,
        document_id: str,
        chunk_types: List[str],
        top_k: int,
        query: str,
        case_spine: Optional[CaseSpine] = None,
    ) -> List[Dict[str, Any]]:
        enhanced_query = query
        if case_spine:
            enhanced_query = f"{query} {case_spine.case_name} {', '.join(case_spine.parties)}"
        search_top_k = min(top_k * 2, settings.CASE_SUMMARY_MAX_CHUNKS_PER_CALL * 2)
        try:
            search_results = self.rag_service.search(
                query=enhanced_query,
                top_k=search_top_k,
                document_id_filter=document_id,
            )
        except Exception:
            search_results = []
        if not search_results:
            return []

        classified = self._backfill_and_classify_chunks(search_results, document_id)
        filtered = [c for c in classified if c.get("chunk_type") in chunk_types]
        filtered.sort(key=lambda c: c.get("score", 0), reverse=True)
        return filtered[: min(top_k, settings.CASE_SUMMARY_MAX_CHUNKS_PER_CALL)]

    def _backfill_and_classify_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
        # Reuse deterministic classifier with backfill for chunk_id/chunk_type
        for i, chunk in enumerate(chunks):
            if not chunk.get("chunk_id"):
                page = chunk.get("page_number", 0)
                chunk_idx = chunk.get("chunk_index", i)
                chunk["chunk_id"] = f"c_p{page:04d}_i{chunk_idx:04d}"
            chunk["document_id"] = chunk.get("document_id", document_id)
        return self.chunk_classifier.classify_chunks_batch(chunks)

    def _group_chunks_by_type(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.get("chunk_type", "background"), []).append(chunk)
        return grouped

    def _has_any_required_types(self, chunks: List[Dict[str, Any]], required: List[str]) -> bool:
        available = {c.get("chunk_type") for c in chunks}
        return any(r in available for r in required)

    def _section_failure(
        self,
        code: str,
        message: str,
        *,
        required_types: List[str],
        allowed_mode: str,
        mandatory: bool,
        force_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        if force_status:
            status = force_status
        else:
            status = "failed" if (mandatory and allowed_mode == "full") else "skipped"
        return {
            "status": status,
            "output": None,
            "error": {
                "code": code,
                "message": message,
                "details": {"required_types": required_types},
            },
            "warnings": [] if status == "failed" else [message],
        }

    def _filter_items_by_chunk_ids(self, items: List[Any], chunks: List[Dict[str, Any]]) -> List[Any]:
        chunk_ids = {c.get("chunk_id") for c in chunks}
        filtered = []
        for item in items or []:
            source = getattr(item, "source", None)
            if source and getattr(source, "chunk_id", None) in chunk_ids:
                filtered.append(item)
        return filtered

    def _collect_citations_from_sections(self, section_results: Dict[str, Dict[str, Any]]) -> List[CitationMetadata]:
        citations: List[CitationMetadata] = []
        seen: set = set()
        for key, section in section_results.items():
            if section.get("status") != "completed":
                continue
            output = section.get("output")
            if output is None:
                continue
            items = output
            if isinstance(output, KeyArguments):
                items = output.claimant + output.defendant
            for item in items:
                source = getattr(item, "source", None)
                if not source:
                    continue
                citation_key = (source.document, source.page, source.chunk_id)
                if citation_key in seen:
                    continue
                seen.add(citation_key)
                citations.append(
                    CitationMetadata(
                        document=source.document,
                        page=source.page,
                        chunk_id=source.chunk_id,
                        chunk_type="unknown",
                    )
                )
        return citations

    def _derive_status(
        self,
        allowed_mode: str,
        section_results: Dict[str, Dict[str, Any]],
        mandatory_sections: List[str],
    ) -> str:
        if allowed_mode == "fail":
            return "failed"
        if allowed_mode == "full":
            for section_name in mandatory_sections:
                section = section_results.get(section_name, {})
                if section.get("status") != "completed":
                    return "failed"
            return "completed"

        # partial mode
        completed = [s for s in section_results.values() if s.get("status") == "completed"]
        failed_or_skipped = [s for s in section_results.values() if s.get("status") in {"failed", "skipped"}]
        if not completed:
            return "failed"
        if failed_or_skipped:
            return "completed_with_warnings"
        return "completed"

    def _build_final_output(self, section_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        output = {}
        for key, section in section_results.items():
            if section.get("status") != "completed":
                output[key] = None
                continue
            value = section.get("output")
            if hasattr(value, "model_dump"):
                output[key] = value.model_dump()
            elif isinstance(value, list):
                output[key] = [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
            else:
                output[key] = value
        return output

    def _clear_section_outputs(self, ctx: WorkflowContext) -> None:
        for key, value in list((ctx.intermediate_results or {}).items()):
            if not key.startswith(f"{STEP_NAME}."):
                continue
            if key == f"{STEP_NAME}.readiness":
                continue
            if isinstance(value, dict):
                value = dict(value)
                value["output"] = None
                ctx.intermediate_results[key] = value
