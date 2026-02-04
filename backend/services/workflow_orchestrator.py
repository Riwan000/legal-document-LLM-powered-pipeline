"""
Workflow orchestrator.

Coordinates workflow steps, passes WorkflowContext through every step,
and returns context (including partial results) on success or failure.
"""
from __future__ import annotations

import os
import random
import uuid
from typing import Optional

from backend.config import settings
from backend.models.workflow import WorkflowContext, DocumentExplorerRequest, EvidenceExplorerRequest
from backend.services.document_explorer_service import DocumentExplorerService
from backend.services.evidence_explorer_service import EvidenceExplorerService
from backend.services.contract_review_service import ContractReviewService
from backend.services.due_diligence_memo_service import DueDiligenceMemoService
from backend.services.guardrails import GuardrailViolation, WORKFLOW_DISCLAIMER


def _apply_determinism(seed: int) -> None:
    """
    Apply best-effort determinism controls.

    Note:
    - Some sources of nondeterminism (e.g., PYTHONHASHSEED) must be set before process start.
    - Contract Review + Document Explorer are primarily deterministic already (no LLM calls),
      but we still seed common RNGs to keep behavior stable as workflows evolve.
    """
    if isinstance(seed, int):
        random.seed(seed)
        # Best-effort; may be too late if set after interpreter start, but harmless.
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        # numpy may not be installed in minimal deployments
        pass


class WorkflowOrchestrator:
    """
    Runs workflow steps in sequence. Each step accepts and returns WorkflowContext.
    On any failure, returns ctx with ctx.status == "failed" and ctx.error set;
    partial results remain in ctx.intermediate_results.
    """

    def __init__(
        self,
        document_explorer_service: DocumentExplorerService,
        contract_review_service: ContractReviewService,
        due_diligence_memo_service: DueDiligenceMemoService,
        evidence_explorer_service: Optional[EvidenceExplorerService] = None,
    ):
        self.document_explorer_service = document_explorer_service
        self.contract_review_service = contract_review_service
        self.due_diligence_memo_service = due_diligence_memo_service
        self.evidence_explorer_service = evidence_explorer_service

    def run_contract_review(
        self,
        contract_id: str,
        contract_type: str,
        jurisdiction: Optional[str] = None,
        review_depth: Optional[str] = None,
    ) -> WorkflowContext:
        """Execute Contract Review workflow. Returns context (success or failed)."""
        _apply_determinism(settings.CONTRACT_REVIEW_SEED)
        ctx = WorkflowContext(
            workflow_id=uuid.uuid4().hex,
            workflow_type="contract_review",
            document_ids=[contract_id],
            jurisdiction=jurisdiction,
            metadata={
                "contract_type": contract_type,
                "jurisdiction": jurisdiction,
                "review_depth": review_depth or "standard",
                "seed": settings.CONTRACT_REVIEW_SEED,
                "temperature": settings.CONTRACT_REVIEW_TEMPERATURE,
            },
        )
        try:
            ctx = self.contract_review_service.run(ctx)

            # Post-step hygiene: ensure the mandatory disclaimer is non-prescriptive.
            # (The plan forbids "should/must/recommend".)
            if ctx.status == "completed":
                resp_key = "contract_review.response"
                resp = (ctx.intermediate_results or {}).get(resp_key)
                if isinstance(resp, dict):
                    resp["disclaimer"] = WORKFLOW_DISCLAIMER
                    ctx.intermediate_results[resp_key] = resp

            return ctx
        except GuardrailViolation as e:
            ctx.fail(
                code=e.code,
                message=e.message,
                step=e.step,
                details=e.details,
            )
            return ctx
        except Exception as e:
            ctx.fail(
                code="WORKFLOW_ERROR",
                message=str(e),
                step="contract_review",
                details={"exception_type": type(e).__name__},
            )
            return ctx

    def run_document_explorer(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> WorkflowContext:
        """Execute Document Explorer workflow. Returns context (success or failed)."""
        ctx = WorkflowContext(
            workflow_id=uuid.uuid4().hex,
            workflow_type="document_explorer",
            document_ids=[document_id],
            metadata={"query": query, "top_k": top_k, "mode": mode or "text"},
        )
        request = DocumentExplorerRequest(
            document_id=document_id,
            query=query,
            top_k=top_k,
            mode=mode or "text",
        )
        try:
            return self.document_explorer_service.run(ctx, request)
        except Exception as e:
            ctx.fail(
                code="WORKFLOW_ERROR",
                message=str(e),
                step="document_explorer",
                details={"exception_type": type(e).__name__},
            )
            return ctx

    def run_due_diligence_memo(self, document_id: str) -> WorkflowContext:
        """Execute Due Diligence Memo workflow. Returns context (success or failed)."""
        _apply_determinism(settings.CASE_SUMMARY_SEED)
        ctx = WorkflowContext(
            workflow_id=uuid.uuid4().hex,
            workflow_type="due_diligence",
            document_ids=[document_id],
            metadata={
                "seed": settings.CASE_SUMMARY_SEED,
                "temperature": settings.CASE_SUMMARY_TEMPERATURE,
            },
        )
        try:
            return self.due_diligence_memo_service.run(ctx, document_id)
        except GuardrailViolation as e:
            ctx.fail(
                code=e.code,
                message=e.message,
                step=e.step,
                details=e.details,
            )
            ctx.final_output = None
            return ctx
        except Exception as e:
            ctx.fail(
                code="WORKFLOW_ERROR",
                message=str(e),
                step="due_diligence",
                details={"exception_type": type(e).__name__},
            )
            ctx.final_output = None
            return ctx

    def run_evidence_explorer(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> WorkflowContext:
        """Execute Evidence Explorer (evidence-only, no LLM). Returns context."""
        if not self.evidence_explorer_service:
            ctx = WorkflowContext(
                workflow_id=uuid.uuid4().hex,
                workflow_type="document_explorer",
                document_ids=[document_id],
                metadata={"query": query, "top_k": top_k, "mode": mode or "text"},
            )
            ctx.fail(
                code="SERVICE_NOT_AVAILABLE",
                message="Evidence Explorer service not initialized.",
                step="evidence_explorer",
                details={},
            )
            return ctx
        ctx = WorkflowContext(
            workflow_id=uuid.uuid4().hex,
            workflow_type="document_explorer",
            document_ids=[document_id],
            metadata={"query": query, "top_k": top_k, "mode": mode or "text"},
        )
        request = EvidenceExplorerRequest(
            document_id=document_id,
            query=query,
            top_k=top_k,
            mode=(mode or "text") if mode in ("text", "clauses", "both") else "text",
        )
        try:
            return self.evidence_explorer_service.run(ctx, request)
        except Exception as e:
            ctx.fail(
                code="WORKFLOW_ERROR",
                message=str(e),
                step="evidence_explorer",
                details={"exception_type": type(e).__name__},
            )
            return ctx
