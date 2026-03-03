"""
Streamlit frontend for Legal Document Intelligence MVP.
Multi-page application connecting to FastAPI backend.
"""
import os
import re as _re
import pandas as _pd
import streamlit as st
import requests
from typing import Dict, Any, Optional
import json

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API base URL (FastAPI). Streamlit acts as a server-side proxy:
# all API calls are made from the Streamlit server, so users only hit the UI port.
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# UI styling
# -----------------------------------------------------------------------------
# Custom CSS for consistent headings/citations.
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .citation {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Backend API helpers
# -----------------------------------------------------------------------------
# These functions wrap HTTP calls to the FastAPI backend and centralize timeouts
# and error handling so the page code stays readable.

def check_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_document(file, document_type: str = "document", display_name: Optional[str] = None) -> Optional[Dict]:
    """Upload document to backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"document_type": document_type}
        if display_name:
            data["display_name"] = display_name
        response = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=files,
            data=data,
            timeout=300  # 5 minutes for large files
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None


def search_documents(query: str, top_k: int = 5, document_id: Optional[str] = None, generate_response: bool = True, response_language: Optional[str] = None) -> Optional[Dict]:
    """Search documents using RAG."""
    try:
        data = {
            "query": query,
            "top_k": top_k,
            "generate_response": generate_response
        }
        if document_id:
            data["document_id"] = document_id
        if response_language:
            data["response_language"] = response_language
        
        response = requests.post(f"{API_BASE_URL}/api/search", data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return None


def extract_clauses(document_id: str) -> Optional[Dict]:
    """Extract clauses from contract."""
    try:
        data = {"document_id": document_id}
        # Increased timeout to 5 minutes (300 seconds) for large documents
        response = requests.post(f"{API_BASE_URL}/api/extract-clauses", data=data, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Clause extraction timed out. The document may be too large. Please try with a smaller document or contact support.")
        return None
    except Exception as e:
        st.error(f"Error extracting clauses: {str(e)}")
        return None


def compare_contracts(contract_id: str, template_id: str) -> Optional[Dict]:
    """Compare contract against template."""
    try:
        data = {
            "contract_id": contract_id,
            "template_id": template_id
        }
        response = requests.post(f"{API_BASE_URL}/api/compare", data=data, timeout=180)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error comparing contracts: {str(e)}")
        return None


def summarize_case_file(document_id: str, top_k: int = 10) -> Optional[Dict]:
    """Summarize case file."""
    try:
        data = {
            "document_id": document_id,
            "top_k": top_k
        }
        response = requests.post(f"{API_BASE_URL}/api/summarize", data=data, timeout=180)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error summarizing: {str(e)}")
        return None


def summarize_case_file_stream(document_id: str) -> Optional[Any]:
    """
    Stream case summary via SSE.

    Yields (event_name, payload_dict).
    Returns None if the request fails.
    """
    try:
        data = {"document_id": document_id}
        with requests.post(
            f"{API_BASE_URL}/api/summarize/stream",
            data=data,
            stream=True,
            timeout=180
        ) as resp:
            resp.raise_for_status()

            event_name = None
            data_lines = []

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if not line:
                    # Dispatch event
                    if data_lines:
                        data_str = "\n".join(data_lines)
                        try:
                            payload = json.loads(data_str) if data_str else {}
                        except Exception:
                            payload = {"raw": data_str}
                        yield (event_name or "message", payload)
                    event_name = None
                    data_lines = []
                    continue

                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                # Ignore other SSE fields (id:, retry:, etc.)

            # Flush if stream ended without blank line
            if data_lines:
                data_str = "\n".join(data_lines)
                try:
                    payload = json.loads(data_str) if data_str else {}
                except Exception:
                    payload = {"raw": data_str}
                yield (event_name or "message", payload)

    except Exception as e:
        st.error(f"Error streaming summary: {str(e)}")
        return None


def due_diligence_memo(document_id: str) -> Optional[Dict]:
    """Due Diligence Memo workflow (WorkflowContext envelope)."""
    try:
        data = {"document_id": document_id}
        response = requests.post(f"{API_BASE_URL}/api/due-diligence-memo", data=data, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err = e.response.json().get("detail", e.response.text)
                st.error(f"Due Diligence Memo failed: {err}")
            except Exception:
                st.error(f"Due Diligence Memo failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error generating due diligence memo: {str(e)}")
        return None

def search_bilingual(query: str, response_language: Optional[str] = None, document_id: Optional[str] = None) -> Optional[Dict]:
    """Bilingual search."""
    try:
        data = {
            "query": query,
            "response_language": response_language
        }
        if document_id:
            data["document_id"] = document_id
        
        response = requests.post(f"{API_BASE_URL}/api/search-bilingual", data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in bilingual search: {str(e)}")
        return None


def get_documents() -> list:
    """Get list of documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents", timeout=10)
        response.raise_for_status()
        return response.json().get("documents", [])
    except Exception as e:
        st.warning(f"Could not load documents: {e}")
        return []


def delete_document(document_id: str) -> bool:
    """Permanently delete a document and its artifacts."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/documents/{document_id}",
            timeout=30,
        )
        if response.status_code == 404:
            st.warning(f"Document {document_id} was not found (it may already be deleted).")
            return False
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False


def get_vector_stats() -> dict:
    """Return the vector_store sub-dict from GET /api/stats, or {} on failure."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json().get("vector_store", {})
    except Exception:
        return {}


def clear_all_documents() -> Optional[dict]:
    """Call POST /api/admin/clear-all and return the response dict, or None on error."""
    try:
        response = requests.post(f"{API_BASE_URL}/api/admin/clear-all", timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error clearing documents: {str(e)}")
        return None


def rename_document(document_id: str, new_display_name: str) -> Optional[Dict]:
    """Rename a document (update display_name)."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/documents/{document_id}/rename",
            json={"display_name": new_display_name},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error renaming document: {str(e)}")
        return None


def contract_review(contract_id: str, contract_type: str, jurisdiction: Optional[str] = None, review_depth: Optional[str] = None) -> Optional[Dict]:
    """Run Contract Review workflow. Returns WorkflowContext as dict."""
    try:
        data = {"contract_id": contract_id, "contract_type": contract_type or "employment"}
        if jurisdiction:
            data["jurisdiction"] = jurisdiction
        if review_depth:
            data["review_depth"] = review_depth
        response = requests.post(f"{API_BASE_URL}/api/contract-review", data=data, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err = e.response.json()
                st.error(f"Contract review failed: {err.get('detail', e.response.text)}")
            except Exception:
                st.error(f"Contract review failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error running contract review: {str(e)}")
        return None


def explore_evidence(document_id: str, query: str, top_k: Optional[int] = None, mode: str = "both") -> Optional[Dict]:
    """Evidence Explorer: deterministic evidence snippets only (no answer). Calls POST /api/explore-evidence."""
    try:
        data = {"document_id": document_id, "query": query, "mode": mode}
        if top_k is not None:
            data["top_k"] = top_k
        response = requests.post(f"{API_BASE_URL}/api/explore-evidence", data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 422:
            detail = response.json().get("detail", {})
            err = detail.get("error", detail) if isinstance(detail, dict) else {}
            st.error(f"Evidence Explorer: {err.get('message', str(detail)) if isinstance(err, dict) else detail}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err = e.response.json().get("detail", e.response.text)
                st.error(f"Evidence Explorer failed: {err}")
            except Exception:
                st.error(f"Evidence Explorer failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def explore_answer(document_id: str, query: str, top_k: Optional[int] = None, response_language: Optional[str] = None) -> Optional[Dict]:
    """RAG Answer Explorer: answer + citations. Calls POST /api/explore-answer."""
    try:
        data = {"document_id": document_id, "query": query}
        if top_k is not None:
            data["top_k"] = top_k
        if response_language:
            data["response_language"] = response_language
        response = requests.post(f"{API_BASE_URL}/api/explore-answer", data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err = e.response.json().get("detail", e.response.text)
                st.error(f"Answer Explorer failed: {err}")
            except Exception:
                st.error(f"Answer Explorer failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# ---------------------------------------------------------------------------
# Chat / Conversational RAG helpers
# ---------------------------------------------------------------------------

def create_chat_session(document_id: str, mode: str = "strict") -> Optional[Dict]:
    """Create a new conversational RAG session. Returns {session_id, document_id, mode, created_at}."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat/session",
            json={"document_id": document_id, "mode": mode},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Could not create chat session: {e}")
        return None


def send_chat_message(session_id: str, message: str, mode: Optional[str] = None) -> Optional[Dict]:
    """Send a message to an existing session. Returns ChatMessageResponse."""
    try:
        payload: Dict = {"message": message}
        if mode:
            payload["mode"] = mode
        response = requests.post(
            f"{API_BASE_URL}/api/chat/{session_id}",
            json=payload,
            timeout=90,
        )
        if response.status_code == 404:
            st.error("Session not found or has expired. Please reset the session.")
            return None
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", "Document mismatch error.")
            except Exception:
                detail = "Document mismatch error."
            st.error(detail)
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Chat request failed: {e}")
        return None


def delete_chat_session_api(session_id: str) -> bool:
    """Delete a session and all its history."""
    try:
        response = requests.delete(f"{API_BASE_URL}/api/chat/{session_id}", timeout=10)
        return response.ok
    except Exception:
        return False


def get_document_classification_api(document_id: str) -> Optional[Dict]:
    """Fetch stored classification for a document from the registry."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/documents/{document_id}/classification", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


# Mandatory disclaimer for workflow outputs
WORKFLOW_DISCLAIMER = "This system does not provide legal advice. All outputs are for review purposes only."


# -----------------------------------------------------------------------------
# Navigation + routing (workflow-first)
# -----------------------------------------------------------------------------
st.sidebar.title("⚖️ Legal Workflows Engine")
st.sidebar.markdown("---")

# Health gate
if not check_backend_health():
    st.sidebar.error("⚠️ Backend not available. Please start the FastAPI server.")
    st.error(
        "**Backend Connection Error**\n\n"
        f"Cannot reach the backend at **{API_BASE_URL}**. Please ensure:\n"
        "1. The FastAPI backend is running: `python run_backend.py` (from the project root).\n"
        "2. If the backend runs on a different host/port, set the **BACKEND_URL** environment variable before starting Streamlit (e.g. `BACKEND_URL=http://127.0.0.1:8000`)."
    )
    st.stop()
else:
    st.sidebar.success("✅ Backend connected")

# Workflow-first navigation (main: contract review, document explorer, upload only)
page = st.sidebar.selectbox(
    "Navigate",
    [
        "🔍 Document Explorer",
        "ℹ️ About",
    ],
)

# -----------------------------------------------------------------------------
# Sidebar: Database admin panel
# -----------------------------------------------------------------------------
with st.sidebar.expander("🗄️ Database", expanded=False):
    # --- Stats row ---
    stats = get_vector_stats()
    col1, col2 = st.columns(2)
    col1.metric("📄 Documents", stats.get("unique_documents", 0))
    col2.metric("🧩 Chunks", stats.get("total_chunks", 0))
    st.caption(f"📐 Vectors: {stats.get('total_vectors', 0)}")
    if st.button("🔄 Refresh", key="db_refresh"):
        st.rerun()

    st.divider()

    # --- Document list ---
    docs = get_documents()
    if docs:
        for doc in docs:
            name = doc.get("display_name") or doc.get("original_filename", doc.get("document_id"))
            chunks = doc.get("total_chunks") or 0
            st.caption(f"• {name} ({chunks} chunks)")
    else:
        st.caption("No documents indexed.")

    st.divider()

    # --- Clear All ---
    confirm_clear = st.checkbox("I understand this will delete all indexed data", key="confirm_clear_all")
    if st.button("🗑️ Clear All Documents", type="primary", key="clear_all_btn", disabled=not confirm_clear):
        result = clear_all_documents()
        if result:
            n = result.get("documents_deleted", 0)
            c = result.get("chunks_deleted", 0)
            st.success(f"Cleared {n} document(s) and {c} chunk(s).")
            st.rerun()

# -----------------------------------------------------------------------------
# Page: Document Explorer — 3-step guided workflow
# Step 1: Upload  →  Step 2: Classification  →  Step 3: Exploration (Q&A)
# -----------------------------------------------------------------------------
if page == "🔍 Document Explorer":
    st.markdown('<div class="main-header">🔍 Document Explorer</div>', unsafe_allow_html=True)
    st.markdown("Upload a document, classify it, then explore it with grounded Q&A.")
    st.caption(WORKFLOW_DISCLAIMER)

    # ── Session-state defaults ────────────────────────────────────────────────
    if "wf_step" not in st.session_state:
        st.session_state.wf_step = 1
    if "wf_document_id" not in st.session_state:
        st.session_state.wf_document_id = None
    if "wf_classification" not in st.session_state:
        st.session_state.wf_classification = None
    if "wf_is_contract" not in st.session_state:
        st.session_state.wf_is_contract = False
    if "explorer_chat_mode" not in st.session_state:
        st.session_state.explorer_chat_mode = "Evidence (Strict)"
    if "explorer_session_id" not in st.session_state:
        st.session_state.explorer_session_id = None
    if "explorer_chat_history" not in st.session_state:
        st.session_state.explorer_chat_history = []
    # Agent state machine keys
    if "wf_agent_state" not in st.session_state:
        st.session_state.wf_agent_state = "idle"
    if "wf_review_result" not in st.session_state:
        st.session_state.wf_review_result = None
    if "wf_review_done" not in st.session_state:
        st.session_state.wf_review_done = False
    if "wf_pdf_answered" not in st.session_state:
        st.session_state.wf_pdf_answered = False
    if "wf_contract_type" not in st.session_state:
        st.session_state.wf_contract_type = "employment"
    if "wf_jurisdiction" not in st.session_state:
        st.session_state.wf_jurisdiction = None

    def _reset_workflow():
        """Reset all wf_* and explorer_* session state keys."""
        if st.session_state.get("explorer_session_id"):
            delete_chat_session_api(st.session_state.explorer_session_id)
        for key in [
            "wf_step", "wf_document_id", "wf_classification", "wf_is_contract",
            "explorer_chat_mode", "explorer_session_id", "explorer_chat_history",
            "wf_agent_state", "wf_review_result", "wf_review_done", "wf_pdf_answered",
            "wf_contract_type", "wf_jurisdiction",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    def _render_stepper(current: int):
        """Render horizontal step badges."""
        cols = st.columns(3)
        steps = ["1  Upload", "2  Classification", "3  Exploration"]
        for i, (col, label) in enumerate(zip(cols, steps), start=1):
            if i < current:
                badge = (
                    f'<span style="background:#22c55e;color:white;padding:5px 14px;'
                    f'border-radius:14px;font-weight:bold;">✓ {label}</span>'
                )
            elif i == current:
                badge = (
                    f'<span style="background:#3b82f6;color:white;padding:5px 14px;'
                    f'border-radius:14px;font-weight:bold;">{label}</span>'
                )
            else:
                badge = (
                    f'<span style="background:#e5e7eb;color:#6b7280;padding:5px 14px;'
                    f'border-radius:14px;">{label}</span>'
                )
            col.markdown(badge, unsafe_allow_html=True)
        st.markdown("---")

    # ── Helper: Render full contract review details (for chat expander) ──────
    def _render_contract_review_details(resp: dict):
        """Render the full contract review UI inside an expander."""
        doc_warning = (resp.get("document_classification_warning") or "").strip()
        if doc_warning:
            st.warning(f"⚠️ {doc_warning}")

        meta_cols = st.columns(4)
        meta_cols[0].metric("Workflow ID", str(resp.get("workflow_id", ""))[:12] + "…")
        meta_cols[1].metric("Document", resp.get("document_id", ""))
        meta_cols[2].metric("Contract type", resp.get("contract_type", ""))
        meta_cols[3].metric("Jurisdiction", resp.get("jurisdiction") or "—")

        # Group 2b: overall risk score metrics
        _all_risks = resp.get("risks", []) or []
        _high_n = sum(1 for r in _all_risks if isinstance(r, dict) and r.get("severity") == "high")
        _med_n  = sum(1 for r in _all_risks if isinstance(r, dict) and r.get("severity") == "medium")
        _low_n  = sum(1 for r in _all_risks if isinstance(r, dict) and r.get("severity") == "low")
        _rlabel = resp.get("risk_label", "low_risk")
        _rlabel_display = {"high_risk": "🔴 High Risk", "medium_risk": "🟡 Medium Risk", "low_risk": "🟢 Low Risk"}.get(_rlabel, _rlabel)
        score_cols = st.columns(4)
        score_cols[0].metric("🔴 High", _high_n)
        score_cols[1].metric("🟡 Medium", _med_n)
        score_cols[2].metric("🟢 Low", _low_n)
        score_cols[3].metric("Overall", _rlabel_display)

        st.markdown("---")

        # Group 3: Clause Coverage Matrix (replaces flat not_detected list)
        exec_items_for_matrix = resp.get("executive_summary", []) or []
        if exec_items_for_matrix:
            st.subheader("Clause Coverage")
            matrix_rows = []
            for item in exec_items_for_matrix:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "")
                sev = item.get("severity") or "—"
                if cat == "confirmation":
                    status_icon = "✅ Detected"
                elif cat == "finding":
                    status_icon = "⚠️ Weak / Implicit"
                else:
                    status_icon = "❌ Not Detected"
                matrix_rows.append({
                    "Clause": item.get("text", ""),
                    "Status": status_icon,
                    "Severity": sev.capitalize() if sev != "—" else "—",
                })
            if matrix_rows:
                st.dataframe(_pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

            # Group 7: statutory notes per clause (inline captions)
            statutory_notes = resp.get("statutory_notes") or {}
            if statutory_notes:
                with st.expander("📜 Statutory References (jurisdiction-specific)", expanded=False):
                    for clause_display, note in statutory_notes.items():
                        if not isinstance(note, dict):
                            continue
                        art = note.get("article", "")
                        text = note.get("text", "")
                        src = note.get("source", "")
                        st.caption(f"**{clause_display}** — {art}: \"{text}\" — *{src}*")

            # Detailed sections (collapsed)
            with st.expander("Details: Not Detected / Implicitly Covered", expanded=False):
                not_detected = resp.get("not_detected_clauses", []) or []
                ct_label = (resp.get("contract_type") or "Contract").strip()
                ct_label = ct_label[0].upper() + ct_label[1:] if ct_label else "Contract"
                st.markdown(f"**Clauses Not Detected ({ct_label} Profile)**")
                if not_detected:
                    for name in not_detected:
                        st.markdown(f"- {name}")
                else:
                    st.info("All expected clauses were detected.")
                implicitly_covered = resp.get("implicitly_covered_clauses", []) or []
                coverage_notes = resp.get("implicit_coverage_notes") or {}
                if implicitly_covered:
                    st.markdown("**Implicitly Covered**")
                    for name in implicitly_covered:
                        note = coverage_notes.get(name)
                        st.markdown(f"- **{name}**: {note}" if note else f"- {name}")
        else:
            # Fallback: original flat not_detected section when no executive_summary
            not_detected = resp.get("not_detected_clauses", []) or []
            ct_label = (resp.get("contract_type") or "Contract").strip()
            ct_label = ct_label[0].upper() + ct_label[1:] if ct_label else "Contract"
            st.subheader(f"Clauses Not Detected ({ct_label} Profile)")
            if not_detected:
                for name in not_detected:
                    st.markdown(f"- {name}")
            else:
                st.info("All expected clauses were detected.")
            implicitly_covered = resp.get("implicitly_covered_clauses", []) or []
            coverage_notes = resp.get("implicit_coverage_notes") or {}
            if implicitly_covered:
                st.subheader("Implicitly Covered")
                for name in implicitly_covered:
                    note = coverage_notes.get(name)
                    st.markdown(f"- **{name}**: {note}" if note else f"- {name}")

        st.markdown("---")

        # Contradiction risks section
        contradiction_risks = resp.get("contradiction_risks", []) or []
        if contradiction_risks:
            st.subheader("Cross-Clause Observations")
            for cr in contradiction_risks:
                if isinstance(cr, dict):
                    st.info(f"ℹ️ {cr.get('description', '')}")

        risks = resp.get("risks", []) or []
        st.subheader("Risk table")
        if not risks:
            st.info("No risks were identified.")
        else:
            def _display_status(s):
                if s == "detected": return "Detected"
                if s == "uncertain": return "Detected (Weak Evidence)"
                if s == "detected_implicit": return "Detected (Implicit Reference)"
                if s == "detected_distributed": return "Detected (Distributed Provisions)"
                if s == "detected_weak": return "Detected (Limited Coverage)"
                return "Not Detected"
            rows = []
            for r in risks:
                if not isinstance(r, dict):
                    continue
                display_names = r.get("display_names", []) or []
                clause_label = ", ".join(display_names) if display_names else ", ".join(r.get("clause_ids", []) or [])
                rows.append({
                    "severity": r.get("severity", ""),
                    "evidence state": _display_status(r.get("status", "")),
                    "description": r.get("description", ""),
                    "clause_types": ", ".join(r.get("clause_types", []) or []),
                    "clauses": clause_label,
                    "pages": ", ".join([str(p) for p in (r.get("page_numbers", []) or [])]),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("Weak evidence: clause found but may lack commonly expected details.")
            # ── Improvement 1 & 2: per-risk detail expanders ─────────────────
            for r in risks:
                if not isinstance(r, dict):
                    continue
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r.get("severity", ""), "⚪")
                exp_label = f"{sev_icon} **{r.get('severity', '').capitalize()}** — {r.get('description', '')}"
                with st.expander(exp_label, expanded=False):
                    # Severity reason
                    if r.get("severity_reason"):
                        st.caption(f"ℹ️ {r['severity_reason']}")
                    # Verbatim evidence with keyword highlighting (Group 4b)
                    snippets = r.get("verbatim_evidence", []) or []
                    if snippets:
                        st.markdown("**Evidence found in document:**")
                        for snippet in snippets:
                            name = snippet.get("display_name") or snippet.get("clause_id", "")
                            page = snippet.get("page_number", "")
                            text = snippet.get("text", "")
                            kw = snippet.get("matched_keyword") or ""
                            if kw:
                                idx = text.lower().find(kw.lower())
                                if idx >= 0:
                                    text = text[:idx] + "**" + text[idx:idx + len(kw)] + "**" + text[idx + len(kw):]
                            st.markdown(
                                f"> **{name}** (p.{page})\n"
                                f"> *\"{text}\"*"
                            )
                    # Recommendation
                    if r.get("recommendation"):
                        st.info(f"💡 **Recommendation:** {r['recommendation']}")

        st.markdown("---")
        exec_items = resp.get("executive_summary", []) or []
        st.subheader("Key Review Observations")
        if exec_items:
            by_cat: dict = {}
            for item in exec_items:
                if isinstance(item, dict):
                    by_cat.setdefault(item.get("category") or "other", []).append(item)
            for section_title, cat_key in [("Risks", "risk"), ("Findings", "finding"), ("Confirmations", "confirmation")]:
                group = by_cat.get(cat_key, [])
                if group:
                    st.markdown(f"**{section_title}**")
                    for item in group:
                        sev = item.get("severity")
                        txt = item.get("text", "")
                        st.markdown(f"- **{sev}**: {txt}" if sev else f"- {txt}")
            if by_cat.get("other"):
                st.markdown("**Other**")
                for item in by_cat["other"]:
                    sev = item.get("severity")
                    txt = item.get("text", "")
                    st.markdown(f"- **{sev}**: {txt}" if sev else f"- {txt}")

    # ── Helper: Call POST /api/contract-review ────────────────────────────────
    def _run_contract_review(doc_id: str, contract_type: str, jurisdiction) -> Optional[dict]:
        """Call contract review API; return the response dict or None on failure."""
        result = contract_review(doc_id, contract_type, jurisdiction, "standard")
        if not result:
            return None
        if result.get("status") == "failed":
            err = result.get("error", {})
            st.error(f"**{err.get('code', 'Error')}:** {err.get('message', '')}")
            return None
        ir = result.get("intermediate_results", {}) or {}
        resp = ir.get("contract_review.response") or {}
        return resp if isinstance(resp, dict) else None

    # ── Helper: Format short review summary for Agent 1 chat bubble ──────────
    def _format_review_summary(resp: dict) -> str:
        lines = []
        ct = resp.get("contract_type") or "contract"
        jur = resp.get("jurisdiction") or "—"
        lines.append(f"**Contract Review Complete** — Type: `{ct}` | Jurisdiction: `{jur}`\n")

        exec_items = resp.get("executive_summary", []) or []
        risks_exec = [i for i in exec_items if isinstance(i, dict) and i.get("category") == "risk"][:5]
        findings = [i for i in exec_items if isinstance(i, dict) and i.get("category") == "finding"][:3]

        if risks_exec:
            lines.append("**Key Risks:**")
            for item in risks_exec:
                sev = item.get("severity", "")
                txt = item.get("text", "")
                lines.append(f"- **{sev}**: {txt}" if sev else f"- {txt}")

        not_detected = resp.get("not_detected_clauses", []) or []
        if not_detected:
            lines.append(f"\n**Clauses Not Detected:** {', '.join(not_detected[:5])}" +
                         (" …and more" if len(not_detected) > 5 else ""))

        if findings:
            lines.append("\n**Findings:**")
            for item in findings:
                lines.append(f"- {item.get('text', '')}")

        lines.append(f"\n*{resp.get('disclaimer', WORKFLOW_DISCLAIMER)}*")
        return "\n".join(lines)

    # ── Helper: Generate PDF report with reportlab ────────────────────────────
    def _generate_pdf_buffer(resp: dict):
        """Build PDF in-memory; return BytesIO. Raises on failure."""
        from io import BytesIO
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        except ImportError as exc:
            raise RuntimeError(f"reportlab not installed: {exc}") from exc

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        normal = styles["Normal"]
        story = []

        # ── 8a: Reusable cell paragraph styles ────────────────────────────────
        cell_style = ParagraphStyle(
            "cell", parent=normal,
            fontSize=7, leading=9,
            wordWrap="LTR",
            spaceAfter=0, spaceBefore=0,
        )
        header_cell_style = ParagraphStyle(
            "header_cell", parent=cell_style,
            textColor=colors.whitesmoke, fontName="Helvetica-Bold",
        )
        toc_style = ParagraphStyle(
            "toc", parent=normal, fontSize=9, leftIndent=12, spaceAfter=2,
        )

        # Title page
        story.append(Paragraph("Contract Review Report", h1))
        story.append(Spacer(1, 0.4*cm))
        ct = resp.get("contract_type") or "—"
        jur = resp.get("jurisdiction") or "—"
        doc_id = resp.get("document_id") or "—"
        risk_label = resp.get("risk_label", "low_risk")
        risk_score = resp.get("risk_score", 0)
        story.append(Paragraph(f"Contract type: {ct} | Jurisdiction: {jur}", normal))
        story.append(Paragraph(f"Document: {doc_id}", normal))
        story.append(Paragraph(f"Overall Risk: {risk_label.replace('_', ' ').title()} (score: {risk_score})", normal))
        story.append(Spacer(1, 0.4*cm))

        # ── 8h: Table of contents ─────────────────────────────────────────────
        statutory_notes = resp.get("statutory_notes") or {}
        exec_items = resp.get("executive_summary", []) or []
        risks = resp.get("risks", []) or []
        toc_items = ["Executive Summary", "Clause Coverage", "Risk Analysis", "Evidence Excerpts", "Clauses Not Detected"]
        if statutory_notes:
            toc_items.append("Statutory References")
        for toc_name in toc_items:
            story.append(Paragraph(f"• {toc_name}", toc_style))
        story.append(Spacer(1, 0.6*cm))

        # Executive Summary
        if exec_items:
            story.append(Paragraph("Executive Summary", h2))
            by_cat: dict = {}
            for item in exec_items:
                if isinstance(item, dict):
                    by_cat.setdefault(item.get("category") or "other", []).append(item)
            for section_title, cat_key in [("Risks", "risk"), ("Findings", "finding"), ("Confirmations", "confirmation")]:
                group = by_cat.get(cat_key, [])
                if group:
                    story.append(Paragraph(f"<b>{section_title}</b>", normal))
                    for item in group:
                        sev = item.get("severity", "")
                        txt = item.get("text", "")
                        bullet = f"• [{sev}] {txt}" if sev else f"• {txt}"
                        story.append(Paragraph(bullet, normal))
            story.append(Spacer(1, 0.4*cm))

        # ── 8f: Clause Coverage Table ─────────────────────────────────────────
        if exec_items:
            story.append(Paragraph("Clause Coverage", h2))
            coverage_data = [[
                Paragraph("Clause", header_cell_style),
                Paragraph("Status", header_cell_style),
                Paragraph("Severity", header_cell_style),
            ]]
            for item in exec_items:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "")
                icon = "Confirmed" if cat == "confirmation" else ("Finding" if cat == "finding" else "Risk")
                sev_text = (item.get("severity") or "").capitalize() or "—"
                coverage_data.append([
                    Paragraph(item.get("text", ""), cell_style),
                    Paragraph(icon, cell_style),
                    Paragraph(sev_text, cell_style),
                ])
            cov_tbl = Table(coverage_data, repeatRows=1, colWidths=[9.5*cm, 4.5*cm, 3.0*cm])
            cov_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(cov_tbl)
            story.append(Spacer(1, 0.4*cm))

        # ── 8b/8c/8d: Risk Analysis Table with Paragraph cells + row colors ──
        if risks:
            story.append(Paragraph("Risk Analysis", h2))
            # ── 8b: fixed 17cm total ──────────────────────────────────────────
            COL_WIDTHS = [1.5*cm, 5.5*cm, 3.0*cm, 2.5*cm, 1.0*cm, 3.5*cm]
            table_data = [[
                Paragraph("Severity", header_cell_style),
                Paragraph("Description", header_cell_style),
                Paragraph("Status", header_cell_style),
                Paragraph("Clauses", header_cell_style),
                Paragraph("Pages", header_cell_style),
                Paragraph("Recommendation", header_cell_style),
            ]]
            # ── 8c: wrap all cells in Paragraph — no truncation ───────────────
            for r in risks:
                if not isinstance(r, dict):
                    continue
                display_names = r.get("display_names", []) or r.get("clause_ids", []) or []
                pages = r.get("page_numbers", []) or []
                table_data.append([
                    Paragraph(r.get("severity", "").capitalize(), cell_style),
                    Paragraph(r.get("description", ""), cell_style),
                    Paragraph(r.get("status", "").replace("_", " ").title(), cell_style),
                    Paragraph(", ".join(display_names[:3]), cell_style),
                    Paragraph(", ".join(str(p) for p in pages[:5]), cell_style),
                    Paragraph(r.get("recommendation") or "", cell_style),
                ])
            tbl = Table(table_data, repeatRows=1, colWidths=COL_WIDTHS)
            # ── 8d: color-code rows by severity ──────────────────────────────
            row_colors = []
            for i, r in enumerate(risks, start=1):
                if not isinstance(r, dict):
                    continue
                bg = (
                    colors.HexColor("#FFD5D5") if r.get("severity") == "high" else
                    colors.HexColor("#FFF3CD") if r.get("severity") == "medium" else
                    colors.HexColor("#D4EDDA")
                )
                row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                *row_colors,
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.4*cm))

            # ── 8e: Evidence Excerpts with split header/body styles ────────────
            has_snippets = any(r.get("verbatim_evidence") for r in risks if isinstance(r, dict))
            if has_snippets:
                story.append(Paragraph("Evidence Excerpts", h2))
                ev_header_style = ParagraphStyle(
                    "ev_h", parent=normal,
                    fontSize=7, fontName="Helvetica-Bold", leftIndent=12, spaceAfter=1,
                )
                ev_body_style = ParagraphStyle(
                    "ev_b", parent=normal,
                    fontSize=7, leftIndent=12, leading=9,
                    textColor=colors.HexColor("#1A3C6B"), spaceAfter=4,
                )
                for r in risks:
                    if not isinstance(r, dict):
                        continue
                    snippets = r.get("verbatim_evidence", []) or []
                    if not snippets:
                        continue
                    story.append(Paragraph(f"<b>{r.get('description', '')}</b>", normal))
                    if r.get("severity_reason"):
                        story.append(Paragraph(f"<i>{r['severity_reason']}</i>", normal))
                    for snippet in snippets:
                        s_name = snippet.get("display_name") or snippet.get("clause_id", "")
                        s_page = snippet.get("page_number", "")
                        s_text = snippet.get("text", "")
                        story.append(Paragraph(f"{s_name} — Page {s_page}", ev_header_style))
                        story.append(Paragraph(f'"{s_text}"', ev_body_style))
                    story.append(Spacer(1, 0.2*cm))

        # Not Detected
        not_detected = resp.get("not_detected_clauses", []) or []
        if not_detected:
            story.append(Paragraph("Clauses Not Detected", h2))
            for name in not_detected:
                story.append(Paragraph(f"• {name}", normal))
            story.append(Spacer(1, 0.3*cm))

        # ── 8g: Statutory References ──────────────────────────────────────────
        if statutory_notes:
            story.append(Paragraph("Statutory References", h2))
            story.append(Paragraph(
                "<i>[Reference only — not legal advice. Verify with qualified counsel.]</i>", normal
            ))
            stat_data = [[
                Paragraph("Clause", header_cell_style),
                Paragraph("Reference", header_cell_style),
            ]]
            for clause_display, note in statutory_notes.items():
                if not isinstance(note, dict):
                    continue
                art = note.get("article", "")
                text_val = note.get("text", "")
                src = note.get("source", "")
                stat_data.append([
                    Paragraph(str(clause_display), cell_style),
                    Paragraph(f"{art}: {text_val} — {src}", cell_style),
                ])
            stat_tbl = Table(stat_data, repeatRows=1, colWidths=[4.5*cm, 12.5*cm])
            stat_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(stat_tbl)
            story.append(Spacer(1, 0.3*cm))

        # Disclaimer
        story.append(Spacer(1, 0.6*cm))
        story.append(Paragraph(resp.get("disclaimer", WORKFLOW_DISCLAIMER), ParagraphStyle("small", parent=normal, fontSize=8)))

        doc.build(story)
        buf.seek(0)
        return buf

    # ── Arabic PDF helpers ─────────────────────────────────────────────────────
    def _ar(text: str) -> str:
        """Reshape + bidi-correct Arabic text for ReportLab rendering."""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(str(text)))
        except ImportError:
            return str(text)

    def _translate_texts(texts: list, api_base: str) -> list:
        """Call backend /api/translate; returns translations in same order."""
        import json as _json
        try:
            r = requests.post(
                f"{api_base}/api/translate",
                data={"texts": _json.dumps(texts), "target_lang": "ar", "source_lang": "en"},
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("translations", texts)
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Arabic translation failed, falling back to source text: {_e}")
            return texts  # fallback: untranslated

    _AR = {
        "title": "تقرير مراجعة العقد",
        "contract_type": "نوع العقد",
        "jurisdiction": "الولاية القضائية",
        "document": "المستند",
        "overall_risk": "المخاطر الإجمالية",
        "score": "النتيجة",
        "exec_summary": "الملخص التنفيذي",
        "clause_coverage": "تغطية البنود",
        "risk_analysis": "تحليل المخاطر",
        "evidence_excerpts": "مقتطفات الأدلة",
        "not_detected": "البنود غير المكتشفة",
        "statutory_refs": "المراجع القانونية",
        "risks_cat": "المخاطر",
        "findings_cat": "النتائج",
        "confirmations_cat": "التأكيدات",
        "confirmed": "مؤكد",
        "finding": "نتيجة",
        "risk_status": "خطر",
        "clause": "البند",
        "status": "الحالة",
        "severity": "الخطورة",
        "description": "الوصف",
        "clauses": "البنود",
        "pages": "الصفحات",
        "recommendation": "التوصية",
        "reference": "المرجع",
        "high_risk": "مخاطرة عالية",
        "medium_risk": "مخاطرة متوسطة",
        "low_risk": "مخاطرة منخفضة",
        "high": "مرتفع",
        "medium": "متوسط",
        "low": "منخفض",
        "ref_disclaimer": "[للمرجعية فقط — لا تمثل استشارة قانونية. يُرجى التحقق مع مستشار قانوني مؤهل.]",
        "disclaimer": "لا يقدم هذا النظام استشارات قانونية. جميع المخرجات لأغراض المراجعة فقط.",
        "page_label": "صفحة",
        "toc_items": ["الملخص التنفيذي", "تغطية البنود", "تحليل المخاطر", "مقتطفات الأدلة", "البنود غير المكتشفة", "المراجع القانونية"],
    }

    def _generate_arabic_pdf_buffer(resp: dict, api_base: str):
        """Build Arabic PDF in-memory; return BytesIO. Raises on failure."""
        import os
        from io import BytesIO
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_RIGHT
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError as exc:
            raise RuntimeError(f"reportlab not installed: {exc}") from exc

        # Register Arabic-capable font
        font_path = r"C:\Windows\Fonts\arial.ttf"
        if os.path.exists(font_path):
            try:
                if "Arabic" not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont("Arabic", font_path))
                AR_FONT = "Arabic"
            except Exception:
                AR_FONT = "Helvetica"
        else:
            AR_FONT = "Helvetica"

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        ar_h1 = ParagraphStyle(
            "ar_h1", parent=normal,
            fontName=AR_FONT, fontSize=16, leading=20,
            wordWrap="RTL", alignment=TA_RIGHT, spaceAfter=8,
        )
        ar_h2 = ParagraphStyle(
            "ar_h2", parent=normal,
            fontName=AR_FONT, fontSize=12, leading=15,
            wordWrap="RTL", alignment=TA_RIGHT, spaceAfter=4,
        )
        ar_normal = ParagraphStyle(
            "ar_normal", parent=normal,
            fontName=AR_FONT, fontSize=9, leading=12,
            wordWrap="RTL", alignment=TA_RIGHT,
        )
        ar_cell = ParagraphStyle(
            "ar_cell", parent=normal,
            fontName=AR_FONT, fontSize=7, leading=9,
            wordWrap="RTL", alignment=TA_RIGHT,
            spaceAfter=0, spaceBefore=0,
        )
        ar_header_cell = ParagraphStyle(
            "ar_header_cell", parent=ar_cell,
            textColor=colors.whitesmoke, fontName=AR_FONT,
        )
        ar_toc_style = ParagraphStyle(
            "ar_toc", parent=ar_normal, fontSize=9, spaceAfter=2,
        )
        ar_small = ParagraphStyle(
            "ar_small", parent=ar_normal, fontSize=8,
        )

        # ── Collect all dynamic strings to translate ──────────────────────────
        exec_items = resp.get("executive_summary", []) or []
        risks = resp.get("risks", []) or []
        statutory_notes = resp.get("statutory_notes") or {}
        not_detected = resp.get("not_detected_clauses", []) or []

        dyn_strings = []
        # 1: contract_type, jurisdiction
        dyn_strings.append(resp.get("contract_type") or "")
        dyn_strings.append(resp.get("jurisdiction") or "")
        # 2: executive_summary texts
        for item in exec_items:
            if isinstance(item, dict):
                dyn_strings.append(item.get("text", ""))
        # 3: risk descriptions, recommendations, severity_reason, display_names, snippets
        for r in risks:
            if not isinstance(r, dict):
                continue
            dyn_strings.append(r.get("description", ""))
            dyn_strings.append(r.get("recommendation") or "")
            dyn_strings.append(r.get("severity_reason") or "")
            for dn in (r.get("display_names") or []):
                dyn_strings.append(str(dn))
            for snippet in (r.get("verbatim_evidence") or []):
                dyn_strings.append(snippet.get("text", ""))
                dyn_strings.append(snippet.get("display_name") or snippet.get("clause_id", ""))
        # 4: not_detected_clauses
        for name in not_detected:
            dyn_strings.append(str(name))
        # 5: statutory_notes keys and note fields
        for clause_display, note in statutory_notes.items():
            dyn_strings.append(str(clause_display))
            if isinstance(note, dict):
                dyn_strings.append(note.get("text", ""))
                dyn_strings.append(note.get("article", ""))
                dyn_strings.append(note.get("source", ""))
        # 6: disclaimer
        dyn_strings.append(resp.get("disclaimer", WORKFLOW_DISCLAIMER) or "")

        # Translate all at once
        translated = _translate_texts(dyn_strings, api_base)

        # ── Reconstruct translated values ─────────────────────────────────────
        idx = 0
        def _t():
            nonlocal idx
            val = translated[idx] if idx < len(translated) else ""
            idx += 1
            return val or ""

        t_contract_type = _t()
        t_jurisdiction = _t()

        t_exec_items = []
        for item in exec_items:
            if isinstance(item, dict):
                t_exec_items.append({**item, "text": _t()})

        t_risks = []
        for r in risks:
            if not isinstance(r, dict):
                continue
            t_desc = _t()
            t_rec = _t()
            t_sev_reason = _t()
            t_display_names = [_t() for _ in (r.get("display_names") or [])]
            t_snippets = []
            for snippet in (r.get("verbatim_evidence") or []):
                t_text = _t()
                t_dname = _t()
                t_snippets.append({**snippet, "text": t_text,
                                    "display_name": t_dname or snippet.get("clause_id", "")})
            t_risks.append({**r, "description": t_desc, "recommendation": t_rec,
                            "severity_reason": t_sev_reason,
                            "display_names": t_display_names,
                            "verbatim_evidence": t_snippets})

        t_not_detected = [_t() for _ in not_detected]

        t_statutory = {}
        for clause_display, note in statutory_notes.items():
            t_clause = _t()
            if isinstance(note, dict):
                t_text_val = _t()
                t_art = _t()
                t_src = _t()
                t_statutory[t_clause] = {**note, "text": t_text_val, "article": t_art, "source": t_src}
            else:
                t_statutory[t_clause] = note

        t_disclaimer = _t()

        # ── Build PDF ─────────────────────────────────────────────────────────
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
        )
        story = []

        # Title page
        story.append(Paragraph(_ar(_AR["title"]), ar_h1))
        story.append(Spacer(1, 0.4*cm))
        risk_label = resp.get("risk_label", "low_risk")
        risk_score = resp.get("risk_score", 0)
        doc_id = resp.get("document_id") or "—"
        risk_label_ar = _AR.get(risk_label, risk_label.replace("_", " "))
        story.append(Paragraph(
            _ar(f"{_AR['contract_type']}: {t_contract_type or '—'} | {_AR['jurisdiction']}: {t_jurisdiction or '—'}"),
            ar_normal,
        ))
        story.append(Paragraph(_ar(f"{_AR['document']}: {doc_id}"), ar_normal))
        story.append(Paragraph(
            _ar(f"{_AR['overall_risk']}: {risk_label_ar} ({_AR['score']}: {risk_score})"),
            ar_normal,
        ))
        story.append(Spacer(1, 0.4*cm))

        # Table of contents
        toc_labels = list(_AR["toc_items"])
        if not t_statutory:
            toc_labels = toc_labels[:5]
        for toc_name in toc_labels:
            story.append(Paragraph(_ar(f"• {toc_name}"), ar_toc_style))
        story.append(Spacer(1, 0.6*cm))

        # Executive Summary
        if t_exec_items:
            story.append(Paragraph(_ar(_AR["exec_summary"]), ar_h2))
            by_cat: dict = {}
            for item in t_exec_items:
                if isinstance(item, dict):
                    by_cat.setdefault(item.get("category") or "other", []).append(item)
            for section_title_key, cat_key in [
                ("risks_cat", "risk"), ("findings_cat", "finding"), ("confirmations_cat", "confirmation")
            ]:
                group = by_cat.get(cat_key, [])
                if group:
                    story.append(Paragraph(f"<b>{_ar(_AR[section_title_key])}</b>", ar_normal))
                    for item in group:
                        sev = item.get("severity", "")
                        txt = item.get("text", "")
                        bullet = f"• [{sev}] {txt}" if sev else f"• {txt}"
                        story.append(Paragraph(_ar(bullet), ar_normal))
            story.append(Spacer(1, 0.4*cm))

        # Clause Coverage Table
        if t_exec_items:
            story.append(Paragraph(_ar(_AR["clause_coverage"]), ar_h2))
            coverage_data = [[
                Paragraph(_ar(_AR["clause"]), ar_header_cell),
                Paragraph(_ar(_AR["status"]), ar_header_cell),
                Paragraph(_ar(_AR["severity"]), ar_header_cell),
            ]]
            for item in t_exec_items:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "")
                icon = _AR["confirmed"] if cat == "confirmation" else (_AR["finding"] if cat == "finding" else _AR["risk_status"])
                sev_raw = (item.get("severity") or "").lower()
                sev_text = _AR.get(sev_raw, sev_raw.capitalize()) or "—"
                coverage_data.append([
                    Paragraph(_ar(item.get("text", "")), ar_cell),
                    Paragraph(_ar(icon), ar_cell),
                    Paragraph(_ar(sev_text), ar_cell),
                ])
            cov_tbl = Table(coverage_data, repeatRows=1, colWidths=[9.5*cm, 4.5*cm, 3.0*cm])
            cov_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(cov_tbl)
            story.append(Spacer(1, 0.4*cm))

        # Risk Analysis Table
        if t_risks:
            story.append(Paragraph(_ar(_AR["risk_analysis"]), ar_h2))
            COL_WIDTHS = [1.5*cm, 5.5*cm, 3.0*cm, 2.5*cm, 1.0*cm, 3.5*cm]
            table_data = [[
                Paragraph(_ar(_AR["severity"]), ar_header_cell),
                Paragraph(_ar(_AR["description"]), ar_header_cell),
                Paragraph(_ar(_AR["status"]), ar_header_cell),
                Paragraph(_ar(_AR["clauses"]), ar_header_cell),
                Paragraph(_ar(_AR["pages"]), ar_header_cell),
                Paragraph(_ar(_AR["recommendation"]), ar_header_cell),
            ]]
            _status_ar = {
                "risk": _AR["risk_status"],
                "finding": _AR["finding"],
                "confirmation": _AR["confirmed"],
            }
            for r in t_risks:
                if not isinstance(r, dict):
                    continue
                display_names = r.get("display_names") or []
                pages = r.get("page_numbers", []) or []
                sev = r.get("severity", "")
                sev_ar = _AR.get(sev, sev).capitalize()
                table_data.append([
                    Paragraph(_ar(sev_ar), ar_cell),
                    Paragraph(_ar(r.get("description", "")), ar_cell),
                    Paragraph(_ar(_status_ar.get(r.get("status", "").lower(), r.get("status", ""))), ar_cell),
                    Paragraph(_ar(", ".join(str(d) for d in display_names[:3])), ar_cell),
                    Paragraph(_ar(", ".join(str(p) for p in pages[:5])), ar_cell),
                    Paragraph(_ar(r.get("recommendation") or ""), ar_cell),
                ])
            tbl = Table(table_data, repeatRows=1, colWidths=COL_WIDTHS)
            row_colors = []
            for i, r in enumerate(t_risks, start=1):
                if not isinstance(r, dict):
                    continue
                bg = (
                    colors.HexColor("#FFD5D5") if r.get("severity") == "high" else
                    colors.HexColor("#FFF3CD") if r.get("severity") == "medium" else
                    colors.HexColor("#D4EDDA")
                )
                row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTNAME", (0, 0), (-1, 0), AR_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                *row_colors,
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.4*cm))

            # Evidence Excerpts
            has_snippets = any(r.get("verbatim_evidence") for r in t_risks if isinstance(r, dict))
            if has_snippets:
                story.append(Paragraph(_ar(_AR["evidence_excerpts"]), ar_h2))
                ar_ev_header = ParagraphStyle(
                    "ar_ev_h", parent=ar_normal,
                    fontSize=7, spaceAfter=1,
                )
                ar_ev_body = ParagraphStyle(
                    "ar_ev_b", parent=ar_normal,
                    fontSize=7, leading=9,
                    textColor=colors.HexColor("#1A3C6B"), spaceAfter=4,
                )
                for r in t_risks:
                    if not isinstance(r, dict):
                        continue
                    snippets = r.get("verbatim_evidence", []) or []
                    if not snippets:
                        continue
                    story.append(Paragraph(f"<b>{_ar(r.get('description', ''))}</b>", ar_normal))
                    if r.get("severity_reason"):
                        story.append(Paragraph(f"<i>{_ar(r['severity_reason'])}</i>", ar_normal))
                    for snippet in snippets:
                        s_name = snippet.get("display_name") or snippet.get("clause_id", "")
                        s_page = snippet.get("page_number", "")
                        s_text = snippet.get("text", "")
                        story.append(Paragraph(
                            _ar(f"{s_name} — {_AR['page_label']} {s_page}"), ar_ev_header
                        ))
                        story.append(Paragraph(_ar(f'"{s_text}"'), ar_ev_body))
                    story.append(Spacer(1, 0.2*cm))

        # Clauses Not Detected
        if t_not_detected:
            story.append(Paragraph(_ar(_AR["not_detected"]), ar_h2))
            for name in t_not_detected:
                story.append(Paragraph(_ar(f"• {name}"), ar_normal))
            story.append(Spacer(1, 0.3*cm))

        # Statutory References
        if t_statutory:
            story.append(Paragraph(_ar(_AR["statutory_refs"]), ar_h2))
            story.append(Paragraph(_ar(_AR["ref_disclaimer"]), ar_normal))
            stat_data = [[
                Paragraph(_ar(_AR["clause"]), ar_header_cell),
                Paragraph(_ar(_AR["reference"]), ar_header_cell),
            ]]
            for clause_display, note in t_statutory.items():
                if not isinstance(note, dict):
                    continue
                art = note.get("article", "")
                text_val = note.get("text", "")
                src = note.get("source", "")
                stat_data.append([
                    Paragraph(_ar(str(clause_display)), ar_cell),
                    Paragraph(_ar(f"{art}: {text_val} — {src}"), ar_cell),
                ])
            stat_tbl = Table(stat_data, repeatRows=1, colWidths=[4.5*cm, 12.5*cm])
            stat_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(stat_tbl)
            story.append(Spacer(1, 0.3*cm))

        # Disclaimer
        story.append(Spacer(1, 0.6*cm))
        story.append(Paragraph(_ar(t_disclaimer or _AR["disclaimer"]), ar_small))

        doc.build(story)
        buf.seek(0)
        return buf

    # ── Helper: Render action buttons for current agent state ─────────────────
    def _render_agent_action_buttons(state: str):
        """Render the correct set of 'what next?' buttons for the current agent state."""
        if state in ("reviewed",):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Generate PDF report", key="btn_pdf_from_reviewed", use_container_width=True):
                    st.session_state.wf_agent_state = "pdf_pending"
                    st.rerun()
            with col2:
                if st.button("💬 Ask me questions about this contract", key="btn_qa_from_reviewed", use_container_width=True):
                    st.session_state.wf_agent_state = "qa_active"
                    st.rerun()

        elif state == "pdf_pending":
            if st.button("💬 Ask me questions about this contract", key="btn_qa_from_pdf", use_container_width=True):
                st.session_state.wf_agent_state = "qa_active"
                st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Upload
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.wf_step == 1:
        _render_stepper(1)
        st.subheader("Upload a document")
        display_name_input = st.text_input("Display Name (optional)", placeholder="e.g., Employment Contract (2025)")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx", "doc"],
            help="Upload the legal document you want to explore.",
        )
        if st.button("Upload & Classify", type="primary", disabled=uploaded_file is None):
            if uploaded_file:
                with st.spinner("Uploading and classifying… Please wait, do not click again."):
                    result = upload_document(
                        uploaded_file,
                        "document",
                        display_name=display_name_input.strip() or None,
                    )
                if result:
                    doc_status = result.get("status", "")
                    doc_id = result.get("document_id", "")
                    if doc_status == "rejected":
                        # Non-legal: skip classification fetch, go to step 2 with rejection data
                        st.session_state.wf_document_id = doc_id
                        st.session_state.wf_classification = {"classification": "non_legal"}
                        st.session_state.wf_is_contract = False
                        st.session_state.wf_step = 2
                        st.rerun()
                    elif doc_id:
                        # Fetch classification from registry
                        cls_data = get_document_classification_api(doc_id)
                        st.session_state.wf_document_id = doc_id
                        st.session_state.wf_classification = cls_data or {"classification": "uncertain"}
                        st.session_state.wf_is_contract = bool((cls_data or {}).get("is_contract", False))
                        st.session_state.wf_step = 2
                        st.rerun()

        # ── Or use an existing document from the library ──────────────────────
        _lib_docs = [d for d in get_documents() if d.get("document_type", "document") == "document"]
        if _lib_docs:
            st.markdown("---")
            st.markdown("**Or select an existing document:**")
            _doc_options = {
                d.get("display_name") or d.get("document_id", "Unknown"): d.get("document_id")
                for d in _lib_docs
            }
            _selected_label = st.selectbox(
                "Choose from library",
                list(_doc_options.keys()),
                key="wf_library_select",
            )
            if st.button("Use selected document →", type="secondary", key="btn_use_library_doc"):
                _selected_id = _doc_options[_selected_label]
                with st.spinner("Loading classification…"):
                    _cls_data = get_document_classification_api(_selected_id)
                st.session_state.wf_document_id = _selected_id
                st.session_state.wf_classification = _cls_data or {"classification": "uncertain"}
                st.session_state.wf_is_contract = bool((_cls_data or {}).get("is_contract", False))
                st.session_state.wf_step = 2
                st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Classification result
    # ═══════════════════════════════════════════════════════════════════════════
    elif st.session_state.wf_step == 2:
        _render_stepper(2)
        cls = st.session_state.wf_classification or {}
        classification = cls.get("classification", "uncertain")
        confidence = cls.get("classification_confidence") or cls.get("confidence")
        method = cls.get("classification_method") or cls.get("method", "—")

        if classification == "non_legal":
            st.error(
                "**Document rejected — not a legal document.**\n\n"
                "Only contracts, NDAs, statutes, and similar legal instruments are accepted."
            )
            if st.button("Upload a different document"):
                _reset_workflow()
                st.rerun()

        elif classification == "legal_contract":
            st.success("**Legal contract detected.** Confirm the detected details, then proceed.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Type", "Contract")
            m2.metric("Confidence", f"{confidence:.0%}" if confidence is not None else "—")
            m3.metric("Method", method)
            st.markdown("")
            st.markdown("**📋 Detected contract details — confirm or adjust before proceeding:**")
            _cls = st.session_state.wf_classification or {}
            detected_type = _cls.get("detected_contract_type") or "employment"
            detected_juris = _cls.get("detected_jurisdiction") or "(none)"
            _ct_options = ["employment", "nda", "msa", "other"]
            _jur_options = ["(none)", "Generic GCC", "KSA", "UAE", "International"]
            col1, col2 = st.columns(2)
            with col1:
                _ct_idx = _ct_options.index(detected_type) if detected_type in _ct_options else 0
                wf_contract_type_sel = st.selectbox(
                    "Contract type", _ct_options, index=_ct_idx, key="wf_contract_type_select",
                    help="Automatically detected from document text. Select 'other' to skip contract analysis and go directly to Document Explorer.",
                )
                if _cls.get("detected_contract_type"):
                    st.caption(f"🤖 Auto-detected: **{_cls['detected_contract_type']}**")
                else:
                    st.caption("⚠️ Auto-detection unavailable")
            with col2:
                _jur_idx = _jur_options.index(detected_juris) if detected_juris in _jur_options else 0
                wf_jurisdiction_sel = st.selectbox(
                    "Jurisdiction", _jur_options, index=_jur_idx, key="wf_jurisdiction_select",
                    help="Inferred from governing law clauses or party references",
                )
                if _cls.get("detected_jurisdiction") and detected_juris != "(none)":
                    st.caption(f"🤖 Auto-detected: **{_cls['detected_jurisdiction']}**")
                else:
                    st.caption("⚠️ Auto-detection unavailable")
            st.session_state.wf_contract_type = wf_contract_type_sel
            st.session_state.wf_jurisdiction = None if wf_jurisdiction_sel == "(none)" else wf_jurisdiction_sel
            if st.button("Continue to Exploration →", type="primary"):
                st.session_state.wf_step = 3
                st.rerun()

        elif classification == "legal_non_contract":
            st.info(
                "**Legal document detected** (not classified as a contract).\n\n"
                "Clause-specific analysis may be limited; general Q&A is fully supported."
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Type", "Legal Doc")
            m2.metric("Confidence", f"{confidence:.0%}" if confidence is not None else "—")
            m3.metric("Method", method)
            if st.button("Continue to Exploration →", type="primary"):
                st.session_state.wf_step = 3
                st.rerun()

        else:  # uncertain
            st.warning(
                "**Classification uncertain.** The document could not be confidently classified.\n\n"
                "You can proceed anyway, but results may vary."
            )
            col_proceed, col_restart = st.columns(2)
            with col_proceed:
                if st.button("Proceed to Exploration →", type="primary"):
                    st.session_state.wf_step = 3
                    st.rerun()
            with col_restart:
                if st.button("Start Over"):
                    _reset_workflow()
                    st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Exploration — Agent state machine (contracts) or direct Q&A
    # ═══════════════════════════════════════════════════════════════════════════
    elif st.session_state.wf_step == 3:
        _render_stepper(3)

        document_id = st.session_state.wf_document_id

        # Classification badge + Start Over
        badge_col, over_col = st.columns([6, 1])
        with badge_col:
            if st.session_state.wf_is_contract:
                st.markdown(
                    '<span style="background:#22c55e;color:white;padding:3px 10px;border-radius:10px;font-size:0.85rem;">CONTRACT</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="background:#3b82f6;color:white;padding:3px 10px;border-radius:10px;font-size:0.85rem;">LEGAL DOC</span>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Document: `{document_id}`")
        with over_col:
            if st.button("Start Over", key="btn_start_over"):
                _reset_workflow()
                st.rerun()

        st.markdown("---")

        # ── Helper: render chat history (handles all message types) ──────────
        def _render_chat_history():
            for turn in st.session_state.explorer_chat_history:
                role = turn.get("role", "user")
                msg_type = turn.get("msg_type", "default")
                with st.chat_message(role):
                    if msg_type == "contract_review":
                        st.markdown(turn.get("content", ""))
                        st.markdown("**📄 Full review details**")
                        _render_contract_review_details(turn.get("review_result", {}))
                    elif msg_type == "agent_prompt":
                        st.markdown(turn.get("content", ""))
                    elif msg_type == "risk_analysis":
                        st.markdown(turn.get("content", ""))
                    elif msg_type == "agent_result":
                        st.markdown(turn.get("content", ""))
                    else:
                        st.write(turn.get("content", ""))
                        if role == "assistant":
                            meta_parts = []
                            ev_score = turn.get("evidence_score")
                            guardrail = turn.get("guardrail_decision")
                            rewritten = turn.get("rewritten_query")
                            status = turn.get("status")
                            if ev_score:
                                score_emoji = {"strong": "🟢", "moderate": "🟡", "weak": "🟠", "none": "🔴"}.get(ev_score, "⚪")
                                meta_parts.append(f"{score_emoji} Evidence: **{ev_score}**")
                            if guardrail:
                                meta_parts.append(f"Guardrail: `{guardrail}`")
                            if status == "refused":
                                meta_parts.append("⚠️ Response refused — insufficient evidence")
                            if rewritten and rewritten != turn.get("original_query"):
                                meta_parts.append(f"*Query rewritten:* {rewritten}")
                            if meta_parts:
                                st.caption(" · ".join(meta_parts))
                            sources = turn.get("sources") or []
                            if sources:
                                with st.expander(f"Sources ({len(sources)})"):
                                    for src in sources[:8]:
                                        pg = src.get("page_number") or src.get("page")
                                        snippet = src.get("text_snippet") or src.get("text") or ""
                                        label = f"Page {pg}" if pg else "Source"
                                        st.markdown(f"**{label}** — {snippet[:300]}{'…' if len(snippet) > 300 else ''}")

        # ── Helper: Q&A conversational chat loop (Agent 4) ───────────────────
        def _run_qa_chat():
            session_col, reset_col = st.columns([5, 1])
            with session_col:
                if st.session_state.explorer_session_id:
                    st.caption(
                        f"Session active · ID: `{st.session_state.explorer_session_id[:12]}…` · "
                        f"{len([m for m in st.session_state.explorer_chat_history if m.get('role') == 'user'])} turn(s)"
                    )
                else:
                    st.caption("No session yet — send a message to start.")
            with reset_col:
                if st.button("🔄 Reset", key="btn_reset_session", help="Clear session history"):
                    if st.session_state.explorer_session_id:
                        delete_chat_session_api(st.session_state.explorer_session_id)
                    st.session_state.explorer_session_id = None
                    st.session_state.explorer_chat_history = []
                    if st.session_state.wf_is_contract:
                        st.session_state.wf_agent_state = "idle"
                        st.session_state.wf_review_done = False
                        st.session_state.wf_review_result = None
                    st.rerun()

            _render_chat_history()

            user_input = st.chat_input("Ask a question about the document…")
            if user_input and user_input.strip():
                user_msg = user_input.strip()
                with st.chat_message("user"):
                    st.write(user_msg)
                st.session_state.explorer_chat_history.append({"role": "user", "content": user_msg})

                if not st.session_state.explorer_session_id:
                    with st.spinner("Starting session…"):
                        sess = create_chat_session(document_id, mode="conversational")
                    if not sess:
                        st.stop()
                    st.session_state.explorer_session_id = sess["session_id"]

                with st.spinner("Thinking…"):
                    resp = send_chat_message(
                        st.session_state.explorer_session_id,
                        user_msg,
                        mode="conversational",
                    )

                if resp is not None:
                    answer = resp.get("answer") or "No answer generated."
                    ev_score = resp.get("evidence_score")
                    guardrail = resp.get("guardrail_decision")
                    status = resp.get("status", "ok")
                    sources = resp.get("sources") or []
                    trace = resp.get("trace") or {}
                    rewritten = trace.get("rewritten_query")
                    with st.chat_message("assistant"):
                        st.write(answer)
                        meta_parts = []
                        if ev_score:
                            score_emoji = {"strong": "🟢", "moderate": "🟡", "weak": "🟠", "none": "🔴"}.get(ev_score, "⚪")
                            meta_parts.append(f"{score_emoji} Evidence: **{ev_score}**")
                        if guardrail:
                            meta_parts.append(f"Guardrail: `{guardrail}`")
                        if status == "refused":
                            meta_parts.append("⚠️ Response refused — insufficient evidence")
                        if rewritten and rewritten != user_msg:
                            meta_parts.append(f"*Query rewritten:* {rewritten}")
                        if meta_parts:
                            st.caption(" · ".join(meta_parts))
                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for src in sources[:8]:
                                    pg = src.get("page_number") or src.get("page")
                                    snippet = src.get("text_snippet") or src.get("text") or ""
                                    label = f"Page {pg}" if pg else "Source"
                                    st.markdown(f"**{label}** — {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
                    st.session_state.explorer_chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "evidence_score": ev_score,
                        "guardrail_decision": guardrail,
                        "status": status,
                        "sources": sources,
                        "rewritten_query": rewritten,
                        "original_query": user_msg,
                    })

        # ════════════════════════════════════════════════════════════════════
        # CONTRACT PATH — Agent state machine
        # ════════════════════════════════════════════════════════════════════
        if st.session_state.wf_is_contract:
            agent_state = st.session_state.wf_agent_state

            # ── Agent 1: Contract Analysis (auto-runs on entry) ───────────
            _REVIEWABLE_TYPES = {"employment", "nda", "msa"}
            if not st.session_state.wf_review_done:
                _selected_ct = st.session_state.get("wf_contract_type", "other")
                if _selected_ct not in _REVIEWABLE_TYPES:
                    # Skip contract review — unsupported type, go straight to Q&A
                    st.session_state.wf_review_result = None
                    st.session_state.wf_review_done = True
                    st.session_state.wf_agent_state = "qa_active"
                    st.session_state.explorer_chat_history.append({
                        "role": "assistant",
                        "content": (
                            f"Document type **{_selected_ct}** is not eligible for contract review. "
                            "You can ask any question about this document below."
                        ),
                        "msg_type": "agent_prompt",
                    })
                    st.rerun()

                st.session_state.wf_agent_state = "reviewing"
                with st.spinner("Running contract analysis…"):
                    review_result = _run_contract_review(
                        document_id,
                        st.session_state.wf_contract_type,
                        st.session_state.wf_jurisdiction,
                    )
                if review_result:
                    st.session_state.wf_review_result = review_result
                    summary_md = _format_review_summary(review_result)
                    st.session_state.explorer_chat_history.append({
                        "role": "assistant",
                        "content": summary_md,
                        "msg_type": "contract_review",
                        "review_result": review_result,
                    })
                    st.session_state.explorer_chat_history.append({
                        "role": "assistant",
                        "content": "✅ Contract review complete. What would you like to do next?",
                        "msg_type": "agent_prompt",
                    })
                    st.session_state.wf_review_done = True
                    st.session_state.wf_agent_state = "reviewed"
                else:
                    # Agent 1 failed — show error, offer PDF (blank) and Q&A
                    st.error("Contract review could not be completed. You can still explore the document via Q&A.")
                    st.session_state.wf_review_done = True
                    st.session_state.wf_agent_state = "reviewed"
                st.rerun()

            # ── Render chat history ───────────────────────────────────────
            _render_chat_history()

            # ── Show action buttons per current state ─────────────────────
            agent_state = st.session_state.wf_agent_state

            if agent_state == "reviewed":
                st.markdown("**What would you like to do next?**")
                _render_agent_action_buttons("reviewed")

            elif agent_state == "pdf_pending":
                # ── Agent 3: PDF ──────────────────────────────────────────
                try:
                    col_en, col_ar = st.columns(2)
                    with col_en:
                        pdf_buf = _generate_pdf_buffer(st.session_state.wf_review_result or {})
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buf,
                            file_name="contract_review_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with col_ar:
                        with st.spinner("Translating to Arabic..."):
                            ar_pdf_buf = _generate_arabic_pdf_buffer(
                                st.session_state.wf_review_result or {}, API_BASE_URL
                            )
                        st.download_button(
                            label="⬇️ تحميل التقرير بالعربية",
                            data=ar_pdf_buf,
                            file_name="contract_review_report_ar.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    if not st.session_state.wf_pdf_answered:
                        st.session_state.explorer_chat_history.append({
                            "role": "assistant",
                            "content": "📄 PDF report generated. Click the buttons above to download in English or Arabic.",
                            "msg_type": "agent_result",
                        })
                        st.session_state.wf_pdf_answered = True
                except Exception as pdf_err:
                    st.error(f"Could not generate PDF: {pdf_err}")
                _render_agent_action_buttons("pdf_pending")

            elif agent_state == "qa_active":
                # ── Agent 4: Q&A ──────────────────────────────────────────
                _run_qa_chat()

        # ════════════════════════════════════════════════════════════════════
        # NON-CONTRACT PATH — Mode toggle + Q&A
        # ════════════════════════════════════════════════════════════════════
        else:
            mode_choice = st.radio(
                "Mode",
                ["Evidence (Strict)", "Conversational (Evidence-Grounded)"],
                index=0 if st.session_state.explorer_chat_mode == "Evidence (Strict)" else 1,
                horizontal=True,
                help=(
                    "**Evidence (Strict)**: stateless RAG answer per query.\n\n"
                    "**Conversational**: session-based with query rewriting, dual retrieval, and evidence guardrails."
                ),
            )
            if mode_choice != st.session_state.explorer_chat_mode:
                if st.session_state.explorer_session_id:
                    delete_chat_session_api(st.session_state.explorer_session_id)
                st.session_state.explorer_chat_mode = mode_choice
                st.session_state.explorer_session_id = None
                st.session_state.explorer_chat_history = []

            if mode_choice == "Evidence (Strict)":
                top_k = st.slider("Max results", min_value=1, max_value=25, value=10)
                query = st.text_input(
                    "Query",
                    placeholder="Where is… / Show clauses related to…",
                    key="strict_query",
                )
                if st.button("Get answer", type="primary", key="btn_answer"):
                    if not query or not query.strip():
                        st.warning("Enter a query.")
                    else:
                        with st.spinner("Generating answer..."):
                            result = explore_answer(document_id, query, top_k=top_k)
                        if result is not None:
                            st.subheader("Answer")
                            st.write(result.get("answer") or "No answer generated.")
                            st.caption(
                                f"Status: {result.get('status', '—')} | Confidence: {result.get('confidence', '—')}"
                            )
                            sources = result.get("sources", []) or []
                            if sources:
                                st.subheader("Sources / citations")
                                for i, src in enumerate(sources[:10], 1):
                                    with st.expander(f"Source {i} — Page {src.get('page_number', 0)}"):
                                        text = src.get("text", "") or ""
                                        st.write(text[:800] + ("..." if len(text) > 800 else ""))
                                        st.caption(
                                            f"Document: {src.get('display_name', src.get('document_id', '—'))}"
                                        )
            else:
                _run_qa_chat()

    if st.session_state.wf_step == 1:
        # ── Document Library ─────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📁 Document Library", expanded=False):
            all_docs = get_documents()
            documents = [d for d in all_docs if d.get("document_type", "document") == "document"]
            if documents:
                for doc in documents:
                    version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
                    doc_title = f"📄 {doc.get('display_name', doc.get('document_id', 'Unknown'))}{version_label}"
                    st.subheader(doc_title)
                    st.write(f"**Display Name:** {doc.get('display_name', 'N/A')}")
                    st.write(f"**Document ID:** {doc.get('document_id', 'N/A')}")
                    st.write(f"**Version:** {doc.get('version', 1)}")
                    if not doc.get('is_latest', True):
                        st.info("⚠️ This is not the latest version")
                    st.write(f"**Original Filename:** {doc.get('original_filename', 'N/A')}")
                    st.write(f"**Chunks:** {doc.get('total_chunks', 0)}")
                    st.write(f"**Pages:** {doc.get('total_pages', 0)}")
                    st.write(f"**Uploaded:** {doc.get('created_at', 'N/A')}")
                    st.markdown("---")
                    with st.form(key=f"rename_{doc.get('document_id')}"):
                        new_name = st.text_input("Rename Document", value=doc.get('display_name', ''),
                                                 key=f"rename_input_{doc.get('document_id')}")
                        if st.form_submit_button("Rename"):
                            if new_name and new_name.strip() != doc.get('display_name'):
                                rename_result = rename_document(doc.get('document_id'), new_name.strip())
                                if rename_result:
                                    st.success("✅ Document renamed successfully!")
                                    st.rerun()
                            else:
                                st.warning("Please enter a different name")
                    st.markdown("---")
                    if st.button("🗑️ Delete this document", key=f"delete_{doc.get('document_id')}", type="secondary"):
                        doc_id = doc.get("document_id")
                        if doc_id:
                            confirm_key = f"confirm_delete_{doc_id}"
                            if not st.session_state.get(confirm_key):
                                st.warning(f"Click delete again to permanently remove {doc.get('display_name', doc_id)}. This cannot be undone.")
                                st.session_state[confirm_key] = True
                            else:
                                if delete_document(doc_id):
                                    st.success("✅ Document deleted successfully.")
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
            else:
                st.info("No documents uploaded yet.")

# -----------------------------------------------------------------------------
# Page: About
# -----------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About Legal Document Intelligence</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## What is RAG (Retrieval-Augmented Generation)?
    
    RAG is a technique where the LLM retrieves relevant documents at query time instead of being trained on client data.
    
    ### How It Works:
    1. **Document Ingestion**: Documents are chunked, embedded, and stored in a vector database
    2. **Query Processing**: User query is embedded and matched against document chunks
    3. **Context Retrieval**: Most relevant chunks are retrieved
    4. **Response Generation**: LLM generates answer using retrieved context only
    
    ### Why RAG Matters:
    - ✅ **No Data Leakage**: Client documents never become training data
    - ✅ **No Retraining**: Model doesn't learn from client data
    - ✅ **Auditable Outputs**: Every answer has source citations
    - ✅ **Legally Safe**: Compliant for regulated environments
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Access vs Training: What's the Difference?
    
    ### Access (What This System Does):
    - LLM **reads** documents during queries
    - Documents are stored locally in vector database
    - LLM uses documents to answer questions
    - **No permanent learning** - documents are not used to update the model
    
    ### Training (What This System Does NOT Do):
    - Model weights are NOT updated with client data
    - Client documents are NOT used to retrain the model
    - No data is sent to external training services
    - Model remains unchanged by client data
    
    ### Compliance Statement:
    > **Client data is never used to retrain the model.**
    > 
    > The model can read documents during a query without learning from or storing them permanently.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## System Capabilities
    
    ### ✅ What This System Can Do:
    - Document ingestion and indexing
    - Contract Review workflow (risks, evidence, executive summary)
    - Contract Comparison workflow
    - Due Diligence Memo workflow
    - Document Explorer (RAG answer + citations, Arabic/English)
    - Clause extraction (verbatim)
    
    ### ❌ What This System Cannot Do:
    - Legal advice or interpretation
    - Law interpretation
    - Outcome prediction
    - Contract drafting
    - Internet access
    - Cross-client data usage
    - Model training on client data
    
    ### ⚠️ Important Limitations:
    This is a **demo MVP** with the following explicit non-capabilities:
    - Not for production use
    - No enterprise-grade security
    - No compliance certifications
    - Demo-only stack
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Technical Architecture
    
    ### Components:
    - **Backend**: FastAPI (Python)
    - **Frontend**: Streamlit (Python)
    - **Vector Database**: FAISS
    - **Embeddings**: Sentence Transformers (multilingual)
    - **LLM**: Ollama (local, private)
    
    ### Data Flow:
    1. Upload → Parse → Chunk → Embed → Store
    2. Query → Embed → Search → Retrieve → Generate
    
    ### Privacy:
    - All processing is local
    - No external API calls (except local Ollama)
    - Documents stored locally
    - No data sent to cloud services
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Demo Script
    
    ### 5-Minute Walkthrough:
    1. **Upload a Contract**: Navigate to Document Explorer, upload a PDF/DOCX
    2. **Document Explorer**: Go to Document Explorer, ask \"Where are the payment terms?\" and review the answer + citations
    3. **Contract Review**: Run Contract Review workflow for risks and clause evidence
    4. **Clause Extraction**: Navigate to Clause Extraction, view extracted clauses
    5. **Contract Comparison**: Compare contract against template
    6. **Due Diligence Memo**: Generate summary of a case file
    7. **Explain RAG**: Review this About page
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Legal Document Intelligence MVP**")
st.sidebar.markdown("Demo-only system")
st.sidebar.markdown("Not for production use")

