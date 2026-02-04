"""
Streamlit frontend for Legal Document Intelligence MVP.
Multi-page application connecting to FastAPI backend.
"""
import os
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
    except:
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
    except:
        return []


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
    st.error("**Backend Connection Error**\n\nPlease ensure the FastAPI backend is running:\n```bash\npython run_backend.py\n```")
    st.stop()
else:
    st.sidebar.success("✅ Backend connected")

# Workflow-first navigation
page = st.sidebar.selectbox(
    "Navigate",
    [
        "📋 Contract Review",
        "⚖️ Contract Comparison",
        "📄 Due Diligence Memo",
        "🔍 Document Explorer",
        "📤 Upload Document",
        "📑 Clause Extraction",
        "ℹ️ About",
    ],
)

# -----------------------------------------------------------------------------
# Page: Contract Review (workflow)
# -----------------------------------------------------------------------------
if page == "📋 Contract Review":
    st.markdown('<div class="main-header">📋 Contract Review</div>', unsafe_allow_html=True)
    st.markdown("Identify risks, missing clauses, and evidence for senior review. No legal advice.")
    st.caption(WORKFLOW_DISCLAIMER)
    documents = get_documents()
    if not documents:
        st.warning("Upload a contract first.")
    else:
        doc_options = {d["document_id"]: d.get("display_name", d["document_id"]) for d in documents}
        contract_id = st.selectbox("Contract", options=list(doc_options.keys()), format_func=lambda x: doc_options[x])
        contract_type = st.selectbox("Contract type", ["employment", "nda", "msa"])
        jurisdiction = st.selectbox("Jurisdiction", [None, "Generic GCC", "KSA", "UAE"], format_func=lambda x: x or "—")
        review_depth = st.selectbox("Review depth", ["standard", "quick"])
        if st.button("Run Contract Review", type="primary"):
            with st.spinner("Running contract review..."):
                result = contract_review(contract_id, contract_type, jurisdiction, review_depth)
            if result:
                if result.get("status") == "failed":
                    err = result.get("error", {})
                    st.error(f"**{err.get('code', 'Error')}:** {err.get('message', '')}")
                    if err.get("details"):
                        st.json(err["details"])
                else:
                    ir = result.get("intermediate_results", {}) or {}
                    resp = ir.get("contract_review.response") or {}
                    if not isinstance(resp, dict) or not resp:
                        st.warning("Workflow completed but no Contract Review response was found in intermediate results.")
                        st.json(result)
                    else:
                        st.success("Contract review completed.")

                        doc_warning = (resp.get("document_classification_warning") or "").strip()
                        if doc_warning:
                            st.warning(f"⚠️ {doc_warning}")

                        # Metadata
                        meta_cols = st.columns(4)
                        meta_cols[0].metric("Workflow ID", str(resp.get("workflow_id", ""))[:12] + "…")
                        meta_cols[1].metric("Document", resp.get("document_id", ""))
                        meta_cols[2].metric("Contract type", resp.get("contract_type", ""))
                        meta_cols[3].metric("Jurisdiction", resp.get("jurisdiction") or "—")

                        st.markdown("---")
                        # Risks table (use display_status: Detected / Detected (Weak Evidence) / Not Detected)
                        risks = resp.get("risks", []) or []
                        st.subheader("Risk table")
                        if not risks:
                            st.info("No risks were identified for the selected profile and evidence.")
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
                                rows.append(
                                    {
                                        "severity": r.get("severity", ""),
                                        "evidence state": _display_status(r.get("status", "")),
                                        "description": r.get("description", ""),
                                        "clause_types": ", ".join(r.get("clause_types", []) or []),
                                        "clauses": clause_label,
                                        "pages": ", ".join([str(p) for p in (r.get("page_numbers", []) or [])]),
                                    }
                                )
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                            st.caption("Weak evidence indicates that a clause was found but may lack commonly expected details.")

                        st.markdown("")
                        # Clauses Not Detected (explicit list)
                        not_detected = resp.get("not_detected_clauses", []) or []
                        contract_type_label = (resp.get("contract_type") or "Contract").strip()
                        if contract_type_label and contract_type_label[0].islower():
                            contract_type_label = contract_type_label[0].upper() + contract_type_label[1:]
                        st.subheader("Clauses Not Detected (Based on " + contract_type_label + " Contract Profile)")
                        st.caption("Global contracts may express obligations implicitly or across multiple provisions. This review surfaces such structures conservatively.")
                        if not_detected:
                            for name in not_detected:
                                st.markdown(f"- {name}")
                            st.caption("Not detected means no supporting evidence was found in the document. This does not confirm absence.")
                        else:
                            st.info("All expected clauses for this profile were detected or had weak evidence.")

                        st.markdown("---")

                        # Evidence blocks
                        evidence = resp.get("evidence", []) or []
                        st.markdown("")
                        st.subheader("Evidence by clause")
                        if not evidence:
                            st.info("No clause evidence blocks were produced.")
                        else:
                            for ev in evidence:
                                if not isinstance(ev, dict):
                                    continue
                                display_name = ev.get("display_name") or ev.get("clause_id", "")
                                page_num = ev.get("page_number", 0)
                                if ev.get("semantic_label"):
                                    heading = f"Section: {ev['semantic_label']} — Page {page_num}"
                                elif ev.get("is_non_contractual"):
                                    heading = f"Section (Non-contractual): {display_name} — Page {page_num}"
                                else:
                                    heading = f"Section (Non-standard) — Page {page_num}"
                                with st.expander(heading):
                                    st.write("**Cleaned text:**")
                                    st.write(ev.get("clean_text", ""))
                                    st.write("**Raw text:**")
                                    st.write(ev.get("raw_text", ""))

                        st.markdown("")
                        # Key Review Observations
                        st.subheader("Key Review Observations")
                        exec_items = resp.get("executive_summary", []) or []
                        if not exec_items:
                            st.info("No executive summary items were produced.")
                        else:
                            for item in exec_items:
                                if isinstance(item, dict):
                                    sev = item.get("severity")
                                    text = item.get("text", "")
                                    if sev:
                                        st.markdown(f"- **{sev}**: {text}")
                                    else:
                                        st.markdown(f"- {text}")
                        st.caption("This review prioritizes evidence recall and conservative interpretation; ambiguity is surfaced rather than resolved.")

                        st.markdown("---")
                        st.caption(resp.get("disclaimer", WORKFLOW_DISCLAIMER))

                        with st.expander("Raw workflow output (JSON)"):
                            st.json(resp)

# -----------------------------------------------------------------------------
# Page: Document Explorer (evidence snippets + RAG answer)
# -----------------------------------------------------------------------------
elif page == "🔍 Document Explorer":
    st.markdown('<div class="main-header">🔍 Document Explorer</div>', unsafe_allow_html=True)
    st.markdown("Locate evidence (snippets) and get a RAG answer with citations within a single document.")
    st.caption(WORKFLOW_DISCLAIMER)
    documents = get_documents()
    if not documents:
        st.warning("Upload a document first.")
    else:
        doc_options = {d["document_id"]: d.get("display_name", d["document_id"]) for d in documents}
        document_id = st.selectbox("Document", options=list(doc_options.keys()), format_func=lambda x: doc_options[x])
        query = st.text_input("Query (e.g. Where is termination notice?)", placeholder="Where is… / Show clauses related to…")
        top_k = st.slider("Max results", min_value=1, max_value=25, value=10)

        tab_evidence, tab_answer = st.tabs(["Evidence (snippets)", "Answer (RAG)"])

        with tab_evidence:
            st.markdown("**Evidence Explorer**: deterministic snippets only (no LLM). Modes: text chunks, extracted clauses, or both.")
            mode = st.radio("Mode", ["both", "text", "clauses"], format_func=lambda x: {"both": "Both (clauses first, then text)", "text": "Text chunks", "clauses": "Extracted clauses"}[x], horizontal=True)
            if st.button("Search evidence", type="primary", key="btn_evidence"):
                if not query or not query.strip():
                    st.warning("Enter a query.")
                else:
                    with st.spinner("Searching evidence..."):
                        result = explore_evidence(document_id, query, top_k=top_k, mode=mode)
                    if result is not None:
                        status = result.get("status")
                        results = result.get("results", [])
                        reason = result.get("reason")
                        if status == "not_found":
                            st.info(reason or "No relevant text or clauses found in the document.")
                        elif results:
                            for i, hit in enumerate(results, 1):
                                src = hit.get("source_type", "chunk")
                                label = f"Page {hit.get('page_number', 0)} — Score {hit.get('score', 0):.2f} — {src}"
                                with st.expander(label):
                                    st.write(hit.get("text_snippet", ""))
                        else:
                            st.info(reason or "No results.")
                        with st.expander("Debug (QA)", expanded=False):
                            st.json(result.get("debug") or {})

        with tab_answer:
            st.markdown("**RAG Answer Explorer**: LLM-generated answer with citations from the document.")
            if st.button("Get answer", type="primary", key="btn_answer"):
                if not query or not query.strip():
                    st.warning("Enter a query.")
                else:
                    with st.spinner("Generating answer..."):
                        result = explore_answer(document_id, query, top_k=top_k)
                    if result is not None:
                        st.subheader("Answer")
                        st.write(result.get("answer") or "No answer generated.")
                        st.caption(f"Status: {result.get('status', '—')} | Confidence: {result.get('confidence', '—')}")
                        sources = result.get("sources", []) or []
                        if sources:
                            st.subheader("Sources / citations")
                            for i, src in enumerate(sources[:10], 1):
                                with st.expander(f"Source {i} — Page {src.get('page_number', 0)}"):
                                    st.write(src.get("text", "")[:800] + ("..." if len(src.get("text", "")) > 800 else ""))
                                    st.caption(f"Document: {src.get('display_name', src.get('document_id', '—'))}")

# -----------------------------------------------------------------------------
# Page: Upload Document
# -----------------------------------------------------------------------------
elif page == "📤 Upload Document":
    st.markdown('<div class="main-header">📤 Document Upload</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload PDF or DOCX documents to the system. Documents will be:
    - Parsed and chunked
    - Embedded and indexed
    - Made searchable via RAG
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc'],
            help="Upload PDF or DOCX files"
        )
        
        document_type = st.radio(
            "Document Type",
            ["document", "template"],
            help="Regular document or firm template"
        )
        
        # Optional display name input
        display_name_input = st.text_input(
            "Display Name (optional)",
            help="User-friendly name for this document. If not provided, filename will be used.",
            placeholder="e.g., Employment Contract (2023)"
        )
        
        if st.button("Upload & Ingest", type="primary"):
            if uploaded_file:
                with st.spinner("Uploading and processing document..."):
                    result = upload_document(
                        uploaded_file, 
                        document_type,
                        display_name=display_name_input.strip() if display_name_input else None
                    )
                    
                    if result:
                        st.success("✅ Document uploaded successfully!")
                        st.json(result)
                    else:
                        st.error("Failed to upload document")
            else:
                st.warning("Please select a file to upload")
    
    with col2:
        st.subheader("Uploaded Documents")
        documents = get_documents()
        
        if documents:
            for doc in documents:
                # Use display_name for expander title, show version if not latest
                version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
                expander_title = f"📄 {doc.get('display_name', doc.get('document_id', 'Unknown'))}{version_label}"
                
                with st.expander(expander_title):
                    st.write(f"**Display Name:** {doc.get('display_name', 'N/A')}")
                    st.write(f"**Document ID:** {doc.get('document_id', 'N/A')}")
                    st.write(f"**Version:** {doc.get('version', 1)}")
                    if not doc.get('is_latest', True):
                        st.info("⚠️ This is not the latest version")
                    st.write(f"**Original Filename:** {doc.get('original_filename', 'N/A')}")
                    st.write(f"**Chunks:** {doc.get('total_chunks', 0)}")
                    st.write(f"**Pages:** {doc.get('total_pages', 0)}")
                    st.write(f"**Uploaded:** {doc.get('created_at', 'N/A')}")
                    
                    # Rename functionality
                    st.markdown("---")
                    with st.form(key=f"rename_{doc.get('document_id')}"):
                        new_name = st.text_input(
                            "Rename Document",
                            value=doc.get('display_name', ''),
                            key=f"rename_input_{doc.get('document_id')}"
                        )
                        if st.form_submit_button("Rename"):
                            if new_name and new_name.strip() != doc.get('display_name'):
                                rename_result = rename_document(doc.get('document_id'), new_name.strip())
                                if rename_result:
                                    st.success("✅ Document renamed successfully!")
                                    st.rerun()
                            else:
                                st.warning("Please enter a different name")
        else:
            st.info("No documents uploaded yet")

# -----------------------------------------------------------------------------
# Page: Clause Extraction
# -----------------------------------------------------------------------------
elif page == "📑 Clause Extraction":
    st.markdown('<div class="main-header">📋 Clause Extraction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Extract contract clauses from uploaded documents. The system identifies clauses verbatim
    with page references. No legal interpretation - pure extraction.
    """)
    
    documents = get_documents()
    if not documents:
        st.warning("No documents available. Please upload a document first.")
    else:
        # Use display_name in selector
        document_options = {}
        for doc in documents:
            display_name = doc.get('display_name', doc.get('document_id', 'Unknown'))
            version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
            label = f"{display_name}{version_label} ({doc.get('total_chunks', 0)} chunks)"
            document_options[doc['document_id']] = label
        
        selected_doc = st.selectbox(
            "Select Document",
            options=list(document_options.keys()),
            format_func=lambda x: document_options[x]
        )
        
        if st.button("Extract Clauses", type="primary"):
            with st.spinner("Extracting clauses (this may take a minute)..."):
                result = extract_clauses(selected_doc)
                
                if result and result.get('clauses'):
                    clauses = result.get('clauses', [])
                    st.success(f"✅ Extracted {len(clauses)} clause(s)")
                    # Group by document_section (new schema)
                    clauses_by_section = {}
                    for clause in clauses:
                        if not isinstance(clause, dict):
                            continue
                        section = clause.get('document_section', 'ambiguous')
                        clause_text = clause.get('verbatim_text', '')
                        page_start = clause.get('page_start', 0)
                        clause_heading = clause.get('clause_heading')
                        clause_id = clause.get('clause_id')
                        
                        if section not in clauses_by_section:
                            clauses_by_section[section] = []
                        clauses_by_section[section].append({
                            'section': section,
                            'text': clause_text,
                            'page_start': page_start,
                            'clause_id': clause_id,
                            'heading': clause_heading
                        })
                    
                    for section, section_clauses in clauses_by_section.items():
                        st.subheader(f"📌 {section} ({len(section_clauses)} clause(s))")
                        for clause in section_clauses:
                            expander_label = f"Page {clause['page_start']}"
                            if clause.get('heading'):
                                expander_label = f"{clause['heading']} - Page {clause['page_start']}"
                            elif clause.get('clause_id'):
                                expander_label = f"{clause['clause_id']} - Page {clause['page_start']}"
                            
                            with st.expander(expander_label):
                                if clause.get('text'):
                                    st.write(clause['text'])
                                else:
                                    st.info("No text available for this clause")
                elif result:
                    # Result exists but no clauses - show message
                    error_msg = result.get('error') or result.get('message', 'No clauses found')
                    st.warning(f"⚠️ {error_msg}")
                else:
                    st.info("No clauses found or extraction failed")

# -----------------------------------------------------------------------------
# Page: Contract Comparison
# -----------------------------------------------------------------------------
elif page == "⚖️ Contract Comparison":
    st.markdown('<div class="main-header">⚖️ Contract Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare an uploaded contract against a firm template. The system identifies:
    - Matched clauses (identical)
    - Modified clauses (differences)
    - Missing clauses (in template but not in contract)
    - Extra clauses (in contract but not in template)
    """)
    st.caption(WORKFLOW_DISCLAIMER)
    
    documents = get_documents()
    if len(documents) < 2:
        st.warning("Need at least 2 documents for comparison. Please upload documents first.")
    else:
        # Use display_name in selector
        document_options = {}
        for doc in documents:
            display_name = doc.get('display_name', doc.get('document_id', 'Unknown'))
            version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
            label = f"{display_name}{version_label} ({doc.get('total_chunks', 0)} chunks)"
            document_options[doc['document_id']] = label
        
        col1, col2 = st.columns(2)
        with col1:
            contract_id = st.selectbox(
                "Select Contract",
                options=list(document_options.keys()),
                format_func=lambda x: document_options[x]
            )
        with col2:
            template_id = st.selectbox(
                "Select Template",
                options=list(document_options.keys()),
                format_func=lambda x: document_options[x]
            )
        
        if st.button("Compare Contracts", type="primary"):
            if contract_id == template_id:
                st.error("Please select different documents for comparison")
            else:
                with st.spinner("Comparing contracts (this may take a few minutes)..."):
                    result = compare_contracts(contract_id, template_id)
                    
                    if result:
                        comparison = result.get('comparison', {})
                        summary = comparison.get('summary', {})
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Matched", summary.get('matched_count', 0))
                        with col2:
                            st.metric("Modified", summary.get('modified_count', 0))
                        with col3:
                            st.metric("Missing", summary.get('missing_count', 0))
                        with col4:
                            st.metric("Extra", summary.get('extra_count', 0))
                        
                        # Detailed results
                        if comparison.get('modified_clauses'):
                            st.subheader("⚠️ Modified Clauses")
                            for mod in comparison['modified_clauses']:
                                with st.expander(f"{mod['template_clause']['type']} (Similarity: {mod['similarity']:.2%})"):
                                    st.write("**Template:**")
                                    st.write(mod['template_clause']['text'])
                                    st.write("**Contract:**")
                                    st.write(mod['contract_clause']['text'])
                        
                        if comparison.get('missing_clauses'):
                            st.subheader("❌ Missing Clauses")
                            for missing in comparison['missing_clauses']:
                                with st.expander(f"{missing['template_clause']['type']}"):
                                    st.write(missing['template_clause']['text'])
                        
                        if comparison.get('extra_clauses'):
                            st.subheader("➕ Extra Clauses")
                            for extra in comparison['extra_clauses']:
                                with st.expander(f"{extra['contract_clause']['type']}"):
                                    st.write(extra['contract_clause']['text'])
                        
                        # Full report
                        st.subheader("Full Report")
                        st.markdown(result.get('report', ''))

# -----------------------------------------------------------------------------
# Page: Case Summary
# -----------------------------------------------------------------------------
elif page == "📄 Due Diligence Memo":
    st.markdown('<div class="main-header">📄 Due Diligence Memo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate a structured, deterministic due diligence memo from case documents:
    - Case spine, executive summary, timeline, key arguments, open issues
    - All sections with mandatory citations. No legal advice.
    """)
    st.caption(WORKFLOW_DISCLAIMER)
    
    documents = get_documents()
    if not documents:
        st.warning("No documents available. Please upload a document first.")
    else:
        # Use display_name in selector
        document_options = {}
        for doc in documents:
            display_name = doc.get('display_name', doc.get('document_id', 'Unknown'))
            version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
            label = f"{display_name}{version_label} ({doc.get('total_chunks', 0)} chunks)"
            document_options[doc['document_id']] = label
        
        selected_doc = st.selectbox(
            "Select Case File",
            options=list(document_options.keys()),
            format_func=lambda x: document_options[x]
        )

        tab_workflow, tab_stream = st.tabs(["Workflow output", "Streaming (advanced)"])

        with tab_workflow:
            st.markdown("Generate a due diligence memo via the workflow endpoint `/api/due-diligence-memo`.")
            if st.button("Generate Due Diligence Memo", type="primary"):
                with st.spinner("Generating due diligence memo..."):
                    result = due_diligence_memo(selected_doc)
                if result:
                    status = result.get("status")
                    error = result.get("error", {}) if isinstance(result.get("error"), dict) else {}
                    final_output = result.get("final_output")
                    intermediate = result.get("intermediate_results", {}) or {}
                    render_output = True

                    if status == "failed":
                        st.error(f"**{error.get('code', 'ERROR')}**: {error.get('message', '')}")
                        if error.get("details"):
                            st.json(error.get("details"))
                        readiness_output = (intermediate.get("due_diligence.readiness") or {}).get("output", {})
                        if isinstance(readiness_output, dict) and readiness_output.get("case_spine_readiness"):
                            st.info("Case spine readiness")
                            st.json(readiness_output.get("case_spine_readiness"))
                        st.info("No memo content was generated.")
                        render_output = False

                    if status == "completed_with_warnings":
                        st.warning("Partial memo generated with warnings.")
                        warnings = []
                        for key, section in intermediate.items():
                            if isinstance(section, dict) and section.get("status") in ("skipped", "failed"):
                                warnings.append(f"{key}: {section.get('error', {}).get('message', 'Section skipped')}")
                        if warnings:
                            st.write("\n".join(warnings))

                    if render_output and not isinstance(final_output, dict):
                        st.warning("Workflow completed but no final output was produced.")
                        st.json(result)
                        render_output = False

                    if render_output:
                        spine = final_output.get("case_spine") or {}
                        with st.expander("Case Spine", expanded=True):
                            st.write(f"**Case:** {spine.get('case_name', 'N/A')}")
                            st.write(f"**Court:** {spine.get('court', 'N/A')}")
                            st.write(f"**Date:** {spine.get('date', 'N/A')}")
                            st.write(f"**Parties:** {', '.join(spine.get('parties', []))}")
                            st.write(f"**Procedural Posture:** {spine.get('procedural_posture', 'N/A')}")
                            if spine.get("core_issues"):
                                st.write("**Core Issues:**")
                                for issue in spine.get("core_issues", []):
                                    st.write(f"- {issue}")

                        st.subheader("Executive Summary")
                        exec_items = final_output.get("executive_summary") or []
                        if exec_items:
                            for item in exec_items:
                                source = (item or {}).get("source", {})
                                st.write((item or {}).get("text", ""))
                                st.caption(f"Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}")
                        else:
                            exec_meta = intermediate.get("due_diligence.executive_summary") or {}
                            exec_warnings = exec_meta.get("warnings") if isinstance(exec_meta, dict) else []
                            st.info(exec_warnings[0] if exec_warnings else "No executive summary items.")

                        st.subheader("Timeline of Events")
                        timeline = final_output.get("timeline") or []
                        if timeline:
                            table = []
                            for ev in timeline:
                                source = (ev or {}).get("source", {})
                                table.append(
                                    {
                                        "Date": (ev or {}).get("date", "N/A"),
                                        "Event": (ev or {}).get("event", ""),
                                        "Source": f"P{source.get('page', 0)} ({(source.get('chunk_id', '') or '')[:12]}...)",
                                    }
                                )
                            st.table(table)
                        else:
                            timeline_meta = intermediate.get("due_diligence.timeline") or {}
                            timeline_warnings = timeline_meta.get("warnings") if isinstance(timeline_meta, dict) else []
                            st.info(timeline_warnings[0] if timeline_warnings else "No timeline events.")

                        st.subheader("Key Arguments")
                        args = final_output.get("key_arguments") or {}
                        claimant = args.get("claimant", []) if isinstance(args, dict) else []
                        defendant = args.get("defendant", []) if isinstance(args, dict) else []
                        if claimant:
                            st.write("**Claimant/Plaintiff**")
                            for arg in claimant:
                                source = (arg or {}).get("source", {})
                                st.write(f"- {(arg or {}).get('text', '')}")
                                st.caption(f"Source: Page {source.get('page', 0)}, Chunk {(source.get('chunk_id', '') or '')[:12]}...")
                        if defendant:
                            st.write("**Defendant/Respondent**")
                            for arg in defendant:
                                source = (arg or {}).get("source", {})
                                st.write(f"- {(arg or {}).get('text', '')}")
                                st.caption(f"Source: Page {source.get('page', 0)}, Chunk {(source.get('chunk_id', '') or '')[:12]}...")
                        if not claimant and not defendant:
                            args_meta = intermediate.get("due_diligence.key_arguments") or {}
                            args_warnings = args_meta.get("warnings") if isinstance(args_meta, dict) else []
                            st.info(args_warnings[0] if args_warnings else "No arguments extracted.")

                        st.subheader("Open Issues")
                        issues = final_output.get("open_issues") or []
                        if issues:
                            for issue in issues:
                                source = (issue or {}).get("source", {})
                                st.write(f"- {(issue or {}).get('text', '')}")
                                st.caption(f"Source: Page {source.get('page', 0)}, Chunk {(source.get('chunk_id', '') or '')[:12]}...")
                        else:
                            issues_meta = intermediate.get("due_diligence.open_issues") or {}
                            issues_warnings = issues_meta.get("warnings") if isinstance(issues_meta, dict) else []
                            st.info(issues_warnings[0] if issues_warnings else "No open issues.")

                        citations = final_output.get("citations") or []
                        if citations:
                            with st.expander("All citations"):
                                for c in citations:
                                    st.write(
                                        f"**{(c or {}).get('chunk_id', '')}** "
                                        f"(Page {(c or {}).get('page', 0)}, Type: {(c or {}).get('chunk_type', 'unknown')})"
                                    )

                        st.caption(WORKFLOW_DISCLAIMER)
                        with st.expander("Raw workflow output (JSON)"):
                            st.json(result)

        with tab_stream:
            st.markdown("Streaming mode uses `/api/summarize/stream` (SSE).")
            if st.button("Stream Summary (SSE)", type="secondary"):
                with st.spinner("Streaming summary (this may take a few minutes)..."):
                    spine_box = st.empty()
                    exec_box = st.empty()
                    timeline_box = st.empty()
                    args_box = st.empty()
                    issues_box = st.empty()
                    citations_box = st.empty()

                    spine = None
                    executive_summary_items = []
                    timeline_events = []
                    claimant_args = []
                    defendant_args = []
                    open_issues = []
                    citations = []

                    stream = summarize_case_file_stream(selected_doc)
                    if stream is None:
                        st.error("Streaming failed. Try the workflow output tab instead.")
                    else:
                        for event_name, payload in stream:
                            if event_name == "error":
                                st.error(f"**Error {payload.get('code', 'UNKNOWN')}:** {payload.get('message', '')}")
                                if payload.get("details"):
                                    st.json(payload["details"])
                                break

                            if event_name == "case_spine":
                                spine = payload
                            elif event_name == "executive_summary_item":
                                executive_summary_items.append(payload)
                            elif event_name == "timeline_event":
                                timeline_events.append(payload)
                            elif event_name == "claimant_argument_item":
                                claimant_args.append(payload)
                            elif event_name == "defendant_argument_item":
                                defendant_args.append(payload)
                            elif event_name == "open_issue_item":
                                open_issues.append(payload)
                            elif event_name == "citations":
                                citations = payload.get("citations", [])
                            elif event_name == "done":
                                pass

                            if spine:
                                with spine_box.container():
                                    with st.expander("Case Spine", expanded=True):
                                        st.write(f"**Case:** {spine.get('case_name', 'N/A')}")
                                        st.write(f"**Court:** {spine.get('court', 'N/A')}")
                                        st.write(f"**Date:** {spine.get('date', 'N/A')}")
                                        st.write(f"**Parties:** {', '.join(spine.get('parties', []))}")
                                        st.write(f"**Procedural Posture:** {spine.get('procedural_posture', 'N/A')}")
                                        if spine.get('core_issues'):
                                            st.write("**Core Issues:**")
                                            for issue in spine['core_issues']:
                                                st.write(f"- {issue}")

                            with exec_box.container():
                                st.subheader("Executive Summary")
                                if executive_summary_items:
                                    for item in executive_summary_items:
                                        source = item.get("source", {})
                                        st.write(item.get("text", ""))
                                        st.caption(f"Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')}")
                                else:
                                    st.write("Waiting for executive summary...")

                            with timeline_box.container():
                                st.subheader("Timeline of Events")
                                if timeline_events:
                                    table = []
                                    for ev in timeline_events:
                                        source = ev.get("source", {})
                                        table.append({
                                            "Date": ev.get("date", "N/A"),
                                            "Event": ev.get("event", ""),
                                            "Source": f"P{source.get('page', 0)} ({source.get('chunk_id', '')[:12]}...)"
                                        })
                                    st.table(table)
                                else:
                                    st.write("Waiting for timeline...")

                            with args_box.container():
                                st.subheader("Key Arguments")
                                if claimant_args:
                                    st.write("**Claimant/Plaintiff Arguments:**")
                                    for arg in claimant_args:
                                        source = arg.get("source", {})
                                        st.write(f"- {arg.get('text', '')}")
                                        st.caption(f"Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')[:12]}...")
                                if defendant_args:
                                    st.write("**Defendant/Respondent Arguments:**")
                                    for arg in defendant_args:
                                        source = arg.get("source", {})
                                        st.write(f"- {arg.get('text', '')}")
                                        st.caption(f"Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')[:12]}...")
                                if not claimant_args and not defendant_args:
                                    st.write("Waiting for arguments (may be empty)...")

                            with issues_box.container():
                                st.subheader("Open Issues")
                                if open_issues:
                                    for issue in open_issues:
                                        source = issue.get("source", {})
                                        st.write(f"- {issue.get('text', '')}")
                                        st.caption(f"Source: Page {source.get('page', 0)}, Chunk {source.get('chunk_id', '')[:12]}...")
                                else:
                                    st.write("Waiting for open issues (may be empty)...")

                            if citations:
                                with citations_box.container():
                                    with st.expander("All Citations"):
                                        for c in citations:
                                            st.write(f"**{c.get('chunk_id', '')}** (Page {c.get('page', 0)}, Type: {c.get('chunk_type', 'unknown')})")

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
    - Document Explorer (evidence-only search, Arabic/English)
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
    1. **Upload a Contract**: Navigate to Upload Document, upload a PDF/DOCX
    2. **Document Explorer**: Go to Document Explorer, ask "Where are the payment terms?" (evidence only)
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

