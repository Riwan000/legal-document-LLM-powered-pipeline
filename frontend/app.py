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

# Workflow-first navigation (main: contract review, document explorer, upload only)
page = st.sidebar.selectbox(
    "Navigate",
    [
        "📋 Contract Review",
        "🔍 Document Explorer",
        "📤 Upload Document",
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

