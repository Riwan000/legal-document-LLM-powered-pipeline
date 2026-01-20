"""
Streamlit frontend for Legal Document Intelligence MVP.
Multi-page application connecting to FastAPI backend.
"""
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

# Backend API base URL (FastAPI). If you change backend port/host, update this.
API_BASE_URL = "http://localhost:8000"

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


def search_documents(query: str, top_k: int = 5, document_id: Optional[str] = None, generate_response: bool = True) -> Optional[Dict]:
    """Search documents using RAG."""
    try:
        data = {
            "query": query,
            "top_k": top_k,
            "generate_response": generate_response
        }
        if document_id:
            data["document_id"] = document_id
        
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


# -----------------------------------------------------------------------------
# Navigation + routing
# -----------------------------------------------------------------------------
# We use a sidebar selectbox to route between "pages" inside a single Streamlit app.
st.sidebar.title("⚖️ Legal Document Intelligence")
st.sidebar.markdown("---")

# Health gate: stop early if the backend isn't reachable so the UI doesn't spam errors.
if not check_backend_health():
    st.sidebar.error("⚠️ Backend not available. Please start the FastAPI server.")
    st.error("**Backend Connection Error**\n\nPlease ensure the FastAPI backend is running:\n```bash\ncd backend\nuvicorn main:app --reload\n```")
    st.stop()
else:
    st.sidebar.success("✅ Backend connected")

# Page selection (acts like a router).
page = st.sidebar.selectbox(
    "Navigate",
    ["📤 Upload Document", "🔍 RAG Search", "📋 Clause Extraction", "⚖️ Contract Comparison", 
     "📄 Case Summary", "🌐 Bilingual Search", "ℹ️ About"]
)

# -----------------------------------------------------------------------------
# Page: Upload Document
# -----------------------------------------------------------------------------
if page == "📤 Upload Document":
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
# Page: RAG Search
# -----------------------------------------------------------------------------
elif page == "🔍 RAG Search":
    st.markdown('<div class="main-header">🔍 RAG Semantic Search</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Search your documents using natural language queries. The system uses RAG (Retrieval-Augmented Generation)
    to find relevant information and generate answers with citations.
    """)
    
    # Get documents for filtering
    documents = get_documents()
    document_options = {None: "All Documents"}
    for doc in documents:
        # Use display_name, show version if not latest
        display_name = doc.get('display_name', doc.get('document_id', 'Unknown'))
        version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
        label = f"{display_name}{version_label} ({doc.get('total_chunks', 0)} chunks)"
        document_options[doc['document_id']] = label
    
    selected_doc = st.selectbox(
        "Filter by Document (optional)",
        options=list(document_options.keys()),
        format_func=lambda x: document_options[x]
    )
    
    query = st.text_area(
        "Enter your query",
        height=100,
        placeholder="e.g., What are the payment terms? What are the termination conditions?"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", 1, 20, 5)
    with col2:
        generate_response = st.checkbox("Generate LLM response", value=True)
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching documents..."):
                result = search_documents(
                    query=query,
                    top_k=top_k,
                    document_id=selected_doc,
                    generate_response=generate_response
                )
                
                if result:
                    if result.get('answer'):
                        st.subheader("Answer")
                        st.write(result['answer'])
                    
                    if result.get('sources'):
                        st.subheader("Sources")
                        for i, source in enumerate(result['sources'], 1):
                            # Use display_name in citation (never show document_hash)
                            doc_name = source.get('display_name', source.get('document_id', 'Unknown'))
                            with st.expander(f"Source {i} - {doc_name}, Page {source['page_number']} (Score: {source['score']:.2f})"):
                                st.write(f"**Document:** {doc_name}")
                                st.write(f"**Document ID:** {source.get('document_id', 'N/A')}")
                                st.write(f"**Page:** {source['page_number']}")
                                st.write(f"**Text:**")
                                st.write(source['text'])
        else:
            st.warning("Please enter a query")

# -----------------------------------------------------------------------------
# Page: Clause Extraction
# -----------------------------------------------------------------------------
elif page == "📋 Clause Extraction":
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
                    # Get total clauses count - handle different response formats
                    total_clauses = result.get('total_clauses') or result.get('count') or len(result.get('clauses', []))
                    if total_clauses:
                        st.success(f"✅ Extracted {total_clauses} clause(s)")
                    else:
                        st.success(f"✅ Extracted {len(result.get('clauses', []))} clause(s)")
                    
                    # Group by clause type
                    clauses_by_type = {}
                    for clause in result['clauses']:
                        # Handle both dict format and structured clause format
                        if isinstance(clause, dict):
                            clause_type = clause.get('type', 'Unknown')
                            # Handle structured clauses with type enum
                            if isinstance(clause_type, dict):
                                clause_type = clause_type.get('value', 'Unknown')
                            elif hasattr(clause_type, 'value'):
                                clause_type = clause_type.value
                            
                            page_num = clause.get('page_number')
                            # Handle evidence blocks in structured clauses
                            if not page_num and clause.get('evidence'):
                                evidence = clause['evidence'][0] if isinstance(clause['evidence'], list) else clause['evidence']
                                page_num = evidence.get('page', 0) if isinstance(evidence, dict) else 0
                            
                            clause_text = clause.get('text', '')
                            # Handle structured clauses with evidence blocks
                            if not clause_text and clause.get('evidence'):
                                evidence = clause['evidence'][0] if isinstance(clause['evidence'], list) else clause['evidence']
                                if isinstance(evidence, dict):
                                    clause_text = evidence.get('clean_text', '') or evidence.get('raw_text', '')
                            
                            if clause_type not in clauses_by_type:
                                clauses_by_type[clause_type] = []
                            clauses_by_type[clause_type].append({
                                'type': clause_type,
                                'text': clause_text,
                                'page_number': page_num or 0,
                                'clause_id': clause.get('clause_id'),
                                'title': clause.get('title')
                            })
                    
                    for clause_type, clauses in clauses_by_type.items():
                        st.subheader(f"📌 {clause_type} ({len(clauses)} clause(s))")
                        for clause in clauses:
                            expander_label = f"Page {clause['page_number']}"
                            if clause.get('title'):
                                expander_label = f"{clause['title']} - Page {clause['page_number']}"
                            elif clause.get('clause_id'):
                                expander_label = f"{clause['clause_id']} - Page {clause['page_number']}"
                            
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
elif page == "📄 Case Summary":
    st.markdown('<div class="main-header">📄 Case File Summary</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate comprehensive summaries of case files including:
    - Executive summary
    - Timeline of events
    - Key arguments
    - Open issues
    - Source citations
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
            "Select Case File",
            options=list(document_options.keys()),
            format_func=lambda x: document_options[x]
        )
        
        top_k = st.slider("Number of chunks to analyze", 5, 20, 10)
        
        if st.button("Generate Summary", type="primary"):
            # Streaming mode via SSE
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
                    st.error("Streaming failed. Try again or use the non-streaming endpoint.")
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
                            # Final summary is also included in done (if status=ok)
                            pass

                        # Render progressively
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
# Page: Bilingual Search
# -----------------------------------------------------------------------------
elif page == "🌐 Bilingual Search":
    st.markdown('<div class="main-header">🌐 Bilingual Search</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Search documents in Arabic or English. The system uses multilingual embeddings to enable
    cross-language search - Arabic queries can retrieve English documents and vice versa.
    """)
    
    documents = get_documents()
    document_options = {None: "All Documents"}
    for doc in documents:
        # Use display_name, show version if not latest
        display_name = doc.get('display_name', doc.get('document_id', 'Unknown'))
        version_label = f" (v{doc.get('version', 1)})" if not doc.get('is_latest', True) else ""
        label = f"{display_name}{version_label} ({doc.get('total_chunks', 0)} chunks)"
        document_options[doc['document_id']] = label
    
    selected_doc = st.selectbox(
        "Filter by Document (optional)",
        options=list(document_options.keys()),
        format_func=lambda x: document_options[x]
    )
    
    query = st.text_area(
        "Enter your query (Arabic or English)",
        height=100,
        placeholder="e.g., ما هي شروط الدفع؟ or What are the payment terms?"
    )
    
    response_language = st.selectbox(
        "Response Language",
        [None, "ar", "en"],
        format_func=lambda x: "Auto (match query)" if x is None else ("Arabic" if x == "ar" else "English")
    )
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching documents..."):
                result = search_bilingual(query, response_language, selected_doc)
                
                if result:
                    if result.get('query_language'):
                        st.info(f"Detected query language: {result['query_language']}")
                    
                    if result.get('answer'):
                        st.subheader("Answer")
                        st.write(result['answer'])
                    
                    if result.get('sources'):
                        st.subheader("Sources")
                        for i, source in enumerate(result['sources'], 1):
                            # Use display_name in citation (never show document_hash)
                            doc_name = source.get('display_name', source.get('document_id', 'Unknown'))
                            with st.expander(f"Source {i} - {doc_name}, Page {source['page_number']}"):
                                st.write(f"**Document:** {doc_name}")
                                st.write(f"**Document ID:** {source.get('document_id', 'N/A')}")
                                st.write(f"**Text:**")
                                st.write(source['text'])
        else:
            st.warning("Please enter a query")

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
    - Semantic search across documents
    - Clause extraction (verbatim)
    - Contract comparison (textual differences)
    - Case file summarization
    - Bilingual Arabic-English search
    
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
    2. **RAG Search**: Go to RAG Search, ask "What are the payment terms?"
    3. **Clause Extraction**: Navigate to Clause Extraction, view extracted clauses
    4. **Contract Comparison**: Compare contract against template
    5. **Case Summary**: Generate summary of a case file
    6. **Bilingual Search**: Try Arabic query: "ما هي شروط الدفع؟"
    7. **Explain RAG**: Review this About page
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Legal Document Intelligence MVP**")
st.sidebar.markdown("Demo-only system")
st.sidebar.markdown("Not for production use")

