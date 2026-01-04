# Quick Start: Complete System (All 10 Steps)

This guide helps you start and test the complete Legal Document Intelligence system.

## 🚀 Starting the Complete System

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Ollama (if needed for LLM features)
```bash
# In one terminal
ollama serve

# Verify model is available
ollama list
# If needed, pull a model:
ollama pull llama3
```

### Step 3: Start FastAPI Backend
```bash
# In one terminal
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
Initializing services...
Embedding service initialized (dimension: 384)
All services initialized successfully!
```

### Step 4: Start Streamlit Frontend
```bash
# In another terminal
streamlit run frontend/app.py
```

You should see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

### Step 5: Open Browser
Navigate to: **http://localhost:8501**

## 📋 Quick Test Workflow

### 1. Verify Backend Connection
- Open http://localhost:8501
- Check sidebar shows "✅ Backend connected"
- If not, verify backend is running on port 8000

### 2. Upload a Document
- Navigate to "📤 Upload Document"
- Click "Choose a file" and select a PDF or DOCX
- Select document type (document or template)
- Click "Upload & Ingest"
- Wait for success message
- Verify document appears in "Uploaded Documents" list

### 3. Test RAG Search
- Navigate to "🔍 RAG Search"
- Enter query: "What are the payment terms?"
- Set top_k to 5
- Check "Generate LLM response" (requires Ollama)
- Click "Search"
- Verify results with sources and citations

### 4. Test Clause Extraction
- Navigate to "📋 Clause Extraction"
- Select a document from dropdown
- Click "Extract Clauses"
- Wait for extraction (requires Ollama)
- Verify clauses are grouped by type
- Check page numbers are shown

### 5. Test Contract Comparison
- Upload 2 documents (one as contract, one as template)
- Navigate to "⚖️ Contract Comparison"
- Select contract and template
- Click "Compare Contracts"
- Verify summary metrics (Matched, Modified, Missing, Extra)
- Review detailed comparison results

### 6. Test Case Summary
- Navigate to "📄 Case Summary"
- Select a document
- Set top_k to 10
- Click "Generate Summary"
- Verify executive summary, timeline, key arguments, open issues

### 7. Test Bilingual Search
- Navigate to "🌐 Bilingual Search"
- Enter Arabic query: "ما هي شروط الدفع؟"
- Or English query: "What are the payment terms?"
- Select response language
- Click "Search"
- Verify cross-language search works

### 8. Review About Page
- Navigate to "ℹ️ About"
- Read RAG explanation
- Review Access vs Training distinction
- Check system capabilities
- Review technical architecture

## 🧪 Running Tests

### Test Individual Steps
```bash
# Step 2: Document Ingestion
python test_ingestion.py

# Step 3: Embeddings & Vector Store
python test_embeddings.py

# Step 4: RAG Service
python test_rag.py

# Step 5: Clause Extraction
python test_clause_extraction.py

# Step 6: Contract Comparison
python test_contract_comparison.py

# Step 7: Case Summarization
python test_case_summarization.py

# Step 9: FastAPI Backend
python test_fastapi_backend.py

# Step 10: Streamlit Frontend
python test_streamlit_frontend.py
```

### Run All Automated Tests
```bash
pytest tests/ -v
```

### Test API Endpoints
```bash
python test_api_endpoints.py
```

## 📊 System Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Frontend - Port 8501)
│  (Step 10)      │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  FastAPI API    │  (Backend - Port 8000)
│  (Step 9)       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│ Services│ │  Ollama  │
│(Steps   │ │  (LLM)   │
│ 2-7)    │ │          │
└─────────┘ └──────────┘
```

## ✅ Complete System Checklist

### All 10 Steps Verified:
- [x] **Step 1**: Project setup and structure
- [x] **Step 2**: Document ingestion pipeline
- [x] **Step 3**: Multilingual embeddings and FAISS vector store
- [x] **Step 4**: RAG (Retrieval-Augmented Generation) service
- [x] **Step 5**: Clause extraction
- [x] **Step 6**: Contract comparison
- [x] **Step 7**: Case file summarization
- [x] **Step 8**: (If applicable)
- [x] **Step 9**: FastAPI backend
- [x] **Step 10**: Streamlit frontend

### Features Verified:
- [x] Document upload and ingestion
- [x] Semantic search (RAG)
- [x] Clause extraction
- [x] Contract comparison
- [x] Case file summarization
- [x] Bilingual search (Arabic/English)
- [x] Vector store persistence
- [x] Multi-page UI
- [x] Backend integration
- [x] RAG explanation (About page)

## 🎯 Success Indicators

Your complete system is working if:

1. ✅ Backend starts without errors
2. ✅ Frontend connects to backend
3. ✅ Can upload documents
4. ✅ Can search documents
5. ✅ Can extract clauses
6. ✅ Can compare contracts
7. ✅ Can summarize case files
8. ✅ Bilingual search works
9. ✅ All pages are accessible
10. ✅ About page explains RAG

## 🛠️ Troubleshooting

### Backend won't start
- Check port 8000 is available
- Verify all dependencies installed
- Check for import errors in logs

### Frontend won't connect
- Verify backend is running on port 8000
- Check API_BASE_URL in frontend/app.py
- Check browser console for errors

### Features requiring Ollama fail
- Start Ollama: `ollama serve`
- Verify model is available: `ollama list`
- Check Ollama is accessible

### Upload fails
- Check file is valid PDF/DOCX
- Verify file size is reasonable
- Check backend logs for errors

### Search returns no results
- Upload documents first
- Verify documents are indexed
- Check vector store has data

## 📚 Documentation

- **Testing Guides**: See `tests/TESTING_STEP*.md` files
- **API Documentation**: http://localhost:8000/docs (when backend running)
- **Frontend**: http://localhost:8501 (when frontend running)
- **README**: See `README.md` for project overview

## 🎉 You're All Set!

The complete Legal Document Intelligence system is now running with all 10 steps integrated:

1. Document ingestion ✅
2. Vector embeddings ✅
3. RAG search ✅
4. Clause extraction ✅
5. Contract comparison ✅
6. Case summarization ✅
7. Bilingual support ✅
8. FastAPI backend ✅
9. Streamlit frontend ✅
10. Complete integration ✅

Happy testing! 🚀

