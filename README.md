# Legal Document Intelligence MVP

A demo-ready MVP of a Private Legal Document Intelligence system using RAG (Retrieval-Augmented Generation), clause extraction, contract comparison, and bilingual Arabic-English support.

## 🎯 Core Concepts

### RAG (Retrieval-Augmented Generation)
- Documents are retrieved at query time, not trained into the model
- No data leakage - client documents never become training data
- Auditable outputs with source citations
- Legally safe for regulated environments

### Clause Comparison via Embeddings
- Contracts broken into clauses and embedded
- Semantic similarity + text diff for comparison
- No legal reasoning - pure textual comparison

### Access Without Training
- Model reads documents during queries only
- Client data is never used to retrain the model
- Compliant with data privacy requirements

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Ollama installed and running locally
- Ollama model downloaded (e.g., `ollama pull llama3`)

### Installation

1. **Clone and navigate to the project:**
```bash
cd legal-document-LLM-powered-pipeline
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env if needed (defaults should work)
```

5. **Ensure Ollama is running:**
```bash
ollama serve
# In another terminal, verify model is available:
ollama list
```

### Running the Application

**Start the FastAPI backend:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start the Streamlit frontend (in a new terminal):**
```bash
cd frontend
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

## 📁 Project Structure

```
legal-document-LLM-powered-pipeline/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── config.py        # Configuration
│   ├── models/          # Data models
│   ├── services/        # Business logic
│   └── utils/           # Utilities
├── frontend/            # Streamlit UI
├── data/                # Document storage
│   ├── documents/       # Uploaded documents
│   ├── templates/       # Contract templates
│   └── vector_store/    # FAISS indices
└── requirements.txt     # Dependencies
```

## 🎬 Demo Script

### 5-Minute Walkthrough

1. **Upload a Contract**
   - Navigate to "Document Upload"
   - Upload a PDF or DOCX contract
   - System ingests and indexes the document

2. **RAG Search**
   - Go to "RAG Search"
   - Enter a query (e.g., "What are the payment terms?")
   - View results with citations (document, page number)

3. **Clause Extraction**
   - Navigate to "Clause Extraction"
   - Select a document
   - View extracted clauses with page references

4. **Contract Comparison**
   - Go to "Contract Comparison"
   - Select uploaded contract and template
   - View highlighted differences and missing clauses

5. **Case File Summarization**
   - Navigate to "Case Summary"
   - Select a case file
   - View executive summary, timeline, arguments, and open issues

6. **Bilingual Search**
   - Try Arabic query: "ما هي شروط الدفع؟"
   - System retrieves English documents via semantic similarity
   - Response in Arabic

## ⚠️ Important Limitations

This is a **demo MVP** with the following explicit non-capabilities:

- ❌ Legal advice or interpretation
- ❌ Law interpretation
- ❌ Outcome prediction
- ❌ Contract drafting
- ❌ Internet access
- ❌ Cross-client data usage
- ❌ Model training on client data
- ❌ Enterprise-grade security (demo-only)

## 🔧 Configuration

Edit `.env` file to customize:
- Ollama model (`llama3`, `mistral`, etc.)
- Embedding model
- Chunk sizes
- Similarity thresholds

## 📚 Understanding RAG

**How it works:**
1. Documents are chunked and embedded at upload time
2. Queries are embedded and matched against document chunks
3. Relevant chunks are retrieved and passed to LLM
4. LLM generates response using retrieved context only

**Why it's safe:**
- Documents are stored locally in vector database
- LLM only reads documents during queries
- No training data is created from client documents
- All operations are auditable

## 🛠️ Development

### Adding New Features
- Backend services go in `backend/services/`
- API endpoints in `backend/main.py`
- Frontend pages in `frontend/app.py`

### Testing
- Manual testing with sample legal documents
- Verify RAG search accuracy
- Test bilingual search functionality

## 📝 License

Demo MVP - Not for production use.

