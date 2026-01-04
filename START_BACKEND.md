# Quick Start: FastAPI Backend

This guide helps you start and test the FastAPI backend yourself.

## 🚀 Starting the Backend

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
# From project root directory
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
Initializing services...
Embedding service initialized (dimension: 384)
Loaded existing vector store (X vectors)
All services initialized successfully!
```

### Step 3: Access the API

**API Base URL:** `http://localhost:8000`

**Interactive API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📋 Testing Endpoints

### Option 1: Use the Interactive Docs (Easiest)

1. Open http://localhost:8000/docs in your browser
2. You'll see all available endpoints
3. Click "Try it out" on any endpoint
4. Fill in the parameters
5. Click "Execute"
6. See the response

### Option 2: Use curl

```bash
# Health check
curl http://localhost:8000/api/health

# Upload a document
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/documents/sample_test.pdf" \
  -F "document_type=document"

# Search documents
curl -X POST http://localhost:8000/api/search \
  -F "query=What are the payment terms?" \
  -F "top_k=5"

# List all documents
curl http://localhost:8000/api/documents

# Get system stats
curl http://localhost:8000/api/stats
```

### Option 3: Use Python

```python
import requests

base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/api/health")
print(response.json())

# Upload document
with open("data/documents/sample_test.pdf", "rb") as f:
    files = {"file": f}
    data = {"document_type": "document"}
    response = requests.post(f"{base_url}/api/upload", files=files, data=data)
    print(response.json())
    document_id = response.json()["document_id"]

# Search
data = {"query": "payment terms", "top_k": 5}
response = requests.post(f"{base_url}/api/search", data=data)
print(response.json())
```

## 🧪 Quick Test Checklist

### Basic Functionality
- [ ] Server starts without errors
- [ ] Health check returns "healthy"
- [ ] Can access /docs (Swagger UI)
- [ ] Can access /redoc (ReDoc)

### Document Upload
- [ ] Can upload a PDF file
- [ ] Can upload a DOCX file
- [ ] Upload returns document_id
- [ ] File is stored in data/documents/
- [ ] Document is embedded and indexed

### Search
- [ ] Can search with English query
- [ ] Can search with Arabic query
- [ ] Search returns results with sources
- [ ] Can filter by document_id

### Other Endpoints
- [ ] Can list documents
- [ ] Can get system stats
- [ ] Can extract clauses (if Ollama running)
- [ ] Can summarize case files (if Ollama running)
- [ ] Can compare contracts (if files exist)

## 📝 Example Workflow

1. **Start the server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Check health:**
   ```bash
   curl http://localhost:8000/api/health
   ```

3. **Upload a document:**
   ```bash
   curl -X POST http://localhost:8000/api/upload \
     -F "file=@path/to/your/document.pdf" \
     -F "document_type=document"
   ```
   Save the `document_id` from the response.

4. **Search the document:**
   ```bash
   curl -X POST http://localhost:8000/api/search \
     -F "query=What is this document about?" \
     -F "document_id=YOUR_DOCUMENT_ID"
   ```

5. **List all documents:**
   ```bash
   curl http://localhost:8000/api/documents
   ```

## 🔍 Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed
- Check for import errors in console

### "Service not initialized" errors
- Check startup logs for errors
- Verify Ollama is running (if needed)
- Check vector store directory exists

### Upload fails
- Verify file is valid PDF or DOCX
- Check file permissions
- Look at server logs for details

### Search returns no results
- Make sure documents are uploaded first
- Check vector store has data: `curl http://localhost:8000/api/stats`
- Verify document_id is correct

## 📚 API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload and ingest document |
| `/api/search` | POST | Search documents (RAG) |
| `/api/extract-clauses` | POST | Extract contract clauses |
| `/api/compare` | POST | Compare contract vs template |
| `/api/summarize` | POST | Summarize case file |
| `/api/search-bilingual` | POST | Bilingual search |
| `/api/documents` | GET | List all documents |
| `/api/stats` | GET | System statistics |

## 🎯 Next Steps

Once the backend is running:
1. Test all endpoints using `/docs`
2. Upload some test documents
3. Try searching and other features
4. Check the logs for any issues
5. Review the API responses

Happy testing! 🚀

