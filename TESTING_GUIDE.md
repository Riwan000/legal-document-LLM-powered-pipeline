# Testing Guide: Document Ingestion Pipeline

This guide explains how to test your document ingestion pipeline to verify:
- ✅ PDF and DOCX file parsing
- ✅ Page number tracking
- ✅ Text chunking with overlap
- ✅ Metadata preservation (page, document ID, chunk index)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Create Sample Test Documents (Optional)
```bash
python tests/create_test_documents.py
```
This creates sample PDF and DOCX files in `data/documents/` for testing.

### Step 3: Run Tests

**Option A: Quick Manual Test** (Recommended)
```bash
# Test with sample documents
python test_ingestion.py

# Or test with your own document
python tests/manual_test_ingestion.py path/to/your/document.pdf
```

**Option B: Automated pytest Tests**
```bash
pytest tests/test_document_ingestion.py -v
```

## 📋 What Each Test Verifies

### 1. File Parsing Tests
- ✅ **PDF Parsing**: Extracts text from PDF files with accurate page numbers
- ✅ **DOCX Parsing**: Extracts text from DOCX files with estimated page numbers
- ✅ **Page Tracking**: Verifies page numbers are sequential and 1-indexed
- ✅ **Text Extraction**: Ensures all pages have non-empty text content

### 2. Chunking Tests
- ✅ **Chunk Creation**: Splits long text into multiple chunks
- ✅ **Overlap**: Verifies consecutive chunks have overlapping content
- ✅ **Metadata**: Checks that page number, document ID, and chunk index are preserved
- ✅ **Multi-page**: Tests chunking across multiple pages

### 3. Metadata Preservation Tests
- ✅ **Document ID**: All chunks have the correct document ID
- ✅ **Page Numbers**: Page numbers are correctly tracked and preserved
- ✅ **Chunk Indices**: Chunk indices are sequential and unique
- ✅ **Text Content**: All chunks contain non-empty text

### 4. Full Pipeline Tests
- ✅ **End-to-End**: Tests the complete ingestion workflow
- ✅ **Error Handling**: Verifies graceful handling of edge cases

## 📊 Expected Test Results

### Manual Test Output
When you run `python test_ingestion.py`, you should see:

```
======================================================================
  DOCUMENT INGESTION PIPELINE - MANUAL TEST SUITE
======================================================================

======================================================================
  TEST 1: File Parsing (PDF/DOCX)
======================================================================
✓ Successfully parsed: sample_test.pdf
✓ Total pages extracted: 3
✓ All page numbers are correctly tracked!

======================================================================
  TEST 2: Text Chunking with Overlap
======================================================================
✓ Total chunks created: 15
✓ Document ID: test-doc-manual-001
✓ Overlap found between consecutive chunks
✓ All chunks have correct metadata!

======================================================================
  TEST 3: Metadata Preservation
======================================================================
✓ Document ID: PASS
✓ Page Numbers: PASS
✓ Chunk Indices: PASS
✓ Text Content: PASS
✓ Metadata Dict: PASS

======================================================================
  TEST 4: Full Document Ingestion Pipeline
======================================================================
✓ Full ingestion pipeline test passed!
✓ Retrieved 15 chunks from document

======================================================================
  TEST SUMMARY
======================================================================
  ✓ PASS - File Parsing
  ✓ PASS - Text Chunking
  ✓ PASS - Metadata Preservation
  ✓ PASS - Full Ingestion

======================================================================
  ✓ ALL TESTS PASSED!
======================================================================
```

### Pytest Output
When you run `pytest tests/test_document_ingestion.py -v`, you should see:

```
tests/test_document_ingestion.py::TestFileParser::test_parse_pdf_with_page_numbers PASSED
tests/test_document_ingestion.py::TestFileParser::test_parse_docx_with_page_numbers PASSED
tests/test_document_ingestion.py::TestFileParser::test_parse_file_pdf PASSED
tests/test_document_ingestion.py::TestFileParser::test_parse_file_docx PASSED
tests/test_document_ingestion.py::TestTextChunker::test_chunk_text_creates_chunks PASSED
tests/test_document_ingestion.py::TestTextChunker::test_chunk_text_preserves_metadata PASSED
tests/test_document_ingestion.py::TestTextChunker::test_chunk_text_has_overlap PASSED
tests/test_document_ingestion.py::TestTextChunker::test_chunk_pages_processes_all_pages PASSED
tests/test_document_ingestion.py::TestDocumentIngestionService::test_ingest_pdf_document PASSED
tests/test_document_ingestion.py::TestDocumentIngestionService::test_ingest_docx_document PASSED
tests/test_document_ingestion.py::TestDocumentIngestionService::test_get_chunks_from_document_preserves_metadata PASSED

============== 11 passed in X.XXs ==============
```

## 🔍 Detailed Test Explanations

### Test 1: File Parsing
**What it does:**
- Parses your PDF or DOCX file
- Extracts text from each page
- Tracks page numbers (1-indexed for PDF, estimated for DOCX)

**What to check:**
- All pages are extracted
- Page numbers start at 1 and are sequential
- Each page has text content

### Test 2: Text Chunking
**What it does:**
- Splits document text into chunks
- Creates overlapping chunks (last part of chunk N appears in chunk N+1)
- Assigns chunk indices sequentially

**What to check:**
- Multiple chunks are created for long documents
- Consecutive chunks have overlapping content
- Chunk indices are sequential (0, 1, 2, ...)

### Test 3: Metadata Preservation
**What it does:**
- Verifies all metadata fields are correctly preserved:
  - `document_id`: Unique identifier for the document
  - `page_number`: Page where chunk appears
  - `chunk_index`: Sequential index of chunk
  - `text`: The actual chunk content
  - `metadata`: Additional metadata dictionary

**What to check:**
- All chunks have the same document ID
- Page numbers match the source pages
- Chunk indices are unique and sequential
- All chunks have non-empty text

### Test 4: Full Ingestion Pipeline
**What it does:**
- Runs the complete ingestion workflow
- Tests the `DocumentIngestionService.ingest_document()` method
- Verifies the response contains correct information

**What to check:**
- Status is "success"
- Pages processed > 0
- Chunks created > 0
- Document ID is generated

## 🛠️ Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'backend'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd legal-document-LLM-powered-pipeline
python test_ingestion.py
```

### Problem: "reportlab not installed"
**Solution:** Install reportlab for PDF creation:
```bash
pip install reportlab
```

### Problem: Tests fail with "No text content found"
**Solution:** 
- Check that your PDF/DOCX file is not corrupted
- Verify the file contains actual text (not just images)
- Try with a different document

### Problem: "No overlap detected"
**Solution:** 
- This is normal if chunks span page boundaries
- Overlap only applies to chunks within the same page
- If all chunks are on different pages, no overlap is expected

### Problem: pytest fixtures fail
**Solution:**
- Make sure `reportlab` is installed: `pip install reportlab`
- The fixtures will skip if dependencies are missing (this is expected)

## 📝 Testing Your Own Documents

To test with your own PDF or DOCX files:

1. **Place your file** in `data/documents/` or any accessible location

2. **Run the manual test:**
```bash
python tests/manual_test_ingestion.py data/documents/your_file.pdf
```

3. **Or use pytest** by modifying the fixtures in `tests/test_document_ingestion.py`

## ✅ Success Criteria

Your pipeline is working correctly if:

1. ✅ **File Parsing**: PDFs and DOCX files are parsed successfully
2. ✅ **Page Numbers**: Page numbers are tracked correctly (1, 2, 3, ...)
3. ✅ **Chunking**: Text is split into chunks with appropriate size
4. ✅ **Overlap**: Consecutive chunks (on same page) have overlapping content
5. ✅ **Metadata**: All chunks have:
   - Correct `document_id`
   - Correct `page_number`
   - Sequential `chunk_index` (0, 1, 2, ...)
   - Non-empty `text`
   - Valid `metadata` dictionary

## 🎯 Next Steps

After verifying your ingestion pipeline works:

1. **Test with real legal documents** to ensure it handles your use case
2. **Check chunk sizes** - adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `backend/config.py` if needed
3. **Monitor performance** - test with large documents to ensure reasonable processing time
4. **Move to Step 3** - Once ingestion is verified, proceed to embedding and vector store setup

## 📚 Additional Resources

- See `tests/README.md` for more detailed testing information
- Check `backend/config.py` for chunking configuration options
- Review `backend/services/document_ingestion.py` for the ingestion logic

