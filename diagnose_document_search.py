"""
Diagnostic script to verify document content and search functionality.
Helps identify why RAG search might be missing content.
"""
from pathlib import Path
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.rag_service import RAGService
from backend.config import settings

def diagnose_document(document_id: str, query: str):
    """Diagnose why a query might not be finding content in a document."""
    
    print("=" * 70)
    print(f"DIAGNOSING DOCUMENT SEARCH")
    print("=" * 70)
    print(f"Document ID: {document_id}")
    print(f"Query: {query}")
    print()
    
    # Initialize services
    embedding_service = EmbeddingService()
    embedding_dim = embedding_service.get_embedding_dimension()
    vector_store = VectorStore(embedding_dim)
    rag_service = RAGService(embedding_service, vector_store)
    
    # Load vector store
    vector_store_path = settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
    if not vector_store_path.exists():
        print("❌ Vector store not found! Document may not be ingested.")
        return
    
    try:
        vector_store.load()
        print("✅ Vector store loaded")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return
    
    # Get all chunks for this document
    print("\n1. Checking Document Chunks:")
    print("-" * 70)
    doc_chunks = vector_store.get_chunks_by_document(document_id)
    
    if not doc_chunks:
        print(f"❌ No chunks found for document {document_id}")
        print("   This means the document was not properly ingested.")
        print("   Solution: Re-upload and ingest the document.")
        return
    
    print(f"✅ Found {len(doc_chunks)} chunk(s) for this document")
    
    # Check for keywords in chunks
    print("\n2. Searching for Query Keywords in Chunks:")
    print("-" * 70)
    import re
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'this', 'that', 'these', 'those'}
    query_words = [w for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop_words and len(w) > 2]
    
    print(f"   Query keywords: {query_words}")
    
    matching_chunks = []
    for i, chunk in enumerate(doc_chunks, 1):
        chunk_text = chunk.get('text', '').lower()
        matches = [word for word in query_words if word in chunk_text]
        if matches:
            matching_chunks.append((i, chunk, matches))
            print(f"\n   ✅ Chunk {i} (Page {chunk.get('page_number', 'N/A')}) matches:")
            print(f"      Keywords found: {matches}")
            print(f"      Text preview: {chunk_text[:200]}...")
    
    if not matching_chunks:
        print("   ❌ No chunks contain query keywords!")
        print("   This suggests the content might not be in the document or wasn't extracted properly.")
    else:
        print(f"\n   ✅ Found {len(matching_chunks)} chunk(s) with matching keywords")
    
    # Test semantic search
    print("\n3. Testing Semantic Search:")
    print("-" * 70)
    try:
        query_embedding = embedding_service.embed_text(query)
        results = vector_store.search(
            query_embedding,
            top_k=10,
            document_id_filter=document_id,
            similarity_threshold=None  # Bypass threshold for testing
        )
        
        if results:
            print(f"   ✅ Semantic search found {len(results)} result(s):")
            for i, result in enumerate(results[:5], 1):
                print(f"\n   Result {i}:")
                print(f"      Score: {result['score']:.3f}")
                print(f"      Page: {result['page_number']}")
                print(f"      Text: {result['text'][:150]}...")
        else:
            print("   ❌ Semantic search found no results")
            print("   This might indicate low semantic similarity.")
    except Exception as e:
        print(f"   ❌ Error in semantic search: {e}")
    
    # Test keyword fallback
    print("\n4. Testing Keyword Fallback Search:")
    print("-" * 70)
    try:
        keyword_results = rag_service._keyword_fallback_search(
            query=query,
            document_id=document_id,
            top_k=10
        )
        
        if keyword_results:
            print(f"   ✅ Keyword fallback found {len(keyword_results)} result(s):")
            for i, result in enumerate(keyword_results[:5], 1):
                print(f"\n   Result {i}:")
                print(f"      Score: {result['score']:.3f} (keyword match)")
                print(f"      Page: {result['page_number']}")
                print(f"      Text: {result['text'][:150]}...")
        else:
            print("   ❌ Keyword fallback found no results")
    except Exception as e:
        print(f"   ❌ Error in keyword fallback: {e}")
    
    # Show chunks with ALL keywords (most relevant)
    print("\n5. Finding Chunks with ALL Query Keywords:")
    print("-" * 70)
    all_keyword_chunks = []
    for i, chunk in enumerate(doc_chunks, 1):
        chunk_text = chunk.get('text', '').lower()
        matches = [word for word in query_words if word in chunk_text]
        if len(matches) == len(query_words):  # All keywords present
            all_keyword_chunks.append((i, chunk, matches))
    
    if all_keyword_chunks:
        print(f"   ✅ Found {len(all_keyword_chunks)} chunk(s) with ALL keywords:")
        for i, (chunk_num, chunk, matches) in enumerate(all_keyword_chunks[:3], 1):
            print(f"\n   Chunk {chunk_num} (Page {chunk.get('page_number', 'N/A')}):")
            print(f"      Full text:")
            print(f"      {chunk.get('text', '')}")
            print()
    else:
        print("   ⚠️  No chunks contain ALL keywords")
        print("   This might indicate the clause is split across multiple chunks")
    
    # Test full RAG query with LLM answer
    print("\n6. Testing Full RAG Query with LLM Answer:")
    print("-" * 70)
    try:
        print("   Generating LLM answer (this may take a moment)...")
        result = rag_service.query(
            query=query,
            document_id_filter=document_id,
            generate_response=True  # Generate LLM answer
        )
        
        print(f"\n   Status: {result.get('status')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Sources found: {len(result.get('sources', []))}")
        
        # Show LLM answer
        answer = result.get('answer')
        if answer:
            print(f"\n   📝 LLM GENERATED ANSWER:")
            print("   " + "=" * 66)
            print(f"   {answer}")
            print("   " + "=" * 66)
        else:
            print("\n   ⚠️  No answer generated")
            print(f"   Refusal reason: {result.get('refusal_reason', 'N/A')}")
        
        # Show citation
        citation = result.get('citation')
        if citation:
            print(f"\n   📄 Citation: {citation}")
        
        # Show sources used
        if result.get('sources'):
            print(f"\n   📚 Sources Used ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"\n   Source {i}:")
                print(f"      Page: {source.get('page_number')}")
                print(f"      Score: {source.get('score', 0):.3f}")
                print(f"      Citation: {source.get('citation', 'N/A')}")
                print(f"      Keyword Match: {source.get('keyword_match', False)}")
                print(f"      Text Preview: {source.get('text', '')[:200]}...")
        else:
            print("\n   ⚠️  No sources found")
            print(f"   Refusal reason: {result.get('refusal_reason', 'N/A')}")
        
        # Show hierarchy analysis if available
        hierarchy = result.get('hierarchy_analysis', {})
        if hierarchy:
            print(f"\n   ⚖️  Legal Hierarchy Analysis:")
            if hierarchy.get('has_governing_law'):
                print(f"      ✅ Governing law detected")
            if hierarchy.get('has_conflict'):
                print(f"      ⚠️  Conflict detected between law and contract")
            if hierarchy.get('supremacy_clauses'):
                print(f"      📋 Supremacy clauses found: {len(hierarchy.get('supremacy_clauses', []))}")
        
    except Exception as e:
        print(f"   ❌ Error in RAG query: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY:")
    print("=" * 70)
    
    if not doc_chunks:
        print("❌ CRITICAL: Document not ingested - no chunks found")
        print("   ACTION: Re-upload and ingest the document")
    elif not matching_chunks:
        print("❌ CRITICAL: Content not found in chunks")
        print("   ACTION: Check if document was properly parsed and text extracted")
    elif not results and not keyword_results:
        print("⚠️  WARNING: Search not finding content")
        print("   ACTION: Check similarity thresholds and search parameters")
    else:
        print("✅ Document appears to be properly ingested and searchable")
        print(f"   - {len(doc_chunks)} chunks total")
        print(f"   - {len(matching_chunks)} chunks with keyword matches")
        print(f"   - {len(results)} semantic search results")
        print(f"   - {len(keyword_results)} keyword fallback results")
        print(f"   - RAG Status: {result.get('status') if 'result' in locals() else 'N/A'}")
        
        # Check if exact clause is in top results
        if 'result' in locals() and result.get('sources'):
            top_source = result['sources'][0]
            top_text = top_source.get('text', '').lower()
            has_exact_terms = all(
                word in top_text 
                for word in ['authorized', 'process', 'transferred', 'personal', 'data', 'bound', 'confidentiality']
            )
            if has_exact_terms:
                print("   ✅ Top result contains exact clause terms")
            else:
                print("   ⚠️  Top result may not contain the exact clause")
                print("   💡 The clause might be split across chunks or in a lower-ranked result")
        
        # Check if LLM answer was generated
        if 'result' in locals():
            answer = result.get('answer')
            if answer:
                print(f"\n   ✅ LLM Answer Generated: {len(answer)} characters")
                if 'not specified' in answer.lower() or 'not covered' in answer.lower():
                    print("   ⚠️  WARNING: Answer suggests content not found")
                elif any(word in answer.lower() for word in ['confidentiality', 'authorized', 'bound']):
                    print("   ✅ Answer appears to address the query")
            else:
                print("\n   ⚠️  No LLM answer generated")
                print(f"   Status: {result.get('status')}")
                print(f"   Reason: {result.get('refusal_reason', 'N/A')}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python diagnose_document_search.py <document_id> <query>")
        print("\nExample:")
        print('  python diagnose_document_search.py ef1f5c7d-e055-4f72-be7a-b5fb4a4c6e20 "who ensures persons authorized to process"')
        sys.exit(1)
    
    document_id = sys.argv[1]
    query = " ".join(sys.argv[2:])
    
    diagnose_document(document_id, query)

