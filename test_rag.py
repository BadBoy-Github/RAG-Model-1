"""
Test suite for Document Processor, RAG Tools, and ReAct Agent.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import is_api_key_valid, PDF_FILE_PATH
from document_processor import PDFDocumentProcessor
from rag_tools import RAGTools


def test_offline_components():
    print("=" * 60)
    print("🧪 TEST 1: PDF Document Loading & Chunking")
    print("=" * 60)

    processor = PDFDocumentProcessor(pdf_path=PDF_FILE_PATH)
    pages = processor.load_pdf()
    print(f"✓ Pages loaded: {len(pages)}")
    assert len(pages) > 0, "PDF should have at least 1 page"

    chunks = processor.chunk_document()
    print(f"✓ Chunks created: {len(chunks)}")
    assert len(chunks) > 0, "Document should produce chunks"

    print("\n" + "=" * 60)
    print("🧪 TEST 2: Keyword Search Tool")
    print("=" * 60)
    tools = RAGTools(processor=processor)

    kw_res = tools.lookup_keyword("Paneer")
    print("Keyword search 'Paneer' result snippet:")
    print(kw_res[:300] + "...\n")
    assert "Paneer" in kw_res, "Keyword search should return Paneer"

    print("=" * 60)
    print("🧪 TEST 3: Page Retrieval Tool")
    print("=" * 60)
    page2 = tools.get_page(2)
    print("Page 2 snippet:")
    print(page2[:250] + "...\n")
    assert "NON-VEGETARIAN" in page2 or "Chicken" in page2, "Page 2 should contain non-veg dishes"

    print("=" * 60)
    print("🧪 TEST 4: Document Overview Tool")
    print("=" * 60)
    overview = tools.get_menu_overview()
    print("Overview snippet:")
    print(overview[:250] + "...\n")
    assert "Total Pages: 4" in overview, "Overview should report 4 pages"

    print("✅ All offline document processing and tool tests passed successfully!")
    return processor, tools


def test_online_react(processor, tools):
    print("\n" + "=" * 60)
    print("🧪 TEST 5: Online Gemini Vector Embeddings & ReAct Agent")
    print("=" * 60)

    if not is_api_key_valid():
        print("⚠️ GEMINI_API_KEY is not yet populated in .env.")
        print("Skipping online embedding and ReAct agent test.")
        print("Once you insert your key into .env, rerun: python test_rag.py")
        return

    from react_agent import ReActAgent
    print("✓ Generating / Loading vector embeddings...")
    processor.build_index(verbose=True)

    print("\n✓ Running ReAct agent with test query...")
    agent = ReActAgent(rag_tools=tools, verbose=True)
    result = agent.run("What are the vegetarian paneer dishes and their prices?")
    assert len(result["steps"]) > 0, "Agent should have performed at least 1 step"
    assert result["final_answer"], "Agent should provide a final answer"
    print("\n✅ ReAct Agent test completed successfully!")


if __name__ == "__main__":
    proc, tools = test_offline_components()
    test_online_react(proc, tools)
