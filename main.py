"""
Main entry point for the ReAct RAG Model.
Provides an interactive command-line interface to ask questions about rag_file.pdf.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import (
    DEFAULT_GEMINI_MODEL,
    PDF_FILE_PATH,
    is_api_key_valid,
)
from document_processor import PDFDocumentProcessor
from rag_tools import RAGTools
from react_agent import ReActAgent


SAMPLE_QUESTIONS = [
    "What chicken dishes are available in the starters and what are their prices?",
    "List all vegetarian paneer curries and their exact prices.",
    "What fresh juices and coolers cost ₹ 120 or less?",
    "Does the menu offer any Chef's Starter Special Platters? List names and prices.",
    "Give an overall summary of the Amman Hotel menu, including sections and total item counts.",
]


def print_banner():
    banner = r"""
╔════════════════════════════════════════════════════════════════════════╗
║             🤖  ReAct Agentic RAG — Amman Hotel Menu AI                ║
║           Reasoning & Action with Gemini + PDF Document Tools          ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_prerequisites() -> bool:
    """Validate that rag_file.pdf exists and alert if API key is missing."""
    if not PDF_FILE_PATH.exists():
        print(f"❌ Error: Required document '{PDF_FILE_PATH.name}' not found at:")
        print(f"   {PDF_FILE_PATH.resolve()}")
        print("Please place 'rag_file.pdf' in the root project folder.")
        return False

    if not is_api_key_valid():
        print("\n" + "!" * 72)
        print("⚠️  GEMINI API KEY REQUIRED IN .env")
        print("!" * 72)
        print("To run the Gemini ReAct agent, please update your '.env' file with:")
        print("    GEMINI_API_KEY=your_actual_gemini_api_key")
        print("\nYou can get a free key from Google AI Studio:")
        print("    https://aistudio.google.com/app/apikey")
        print("!" * 72 + "\n")
        return False

    return True


def interactive_loop(agent: ReActAgent, processor: PDFDocumentProcessor):
    """Run an interactive question-and-answer session."""
    print("✨ ReAct Agent is ready! You can ask questions about the Amman Hotel menu.")
    print("Commands:")
    print("  - Type any question directly (e.g. 'What are the mutton dishes?')")
    print("  - Type 'samples' or 'examples' to see recommended test queries")
    print("  - Type 1-5 to run one of the pre-set sample queries")
    print("  - Type 'overview' to inspect document structure")
    print("  - Type 'reindex' to recompute vector embeddings")
    print("  - Type 'exit', 'quit', or 'q' to stop\n")

    while True:
        try:
            user_input = input("💬 Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("👋 Exiting ReAct RAG system. Have a great day!")
            break

        elif cmd in ("samples", "examples", "help"):
            print("\n📋 Sample Questions:")
            for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                print(f"  [{i}] {q}")
            print()
            continue

        elif cmd.isdigit() and 1 <= int(cmd) <= len(SAMPLE_QUESTIONS):
            question = SAMPLE_QUESTIONS[int(cmd) - 1]
            print(f"\n[Selected Sample Question {cmd}]: {question}")
            agent.run(question)
            continue

        elif cmd == "overview":
            summary = processor.get_document_summary()
            print("\n📊 Document Overview:")
            print(f"  - Total Pages: {summary['total_pages']}")
            print(f"  - Total Chunks: {summary['total_chunks']}")
            for p in summary["pages_overview"]:
                print(f"  - Page {p['page']}: {p['char_count']} chars (Starts with: '{p['first_line']}')")
            print()
            continue

        elif cmd == "reindex":
            print("\n🔄 Recomputing embeddings from scratch...")
            processor.build_index(force_reindex=True, verbose=True)
            print("✅ Reindexing complete.\n")
            continue

        # Regular question
        agent.run(user_input)


def main():
    parser = argparse.ArgumentParser(description="ReAct Agentic RAG for PDF Documents")
    parser.add_argument("--query", "-q", type=str, help="Run a single query directly and exit")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild embeddings index")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help="Gemini model to use")
    args = parser.parse_args()

    print_banner()

    # Step 1: Initialize document processor
    print(f"📄 Target Document: {PDF_FILE_PATH.name}")
    processor = PDFDocumentProcessor(pdf_path=PDF_FILE_PATH)

    # Step 2: Validate API key
    if not is_api_key_valid():
        check_prerequisites()
        # Offer offline inspection
        print("💡 The document parser and offline keyword search can still be tested.")
        print("   Running a local document scan now...")
        pages = processor.load_pdf()
        chunks = processor.chunk_document()
        print(f"   ✓ Extracted {len(pages)} pages")
        print(f"   ✓ Generated {len(chunks)} text chunks")
        print("\nOnce you add your GEMINI_API_KEY to '.env', rerun:")
        print("   python main.py\n")
        return

    # Step 3: Index document with vector embeddings
    try:
        processor.build_index(force_reindex=args.reindex, verbose=True)
    except Exception as e:
        print(f"❌ Failed to build RAG vector index: {e}")
        print("Please check your internet connection and GEMINI_API_KEY in .env.")
        return

    # Step 4: Initialize tools and ReAct agent
    tools = RAGTools(processor=processor)
    agent = ReActAgent(rag_tools=tools, model_name=args.model, verbose=True)

    # Step 5: Execute single query or launch interactive session
    if args.query:
        agent.run(args.query)
    else:
        interactive_loop(agent, processor)


if __name__ == "__main__":
    main()
