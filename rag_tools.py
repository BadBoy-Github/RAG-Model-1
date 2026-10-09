"""
RAG Tools module: Exposes functions for the ReAct Agent to interact with the PDF document.
"""

from typing import Any, Callable, Dict, List
from document_processor import PDFDocumentProcessor


class RAGTools:
    """Encapsulates document retrieval tools for the ReAct Agent."""

    def __init__(self, processor: PDFDocumentProcessor, on_action_callback: Optional[Callable[[str, Dict[str, Any], str], None]] = None):
        self.processor = processor
        self.on_action_callback = on_action_callback

    def _notify(self, action_name: str, args: Dict[str, Any], observation: str):
        if self.on_action_callback:
            self.on_action_callback(action_name, args, observation)

    def search_menu(self, query: str) -> str:
        """
        Search the Amman Hotel menu semantically using vector similarity.
        Best for natural language questions, finding dishes by category,
        ingredients, style, or general description.

        Args:
            query: The search phrase or question (e.g. 'chicken starter platter', 'cold juices', 'paneer curries').
        """
        try:
            results = self.processor.semantic_search(query=query, top_k=3)
            if not results:
                res = "No matching chunks found in the menu document."
            else:
                formatted = [
                    f"--- [Page {r['page']} | Match Score: {r['score']:.2f}] ---\n{r['text']}"
                    for r in results
                ]
                res = "\n\n".join(formatted)
        except Exception as e:
            kw_results = self.processor.keyword_search(query, top_k=3)
            if kw_results:
                formatted = [
                    f"--- [Page {r['page']} (Keyword Match Score: {r['score']})] ---\n{r['text']}"
                    for r in kw_results
                ]
                res = "\n\n".join(formatted)
            else:
                res = f"Error during semantic search: {str(e)}"
        self._notify("search_menu", {"query": query}, res)
        return res

    def lookup_keyword(self, keyword: str) -> str:
        """
        Search the Amman Hotel menu for exact keywords, dish names, or ingredients.
        Use this when searching for a specific item (e.g. 'Tandoori', 'Paneer 65', 'Lime Juice', 'Lollipop').

        Args:
            keyword: The exact word or dish name to search for.
        """
        try:
            results = self.processor.keyword_search(keyword=keyword, top_k=3)
            if not results:
                res = f"No occurrences of '{keyword}' found in the document."
            else:
                formatted = [
                    f"--- [Page {r['page']} | Keyword Score: {r['score']}] ---\n{r['text']}"
                    for r in results
                ]
                res = "\n\n".join(formatted)
        except Exception as e:
            res = f"Error during keyword lookup: {str(e)}"
        self._notify("lookup_keyword", {"keyword": keyword}, res)
        return res

    def get_page(self, page_number: int) -> str:
        """
        Retrieve the complete text of a specific page number from the menu (Pages 1 to 4).
        Use this when you need to inspect the full page or list all items in a section.

        Args:
            page_number: The 1-based page number (1, 2, 3, or 4).
        """
        try:
            page_num = int(page_number)
            content = self.processor.get_page_content(page_num)
            res = f"=== Full Content of Page {page_num} ===\n{content}"
        except Exception as e:
            res = f"Error retrieving page {page_number}: {str(e)}"
        self._notify("get_page", {"page_number": page_number}, res)
        return res

    def get_menu_overview(self) -> str:
        """
        Get high-level overview and metadata of the Amman Hotel menu document,
        including total pages, categories, and item counts.
        """
        try:
            summary = self.processor.get_document_summary()
            page_1 = self.processor.get_page_content(1)
            res = (
                f"Document Overview:\n"
                f"- Total Pages: {summary['total_pages']}\n"
                f"- Total Chunks Indexed: {summary['total_chunks']}\n\n"
                f"Page 1 (Menu Summary & Categories):\n{page_1}"
            )
        except Exception as e:
            res = f"Error retrieving menu overview: {str(e)}"
        self._notify("get_menu_overview", {}, res)
        return res

    def get_tool_callables(self) -> List[Callable]:
        """Returns the list of python callable tools for Gemini native function calling."""
        return [
            self.search_menu,
            self.lookup_keyword,
            self.get_page,
            self.get_menu_overview,
        ]

    def execute_tool_by_name(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool dynamically by name with given arguments."""
        tool_map = {
            "search_menu": self.search_menu,
            "lookup_keyword": self.lookup_keyword,
            "get_page": self.get_page,
            "get_menu_overview": self.get_menu_overview,
        }

        func = tool_map.get(tool_name)
        if not func:
            return f"Error: Tool '{tool_name}' does not exist. Available tools: {list(tool_map.keys())}"

        try:
            return str(func(**args))
        except TypeError as te:
            return f"Error: Invalid arguments {args} for tool '{tool_name}': {te}"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
