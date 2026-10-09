"""
ReAct (Reasoning and Action) Agent implementation using Google GenAI SDK and Chat API.
"""

import sys
import time
from typing import Any, Callable, Dict, List, Optional
from google.genai import types

from config import DEFAULT_GEMINI_MODEL, get_gemini_client
from rag_tools import RAGTools


REACT_SYSTEM_INSTRUCTION = """You are an expert ReAct (Reasoning and Action) Agent for the Amman Hotel document knowledge base.

Your mission is to answer user queries accurately based ONLY on the data retrieved from the document ("rag_file.pdf").

Follow the ReAct framework:
1. THOUGHT & ACTION: Analyze what information is requested. Determine which tool is best:
   - Use `search_menu` for conceptual or natural language queries.
   - Use `lookup_keyword` for specific dish names, ingredients, or items.
   - Use `get_page` for inspecting full pages.
   - Use `get_menu_overview` for category and menu structure questions.
2. OBSERVATION: Reflect on the retrieved facts from the tools.
3. FINAL ANSWER: Once you have gathered the necessary facts, synthesize a clear, helpful, and structured answer:
   - Always include page citations (e.g., "[Page 2]", "[Page 3]").
   - Include exact prices in Indian Rupees (₹) as listed in the menu.
   - If an item or information is not found in the menu, clearly state that it is not available in the document. Never hallucinate dishes or prices not present in the menu.
"""


class ReActStep:
    """Represents a single step in the ReAct trajectory."""

    def __init__(
        self,
        step_number: int,
        thought: Optional[str] = None,
        action_name: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
    ):
        self.step_number = step_number
        self.thought = thought
        self.action_name = action_name
        self.action_args = action_args or {}
        self.observation = observation


class ReActAgent:
    """
    ReAct Agent that combines reasoning (Thought) and tool execution (Action)
    to query the RAG document knowledge base.
    """

    def __init__(
        self,
        rag_tools: RAGTools,
        model_name: str = DEFAULT_GEMINI_MODEL,
        verbose: bool = True,
    ):
        self.rag_tools = rag_tools
        self.model_name = model_name
        self.verbose = verbose
        self.client = None
        self.current_steps: List[ReActStep] = []

        # Wire up the callback from RAGTools to intercept actions and observations
        self.rag_tools.on_action_callback = self._on_tool_executed

    def _ensure_client(self):
        if self.client is None:
            self.client = get_gemini_client()

    def _log(self, text: str):
        if self.verbose:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()

    def _on_tool_executed(self, action_name: str, args: Dict[str, Any], observation: str):
        """Callback invoked whenever a retrieval tool is executed by Gemini."""
        step_num = len(self.current_steps) + 1

        self._log(f"\n--- ReAct Action {step_num} ---")
        self._log(f"⚡ [ACTION]: {action_name}({args})")

        # Format observation preview for terminal readability
        lines = observation.strip().splitlines()
        preview = "\n".join(lines[:12])
        if len(lines) > 12:
            preview += f"\n... [{len(lines) - 12} more lines retrieved]"
        self._log(f"📋 [OBSERVATION]:\n{preview}")

        step = ReActStep(
            step_number=step_num,
            thought="Model invoked tool to gather relevant document context.",
            action_name=action_name,
            action_args=args,
            observation=observation,
        )
        self.current_steps.append(step)

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the ReAct retrieval and reasoning loop for a given question.
        Returns a dictionary containing:
        - 'question': Original user question
        - 'steps': List of ReActStep objects (Thought, Action, Observation)
        - 'final_answer': The synthesized answer
        - 'iterations': Number of actions taken
        """
        self._ensure_client()
        self.current_steps = []

        self._log("\n" + "=" * 70)
        self._log(f"🧠 [USER QUERY]: {question}")
        self._log("=" * 70)
        self._log("🤔 [THOUGHT]: Analyzing question and identifying required document tools...")

        # Models to try with fallback in case of temporary 503 spikes
        models_to_try = [
            self.model_name,
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
        ]
        models_to_try = list(dict.fromkeys(models_to_try))

        last_err = None
        final_answer = ""

        for model in models_to_try:
            for attempt in range(2):
                try:
                    chat = self.client.chats.create(
                        model=model,
                        config=types.GenerateContentConfig(
                            system_instruction=REACT_SYSTEM_INSTRUCTION,
                            tools=self.rag_tools.get_tool_callables(),
                            temperature=0.2,
                        ),
                    )

                    response = chat.send_message(question)
                    self.model_name = model
                    final_answer = response.text if response and response.text else "No response generated."
                    break
                except Exception as ex:
                    last_err = ex
                    err_str = str(ex).lower()
                    if "404" in err_str or "not found" in err_str or "no longer available" in err_str:
                        break  # Try next model immediately
                    if "503" in err_str or "429" in err_str or "unavailable" in err_str:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise ex
            if final_answer:
                break
        else:
            final_answer = f"Error during generation: {last_err}"

        self._log("\n" + "-" * 70)
        self._log(f"🎯 [FINAL ANSWER]:\n{final_answer}")
        self._log("=" * 70 + "\n")

        return {
            "question": question,
            "steps": self.current_steps,
            "final_answer": final_answer,
            "iterations": len(self.current_steps),
        }
