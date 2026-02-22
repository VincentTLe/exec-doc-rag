"""Generative QA using Anthropic Claude API with grounded citations.

Provides an alternative to the extractive QA in answer_builder.py,
using Claude to synthesize answers from retrieved passages while
enforcing strict grounding — answers must cite specific passages
and state "Insufficient data" when information is lacking.

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.rag.retriever import RetrievalResult


# System prompt enforcing grounded, traceable outputs
SYSTEM_PROMPT = """You are a regulatory compliance assistant that answers questions about trade execution rules and best execution obligations.

RULES:
1. Answer ONLY using the provided passages below. Do NOT use any external knowledge.
2. If the passages do not contain sufficient information to answer the question, respond with: "Insufficient data — the retrieved passages do not contain enough information to answer this question."
3. ALWAYS cite your sources using the format [Source: <document name>, p.<page>] after each claim.
4. Be concise and precise. Use regulatory language where appropriate.
5. If multiple passages are relevant, synthesize them into a coherent answer.
6. Never fabricate rules, numbers, or regulatory requirements."""


@dataclass
class GenerativeAnswer:
    """A generated answer with metadata."""

    answer_text: str
    model: str
    passages_used: int
    source_docs: list[str]
    query: str

    def format_for_display(self) -> str:
        """Format the answer for Streamlit display."""
        lines = [
            f"**Answer** (via {self.model}):",
            "",
            self.answer_text,
            "",
            f"*Based on {self.passages_used} retrieved passages from: "
            f"{', '.join(set(self.source_docs))}*",
        ]
        return "\n".join(lines)


def _format_passages_for_prompt(results: list[RetrievalResult]) -> str:
    """Format retrieved passages into a structured context block."""
    parts = []
    for r in results:
        parts.append(
            f"--- Passage {r.rank} ---\n"
            f"Source: {r.chunk.source_doc}, Page {r.chunk.page_number}\n"
            f"Section: {r.chunk.section_title}\n"
            f"Similarity: {r.score:.3f}\n"
            f"Text: {r.chunk.text}\n"
        )
    return "\n".join(parts)


def is_api_available() -> bool:
    """Check if the Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_answer(
    query: str,
    results: list[RetrievalResult],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> GenerativeAnswer:
    """Generate a grounded answer using Claude API.

    Args:
        query: The user's question.
        results: Retrieved passages from the RAG pipeline.
        model: Anthropic model to use.
        max_tokens: Maximum tokens in the response.

    Returns:
        GenerativeAnswer with the synthesized response.

    Raises:
        ImportError: If anthropic package is not installed.
        RuntimeError: If API key is missing or API call fails.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for generative QA. "
            "Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=your-key-here"
        )

    if not results:
        return GenerativeAnswer(
            answer_text="No passages were retrieved for this query.",
            model=model,
            passages_used=0,
            source_docs=[],
            query=query,
        )

    # Build the user message with passages
    passages_text = _format_passages_for_prompt(results)
    user_message = (
        f"PASSAGES:\n{passages_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"Please answer the question using only the passages above. "
        f"Include citations in [Source: <doc>, p.<page>] format."
    )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text
    except Exception as e:
        answer_text = f"Error calling Claude API: {e}"

    return GenerativeAnswer(
        answer_text=answer_text,
        model=model,
        passages_used=len(results),
        source_docs=[r.chunk.source_doc for r in results],
        query=query,
    )
