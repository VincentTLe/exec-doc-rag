"""Extractive answer assembly from retrieved passages.

Uses a fine-tuned extractive QA model (TinyRoBERTa on SQuAD 2.0) to extract
answer spans from retrieved passages. This is a principled design choice:

1. **Traceability**: Every answer is a verbatim quote from source text.
2. **No hallucination**: The model can only select existing text, never generate.
3. **Auditability**: Confidence scores indicate model certainty.
4. **No API cost**: Runs entirely locally on CPU (~82MB model).

This directly addresses the JD's requirement for "traceable/grounded outputs"
and is more appropriate for regulatory document Q&A than generative approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import pipeline as hf_pipeline

from src.config import QA_MODEL
from src.rag.retriever import RetrievalResult


@dataclass
class ExtractedAnswer:
    """A single extracted answer with full citation metadata."""

    answer_text: str
    confidence: float  # QA model confidence [0, 1]
    source_passage: str  # The passage text it was extracted from
    source_doc: str  # Document name
    page_number: int
    section_title: str
    similarity_score: float  # Retrieval cosine similarity


@dataclass
class AnswerResponse:
    """Full response to a user query, with multiple answers and passages."""

    query: str
    top_answer: ExtractedAnswer | None
    all_answers: list[ExtractedAnswer]
    retrieved_passages: list[RetrievalResult]

    def format_for_display(self) -> str:
        """Format as a readable string with citations.

        Returns markdown-formatted string for Streamlit display.
        """
        if not self.top_answer:
            return "No answer found in the retrieved passages. Try rephrasing your question."

        lines: list[str] = []

        # Top answer
        lines.append(f"**Answer:** {self.top_answer.answer_text}")
        lines.append("")
        lines.append(
            f"*Source: {self.top_answer.source_doc}, "
            f"p.{self.top_answer.page_number}"
        )
        if self.top_answer.section_title:
            lines.append(f" | Section: {self.top_answer.section_title}")
        lines.append(f" | Confidence: {self.top_answer.confidence:.1%}*")

        # Additional answers
        if len(self.all_answers) > 1:
            lines.append("")
            lines.append("**Additional relevant excerpts:**")
            for i, ans in enumerate(self.all_answers[1:], start=2):
                lines.append(
                    f"{i}. \"{ans.answer_text}\" "
                    f"— {ans.source_doc}, p.{ans.page_number} "
                    f"(confidence: {ans.confidence:.1%})"
                )

        return "\n".join(lines)


class AnswerBuilder:
    """Builds extractive answers from retrieved passages.

    Uses a question-answering pipeline from HuggingFace transformers
    to extract answer spans from each retrieved passage.
    """

    def __init__(self, model_name: str = QA_MODEL):
        """Load the extractive QA pipeline.

        Uses deepset/tinyroberta-squad2 (~82MB) for fast CPU inference.
        The pipeline handles tokenization, inference, and span extraction.
        """
        self.qa_pipeline = hf_pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU
        )

    def build_answer(
        self,
        query: str,
        results: list[RetrievalResult],
        max_answers: int = 3,
        min_confidence: float = 0.01,
    ) -> AnswerResponse:
        """Build extractive answers from retrieved passages.

        For each retrieved passage:
        1. Run QA pipeline with query + passage as context.
        2. Extract answer span and confidence score.
        3. Attach citation metadata from the chunk.

        Answers are ranked by (confidence * similarity_score) to balance
        retrieval relevance and answer extraction quality.

        Args:
            query: The user's question.
            results: Retrieved passages from the Retriever.
            max_answers: Maximum number of answers to return.
            min_confidence: Minimum QA confidence to include an answer.

        Returns:
            AnswerResponse with top answer, all answers, and passages.
        """
        if not results:
            return AnswerResponse(
                query=query,
                top_answer=None,
                all_answers=[],
                retrieved_passages=[],
            )

        answers: list[ExtractedAnswer] = []

        for result in results:
            try:
                qa_output = self.qa_pipeline(
                    question=query,
                    context=result.chunk.text,
                )
            except Exception:
                # Skip passages that cause errors (too short, etc.)
                continue

            confidence = qa_output["score"]  # type: ignore[index]
            answer_text = qa_output["answer"]  # type: ignore[index]

            if confidence < min_confidence:
                continue
            if not answer_text or not answer_text.strip():
                continue

            answers.append(
                ExtractedAnswer(
                    answer_text=answer_text.strip(),
                    confidence=confidence,
                    source_passage=result.chunk.text,
                    source_doc=result.chunk.source_doc,
                    page_number=result.chunk.page_number,
                    section_title=result.chunk.section_title,
                    similarity_score=result.score,
                )
            )

        # Rank by combined score: confidence * similarity
        answers.sort(
            key=lambda a: a.confidence * a.similarity_score,
            reverse=True,
        )

        top_answers = answers[:max_answers]

        return AnswerResponse(
            query=query,
            top_answer=top_answers[0] if top_answers else None,
            all_answers=top_answers,
            retrieved_passages=results,
        )
