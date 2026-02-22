"""Tests for the retrieval evaluation module."""

import pytest

from src.rag.chunker import Chunk
from src.rag.evaluation import EvalQuestion, _is_relevant
from src.rag.retriever import RetrievalResult


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="test::p1::c0",
        text="Rule 605 requires market centers to publish monthly execution quality statistics including effective spread and price improvement data.",
        source_doc="SEC Rule 605 Fact Sheet",
        source_file="sec_rule605_factsheet.pdf",
        page_number=1,
        section_title="Overview of Rule 605",
        chunk_index=0,
        token_count=20,
    )


@pytest.fixture
def sample_result(sample_chunk: Chunk) -> RetrievalResult:
    return RetrievalResult(
        chunk=sample_chunk,
        score=0.85,
        rank=1,
    )


@pytest.fixture
def matching_question() -> EvalQuestion:
    return EvalQuestion(
        question="What does Rule 605 require?",
        relevant_doc="SEC Rule 605",
        relevant_section="Overview",
        relevant_keywords=["execution quality", "statistics", "market center"],
        difficulty="easy",
    )


@pytest.fixture
def non_matching_question() -> EvalQuestion:
    return EvalQuestion(
        question="What is FINRA Rule 5310?",
        relevant_doc="FINRA Rule 5310",
        relevant_section="Rule text",
        relevant_keywords=["best execution", "reasonable diligence"],
        difficulty="easy",
    )


def test_is_relevant_matching(
    sample_result: RetrievalResult, matching_question: EvalQuestion
) -> None:
    """A result from the correct doc with matching section should be relevant."""
    assert _is_relevant(sample_result, matching_question) is True


def test_is_relevant_wrong_doc(
    sample_result: RetrievalResult, non_matching_question: EvalQuestion
) -> None:
    """A result from the wrong document should not be relevant."""
    assert _is_relevant(sample_result, non_matching_question) is False


def test_is_relevant_keyword_match(sample_chunk: Chunk) -> None:
    """Keyword matching should work when section doesn't match."""
    result = RetrievalResult(chunk=sample_chunk, score=0.8, rank=1)
    question = EvalQuestion(
        question="Test",
        relevant_doc="SEC Rule 605",
        relevant_section="WRONG SECTION",  # Section won't match
        relevant_keywords=["execution quality", "effective spread"],  # Keywords will match
        difficulty="easy",
    )
    assert _is_relevant(result, question) is True


def test_is_relevant_insufficient_keywords(sample_chunk: Chunk) -> None:
    """Only 1 keyword match (need >= 2) with wrong section should not be relevant."""
    result = RetrievalResult(chunk=sample_chunk, score=0.8, rank=1)
    question = EvalQuestion(
        question="Test",
        relevant_doc="SEC Rule 605",
        relevant_section="WRONG SECTION",
        relevant_keywords=["execution quality", "NONEXISTENT_WORD"],  # Only 1 match
        difficulty="easy",
    )
    assert _is_relevant(result, question) is False
