"""Tests for the chunker module."""

import pytest

from src.rag.chunker import Chunk, chunk_pages
from src.rag.parser import ParsedPage


@pytest.fixture
def sample_page() -> ParsedPage:
    """A sample parsed page with enough text for multiple chunks."""
    text = (
        "Rule 605 requires market centers to publish monthly reports "
        "containing standardized execution quality statistics. "
        "These reports must include information about covered orders, "
        "which are defined as any market order or limit order received "
        "during regular trading hours. "
        "The rule was originally adopted in 2000 as Rule 11Ac1-5 "
        "and was later redesignated as Rule 605 of Regulation NMS. "
        "In 2024, the SEC adopted amendments to modernize and enhance "
        "the execution quality disclosure framework. "
        "The amendments expand the scope of entities required to report "
        "and update the metrics that must be disclosed. "
        "Key metrics include effective spread, realized spread, "
        "price improvement statistics, and execution speed measures. "
        "Market centers must calculate these metrics on a security-by-security "
        "basis for different order types and size categories. "
        "The information helps investors evaluate the quality of executions "
        "they receive from different market centers and broker-dealers."
    )
    return ParsedPage(
        text=text,
        source_doc="SEC Rule 605 Fact Sheet",
        source_file="sec_rule605_factsheet.pdf",
        page_number=1,
        section_title="Overview",
    )


@pytest.fixture
def short_page() -> ParsedPage:
    """A page shorter than one chunk."""
    return ParsedPage(
        text="This is a very short page with minimal content.",
        source_doc="Test Doc",
        source_file="test.pdf",
        page_number=1,
        section_title="Short",
    )


@pytest.fixture
def empty_page() -> ParsedPage:
    return ParsedPage(
        text="",
        source_doc="Empty Doc",
        source_file="empty.pdf",
        page_number=1,
        section_title="",
    )


def test_chunk_produces_output(sample_page: ParsedPage) -> None:
    """Chunking a page with content should produce at least one chunk."""
    chunks = chunk_pages([sample_page], chunk_size=50, overlap=10)
    assert len(chunks) >= 1


def test_chunk_preserves_metadata(sample_page: ParsedPage) -> None:
    """Chunk metadata should match source page metadata."""
    chunks = chunk_pages([sample_page], chunk_size=50, overlap=10)
    for chunk in chunks:
        assert chunk.source_doc == sample_page.source_doc
        assert chunk.source_file == sample_page.source_file
        assert chunk.page_number == sample_page.page_number
        assert chunk.section_title == sample_page.section_title


def test_chunk_ids_are_unique(sample_page: ParsedPage) -> None:
    """All chunk IDs should be unique."""
    chunks = chunk_pages([sample_page], chunk_size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_token_count_reasonable(sample_page: ParsedPage) -> None:
    """Chunks should not wildly exceed the target size."""
    chunk_size = 50
    chunks = chunk_pages([sample_page], chunk_size=chunk_size, overlap=10)
    for chunk in chunks:
        # Allow 50% overshoot (sentences aren't perfectly splittable)
        assert chunk.token_count <= chunk_size * 1.5 + 20


def test_short_page_produces_single_chunk(short_page: ParsedPage) -> None:
    """Pages shorter than chunk_size should produce exactly one chunk."""
    chunks = chunk_pages([short_page], chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_empty_page_produces_no_chunks(empty_page: ParsedPage) -> None:
    """Empty pages should not produce chunks."""
    chunks = chunk_pages([empty_page], chunk_size=500, overlap=50)
    assert len(chunks) == 0


def test_multiple_pages_chunked(
    sample_page: ParsedPage, short_page: ParsedPage
) -> None:
    """Chunking multiple pages produces chunks from both."""
    chunks = chunk_pages([sample_page, short_page], chunk_size=50, overlap=10)
    sources = {c.source_doc for c in chunks}
    assert len(sources) == 2


def test_chunk_text_not_empty(sample_page: ParsedPage) -> None:
    """No chunk should have empty text."""
    chunks = chunk_pages([sample_page], chunk_size=50, overlap=10)
    for chunk in chunks:
        assert chunk.text.strip() != ""
        assert chunk.token_count > 0
