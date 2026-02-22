"""Section-aware text chunking with metadata preservation.

Splits parsed pages into overlapping chunks suitable for embedding.
Preserves all source metadata for citation traceability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.text_utils import count_tokens_approx, sentence_split
from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from src.rag.parser import ParsedPage


@dataclass
class Chunk:
    """A text chunk ready for embedding, with full citation metadata."""

    chunk_id: str  # "{source_file}::p{page}::c{idx}"
    text: str
    source_doc: str  # Human-readable document name
    source_file: str  # Filename
    page_number: int  # Page in the original document
    section_title: str  # Section heading
    chunk_index: int  # Position within the page
    token_count: int  # Approximate whitespace-split token count


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk parsed pages into overlapping segments.

    Strategy:
    1. For each ParsedPage, split text into sentences.
    2. Accumulate sentences until token_count >= chunk_size.
    3. Save chunk, then rewind by overlap tokens for next chunk.
    4. Preserve all metadata from the source ParsedPage.

    Args:
        pages: List of parsed pages to chunk.
        chunk_size: Target chunk size in whitespace-split tokens.
        overlap: Number of overlap tokens between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    all_chunks: list[Chunk] = []

    for page in pages:
        sentences = sentence_split(page.text)
        if not sentences:
            continue

        page_chunks = _chunk_sentences(
            sentences=sentences,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for idx, chunk_text in enumerate(page_chunks):
            token_count = count_tokens_approx(chunk_text)
            chunk_id = f"{page.source_file}::p{page.page_number}::c{idx}"

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_doc=page.source_doc,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    section_title=page.section_title,
                    chunk_index=idx,
                    token_count=token_count,
                )
            )

    return all_chunks


def _chunk_sentences(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Group sentences into chunks respecting size and overlap constraints.

    Returns list of chunk texts.
    """
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens_approx(sentence)

        # If a single sentence exceeds chunk_size, it becomes its own chunk
        if sent_tokens > chunk_size and not current_sentences:
            chunks.append(sentence)
            continue

        # If adding this sentence would exceed the limit, save current chunk
        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(chunk_text)

            # Rewind for overlap: keep trailing sentences up to overlap tokens
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for sent in reversed(current_sentences):
                st = count_tokens_approx(sent)
                if overlap_tokens + st > overlap:
                    break
                overlap_sentences.insert(0, sent)
                overlap_tokens += st

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        # Only add if it has meaningful content (>= 5 tokens)
        if count_tokens_approx(chunk_text) >= 5:
            chunks.append(chunk_text)

    return chunks


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Save chunks as JSONL for reproducibility and inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[Chunk]:
    """Load chunks from JSONL file."""
    chunks: list[Chunk] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    return chunks
