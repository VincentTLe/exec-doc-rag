"""Shared text cleaning and utility functions."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean extracted text from PDFs/HTML.

    Steps:
    1. Normalize unicode (NFKC)
    2. Fix broken hyphenation at line breaks (e.g., "execu-\\ntion" -> "execution")
    3. Replace multiple whitespace with single space per line
    4. Remove excessive blank lines (keep max 2)
    5. Strip leading/trailing whitespace per line
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Fix broken hyphenation at line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace tabs and multiple spaces with single space (per line)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Remove excessive blank lines (keep max 2 consecutive)
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def sentence_split(text: str) -> list[str]:
    """Split text into sentences.

    Handles common abbreviations (Mr., Mrs., Dr., Inc., Corp., No., vs., etc.)
    and splits on period-space-uppercase, question marks, and paragraph breaks.
    """
    # Protect common abbreviations
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Inc.", "Corp.", "Ltd.", "Co.",
        "Jr.", "Sr.", "vs.", "etc.", "i.e.", "e.g.", "No.", "Vol.",
        "Sec.", "Art.", "Dept.", "Gov.", "U.S.", "S.E.C.",
    ]
    protected = text
    placeholders: dict[str, str] = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    # Split on sentence boundaries
    # Period/question/exclamation followed by space and uppercase letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Also split on double newlines (paragraph breaks)
    expanded: list[str] = []
    for sent in sentences:
        parts = re.split(r"\n{2,}", sent)
        expanded.extend(parts)

    # Restore abbreviations
    result: list[str] = []
    for sent in expanded:
        for placeholder, abbr in placeholders.items():
            sent = sent.replace(placeholder, abbr)
        sent = sent.strip()
        if sent:
            result.append(sent)

    return result


def count_tokens_approx(text: str) -> int:
    """Approximate token count using whitespace splitting.

    ~20% lower than true BPE count, but 100x faster.
    Acceptable for chunking where exact counts don't matter.
    """
    return len(text.split())
