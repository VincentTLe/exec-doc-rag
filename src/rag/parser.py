"""Parse PDF and HTML documents into structured text with metadata.

Uses PyMuPDF (fitz) for PDFs and BeautifulSoup for HTML pages.
Each document is parsed into a list of ParsedPage objects preserving
source metadata for citation traceability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF
from bs4 import BeautifulSoup

from src.common.text_utils import clean_text
from src.config import DOCUMENT_SOURCES, RAW_DIR, DocumentSource


@dataclass
class ParsedPage:
    """A single page/section of a document with metadata."""

    text: str
    source_doc: str  # Human-readable document name
    source_file: str  # Filename
    page_number: int  # 1-indexed page number (or section index for HTML)
    section_title: str  # Detected section heading, or empty string


def _detect_section_heading(text: str) -> str:
    """Heuristic to find the main heading in a block of text.

    Looks for:
    - Lines matching common regulatory patterns (Section, Rule, Article, Part)
    - Lines in ALL CAPS shorter than 100 chars
    - First non-empty line if it's short (<80 chars)
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    for line in lines[:5]:  # Check first 5 lines only
        # Regulatory section patterns
        if re.match(
            r"^(Section|Rule|Article|Part|Chapter|Appendix|RULE|SECTION)\s+[\dIVXivx]",
            line,
        ):
            return line[:120]

        # Roman numeral sections: "I.", "II.", "III.", "IV." etc.
        if re.match(r"^[IVX]+\.\s+\w", line) and len(line) < 100:
            return line[:120]

        # Numbered sections: "1.", "2.3", "(a)", etc.
        if re.match(r"^\d+\.[\d.]*\s+[A-Z]", line) and len(line) < 100:
            return line[:120]

        # ALL CAPS heading
        if line.isupper() and 5 < len(line) < 100:
            return line

    # Fall back to first short line
    if len(lines[0]) < 80:
        return lines[0]

    return ""


def parse_pdf(
    pdf_path: Path,
    doc_name: str,
    max_pages: int | None = None,
) -> list[ParsedPage]:
    """Extract text from PDF using PyMuPDF, page by page.

    Args:
        pdf_path: Path to the PDF file.
        doc_name: Human-readable document name for citations.
        max_pages: Maximum number of pages to extract (None = all).

    Returns:
        List of ParsedPage objects, one per page.
    """
    pages: list[ParsedPage] = []

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        print(f"  [ERROR] Cannot open PDF {pdf_path}: {e}")
        return pages

    n_pages = min(len(doc), max_pages) if max_pages else len(doc)

    for i in range(n_pages):
        page = doc[i]
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        # Skip near-empty pages (headers/footers only)
        if len(text.split()) < 20:
            continue

        section_title = _detect_section_heading(text)

        pages.append(
            ParsedPage(
                text=text,
                source_doc=doc_name,
                source_file=pdf_path.name,
                page_number=i + 1,  # 1-indexed
                section_title=section_title,
            )
        )

    doc.close()
    return pages


def parse_html(
    html_path: Path,
    doc_name: str,
) -> list[ParsedPage]:
    """Parse saved HTML file into sections.

    Splits on heading elements (h1-h4) or <strong> tags that act as section markers.
    Each section becomes a ParsedPage.
    """
    pages: list[ParsedPage] = []

    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [ERROR] Cannot read HTML {html_path}: {e}")
        return pages

    soup = BeautifulSoup(raw_html, "html.parser")

    # Try to find the main content area
    # Order matters: prefer main/section containers over article (which may be
    # just navigation on some sites like FINRA).
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"field-items|content|main|body"))
        or soup.body
        or soup
    )

    if main_content is None:
        return pages

    # Split into sections based on headings
    current_section_title = doc_name
    current_text_parts: list[str] = []
    section_idx = 1

    # Detect if h1-h4 headings meaningfully subdivide the content.
    # If there are <=2 headings, they're likely just the page title —
    # also treat <strong> tags that look like section headers as dividers.
    heading_tags = {"h1", "h2", "h3", "h4"}
    real_headings = [
        h for h in main_content.find_all(heading_tags)
        if h.get_text(strip=True) and len(h.get_text(strip=True)) > 3
        and "no results" not in h.get_text(strip=True).lower()
    ]
    has_headings = len(real_headings) >= 3  # Need multiple headings to use as dividers

    def _is_section_divider(el) -> bool:
        """Check if an element acts as a section heading."""
        if el.name in heading_tags:
            return True
        # Use <strong> tags as dividers when no real headings exist
        # Must look like a title: starts with a number/dot pattern, or is short
        if not has_headings and el.name == "strong":
            text = el.get_text(strip=True)
            if text and len(text) < 120:
                # Supplementary material markers (.01, .02, etc.)
                if re.match(r"^\.\d{2}\s", text):
                    return True
                # Numbered patterns like "(a)", "Rule 5310"
                if re.match(r"^(\(\w+\)|Rule\s|Section\s|RULE\s)", text):
                    return True
                # All-caps short text acting as heading
                if text.isupper() and len(text) > 5:
                    return True
                # Separator lines
                if "---" in text or "Supplementary Material" in text:
                    return True
        return False

    for element in main_content.descendants:
        if _is_section_divider(element):
            # Save previous section if it has content
            if current_text_parts:
                text = clean_text("\n".join(current_text_parts))
                if len(text.split()) >= 20:
                    pages.append(
                        ParsedPage(
                            text=text,
                            source_doc=doc_name,
                            source_file=html_path.name,
                            page_number=section_idx,
                            section_title=current_section_title,
                        )
                    )
                    section_idx += 1

            current_section_title = element.get_text(strip=True)[:120]
            current_text_parts = []

        elif element.name in ("p", "li", "td"):
            text = element.get_text(strip=True)
            if text:
                current_text_parts.append(text)

        elif element.name == "div":
            # Capture text from content divs that don't contain sub-elements
            # (e.g., FINRA's <div class="indent_firstpara">)
            classes = element.get("class", [])
            is_leaf_div = not element.find(["p", "li", "td", "div", "table"])
            has_content_class = any(
                kw in " ".join(classes).lower()
                for kw in ("indent", "paragraph", "text", "field__item")
            )
            if is_leaf_div or has_content_class:
                text = element.get_text(strip=True)
                # Only add if this div's text isn't already captured by child p/li
                if text and len(text) > 30 and not element.find(["p", "li"]):
                    current_text_parts.append(text)

    # Don't forget the last section
    if current_text_parts:
        text = clean_text("\n".join(current_text_parts))
        if len(text.split()) >= 20:
            pages.append(
                ParsedPage(
                    text=text,
                    source_doc=doc_name,
                    source_file=html_path.name,
                    page_number=section_idx,
                    section_title=current_section_title,
                )
            )

    return pages


def _find_source_for_file(filename: str) -> DocumentSource | None:
    """Find the DocumentSource config for a given filename."""
    for source in DOCUMENT_SOURCES:
        if source.filename == filename:
            return source
    return None


def parse_all_documents(raw_dir: Path = RAW_DIR) -> list[ParsedPage]:
    """Parse all downloaded documents in the raw directory.

    Dispatches to parse_pdf or parse_html based on file extension.
    Uses DOCUMENT_SOURCES config for doc_name and max_pages.

    Returns:
        Flat list of all ParsedPage objects across all documents.
    """
    all_pages: list[ParsedPage] = []

    for source in DOCUMENT_SOURCES:
        filepath = raw_dir / source.filename
        if not filepath.exists():
            print(f"  [SKIP] {source.name} — file not found")
            continue

        print(f"  Parsing {source.name}...")

        if source.format == "pdf":
            pages = parse_pdf(filepath, source.name, source.max_pages)
        elif source.format == "html":
            pages = parse_html(filepath, source.name)
        else:
            print(f"  [SKIP] Unknown format: {source.format}")
            continue

        print(f"    -> {len(pages)} pages extracted")
        all_pages.extend(pages)

    print(f"\nTotal: {len(all_pages)} pages from {len(DOCUMENT_SOURCES)} documents")
    return all_pages
