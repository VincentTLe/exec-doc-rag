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


def _find_main_content(soup: BeautifulSoup) -> "Tag | None":
    """Locate the primary content container in an HTML page.

    Strategy: find the largest text-bearing block among candidate containers.
    This avoids picking navigation/login sections on sites like FINRA where
    <main> wraps the entire page.
    """
    # 1. Try domain-specific selectors (FINRA's Drupal layout)
    #    The actual rule text lives inside the largest field--name-body div.
    body_fields = soup.find_all("div", class_=re.compile(r"field--name-body"))
    if body_fields:
        largest = max(body_fields, key=lambda el: len(el.get_text(strip=True)))
        if len(largest.get_text(strip=True)) > 200:
            return largest

    # 2. Look for tab-content containers (FINRA uses Bootstrap tabs)
    tab_content = soup.find("div", class_="tab-content")
    if tab_content:
        panes = tab_content.find_all("div", class_="tab-pane")
        if panes:
            largest_pane = max(panes, key=lambda el: len(el.get_text(strip=True)))
            if len(largest_pane.get_text(strip=True)) > 200:
                return largest_pane

    # 3. Generic selectors — text-formatted fields, article, main
    text_formatted = soup.find("div", class_="text-formatted")
    if text_formatted and len(text_formatted.get_text(strip=True)) > 200:
        return text_formatted

    # 4. Standard HTML5 semantic containers
    for tag in ("article", "main"):
        el = soup.find(tag)
        if el:
            return el

    # 5. Common CMS content divs
    content_div = soup.find(
        "div", class_=re.compile(r"field-items|content|main|body")
    )
    if content_div:
        return content_div

    return soup.body or soup


def parse_html(
    html_path: Path,
    doc_name: str,
) -> list[ParsedPage]:
    """Parse saved HTML file into sections.

    Splits on heading elements (h1-h4) or <strong> tags that act as section markers.
    Each section becomes a ParsedPage.

    Handles FINRA-style layouts where content lives in div.indent_firstpara /
    div.indent_secondpara rather than <p> tags.
    """
    pages: list[ParsedPage] = []

    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [ERROR] Cannot read HTML {html_path}: {e}")
        return pages

    soup = BeautifulSoup(raw_html, "html.parser")
    main_content = _find_main_content(soup)

    if main_content is None:
        return pages

    # Split into sections based on headings
    current_section_title = doc_name
    current_text_parts: list[str] = []
    section_idx = 1

    # Detect if h1-h4 headings meaningfully subdivide the content.
    heading_tags = {"h1", "h2", "h3", "h4"}
    real_headings = [
        h for h in main_content.find_all(heading_tags)
        if h.get_text(strip=True) and len(h.get_text(strip=True)) > 3
        and "no results" not in h.get_text(strip=True).lower()
    ]
    has_headings = len(real_headings) >= 3

    # Track which elements' text we've already captured via a parent div,
    # so we don't double-count when iterating descendants.
    captured_elements: set[int] = set()

    def _is_section_divider(el) -> bool:
        """Check if an element acts as a section heading."""
        if el.name in heading_tags:
            return True
        # Use <strong> tags as dividers when no real headings exist
        if not has_headings and el.name == "strong":
            text = el.get_text(strip=True)
            if text and len(text) < 120:
                if re.match(r"^\.\d{2}\s", text):
                    return True
                if re.match(r"^(\(\w+\)|Rule\s|Section\s|RULE\s)", text):
                    return True
                if text.isupper() and len(text) > 5:
                    return True
                if "---" in text or "Supplementary Material" in text:
                    return True
        return False

    def _is_content_div(el) -> bool:
        """Check if a div is a leaf-level content carrier (e.g., FINRA indent divs)."""
        if el.name != "div":
            return False
        classes = el.get("class", [])
        cls_str = " ".join(classes).lower()

        # FINRA indent_firstpara / indent_secondpara
        if "indent" in cls_str:
            # Only capture leaf indent divs (no child indent divs)
            child_indents = el.find_all(
                "div", class_=re.compile(r"indent"), recursive=False
            )
            if not child_indents:
                return True
            return False

        # Generic content divs
        is_leaf = not el.find(["p", "li", "td", "div", "table"])
        has_content_class = any(
            kw in cls_str for kw in ("paragraph", "text", "field__item")
        )
        if is_leaf or has_content_class:
            # Don't capture if text is already in child p/li
            if not el.find(["p", "li"]):
                return True
        return False

    for element in main_content.descendants:
        # Skip if already captured by a parent content div
        if id(element) in captured_elements:
            continue

        if _is_section_divider(element):
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
                # Mark descendants as captured
                for desc in element.descendants:
                    captured_elements.add(id(desc))

        elif _is_content_div(element):
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                current_text_parts.append(text)
                for desc in element.descendants:
                    captured_elements.add(id(desc))

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
