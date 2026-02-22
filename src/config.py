"""Central configuration for ExecDocRAG.

All paths, model names, document URLs, and hyperparameters in one place.
No secrets should be stored here — use environment variables for API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─── Project paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
TRADES_DIR = DATA_DIR / "synthetic_trades"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories on import
for _dir in [RAW_DIR, CHUNKS_DIR, INDEX_DIR, EVAL_DIR, TRADES_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Model configuration ────────────────────────────────────────────────────

# Embedding model: 384-dim, ~80MB, fast on CPU
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Extractive QA model: ~82MB, <0.5s per passage on CPU
QA_MODEL = "deepset/tinyroberta-squad2"

# ─── Chunking parameters ────────────────────────────────────────────────────

CHUNK_SIZE_TOKENS = 512  # Target chunk size in whitespace-split tokens
CHUNK_OVERLAP_TOKENS = 64  # Overlap between consecutive chunks

# ─── Retrieval parameters ────────────────────────────────────────────────────

TOP_K_DEFAULT = 5
SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity to return

# ─── FRED API ────────────────────────────────────────────────────────────────

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ─── Document sources ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentSource:
    """A regulatory document to download and index."""

    name: str
    url: str
    format: str  # "pdf" or "html"
    filename: str
    max_pages: int | None = None  # None = all pages
    description: str = ""


DOCUMENT_SOURCES: list[DocumentSource] = [
    DocumentSource(
        name="SEC Rule 605 Fact Sheet",
        url="https://www.sec.gov/files/34-96493-fact-sheet.pdf",
        format="pdf",
        filename="sec_rule605_factsheet.pdf",
        description="Overview of Rule 605 execution quality disclosure requirements",
    ),
    DocumentSource(
        name="SEC Rule 605 Final Rule (2024)",
        url="https://www.sec.gov/files/rules/final/2024/34-99679.pdf",
        format="pdf",
        filename="sec_rule605_final_2024.pdf",
        max_pages=80,
        description="Amendments to enhance disclosure of order execution information",
    ),
    DocumentSource(
        name="SEC Rule 606 Final Rule (2018)",
        url="https://www.sec.gov/files/rules/final/2018/34-84528.pdf",
        format="pdf",
        filename="sec_rule606_final_2018.pdf",
        max_pages=100,
        description="Disclosure of order handling information",
    ),
    DocumentSource(
        name="SEC Rule 606 Risk Alert",
        url="https://www.sec.gov/files/reg-nms-rule-606-disclosures-risk-alert.pdf",
        format="pdf",
        filename="sec_rule606_risk_alert.pdf",
        description="Observations related to Regulation NMS Rule 606 disclosures",
    ),
    DocumentSource(
        name="FINRA Rule 5310 - Best Execution",
        url="https://www.finra.org/rules-guidance/rulebooks/finra-rules/5310",
        format="html",
        filename="finra_rule5310.html",
        description="Best execution and interpositioning obligations",
    ),
    DocumentSource(
        name="FINRA Regulatory Notice 15-46",
        url="https://www.finra.org/sites/default/files/notice_doc_file_ref/Notice_Regulatory_15-46.pdf",
        format="pdf",
        filename="finra_notice_15_46.pdf",
        description="Guidance on best execution obligations in equity, options and fixed income markets",
    ),
    DocumentSource(
        name="FINRA Regulatory Notice 21-23",
        url="https://www.finra.org/sites/default/files/2021-06/Regulatory-Notice-21-23.pdf",
        format="pdf",
        filename="finra_notice_21_23.pdf",
        description="FINRA reminds member firms of best execution obligations",
    ),
    DocumentSource(
        name="SEC Regulation NMS Rule 611 Memo",
        url="https://www.sec.gov/spotlight/emsac/memo-rule-611-regulation-nms.pdf",
        format="pdf",
        filename="sec_reg_nms_rule611_memo.pdf",
        description="Trade-through rule (Order Protection Rule) explanation",
    ),
]

# ─── Execution analytics configuration ───────────────────────────────────────


@dataclass(frozen=True)
class TradeGenConfig:
    """Configuration for synthetic trade data generation."""

    n_orders: int = 1500
    seed: int = 42
    symbols: tuple[str, ...] = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "BAC", "GS", "MS", "V", "MA",
        "JNJ", "PFE", "XOM", "CVX", "SPY", "QQQ",
    )
    anomaly_rate: float = 0.08  # 8% of orders are anomalous


# ─── User-Agent for downloads ────────────────────────────────────────────────

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
        "ExecDocRAG/1.0 (academic research)"
    ),
}
