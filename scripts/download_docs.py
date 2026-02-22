"""Download all SEC/FINRA regulatory documents."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR
from src.rag.downloader import download_all

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = download_all()
    print(f"\nReady: {len(paths)} documents in {RAW_DIR}")
