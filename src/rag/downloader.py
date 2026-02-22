"""Download SEC/FINRA regulatory documents.

Handles both PDF and HTML sources. Implements retry logic and
idempotent downloads (skips if file already exists).
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import DOCUMENT_SOURCES, DOWNLOAD_HEADERS, RAW_DIR, DocumentSource


def download_document(
    source: DocumentSource,
    output_dir: Path = RAW_DIR,
    timeout: int = 60,
) -> Path | None:
    """Download a single document. Returns path to saved file or None on failure.

    Skips download if file already exists (idempotent).
    Retries once with a 3-second delay on failure.
    """
    output_path = output_dir / source.filename
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  [SKIP] {source.name} — already downloaded")
        return output_path

    for attempt in range(2):
        try:
            response = requests.get(
                source.url,
                headers=DOWNLOAD_HEADERS,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            output_path.write_bytes(response.content)
            size_kb = len(response.content) / 1024
            print(f"  [OK]   {source.name} — {size_kb:.0f} KB")
            return output_path

        except requests.RequestException as e:
            if attempt == 0:
                print(f"  [RETRY] {source.name} — {e}")
                time.sleep(3)
            else:
                print(f"  [FAIL] {source.name} — {e}")
                return None

    return None


def download_all(output_dir: Path = RAW_DIR) -> list[Path]:
    """Download all configured documents.

    Returns list of successfully downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(DOCUMENT_SOURCES)} documents to {output_dir}\n")

    downloaded: list[Path] = []
    failed: list[str] = []

    for source in tqdm(DOCUMENT_SOURCES, desc="Downloading", unit="doc"):
        path = download_document(source, output_dir)
        if path is not None:
            downloaded.append(path)
        else:
            failed.append(source.name)

    print(f"\n{'=' * 60}")
    print(f"Downloaded: {len(downloaded)}/{len(DOCUMENT_SOURCES)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")

    return downloaded
