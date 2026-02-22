"""Build the FAISS index from downloaded documents.

Full pipeline: parse -> chunk -> embed -> index -> save
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_DIR, INDEX_DIR, RAW_DIR
from src.rag.chunker import chunk_pages, save_chunks
from src.rag.embedder import Embedder
from src.rag.indexer import FAISSIndex
from src.rag.parser import parse_all_documents

if __name__ == "__main__":
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ExecDocRAG Index Builder")
    print("=" * 60)

    print("\nStep 1: Parsing documents...")
    pages = parse_all_documents(RAW_DIR)
    print(f"  -> {len(pages)} pages parsed")

    print("\nStep 2: Chunking...")
    chunks = chunk_pages(pages)
    save_chunks(chunks, CHUNKS_DIR / "chunks.jsonl")
    print(f"  -> {len(chunks)} chunks created")

    # Print chunk stats
    token_counts = [c.token_count for c in chunks]
    print(f"     Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
    print(f"     Min/Max tokens: {min(token_counts)}/{max(token_counts)}")

    print("\nStep 3: Embedding chunks...")
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, INDEX_DIR / "embeddings.npy")
    print(f"  -> Embedded {embeddings.shape[0]} chunks ({embeddings.shape[1]}-dim)")

    print("\nStep 4: Building FAISS index...")
    index = FAISSIndex(dimension=embedder.dimension)
    index.add(embeddings)
    index.save(INDEX_DIR / "faiss.index")
    print(f"  -> Index built with {index.size} vectors")

    print("\n" + "=" * 60)
    print("Index build complete!")
    print(f"  Chunks: {CHUNKS_DIR / 'chunks.jsonl'}")
    print(f"  Index:  {INDEX_DIR / 'faiss.index'}")
    print("=" * 60)
