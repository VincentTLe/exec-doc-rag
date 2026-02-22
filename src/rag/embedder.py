"""Sentence-transformer embedding wrapper.

Uses all-MiniLM-L6-v2 (384-dim, ~80MB) for encoding chunks and queries.
All embeddings are L2-normalized for cosine similarity via FAISS IndexFlatIP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.rag.chunker import Chunk


class Embedder:
    """Wraps sentence-transformers for encoding chunks and queries."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Load the sentence-transformer model.

        First call downloads ~80MB model to ~/.cache/huggingface/.
        Subsequent calls load from cache.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()  # 384

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed all chunks. Returns (N, 384) float32 array.

        Normalizes embeddings to unit length for cosine similarity.
        """
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (1, 384) float32 array."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed arbitrary texts. Returns (N, 384) float32 array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
        """Save embeddings as .npy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    @staticmethod
    def load_embeddings(path: Path) -> np.ndarray:
        """Load embeddings from .npy file."""
        return np.load(path)
