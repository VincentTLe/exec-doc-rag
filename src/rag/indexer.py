"""FAISS index management.

Uses IndexFlatIP (inner product) with L2-normalized vectors,
which is equivalent to cosine similarity. Exact search is
appropriate for <10K vectors — no approximation needed.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FAISSIndex:
    """Thin wrapper around FAISS IndexFlatIP for cosine similarity search."""

    def __init__(self, dimension: int = 384):
        """Create an empty IndexFlatIP index.

        With L2-normalized vectors, inner product = cosine similarity.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index.

        Args:
            embeddings: (N, dimension) float32 array, L2-normalized.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {embeddings.shape}"
            )
        self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for top_k nearest neighbors.

        Args:
            query_embedding: (1, dimension) float32 array, L2-normalized.
            top_k: Number of results to return.

        Returns:
            (scores, indices): both (1, top_k) arrays.
            Scores are cosine similarities in [0, 1] for normalized vectors.
        """
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )
        return scores, indices

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        """Load index from disk, replacing current index."""
        self.index = faiss.read_index(str(path))
        self.dimension = self.index.d

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal
