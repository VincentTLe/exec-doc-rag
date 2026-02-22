"""Tests for the retriever module.

These tests use synthetic data to verify retriever logic
without requiring actual model downloads.
"""

import numpy as np
import pytest

from src.rag.chunker import Chunk
from src.rag.indexer import FAISSIndex


class TestFAISSIndex:
    """Tests for the FAISS index wrapper."""

    def test_add_and_search(self) -> None:
        """Adding vectors and searching should return valid results."""
        dim = 8
        index = FAISSIndex(dimension=dim)

        # Add 5 normalized vectors
        vectors = np.random.randn(5, dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)

        assert index.size == 5

        # Search with one of the vectors — should find itself
        query = vectors[0:1]
        scores, indices = index.search(query, top_k=3)

        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert indices[0][0] == 0  # Should find itself first
        assert scores[0][0] > 0.99  # Near-perfect match

    def test_empty_index_search(self) -> None:
        """Searching an empty index should return -1 indices."""
        index = FAISSIndex(dimension=8)
        query = np.random.randn(1, 8).astype(np.float32)
        scores, indices = index.search(query, top_k=3)
        assert (indices == -1).all()

    def test_dimension_mismatch_raises(self) -> None:
        """Adding vectors of wrong dimension should raise."""
        index = FAISSIndex(dimension=8)
        wrong_dim = np.random.randn(3, 16).astype(np.float32)
        with pytest.raises(ValueError):
            index.add(wrong_dim)

    def test_save_and_load(self, tmp_path) -> None:
        """Index should survive save/load roundtrip."""
        dim = 8
        index = FAISSIndex(dimension=dim)
        vectors = np.random.randn(10, dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)

        # Save and load
        path = tmp_path / "test.index"
        index.save(path)

        new_index = FAISSIndex(dimension=dim)
        new_index.load(path)

        assert new_index.size == 10

        # Search should give same results
        query = vectors[0:1]
        s1, i1 = index.search(query, top_k=3)
        s2, i2 = new_index.search(query, top_k=3)
        np.testing.assert_array_equal(i1, i2)

    def test_scores_are_descending(self) -> None:
        """Search results should be sorted by descending score."""
        dim = 8
        index = FAISSIndex(dimension=dim)
        vectors = np.random.randn(20, dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)

        query = np.random.randn(1, dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        scores, _ = index.search(query, top_k=10)

        # Scores should be non-increasing
        for i in range(len(scores[0]) - 1):
            assert scores[0][i] >= scores[0][i + 1]
