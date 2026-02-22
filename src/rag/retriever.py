"""Top-k retrieval with citation metadata.

Every result includes full traceability: source document name,
page number, section title, and similarity score. This addresses
the JD's requirement for "traceable/grounded outputs".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import SIMILARITY_THRESHOLD, TOP_K_DEFAULT
from src.rag.chunker import Chunk
from src.rag.embedder import Embedder
from src.rag.indexer import FAISSIndex


@dataclass
class RetrievalResult:
    """A single retrieved passage with full citation metadata."""

    chunk: Chunk
    score: float  # Cosine similarity score [0, 1]
    rank: int  # 1-indexed rank in results

    def format_citation(self) -> str:
        """Format as a citation string for display.

        Example: "[SEC Rule 605 Fact Sheet, p.2, Section: Overview] (score: 0.87)"
        """
        parts = [self.chunk.source_doc]
        parts.append(f"p.{self.chunk.page_number}")
        if self.chunk.section_title:
            parts.append(f"Section: {self.chunk.section_title[:60]}")
        citation = ", ".join(parts)
        return f"[{citation}] (score: {self.score:.3f})"


class Retriever:
    """Retrieves relevant passages for a query with citation metadata."""

    def __init__(
        self,
        embedder: Embedder,
        index: FAISSIndex,
        chunks: list[Chunk],
    ):
        """Initialize retriever.

        Args:
            embedder: The embedding model for encoding queries.
            index: The FAISS index containing chunk embeddings.
            chunks: List of chunks in the SAME ORDER as they were indexed.
        """
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

        if index.size != len(chunks):
            raise ValueError(
                f"Index size ({index.size}) != chunk count ({len(chunks)}). "
                "Ensure chunks are in the same order as indexed embeddings."
            )

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[RetrievalResult]:
        """Retrieve top-k passages for a query.

        Steps:
        1. Embed the query.
        2. Search the FAISS index for nearest neighbors.
        3. Filter by similarity threshold.
        4. Return RetrievalResult objects with full citation metadata.

        Args:
            query: The user's question.
            top_k: Number of passages to retrieve.
            threshold: Minimum cosine similarity to include in results.

        Returns:
            List of RetrievalResult objects, sorted by descending score.
        """
        query_emb = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_emb, top_k)

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            # FAISS returns -1 for missing results
            if idx == -1:
                continue
            # Filter by threshold
            if score < threshold:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(idx)],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int = TOP_K_DEFAULT,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[list[RetrievalResult]]:
        """Retrieve passages for multiple queries at once.

        More efficient than calling retrieve() in a loop because
        it batches the embedding computation.
        """
        all_results: list[list[RetrievalResult]] = []
        for query in queries:
            results = self.retrieve(query, top_k, threshold)
            all_results.append(results)
        return all_results
