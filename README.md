# ExecDocRAG

**Retrieval-Augmented Generation system for SEC/FINRA execution policy documents + synthetic trade execution analytics.**

Built as a demonstration of GenAI applied to trade execution governance — embedding/vector-search pipelines, grounded Q&A with citations, retrieval evaluation, anomaly detection, and human-in-the-loop feedback.

---

## Problem Statement

Trade execution teams at institutional firms need fast, accurate access to regulatory information spread across dense SEC and FINRA documents (Rule 605, Rule 606, FINRA 5310, Regulation NMS). Manual lookup is slow and error-prone. This project builds a RAG system that:

1. **Retrieves** the most relevant regulatory passages for a given question
2. **Answers** with verbatim quotes (extractive) or synthesized responses (generative via Claude API)
3. **Cites** every answer with source document, page number, and section
4. **Evaluates** retrieval quality with Recall@k, MRR, and NDCG metrics
5. **Analyzes** trade execution quality with Implementation Shortfall decomposition and anomaly detection

## Architecture

```
                      ExecDocRAG Architecture
                      ========================

  SEC/FINRA PDFs/HTML
         |
    [Parser] ── PyMuPDF + BeautifulSoup
         |
    [Chunker] ── 512-token sentence-aware chunks with 64-token overlap
         |
    [Embedder] ── all-MiniLM-L6-v2 (384-dim, L2-normalized)
         |
    [FAISS Index] ── IndexFlatIP (exact cosine similarity)
         |
  User Query ──> [Embed Query] ──> [Top-K Search]
                                        |
                                  Retrieved Passages
                                   /            \
                      [Extractive QA]       [Generative QA]
                    TinyRoBERTa-SQuAD2      Claude API (optional)
                    (local, no hallucination)  (grounded, cited)
                                   \            /
                              Answer + Citations
                                        |
                              [Streamlit UI]
                              + Human Feedback
```

**Execution Analytics Pipeline:**
```
  Synthetic Orders/Fills (1,500 orders, 5,389 fills)
         |
    [IS Decomposition] ── Delay + Execution + Opportunity + Fixed costs (bps)
         |
    [Anomaly Detection] ── Z-score flagging (threshold: 2.5 sigma)
         |
    [DuckDB Store] ── SQL analytics (venue performance, daily trends)
         |
    [Tool Registry] ── Natural language -> analytics function mapping
         |
    [Streamlit UI] ── Interactive charts + tables
```

## Key Results

### Retrieval Evaluation (30 hand-crafted questions)

| Metric | Value |
|--------|-------|
| Recall@3 | 63.3% |
| Recall@5 | 66.7% |
| Recall@10 | 70.0% |
| MRR | 0.516 |
| NDCG@5 | 0.465 |

**By Difficulty:**
- Easy (n=10): Recall@3 = 70.0%
- Medium (n=12): Recall@3 = 66.7%
- Hard (n=8): Recall@3 = 50.0%

### Execution Analytics
- **1,500 synthetic orders** across 18 equities, 20 trading days
- **Implementation Shortfall** decomposition (hand-verified against manual calculation)
- **8% injected anomaly rate**, detected via z-score analysis
- Analytics queryable via natural language (regex-based tool registry)

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd ExecDocRAG
pip install -r requirements.txt

# 2. Download regulatory documents (SEC/FINRA)
python scripts/download_docs.py

# 3. Build the vector index (parse -> chunk -> embed -> FAISS)
python scripts/build_index.py

# 4. Generate synthetic trade data
python scripts/generate_trades.py

# 5. Launch the Streamlit app
streamlit run app/app.py

# Optional: Enable generative QA with Claude
export ANTHROPIC_API_KEY=your-key-here
```

## Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| **Extractive QA as default** (TinyRoBERTa-SQuAD2) | Every answer is a verbatim quote from the source. Zero hallucination risk. No API cost. Ideal for regulatory text where precision matters. |
| **Generative QA as optional** (Claude API) | Synthesizes multi-passage answers with citations. Better for complex questions but requires API key and introduces hallucination risk (mitigated by system prompt constraints). |
| **FAISS IndexFlatIP** (exact search) | For <10K vectors, exact search is fast enough and avoids approximation errors from IVF/HNSW. |
| **Sentence-aware chunking** (512 tokens, 64 overlap) | Respects sentence boundaries to preserve meaning. Overlap prevents information loss at chunk borders. |
| **DuckDB over pandas for analytics** | Analytical SQL queries are more expressive and demonstrate data engineering skills. No need for full database server. |
| **Synthetic trade data** | Real execution data is proprietary. Synthetic data with controlled anomalies allows reproducible evaluation. Realistic distributions (log-normal sizes, Dirichlet fills, venue-dependent slippage). |
| **Regex-based tool registry** | Fast, deterministic, debuggable. No LLM call needed for routing analytics queries. Easy to extend. |

## Limitations & Future Work

**Current Limitations:**
- FINRA Rule 5310 (HTML) produces fewer chunks than PDF documents, leading to lower recall for some FINRA-specific questions
- Hard questions requiring multi-document reasoning (e.g., "How do Rule 605 and FINRA 5310 interact?") have only 50% Recall@3
- Extractive QA can only surface spans within individual passages, not synthesize across them
- Anomaly detection uses global z-scores; per-venue or per-time-period baselines could improve precision

**Potential Improvements:**
- Fine-tune embedding model on financial regulatory text (domain adaptation)
- Implement hybrid retrieval (BM25 + dense) for better keyword matching
- Add parent-child chunk retrieval to expand context around hits
- Integrate RAGAS or DeepEval for automated answer quality evaluation
- Connect to real SEC Rule 605 CSV data for live execution quality analysis
- Add MiFID II RTS 27/28 documents for cross-jurisdictional coverage

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector Search | FAISS (`IndexFlatIP`, cosine similarity) |
| Extractive QA | `deepset/tinyroberta-squad2` |
| Generative QA | Anthropic Claude API (optional) |
| Analytics DB | DuckDB |
| UI | Streamlit |
| Document Parsing | PyMuPDF, BeautifulSoup |
| Testing | pytest (32 tests) |

## Project Structure

```
ExecDocRAG/
  README.md
  requirements.txt
  pyproject.toml
  data/
    raw/                    # Downloaded SEC/FINRA documents (8 docs, 6.1 MB)
    chunks/chunks.jsonl     # Processed chunks with metadata (271 chunks)
    index/                  # FAISS index + embeddings
    synthetic_trades/       # Generated orders.csv + fills.csv
    eval/eval_questions.json # 30 hand-crafted evaluation questions
  src/
    config.py               # Central configuration
    common/text_utils.py    # Text cleaning & sentence splitting
    rag/
      parser.py             # PDF/HTML parsing with section detection
      chunker.py            # Sentence-aware chunking with overlap
      embedder.py           # Sentence-transformer embedding wrapper
      indexer.py            # FAISS index management
      retriever.py          # Top-k retrieval with citations
      answer_builder.py     # Extractive QA (TinyRoBERTa)
      generative_answer.py  # Generative QA (Claude API)
      evaluation.py         # Recall@k, MRR, NDCG evaluation
      downloader.py         # Document download with retry
    execution/
      schemas.py            # Order/Fill dataclasses
      data_generator.py     # Synthetic trade data generation
      metrics.py            # Implementation Shortfall decomposition
      anomaly.py            # Z-score anomaly detection
      duckdb_store.py       # DuckDB analytical queries
      tool_registry.py      # NL -> analytics function mapping
  app/
    app.py                  # Streamlit UI (3 tabs)
  scripts/
    download_docs.py        # Download SEC/FINRA documents
    build_index.py          # Build FAISS index pipeline
    generate_trades.py      # Generate synthetic trade data
  tests/                    # 32 passing tests
  reports/
    retrieval_eval.md       # Evaluation results
```

## Document Corpus

| Document | Source | Format | Content |
|----------|--------|--------|---------|
| SEC Rule 605 Fact Sheet | SEC.gov | PDF | Order execution quality disclosure requirements |
| SEC Rule 605 Final Rule (2024) | SEC.gov | PDF | 2024 amendments to execution quality reporting |
| SEC Rule 606 Final Rule (2018) | SEC.gov | PDF | Order routing disclosure requirements |
| SEC Rule 606 Risk Alert | SEC.gov | PDF | Common compliance issues with Rule 606 |
| FINRA Rule 5310 | FINRA.org | HTML | Best Execution and Interpositioning |
| FINRA Notice 15-46 | FINRA.org | PDF | Best execution guidance and review requirements |
| FINRA Notice 21-23 | FINRA.org | PDF | Updated best execution practices |
| SEC Reg NMS Rule 611 | SEC.gov | PDF | Order Protection (Trade-Through) Rule |

## References

- SEC Rule 605: [Disclosure of Order Execution Information](https://www.sec.gov/rules/final/2024/34-99679.pdf)
- SEC Rule 606: [Disclosure of Order Routing Information](https://www.sec.gov/rules/final/2018/34-84528.pdf)
- FINRA Rule 5310: [Best Execution and Interpositioning](https://www.finra.org/rules-guidance/rulebooks/finra-rules/5310)
- Perold, A. (1988). "The Implementation Shortfall: Paper vs. Reality." Journal of Portfolio Management.

---

*project demonstrating GenAI applied to trade execution governance.*
