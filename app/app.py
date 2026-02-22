"""ExecDocRAG — Streamlit Application.

Two-tab layout:
    Tab 1: Regulatory Document Q&A (RAG with extractive answers and citations)
    Tab 2: Execution Quality Analytics (IS decomposition, anomaly detection)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNKS_DIR,
    DOCUMENT_SOURCES,
    EVAL_DIR,
    INDEX_DIR,
    TRADES_DIR,
)

# ─── Page config (must be first Streamlit command) ───────────────────────────

st.set_page_config(
    page_title="ExecDocRAG",
    page_icon="\U0001f4ca",
    layout="wide",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .source-citation { color: #666; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# ─── Cached resource loading ────────────────────────────────────────────────

@st.cache_resource
def load_rag_components():
    """Load embedder, index, chunks, retriever, and answer builder."""
    from src.rag.answer_builder import AnswerBuilder
    from src.rag.chunker import load_chunks
    from src.rag.embedder import Embedder
    from src.rag.indexer import FAISSIndex
    from src.rag.retriever import Retriever

    chunks_path = CHUNKS_DIR / "chunks.jsonl"
    index_path = INDEX_DIR / "faiss.index"

    if not chunks_path.exists() or not index_path.exists():
        return None, None

    chunks = load_chunks(chunks_path)
    embedder = Embedder()

    index = FAISSIndex()
    index.load(index_path)

    retriever = Retriever(embedder, index, chunks)
    answer_builder = AnswerBuilder()

    return retriever, answer_builder


@st.cache_resource
def load_execution_components():
    """Load trade data and analytics components."""
    from src.execution.anomaly import flag_anomalies
    from src.execution.duckdb_store import TradeStore
    from src.execution.metrics import compute_is_batch
    from src.execution.tool_registry import build_default_registry

    orders_path = TRADES_DIR / "orders.csv"
    fills_path = TRADES_DIR / "fills.csv"

    if not orders_path.exists():
        return None, None, None, None

    orders_df = pd.read_csv(orders_path)
    fills_df = pd.read_csv(fills_path)

    # Compute IS if not already present
    if "total_is_bps" not in orders_df.columns:
        orders_df = compute_is_batch(orders_df)

    # Flag anomalies
    orders_df = flag_anomalies(
        orders_df,
        metric_columns=["total_is_bps", "delay_cost_bps", "execution_cost_bps"],
        threshold=2.5,
    )

    store = TradeStore()
    store.load_data(orders_df, fills_df)
    registry = build_default_registry(store)

    return orders_df, fills_df, store, registry


# ─── Main Layout ─────────────────────────────────────────────────────────────

st.title("\U0001f4ca ExecDocRAG")
st.caption(
    "Execution Policy Q&A with RAG + Trade Analytics | "
    "Extractive QA (local) + optional Generative QA (Claude API)"
)

tab_rag, tab_analytics, tab_about = st.tabs([
    "\U0001f4d6 Regulatory Document Q&A",
    "\U0001f4c8 Execution Quality Analytics",
    "\u2139\ufe0f About",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RAG Q&A
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rag:
    st.header("Ask about SEC/FINRA execution regulations")

    retriever, answer_builder = load_rag_components()

    if retriever is None:
        st.error(
            "RAG index not found. Run the following commands first:\n\n"
            "```bash\n"
            "python scripts/download_docs.py\n"
            "python scripts/build_index.py\n"
            "```"
        )
    else:
        # Sidebar info
        with st.sidebar:
            st.subheader("Document Corpus")
            st.metric("Chunks Indexed", retriever.index.size)
            st.metric("Embedding Dim", retriever.embedder.dimension)
            st.markdown("**Sources:**")
            for doc in DOCUMENT_SOURCES:
                st.markdown(f"- {doc.name}")

        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What does Rule 605 require market centers to disclose?",
            key="rag_query",
        )

        col_settings1, col_settings2, col_settings3 = st.columns([1, 1, 1])
        with col_settings1:
            top_k = st.slider("Passages to retrieve", 3, 10, 5, key="top_k")
        with col_settings2:
            min_score = st.slider("Min similarity", 0.1, 0.8, 0.25, key="min_score")
        with col_settings3:
            # Check if generative mode is available
            from src.rag.generative_answer import is_api_available
            gen_available = is_api_available()
            use_generative = st.toggle(
                "Generative QA (Claude API)",
                value=False,
                disabled=not gen_available,
                help=(
                    "Use Claude API for synthesized answers with citations. "
                    "Requires ANTHROPIC_API_KEY env var."
                    if gen_available
                    else "Set ANTHROPIC_API_KEY to enable generative mode."
                ),
                key="use_gen",
            )

        if query:
            with st.spinner("Searching regulatory documents..."):
                results = retriever.retrieve(query, top_k=top_k, threshold=min_score)

            # Generative QA mode
            if use_generative and gen_available:
                from src.rag.generative_answer import generate_answer

                with st.spinner("Generating answer with Claude..."):
                    gen_answer = generate_answer(query, results)

                st.subheader("Answer (Generative)")
                st.info(gen_answer.answer_text)
                st.caption(
                    f"Model: {gen_answer.model} | "
                    f"Passages used: {gen_answer.passages_used} | "
                    f"Sources: {', '.join(set(gen_answer.source_docs))}"
                )

            # Extractive QA mode (default)
            else:
                answer_response = answer_builder.build_answer(query, results)

                if answer_response.top_answer:
                    st.subheader("Answer (Extractive)")
                    ans = answer_response.top_answer
                    st.success(f"**{ans.answer_text}**")
                    st.caption(
                        f"\U0001f4ce Source: **{ans.source_doc}** | "
                        f"Page {ans.page_number} | "
                        f"Section: {ans.section_title[:80]} | "
                        f"Confidence: {ans.confidence:.1%} | "
                        f"Similarity: {ans.similarity_score:.3f}"
                    )

                    # Additional answers
                    if len(answer_response.all_answers) > 1:
                        with st.expander("More relevant excerpts"):
                            for i, extra in enumerate(answer_response.all_answers[1:], 2):
                                st.markdown(
                                    f"**{i}.** \"{extra.answer_text}\" "
                                    f"*\u2014 {extra.source_doc}, p.{extra.page_number} "
                                    f"(conf: {extra.confidence:.1%})*"
                                )
                else:
                    st.warning(
                        "No confident answer found. Try rephrasing or lowering the similarity threshold."
                    )

            # Retrieved passages
            st.subheader("Retrieved Passages")
            for result in results:
                with st.expander(
                    f"[{result.rank}] {result.chunk.source_doc} \u2014 "
                    f"p.{result.chunk.page_number} "
                    f"(sim: {result.score:.3f})"
                ):
                    st.write(result.chunk.text[:800])
                    if len(result.chunk.text) > 800:
                        st.caption("... (truncated)")
                    st.caption(
                        f"Section: {result.chunk.section_title} | "
                        f"Chunk ID: {result.chunk.chunk_id}"
                    )

            # Feedback
            st.divider()
            st.subheader("Feedback (Human-in-the-Loop)")
            col_fb = st.columns(3)
            with col_fb[0]:
                if st.button("\u2705 Relevant", key="fb_good"):
                    _save_feedback(query, "relevant", results)
                    st.toast("Thank you!")
            with col_fb[1]:
                if st.button("\u2796 Partial", key="fb_partial"):
                    _save_feedback(query, "partial", results)
                    st.toast("Feedback saved!")
            with col_fb[2]:
                if st.button("\u274c Not Relevant", key="fb_bad"):
                    _save_feedback(query, "not_relevant", results)
                    st.toast("Feedback saved!")

        # Example questions
        with st.expander("Example questions"):
            examples = [
                "What does Rule 605 require market centers to disclose?",
                "What factors should firms consider for best execution?",
                "What is the Order Protection Rule?",
                "How does Rule 606 address payment for order flow?",
                "What obligations apply when routing orders to another broker?",
            ]
            for ex in examples:
                st.code(ex, language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EXECUTION QUALITY ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.header("Execution Quality Analytics")

    result = load_execution_components()
    orders_df, fills_df, store, registry = (
        result if result[0] is not None else (None, None, None, None)
    )

    if orders_df is None:
        st.error(
            "Trade data not found. Run:\n\n"
            "```bash\npython scripts/generate_trades.py\n```"
        )
    else:
        # Summary metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Orders", f"{len(orders_df):,}")
        with m2:
            st.metric("Avg IS (bps)", f"{orders_df['total_is_bps'].mean():.1f}")
        with m3:
            fill_rate = (orders_df["filled_quantity"] / orders_df["quantity"]).mean()
            st.metric("Avg Fill Rate", f"{fill_rate:.1%}")
        with m4:
            n_anomalies = orders_df["is_anomaly"].sum() if "is_anomaly" in orders_df.columns else 0
            st.metric("Anomalies", int(n_anomalies))

        # Natural language query
        st.subheader("Ask about execution quality")
        nl_query = st.text_input(
            "Ask a question:",
            placeholder="e.g., Which venue has the worst execution quality?",
            key="exec_query",
        )

        if nl_query and registry is not None:
            tool_result = registry.execute(nl_query)
            if tool_result:
                st.success(f"Tool: **{tool_result.tool_name}** \u2014 {tool_result.description}")
                st.dataframe(tool_result.data, use_container_width=True)
            else:
                st.info("No matching analysis. Try one of these:")
                for t in registry.list_tools():
                    st.markdown(
                        f"- **{t['description']}**: _{t['example_questions'][0]}_"
                    )

        # Visualization sub-tabs
        viz1, viz2, viz3, viz4 = st.tabs([
            "IS Decomposition", "Venue Analysis", "Anomalies", "Order Types",
        ])

        with viz1:
            st.subheader("Implementation Shortfall Decomposition")
            if store:
                is_summary = store.is_decomposition_summary()
                st.dataframe(is_summary, use_container_width=True)

            # IS distribution
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].hist(
                orders_df["total_is_bps"].clip(-50, 50),
                bins=50,
                color="#4C78A8",
                edgecolor="white",
                alpha=0.8,
            )
            axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
            axes[0].set_xlabel("Total IS (bps)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("IS Distribution")

            # Stacked bar of IS components
            components = ["delay_cost_bps", "execution_cost_bps", "opportunity_cost_bps", "fixed_cost_bps"]
            component_means = [orders_df[c].mean() for c in components]
            labels = ["Delay", "Execution", "Opportunity", "Fixed"]
            colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

            axes[1].bar(labels, component_means, color=colors, edgecolor="white")
            axes[1].set_ylabel("Average Cost (bps)")
            axes[1].set_title("IS Component Breakdown")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with viz2:
            st.subheader("Venue Analysis")
            if store:
                venue_data = store.avg_is_by_venue()
                st.dataframe(venue_data, use_container_width=True)

                venue_share = store.venue_market_share()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(
                    venue_share["venue"],
                    venue_share["market_share_pct"],
                    color="#4C78A8",
                    edgecolor="white",
                )
                ax.set_xlabel("Market Share (%)")
                ax.set_title("Venue Market Share by Fill Volume")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with viz3:
            st.subheader("Anomaly Detection")
            if "is_anomaly" in orders_df.columns:
                anomalies = orders_df[orders_df["is_anomaly"]]
                st.write(f"**{len(anomalies)} anomalies** detected out of {len(orders_df)} orders ({len(anomalies)/len(orders_df):.1%})")

                if len(anomalies) > 0:
                    # Scatter plot: IS vs anomaly score
                    fig, ax = plt.subplots(figsize=(10, 5))
                    normal = orders_df[~orders_df["is_anomaly"]]
                    ax.scatter(
                        normal["total_is_bps"],
                        normal.get("anomaly_score", 0),
                        alpha=0.3,
                        s=10,
                        color="#4C78A8",
                        label="Normal",
                    )
                    ax.scatter(
                        anomalies["total_is_bps"],
                        anomalies.get("anomaly_score", 0),
                        alpha=0.7,
                        s=30,
                        color="#E45756",
                        label="Anomaly",
                    )
                    ax.set_xlabel("Total IS (bps)")
                    ax.set_ylabel("Anomaly Score (max |z-score|)")
                    ax.set_title("Execution Anomaly Detection")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Anomaly table
                    st.subheader("Top Anomalous Orders")
                    display_cols = [
                        "order_id", "symbol", "side", "order_type",
                        "total_is_bps", "anomaly_score", "anomaly_reasons",
                    ]
                    available_cols = [c for c in display_cols if c in anomalies.columns]
                    st.dataframe(
                        anomalies.nlargest(15, "anomaly_score")[available_cols],
                        use_container_width=True,
                    )

        with viz4:
            st.subheader("Performance by Order Type")
            if store:
                fill_data = store.fill_rate_by_order_type()
                st.dataframe(fill_data, use_container_width=True)

                # Box plot of IS by order type
                fig, ax = plt.subplots(figsize=(10, 5))
                order_types = orders_df["order_type"].unique()
                data_by_type = [
                    orders_df[orders_df["order_type"] == ot]["total_is_bps"].clip(-30, 30)
                    for ot in order_types
                ]
                bp = ax.boxplot(data_by_type, labels=order_types, patch_artist=True)
                colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
                for patch, color in zip(bp["boxes"], colors[:len(order_types)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_ylabel("Total IS (bps)")
                ax.set_title("IS Distribution by Order Type")
                ax.axhline(0, color="red", linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.header("About ExecDocRAG")
    st.markdown("""
### Architecture

**Module A: Regulatory Document Q&A (RAG)**
- Real SEC/FINRA regulatory documents (Rule 605, 606, FINRA 5310, Reg NMS)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local)
- Vector Search: FAISS IndexFlatIP (exact cosine similarity)
- Answer Extraction: `deepset/tinyroberta-squad2` (extractive QA, no hallucination)
- Every answer is a verbatim quote with source, page, and section citations

**Module B: Execution Quality Analytics**
- Implementation Shortfall decomposition (Perold 1988)
- Z-score anomaly detection on execution metrics
- DuckDB analytical queries over synthetic trade data
- Natural language tool registry for analytics queries

### Design Principles

1. **Traceability**: Extractive QA means answers are literal source text, not LLM-generated
2. **No paid APIs**: Entirely local inference with open-source models
3. **Human-in-the-Loop**: Feedback mechanism for continuous quality monitoring
4. **Reproducibility**: All data and models are freely available

### Tech Stack
`Python` | `FAISS` | `sentence-transformers` | `HuggingFace Transformers` |
`DuckDB` | `Streamlit` | `PyMuPDF` | `LightGBM` | `SHAP`
    """)


# ─── Helper functions ────────────────────────────────────────────────────────

def _save_feedback(query: str, rating: str, results: list) -> None:
    """Save user feedback to JSONL file."""
    feedback_path = EVAL_DIR / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "rating": rating,
        "num_results": len(results),
        "top_score": results[0].score if results else 0,
        "top_doc": results[0].chunk.source_doc if results else "",
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
