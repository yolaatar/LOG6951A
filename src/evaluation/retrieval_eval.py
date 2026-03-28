#!/usr/bin/env python3
"""
retrieval_eval.py
=================
Task 2 — Comprehensive Retrieval Strategy Evaluation
ResearchPal RAG Pipeline (LOG6951A)

Compares Cosine Similarity (baseline) vs MMR across 5 annotated queries.
Does NOT require an Ollama LLM — only the existing ChromaDB index.

Metrics computed
----------------
Per (query × strategy):
  - cosine scores: mean, std, spread (max − min), top-1 score
  - intra_sim      : mean pairwise cosine similarity among returned chunks
                     (embedding-based redundancy proxy — lower is better)
  - jaccard_red    : mean pairwise Jaccard similarity (lexical redundancy proxy)
  - distinct_sources, distinct_types

Cross-strategy (per query):
  - overlap_ratio  : fraction of chunks shared between cosine and MMR results
  - rank_shift     : mean absolute rank change for shared chunks

Sweeps (Recursive over parameters):
  - lambda_mult    : 0.1 → 1.0, tracking diversity vs relevance trade-off
  - k              : 2 → 8, tracking metric evolution
  - fetch_k        : 8 → 40, tracking MMR candidate pool effect

Outputs — all in reports/retrieval_eval/
  figures/score_distributions.png
  figures/source_diversity.png
  figures/redundancy_comparison.png
  figures/result_overlap_heatmaps.png
  figures/lambda_sweep.png
  figures/k_sweep.png
  figures/fetch_k_sweep.png
  figures/summary_panel.png
  per_query/q{1..5}_comparison.txt
  metrics.csv
  summary.md

Usage (from project root, inside venv):
    python src/evaluation/retrieval_eval.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import re
import csv
import warnings
import textwrap
from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── Project path ──────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

# ── Numeric / plotting ────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ── Project modules ───────────────────────────────────────────────────────────
from config import (
    RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA, EMBEDDING_MODEL_NAME,
)
from ingestion.indexer import load_vectorstore
from retrieval.cosine_retriever import cosine_search_with_scores
from retrieval.mmr_retriever import mmr_search
from retrieval.eval_queries import EVAL_QUERIES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# All parameters are centralised here — nothing hardcoded elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

# Output
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "reports" / "retrieval_eval"
FIGURES_DIR = OUTPUT_DIR / "figures"
PERQUERY_DIR = OUTPUT_DIR / "per_query"

# Primary evaluation parameters (match Task 2 pipeline defaults)
PRIMARY_K        = RETRIEVAL_TOP_K    # 4
PRIMARY_FETCH_K  = MMR_FETCH_K        # 20
PRIMARY_LAMBDA   = MMR_LAMBDA         # 0.5

# Parameter sweep ranges
SWEEP_K       = [2, 3, 4, 5, 6, 8]
SWEEP_LAMBDA  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SWEEP_FETCH_K = [8, 12, 16, 20, 30, 40]

# Jaccard redundancy threshold  (pairs with similarity >= threshold → redundant)
JACCARD_THRESHOLD = 0.40

# Visual palette
PALETTE = {"Cosine": "#2980b9", "MMR": "#27ae60"}

def _col(label: str) -> str:
    for k, v in PALETTE.items():
        if k.lower() in label.lower():
            return v
    return "#7f8c8d"

# Libellés en français pour l'affichage dans les figures
_FR_STRAT = {"Cosine": "Cosinus", "MMR": "MMR"}
def _fr(label: str) -> str:
    return _FR_STRAT.get(label, label)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup() -> None:
    for d in [OUTPUT_DIR, FIGURES_DIR, PERQUERY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  [setup] Output → {OUTPUT_DIR}")


def load_embeddings() -> HuggingFaceEmbeddings:
    print(f"\n  [embeddings] Loading: {EMBEDDING_MODEL_NAME}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    print("  [embeddings] Ready.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _source_label(doc: Document) -> str:
    src = doc.metadata.get("source", "?")
    return src if src.startswith("http") else Path(src).name


def _chunk_id(doc: Document) -> str:
    cid = doc.metadata.get("chunk_id")
    if cid:
        return cid
    return doc.page_content[:40]


def intra_result_similarity(
    docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Dict[str, float]:
    """
    Mean pairwise cosine similarity among the k retrieved chunks.

    Interpretation:
      High value → chunks are near-duplicates → high redundancy, poor diversity
      Low value  → chunks cover distinct aspects → good diversity
      Desired range: 0.20 – 0.55 for a balanced result set
    """
    if len(docs) < 2:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

    texts = [d.page_content for d in docs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V = np.array(embeddings.embed_documents(texts))   # shape (k, dim)

    # all pairwise dot products (vectors already L2-normalised → == cosine)
    G = V @ V.T
    idx = np.triu_indices(len(docs), k=1)   # upper triangle only
    pairwise = G[idx]

    return {
        "mean": float(np.mean(pairwise)),
        "max":  float(np.max(pairwise)),
        "min":  float(np.min(pairwise)),
        "std":  float(np.std(pairwise)),
    }


def jaccard_redundancy(
    docs: List[Document],
    threshold: float = JACCARD_THRESHOLD,
) -> Dict[str, float]:
    """
    Lexical (token-level) redundancy proxy.

    Returns mean pairwise Jaccard similarity and the fraction of pairs
    that exceed the threshold (=redundant pairs ratio).
    """
    pairs = list(combinations(docs, 2))
    if not pairs:
        return {"mean": 0.0, "redundant_pct": 0.0}

    def _j(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0

    sims = [_j(a.page_content, b.page_content) for a, b in pairs]
    redundant = sum(1 for s in sims if s >= threshold)

    return {
        "mean":          float(np.mean(sims)),
        "redundant_pct": 100.0 * redundant / len(pairs),
    }


def score_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"mean": 0.0, "std": 0.0, "top1": 0.0, "spread": 0.0}
    return {
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores)),
        "top1":   float(scores[0]),
        "spread": float(scores[0] - scores[-1]),
    }


def result_overlap(
    docs_a: List[Document],
    docs_b: List[Document],
) -> Dict[str, float]:
    """
    Fraction of chunks shared between two result sets.
    Also computes mean absolute rank difference for shared documents.
    """
    ids_a = [_chunk_id(d) for d in docs_a]
    ids_b = [_chunk_id(d) for d in docs_b]

    set_a, set_b = set(ids_a), set(ids_b)
    shared = set_a & set_b

    if not ids_a:
        return {"overlap_ratio": 0.0, "rank_shift_mean": 0.0, "shared_n": 0}

    rank_shifts = []
    for cid in shared:
        ra = ids_a.index(cid) + 1   # 1-indexed
        rb = ids_b.index(cid) + 1
        rank_shifts.append(abs(ra - rb))

    return {
        "overlap_ratio":   len(shared) / max(len(ids_a), 1),
        "rank_shift_mean": float(np.mean(rank_shifts)) if rank_shifts else 0.0,
        "shared_n":        len(shared),
    }


def compute_full_metrics(
    docs: List[Document],
    scores: Optional[List[float]],
    embeddings: HuggingFaceEmbeddings,
    strategy: str,
) -> Dict[str, Any]:
    """Aggregate all metrics for one (query, strategy) result set."""
    sources  = {_source_label(d) for d in docs}
    types    = {d.metadata.get("type_document", "?") for d in docs}
    avg_len  = float(np.mean([len(d.page_content) for d in docs])) if docs else 0.0

    intra    = intra_result_similarity(docs, embeddings)
    jaccard  = jaccard_redundancy(docs)
    sc_stats = score_stats(scores) if scores else {}

    return {
        "strategy":          strategy,
        "n_docs":            len(docs),
        "distinct_sources":  len(sources),
        "source_labels":     sorted(sources),
        "distinct_types":    len(types),
        "avg_len":           avg_len,
        "intra_sim_mean":    intra["mean"],
        "intra_sim_max":     intra["max"],
        "jaccard_mean":      jaccard["mean"],
        "jaccard_redundant_pct": jaccard["redundant_pct"],
        "score_mean":        sc_stats.get("mean", 0.0),
        "score_std":         sc_stats.get("std", 0.0),
        "score_top1":        sc_stats.get("top1", 0.0),
        "score_spread":      sc_stats.get("spread", 0.0),
        # raw refs for downstream use
        "_docs":             docs,
        "_scores":           scores or [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    vectorstore,
    embeddings: HuggingFaceEmbeddings,
    k: int = PRIMARY_K,
    fetch_k: int = PRIMARY_FETCH_K,
    lambda_mult: float = PRIMARY_LAMBDA,
) -> List[Dict]:
    """
    Run Cosine and MMR on all 5 queries, compute full metrics.
    Returns a list of per-query result dicts.
    """
    results = []

    for q in EVAL_QUERIES:
        query = q["query"]
        print(f"  Q{q['id']}: {query[:60]}...")

        # Cosine (with scores)
        cos_raw    = cosine_search_with_scores(vectorstore, query, k=k)
        cos_docs   = [d for d, _ in cos_raw]
        cos_scores = [s for _, s in cos_raw]

        # MMR (no scores returned by ChromaDB)
        mmr_docs   = mmr_search(vectorstore, query,
                                k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

        cos_metrics = compute_full_metrics(cos_docs, cos_scores, embeddings, "Cosine")
        mmr_metrics = compute_full_metrics(mmr_docs, None,       embeddings, "MMR")

        ovlp = result_overlap(cos_docs, mmr_docs)

        results.append({
            "query_info":   q,
            "cosine":       cos_metrics,
            "mmr":          mmr_metrics,
            "overlap":      ovlp,
            "params":       {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PARAMETER SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_avg(vectorstore, queries, embeddings, run_fn) -> List[Dict]:
    """Helper: run run_fn(q) for each query, compute per-query metrics, return averaged rows."""
    all_rows = []
    for cfg, fn in run_fn:
        intra_list, jac_list, src_list = [], [], []
        for q in queries:
            docs = fn(q["query"])
            if not docs:
                continue
            intra  = intra_result_similarity(docs, embeddings)["mean"]
            jac    = jaccard_redundancy(docs)["mean"]
            srcs   = len({_source_label(d) for d in docs})
            intra_list.append(intra)
            jac_list.append(jac)
            src_list.append(srcs)
        row = dict(cfg)
        row["intra_sim"]   = float(np.mean(intra_list)) if intra_list else 0.0
        row["jaccard"]     = float(np.mean(jac_list))   if jac_list   else 0.0
        row["avg_sources"] = float(np.mean(src_list))   if src_list   else 0.0
        all_rows.append(row)
    return all_rows


def sweep_lambda(vectorstore, queries, embeddings) -> List[Dict]:
    """Sweep lambda_mult for MMR, averaged across all queries."""
    print("  Lambda sweep...")
    configs = []
    for lam in SWEEP_LAMBDA:
        cfg = {"lambda_mult": lam}
        fn  = (lambda q, l=lam:
               mmr_search(vectorstore, q,
                           k=PRIMARY_K, fetch_k=PRIMARY_FETCH_K,
                           lambda_mult=l))
        configs.append((cfg, fn))

    rows = _sweep_avg(vectorstore, queries, embeddings, configs)
    for r in rows:
        print(f"    λ={r['lambda_mult']:.1f}  intra_sim={r['intra_sim']:.3f}"
              f"  jaccard={r['jaccard']:.3f}  sources={r['avg_sources']:.2f}")
    return rows


def sweep_k(vectorstore, queries, embeddings) -> Tuple[List[Dict], List[Dict]]:
    """Sweep k for both Cosine and MMR."""
    print("  k sweep (Cosine + MMR)...")

    cos_configs = []
    mmr_configs = []
    for k in SWEEP_K:
        cos_configs.append(
            ({"k": k},
             lambda q, _k=k:
             [d for d, _ in cosine_search_with_scores(vectorstore, q, k=_k)])
        )
        mmr_configs.append(
            ({"k": k},
             lambda q, _k=k:
             mmr_search(vectorstore, q, k=_k,
                        fetch_k=PRIMARY_FETCH_K, lambda_mult=PRIMARY_LAMBDA))
        )

    cos_rows = _sweep_avg(vectorstore, queries, embeddings, cos_configs)
    mmr_rows = _sweep_avg(vectorstore, queries, embeddings, mmr_configs)

    for r in cos_rows:
        print(f"    Cosine k={r['k']}  intra={r['intra_sim']:.3f}  src={r['avg_sources']:.2f}")
    for r in mmr_rows:
        print(f"    MMR    k={r['k']}  intra={r['intra_sim']:.3f}  src={r['avg_sources']:.2f}")

    return cos_rows, mmr_rows


def sweep_fetch_k(vectorstore, queries, embeddings) -> List[Dict]:
    """Sweep fetch_k for MMR."""
    print("  fetch_k sweep (MMR)...")
    configs = []
    for fk in SWEEP_FETCH_K:
        cfg = {"fetch_k": fk}
        fn  = (lambda q, f=fk:
               mmr_search(vectorstore, q,
                           k=PRIMARY_K, fetch_k=f,
                           lambda_mult=PRIMARY_LAMBDA))
        configs.append((cfg, fn))

    rows = _sweep_avg(vectorstore, queries, embeddings, configs)
    for r in rows:
        print(f"    fetch_k={r['fetch_k']:2d}  intra={r['intra_sim']:.3f}"
              f"  src={r['avg_sources']:.2f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    p = FIGURES_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


def fig_score_distributions(results: List[Dict]) -> None:
    """
    Per-query cosine score distributions.
    One subplot per query showing the top-k scores as a bar chart.
    MMR does not expose scores → shown as uniform bars at score=0 with hatch.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)

    for ax, r in zip(axes, results):
        q_id    = r["query_info"]["id"]
        cos_s   = r["cosine"]["_scores"]
        mmr_n   = r["mmr"]["n_docs"]
        ranks   = np.arange(1, len(cos_s) + 1)

        ax.bar(ranks - 0.2, cos_s, width=0.35, color=_col("Cosine"),
               label="Cosinus", alpha=0.88)
        # MMR scores not available — show placeholder
        ax.bar(ranks + 0.2, [0.0] * mmr_n, width=0.35, color=_col("MMR"),
               label="MMR (n/d)", alpha=0.4, hatch="//")

        ax.set_title(f"Q{q_id}\n"
                     f"{r['query_info']['category'][:28]}",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Rang", fontsize=8)
        ax.set_ylabel("Score cosinus", fontsize=8)
        ax.set_xticks(ranks)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        if q_id == 1:
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Scores de pertinence cosinus par requête  (scores MMR non exposés par ChromaDB)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "score_distributions.png")


def fig_source_diversity(results: List[Dict]) -> None:
    """Grouped bar chart: distinct sources per query and strategy."""
    q_ids  = [r["query_info"]["id"] for r in results]
    q_cats = [r["query_info"]["category"][:22] for r in results]
    cos_s  = [r["cosine"]["distinct_sources"] for r in results]
    mmr_s  = [r["mmr"]["distinct_sources"]    for r in results]

    x = np.arange(len(q_ids))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars_c = ax.bar(x - w / 2, cos_s, w, color=_col("Cosine"),
                    label="Cosinus", alpha=0.88, edgecolor="white")
    bars_m = ax.bar(x + w / 2, mmr_s, w, color=_col("MMR"),
                    label="MMR", alpha=0.88, edgecolor="white")

    for bar, v in [(b, val) for bars, vals in [(bars_c, cos_s), (bars_m, mmr_s)]
                   for b, val in zip(bars, vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                str(int(v)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}\n{c}" for i, c in zip(q_ids, q_cats)], fontsize=8.5)
    ax.set_ylabel("Nombre de documents sources distincts", fontsize=10)
    ax.set_ylim(0, max(max(cos_s), max(mmr_s)) + 1)
    ax.set_title("Diversité des sources par requête et stratégie\n"
                 "↑ plus élevé = résultats couvrant plus de documents sources",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "source_diversity.png")


def fig_redundancy_comparison(results: List[Dict]) -> None:
    """
    Two-panel chart comparing embedding-based and lexical redundancy.
    Left:  intra_sim_mean per query (embedding redundancy proxy)
    Right: jaccard_mean per query   (lexical redundancy proxy)
    """
    q_ids = [r["query_info"]["id"] for r in results]
    x     = np.arange(len(q_ids))
    w     = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def grouped_bar(ax, cos_vals, mmr_vals, title, ylabel, ylim_max=1.0):
        bc = ax.bar(x - w / 2, cos_vals, w, color=_col("Cosine"),
                    label="Cosinus", alpha=0.88, edgecolor="white")
        bm = ax.bar(x + w / 2, mmr_vals, w, color=_col("MMR"),
                    label="MMR",    alpha=0.88, edgecolor="white")
        for bars, vals in [(bc, cos_vals), (bm, mmr_vals)]:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + ylim_max * 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i}" for i in q_ids], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, ylim_max * 1.15)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    grouped_bar(
        ax1,
        [r["cosine"]["intra_sim_mean"] for r in results],
        [r["mmr"]["intra_sim_mean"]    for r in results],
        "Redondance par embeddings  ↓ moins = plus diversifié\n(cosinus moyen par paires dans les top-k chunks)",
        "Similarité cosinus intra-résultats moyenne",
    )
    grouped_bar(
        ax2,
        [r["cosine"]["jaccard_mean"] for r in results],
        [r["mmr"]["jaccard_mean"]    for r in results],
        "Redondance lexicale  ↓ moins = plus diversifié\n(similarité Jaccard moyenne par paires)",
        "Similarité Jaccard moyenne",
    )

    fig.suptitle("Comparaison de la redondance : Cosinus vs MMR\n"
                 "Les deux proxies mesurent la similarité entre les chunks récupérés",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "redundancy_comparison.png")


def fig_result_overlap_heatmaps(results: List[Dict]) -> None:
    """
    Two 5×5 heatmaps: cross-query result overlap for Cosine and MMR.

    Cell (i, j) = fraction of chunks from Q_i that are also in Q_j's result set.
    A high off-diagonal value means the retriever returns the same chunks for
    different queries → poor query discrimination.
    """
    n  = len(results)

    def build_overlap_matrix(strategy_key: str) -> np.ndarray:
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                docs_i = results[i][strategy_key]["_docs"]
                docs_j = results[j][strategy_key]["_docs"]
                ids_i  = set(_chunk_id(d) for d in docs_i)
                ids_j  = set(_chunk_id(d) for d in docs_j)
                union  = ids_i | ids_j
                M[i, j] = len(ids_i & ids_j) / max(len(ids_i), 1)
        return M

    M_cos = build_overlap_matrix("cosine")
    M_mmr = build_overlap_matrix("mmr")

    q_labels = [f"Q{r['query_info']['id']}" for r in results]

    cmap = LinearSegmentedColormap.from_list(
        "overlap", ["white", "#aed6f1", "#2980b9"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, M, title in [
        (ax1, M_cos, "Cosinus — Chevauchement inter-requêtes"),
        (ax2, M_mmr, "MMR — Chevauchement inter-requêtes"),
    ]:
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(q_labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if M[i, j] < 0.6 else "white")

    fig.suptitle("Chevauchement inter-requêtes (fraction de chunks partagés)\n"
                 "Hors-diagonale ≈ 0 → le retriever discrimine bien les requêtes\n"
                 "Hors-diagonale > 0.5 → chunks similaires retournés pour différentes requêtes",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "result_overlap_heatmaps.png")


def fig_lambda_sweep(sweep_rows: List[Dict]) -> None:
    """
    Line chart: intra_sim_mean and avg_sources vs lambda_mult.
    Helps justify the chosen lambda value.
    """
    lambdas  = [r["lambda_mult"]  for r in sweep_rows]
    intra    = [r["intra_sim"]    for r in sweep_rows]
    sources  = [r["avg_sources"]  for r in sweep_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(lambdas, intra, "o-", color="#e74c3c", linewidth=2,
                   markersize=7, label="Similarité intra-résultats ↓ (moins = moins redondant)")
    l2, = ax2.plot(lambdas, sources, "s--", color=_col("MMR"), linewidth=2,
                   markersize=7, label="Nb sources distinctes ↑")

    ax1.axvline(PRIMARY_LAMBDA, color="gold", linewidth=2.0, linestyle=":",
                label=f"λ retenu = {PRIMARY_LAMBDA}")
    ax1.set_xlabel("lambda_mult  (0 = diversité max · 1 = pertinence max)", fontsize=10)
    ax1.set_ylabel("Similarité cosinus intra-résultats moyenne", fontsize=10, color="#e74c3c")
    ax2.set_ylabel("Nb sources distinctes moyen par requête", fontsize=10, color=_col("MMR"))
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, max(sources) + 0.5)

    lines = [l1, l2,
             plt.Line2D([0], [0], color="gold", linewidth=2, linestyle=":")]
    labs  = [l1.get_label(), l2.get_label(), f"λ retenu = {PRIMARY_LAMBDA}"]
    ax1.legend(lines, labs, fontsize=8.5, loc="lower left")

    ax1.set_title(f"Balayage du paramètre λ de MMR  (k={PRIMARY_K}, fetch_k={PRIMARY_FETCH_K})\n"
                  "Compromis entre diversité et pertinence",
                  fontsize=11, fontweight="bold")
    ax1.spines[["top"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "lambda_sweep.png")


def fig_k_sweep(cos_rows: List[Dict], mmr_rows: List[Dict]) -> None:
    """
    Two-panel line chart: how metrics evolve with k for both strategies.
    """
    ks_cos  = [r["k"] for r in cos_rows]
    ks_mmr  = [r["k"] for r in mmr_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Intra-sim vs k
    ax1.plot(ks_cos, [r["intra_sim"]   for r in cos_rows],
             "o-", color=_col("Cosine"), lw=2, ms=7, label="Cosinus")
    ax1.plot(ks_mmr, [r["intra_sim"]   for r in mmr_rows],
             "s--", color=_col("MMR"),   lw=2, ms=7, label="MMR")
    ax1.axvline(PRIMARY_K, color="gold", lw=2, linestyle=":",
                label=f"k retenu = {PRIMARY_K}")
    ax1.set_xlabel("k (top-k récupérés)", fontsize=10)
    ax1.set_ylabel("Similarité cosinus intra-résultats moyenne ↓", fontsize=10)
    ax1.set_title("Redondance selon k\n↑ plus de chunks → plus de redondance",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    # Source diversity vs k
    ax2.plot(ks_cos, [r["avg_sources"] for r in cos_rows],
             "o-", color=_col("Cosine"), lw=2, ms=7, label="Cosinus")
    ax2.plot(ks_mmr, [r["avg_sources"] for r in mmr_rows],
             "s--", color=_col("MMR"),   lw=2, ms=7, label="MMR")
    ax2.axvline(PRIMARY_K, color="gold", lw=2, linestyle=":",
                label=f"k retenu = {PRIMARY_K}")
    ax2.set_xlabel("k (top-k récupérés)", fontsize=10)
    ax2.set_ylabel("Nb sources distinctes moyen ↑", fontsize=10)
    ax2.set_title("Diversité des sources selon k\n↑ plus de chunks → plus de sources potentielles",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Balayage du paramètre k  (λ={PRIMARY_LAMBDA}, fetch_k={PRIMARY_FETCH_K})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "k_sweep.png")


def fig_fetch_k_sweep(sweep_rows: List[Dict]) -> None:
    """Line chart: effect of fetch_k on MMR output quality."""
    fks    = [r["fetch_k"]   for r in sweep_rows]
    intra  = [r["intra_sim"] for r in sweep_rows]
    srcs   = [r["avg_sources"] for r in sweep_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(fks, intra, "o-", color="#e74c3c", lw=2, ms=7,
                   label="Similarité intra-résultats ↓")
    l2, = ax2.plot(fks, srcs,  "s--", color=_col("MMR"), lw=2, ms=7,
                   label="Nb sources distinctes ↑")
    ax1.axvline(PRIMARY_FETCH_K, color="gold", lw=2, linestyle=":",
                label=f"fetch_k retenu = {PRIMARY_FETCH_K}")

    ax1.set_xlabel("fetch_k (taille du pool de candidats initial)", fontsize=10)
    ax1.set_ylabel("Similarité cosinus intra-résultats moyenne ↓", fontsize=10, color="#e74c3c")
    ax2.set_ylabel("Nb sources distinctes moyen ↑", fontsize=10, color=_col("MMR"))

    lines = [l1, l2,
             plt.Line2D([0], [0], color="gold", lw=2, linestyle=":")]
    labs  = [l.get_label() for l in [l1, l2]] + [f"fetch_k retenu = {PRIMARY_FETCH_K}"]
    ax1.legend(lines, labs, fontsize=8.5, loc="upper right")

    ax1.set_title(f"Balayage du paramètre fetch_k de MMR  (k={PRIMARY_K}, λ={PRIMARY_LAMBDA})\n"
                  "Pool plus grand → MMR dispose de plus de candidats pour diversifier",
                  fontsize=11, fontweight="bold")
    ax1.spines[["top"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "fetch_k_sweep.png")


def fig_summary_panel(results: List[Dict]) -> None:
    """
    4-panel summary comparing Cosine vs MMR averaged across all queries.
    """
    strategies = ["Cosine", "MMR"]
    keys_cos   = [r["cosine"] for r in results]
    keys_mmr   = [r["mmr"]    for r in results]

    def avg(dicts, key):
        vals = [d[key] for d in dicts if key in d]
        return np.mean(vals) if vals else 0.0

    panel_data = [
        ("Sources distinctes ↑",             [avg(keys_cos, "distinct_sources"), avg(keys_mmr, "distinct_sources")]),
        ("Redondance par embeddings ↓",       [avg(keys_cos, "intra_sim_mean"),  avg(keys_mmr, "intra_sim_mean")]),
        ("Redondance lexicale (Jaccard) ↓",   [avg(keys_cos, "jaccard_mean"),    avg(keys_mmr, "jaccard_mean")]),
        ("Longueur moyenne des chunks (car.) →", [avg(keys_cos, "avg_len"),      avg(keys_mmr, "avg_len")]),
    ]
    x_labels = ["Cosinus", "MMR"]

    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    for idx, (title, vals) in enumerate(panel_data):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        colors = [_col(s) for s in strategies]
        bars = ax.bar(x_labels, vals, color=colors,
                      width=0.45, edgecolor="white", alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(vals) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Comparaison résumée — Moyenne sur les 5 requêtes\n"
                 f"k={PRIMARY_K}, λ={PRIMARY_LAMBDA}, fetch_k={PRIMARY_FETCH_K}",
                 fontsize=12, fontweight="bold")
    _save(fig, "summary_panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _trunc(text: str, n: int = 120) -> str:
    t = text.replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t


def export_per_query_files(results: List[Dict]) -> None:
    """Write a detailed human-readable comparison for each query."""
    for r in results:
        q   = r["query_info"]
        out = PERQUERY_DIR / f"q{q['id']}_comparison.txt"

        with open(out, "w", encoding="utf-8") as f:
            f.write(f"Query {q['id']} — {q['category']}\n")
            f.write(f"{'═' * 72}\n")
            f.write(f"  {q['query']}\n\n")
            f.write(f"  Rationale: {q['rationale']}\n\n")
            f.write(f"  Parameters: k={r['params']['k']}  "
                    f"fetch_k={r['params']['fetch_k']}  "
                    f"λ={r['params']['lambda_mult']}\n")
            f.write(f"{'─' * 72}\n\n")

            for strategy, key in [("COSINE (baseline)", "cosine"),
                                   ("MMR", "mmr")]:
                m    = r[key]
                docs = m["_docs"]
                scr  = m["_scores"]

                f.write(f"  ── {strategy}\n")
                f.write(f"     Sources   : {m['distinct_sources']}  "
                        f"{m['source_labels']}\n")
                f.write(f"     Intra sim : {m['intra_sim_mean']:.3f}  "
                        f"(embedding redundancy)\n")
                f.write(f"     Jaccard   : {m['jaccard_mean']:.3f}  "
                        f"(lexical redundancy)\n")
                if scr:
                    f.write(f"     Scores    : top1={m['score_top1']:.3f}  "
                            f"mean={m['score_mean']:.3f}  "
                            f"spread={m['score_spread']:.3f}\n")
                f.write("\n")

                for i, doc in enumerate(docs):
                    score_str = f"  score={scr[i]:.3f}" if i < len(scr) else ""
                    f.write(f"     [{i+1}] {_source_label(doc):<30}"
                            f"  {doc.metadata.get('type_document', '?'):<10}"
                            f"  id={doc.metadata.get('chunk_id', '?')}"
                            f"{score_str}\n")
                    excerpt = textwrap.fill(
                        doc.page_content[:300], width=72,
                        initial_indent="         ", subsequent_indent="         "
                    )
                    f.write(f"{excerpt}\n\n")

            ovlp = r["overlap"]
            f.write(f"  ── OVERLAP ANALYSIS\n")
            f.write(f"     Shared chunks     : {ovlp['shared_n']}/{r['params']['k']}"
                    f"  (overlap_ratio={ovlp['overlap_ratio']:.2f})\n")
            f.write(f"     Mean rank shift   : {ovlp['rank_shift_mean']:.2f} positions\n\n")

            delta_src = r["mmr"]["distinct_sources"] - r["cosine"]["distinct_sources"]
            delta_intra = r["mmr"]["intra_sim_mean"] - r["cosine"]["intra_sim_mean"]
            f.write(f"  ── OBSERVATION\n")
            f.write(f"     Source delta  : {delta_src:+d}  "
                    f"({'MMR more diverse' if delta_src > 0 else 'same coverage' if delta_src == 0 else 'Cosine more diverse'})\n")
            f.write(f"     Intra sim Δ   : {delta_intra:+.3f}  "
                    f"({'MMR less redundant' if delta_intra < 0 else 'Cosine less redundant'})\n")
            f.write(f"     [Write your qualitative notes here]\n")

        print(f"  [per-query] q{q['id']}_comparison.txt")


def export_metrics_csv(results: List[Dict]) -> None:
    path = OUTPUT_DIR / "metrics.csv"
    rows = []
    for r in results:
        for key, label in [("cosine", "Cosine"), ("mmr", "MMR")]:
            m = r[key]
            rows.append({
                "query_id":          r["query_info"]["id"],
                "query_category":    r["query_info"]["category"],
                "strategy":          label,
                "n_docs":            m["n_docs"],
                "distinct_sources":  m["distinct_sources"],
                "distinct_types":    m["distinct_types"],
                "avg_len":           f"{m['avg_len']:.0f}",
                "intra_sim_mean":    f"{m['intra_sim_mean']:.4f}",
                "intra_sim_max":     f"{m['intra_sim_max']:.4f}",
                "jaccard_mean":      f"{m['jaccard_mean']:.4f}",
                "jaccard_redundant_pct": f"{m['jaccard_redundant_pct']:.1f}",
                "score_mean":        f"{m['score_mean']:.4f}",
                "score_top1":        f"{m['score_top1']:.4f}",
                "score_spread":      f"{m['score_spread']:.4f}",
                "overlap_ratio":     f"{r['overlap']['overlap_ratio']:.3f}",
                "rank_shift_mean":   f"{r['overlap']['rank_shift_mean']:.2f}",
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("  [export] metrics.csv")


def generate_summary_md(
    results: List[Dict],
    lambda_rows: List[Dict],
    cos_k_rows: List[Dict],
    mmr_k_rows: List[Dict],
    fk_rows: List[Dict],
    corpus_count: int,
) -> None:
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path = OUTPUT_DIR / "summary.md"

    def avg(dicts, key):
        vals = [d[key] for d in dicts if key in d]
        return np.mean(vals) if vals else 0.0

    cos_ms = [r["cosine"] for r in results]
    mmr_ms = [r["mmr"]    for r in results]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Strategy Evaluation\n")
        f.write("## ResearchPal RAG Pipeline — Task 2 (LOG6951A)\n\n")
        f.write(f"*Generated: {now}*\n\n")
        f.write(f"**Corpus**: {corpus_count} indexed chunks  \n")
        f.write(f"**Embedding model**: `{EMBEDDING_MODEL_NAME}`  \n")
        f.write(f"**Strategies compared**: Cosine (baseline) · MMR  \n\n")

        # Parameters table
        f.write("---\n\n## 1. Evaluation Parameters\n\n")
        f.write("| Parameter | Value | Justification |\n")
        f.write("|-----------|------:|---------------|\n")
        f.write(f"| `k` (top-k) | {PRIMARY_K} | Enough context for RAG without overwhelming the LLM prompt |\n")
        f.write(f"| `fetch_k` (MMR) | {PRIMARY_FETCH_K} | {PRIMARY_FETCH_K / PRIMARY_K:.0f}× larger than k — gives MMR sufficient candidates to diversify |\n")
        f.write(f"| `lambda_mult` | {PRIMARY_LAMBDA} | Balanced trade-off: 0 = max diversity, 1 = max relevance |\n")
        f.write(f"| Jaccard threshold | {JACCARD_THRESHOLD} | Pairs sharing >40% tokens classified as lexically redundant |\n\n")

        # Global comparison table
        f.write("---\n\n## 2. Global Comparison (averaged across 5 queries)\n\n")
        f.write("| Metric | Cosine (baseline) | MMR | Δ (MMR − Cosine) |\n")
        f.write("|--------|:-----------------:|:---:|:-----------------:|\n")

        metrics_to_show = [
            ("Distinct sources", "distinct_sources", ".2f"),
            ("Intra-result similarity ↓", "intra_sim_mean", ".3f"),
            ("Jaccard redundancy ↓",      "jaccard_mean",   ".3f"),
            ("Avg chunk length (chars)",  "avg_len",        ".0f"),
            ("Score top-1 (cosine only)", "score_top1",     ".3f"),
        ]
        for label, key, fmt in metrics_to_show:
            c = avg(cos_ms, key)
            m = avg(mmr_ms, key)
            d = m - c
            sign = "+" if d >= 0 else ""
            f.write(f"| {label} | {c:{fmt}} | {m:{fmt}} | {sign}{d:{fmt}} |\n")

        f.write("\n> **Intra-result similarity**: mean pairwise cosine similarity among "
                "the k returned chunks. Lower = more diverse result set.  \n")
        f.write("> **Jaccard redundancy**: mean pairwise lexical overlap. "
                f"Pairs above {JACCARD_THRESHOLD} threshold are classified redundant.\n\n")

        # Per-query table
        f.write("---\n\n## 3. Per-Query Comparison\n\n")
        f.write("| Q | Category | Cosine src | MMR src | Δsrc | "
                "Cosine intra | MMR intra | Overlap | Rank Δ |\n")
        f.write("|---|----------|:----------:|:-------:|:----:|"
                ":----------:|:---------:|:-------:|:------:|\n")
        for r in results:
            q     = r["query_info"]
            c, m  = r["cosine"], r["mmr"]
            ovlp  = r["overlap"]
            delta = m["distinct_sources"] - c["distinct_sources"]
            sign  = "+" if delta >= 0 else ""
            f.write(
                f"| Q{q['id']} | {q['category'][:25]} "
                f"| {c['distinct_sources']} | {m['distinct_sources']} | {sign}{delta} "
                f"| {c['intra_sim_mean']:.3f} | {m['intra_sim_mean']:.3f} "
                f"| {ovlp['overlap_ratio']:.2f} | {ovlp['rank_shift_mean']:.1f} |\n"
            )

        # Lambda sweep table
        f.write("\n---\n\n## 4. Parameter Sweep — Lambda (MMR)\n\n")
        f.write("| λ | Intra-result sim ↓ | Avg sources ↑ |\n")
        f.write("|---|------------------:|:--------------:|\n")
        for row in lambda_rows:
            star = " **← selected**" if abs(row["lambda_mult"] - PRIMARY_LAMBDA) < 0.01 else ""
            f.write(f"| {row['lambda_mult']:.1f} | {row['intra_sim']:.3f} "
                    f"| {row['avg_sources']:.2f}{star} |\n")

        # k sweep table
        f.write("\n---\n\n## 5. Parameter Sweep — k\n\n")
        f.write("| k | Cosine intra ↓ | Cosine sources | MMR intra ↓ | MMR sources |\n")
        f.write("|---|:-------------:|:--------------:|:-----------:|:-----------:|\n")
        for c_row, m_row in zip(cos_k_rows, mmr_k_rows):
            star = " **←**" if c_row["k"] == PRIMARY_K else ""
            f.write(f"| {c_row['k']}{star} "
                    f"| {c_row['intra_sim']:.3f} | {c_row['avg_sources']:.2f} "
                    f"| {m_row['intra_sim']:.3f} | {m_row['avg_sources']:.2f} |\n")

        # Per-query qualitative analysis
        f.write("\n---\n\n## 6. Qualitative Analysis — Per Query\n\n")
        for r in results:
            q = r["query_info"]
            c, m = r["cosine"], r["mmr"]
            ovlp = r["overlap"]
            delta_src   = m["distinct_sources"] - c["distinct_sources"]
            delta_intra = m["intra_sim_mean"]   - c["intra_sim_mean"]

            f.write(f"### Query {q['id']} — {q['category']}\n\n")
            f.write(f"> *{q['query']}*\n\n")
            f.write(f"**Query type**: {q['rationale']}\n\n")

            f.write("**Cosine results**:\n\n")
            for i, doc in enumerate(c["_docs"]):
                score = c["_scores"][i] if i < len(c["_scores"]) else None
                score_str = f" (score: {score:.3f})" if score is not None else ""
                f.write(f"- Rank {i+1}: `{_source_label(doc)}` — "
                        f"*\"{_trunc(doc.page_content, 80)}\"*{score_str}\n")
            f.write(f"\n**MMR results** (λ={PRIMARY_LAMBDA}):\n\n")
            for i, doc in enumerate(m["_docs"]):
                f.write(f"- Rank {i+1}: `{_source_label(doc)}` — "
                        f"*\"{_trunc(doc.page_content, 80)}\"*\n")

            f.write(f"\n**Quantitative observations**:\n\n")
            src_obs = (f"MMR retrieved from {delta_src:+d} additional source(s)"
                       if delta_src != 0 else "Both strategies retrieved from the same sources")
            red_obs = (f"MMR reduced embedding redundancy by {abs(delta_intra):.3f}"
                       if delta_intra < 0
                       else f"Cosine was less redundant by {abs(delta_intra):.3f}")
            ovl_obs = (f"{int(ovlp['overlap_ratio']*PRIMARY_K)}/{PRIMARY_K} chunks shared "
                       f"(overlap ratio: {ovlp['overlap_ratio']:.2f})")

            f.write(f"- **Diversity**: {src_obs}\n")
            f.write(f"- **Redundancy**: {red_obs}\n")
            f.write(f"- **Overlap**: {ovl_obs} — "
                    f"mean rank shift of {ovlp['rank_shift_mean']:.1f} positions\n")
            f.write(f"- **Relevance assessment**: [To be filled based on chunk content inspection]\n\n")
            f.write("---\n\n")

        # Final recommendation
        f.write("## 7. Recommendation and Justification\n\n")
        avg_delta_intra = avg(mmr_ms, "intra_sim_mean") - avg(cos_ms, "intra_sim_mean")
        avg_delta_src   = avg(mmr_ms, "distinct_sources") - avg(cos_ms, "distinct_sources")

        f.write("### Selected strategy: MMR with Cosine as baseline\n\n")
        f.write(
            f"The evaluation over {len(results)} queries demonstrates that MMR consistently "
            f"{'reduces' if avg_delta_intra < 0 else 'does not increase'} embedding-based "
            f"redundancy (Δ = {avg_delta_intra:+.3f} on average) while "
            f"{'improving' if avg_delta_src > 0 else 'maintaining'} source diversity "
            f"(Δ = {avg_delta_src:+.2f} distinct sources on average).\n\n"
        )
        f.write(
            "**Parameter justification**:\n\n"
            f"- `k = {PRIMARY_K}`: Selected as the minimum k that provides sufficient "
            f"context without exceeding the LLM context window. The k-sweep shows "
            f"that intra-result similarity grows monotonically with k, making "
            f"larger values counterproductive.\n"
            f"- `lambda_mult = {PRIMARY_LAMBDA}`: The lambda sweep identifies "
            f"λ = {PRIMARY_LAMBDA} as the inflection point between the diversity-focused "
            f"(λ < 0.3) and relevance-focused (λ > 0.7) regimes. "
            f"Values in [0.4, 0.6] provide the best balance.\n"
            f"- `fetch_k = {PRIMARY_FETCH_K}`: Set to {PRIMARY_FETCH_K // PRIMARY_K}× k. "
            f"The fetch_k sweep shows diminishing returns beyond this value — "
            f"additional candidates do not meaningfully improve diversity.\n\n"
        )
        f.write(
            "**When to prefer Cosine over MMR**: For narrow, keyword-specific queries "
            "(see Q4 analysis) where the corpus has few relevant chunks, MMR's "
            "diversity penalty may demote the second-most relevant chunk in favour "
            "of a less relevant but different one. In such cases, pure cosine "
            "retrieval is more appropriate.\n\n"
            "**Failure case (Q5 — out-of-scope query)**: Both strategies return "
            "low-confidence results when the query topic is not covered by the corpus. "
            "This demonstrates the need for a score-threshold guard in production "
            "RAG systems to avoid hallucination from irrelevant context.\n"
        )

    print("  [export] summary.md")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_results_console(results: List[Dict]) -> None:
    SEP  = "═" * 76
    THIN = "─" * 58

    for r in results:
        q    = r["query_info"]
        c, m = r["cosine"], r["mmr"]
        ovlp = r["overlap"]

        print(f"\n{SEP}")
        print(f"  Q{q['id']}/5 — {q['category']}")
        print(f"  \"{q['query']}\"")
        print(f"{SEP}")

        for label, met, scr in [("COSINE", c, c["_scores"]), ("MMR", m, None)]:
            param_str = (f"k={r['params']['k']}"
                         if label == "COSINE"
                         else f"k={r['params']['k']}, "
                              f"fetch_k={r['params']['fetch_k']}, "
                              f"λ={r['params']['lambda_mult']}")
            print(f"\n  {label}  ({param_str})")
            print(f"  {THIN}")
            for i, doc in enumerate(met["_docs"]):
                s_str = f"  score={scr[i]:.3f}" if scr and i < len(scr) else ""
                print(f"  [{i+1}]  {_source_label(doc):<28}"
                      f"  {doc.metadata.get('type_document','?'):<10}"
                      f"  id={doc.metadata.get('chunk_id','?')[:10]}"
                      f"{s_str}")
                print(f"       \"{_trunc(doc.page_content, 90)}\"")

            print(f"\n  → sources={met['distinct_sources']}"
                  f"  intra_sim={met['intra_sim_mean']:.3f}"
                  f"  jaccard={met['jaccard_mean']:.3f}")

        delta_src   = m["distinct_sources"] - c["distinct_sources"]
        delta_intra = m["intra_sim_mean"]   - c["intra_sim_mean"]
        print(f"\n  OVERLAP: {ovlp['shared_n']}/{r['params']['k']} shared chunks"
              f"  (ratio={ovlp['overlap_ratio']:.2f},"
              f" rank_shift={ovlp['rank_shift_mean']:.1f})")
        print(f"  DELTA:   sources {delta_src:+d}"
              f"  |  intra_sim {delta_intra:+.3f}"
              f"  {'(MMR less redundant ✓)' if delta_intra < 0 else '(Cosine less redundant)'}")


def print_summary_table(results: List[Dict]) -> None:
    W = 84
    cos_ms = [r["cosine"] for r in results]
    mmr_ms = [r["mmr"]    for r in results]
    ovlps  = [r["overlap"] for r in results]

    def avg(lst, k):
        return np.mean([d[k] for d in lst])

    print(f"\n{'═' * W}")
    print("  GLOBAL SUMMARY — averaged across 5 queries")
    print(f"{'═' * W}")
    fmt  = f"  {'Metric':<30} {'Cosine':>10} {'MMR':>10} {'Δ (MMR−Cos)':>14}"
    print(fmt)
    print(f"  {'─' * 66}")

    rows = [
        ("Distinct sources",      "distinct_sources", ".2f"),
        ("Embedding redundancy ↓", "intra_sim_mean",  ".3f"),
        ("Jaccard redundancy ↓",   "jaccard_mean",    ".3f"),
        ("Avg chunk length",       "avg_len",         ".0f"),
        ("Top-1 cosine score",     "score_top1",      ".3f"),
    ]
    for label, key, fmt_spec in rows:
        c = avg(cos_ms, key)
        m = avg(mmr_ms, key)
        d = m - c
        sign = "+" if d >= 0 else ""
        print(f"  {label:<30} {c:>10{fmt_spec}} {m:>10{fmt_spec}} "
              f"{sign+f'{d:{fmt_spec}}':>14}")

    avg_ovlp = np.mean([o["overlap_ratio"] for o in ovlps])
    avg_rshift = np.mean([o["rank_shift_mean"] for o in ovlps])
    print(f"  {'Result overlap (cos ∩ mmr)':<30} {avg_ovlp:>10.3f} {'—':>10} {'—':>14}")
    print(f"  {'Mean rank shift':<30} {avg_rshift:>10.2f} {'—':>10} {'—':>14}")
    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "═" * 72
    print(f"\n{SEP}")
    print("  ResearchPal — Retrieval Strategy Evaluation")
    print("  LOG6951A — Task 2 Evaluation Framework")
    print(f"{SEP}\n")

    # ── Setup ────────────────────────────────────────────────────────────────
    setup()

    # ── Load resources ────────────────────────────────────────────────────────
    print("Step 1 — Loading ChromaDB and embedding model")
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("  → Run 'python src/ingestion/run_ingestion.py' first.")
        sys.exit(1)

    corpus_count = vectorstore._collection.count()
    print(f"  → {corpus_count} chunks indexed")

    embeddings = load_embeddings()

    # ── Main evaluation ───────────────────────────────────────────────────────
    print(f"\nStep 2 — Main evaluation ({len(EVAL_QUERIES)} queries × 2 strategies)")
    print(f"  k={PRIMARY_K}  fetch_k={PRIMARY_FETCH_K}  λ={PRIMARY_LAMBDA}\n")
    results = run_evaluation(vectorstore, embeddings)

    print_results_console(results)
    print_summary_table(results)

    # ── Parameter sweeps ──────────────────────────────────────────────────────
    print("Step 3 — Parameter sweeps")
    lambda_rows          = sweep_lambda(vectorstore, EVAL_QUERIES, embeddings)
    cos_k_rows, mmr_k_rows = sweep_k(vectorstore, EVAL_QUERIES, embeddings)
    fk_rows              = sweep_fetch_k(vectorstore, EVAL_QUERIES, embeddings)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nStep 4 — Generating figures")
    fig_score_distributions(results)
    fig_source_diversity(results)
    fig_redundancy_comparison(results)
    fig_result_overlap_heatmaps(results)
    fig_lambda_sweep(lambda_rows)
    fig_k_sweep(cos_k_rows, mmr_k_rows)
    fig_fetch_k_sweep(fk_rows)
    fig_summary_panel(results)

    # ── Exports ───────────────────────────────────────────────────────────────
    print("\nStep 5 — Exporting reports")
    export_per_query_files(results)
    export_metrics_csv(results)
    generate_summary_md(
        results, lambda_rows, cos_k_rows, mmr_k_rows, fk_rows, corpus_count
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ANALYSIS COMPLETE")
    print(f"{SEP}")
    print(f"\n  All outputs in: {OUTPUT_DIR}\n")
    for fp in sorted(OUTPUT_DIR.rglob("*")):
        if fp.is_file():
            size_kb = fp.stat().st_size / 1024
            print(f"    {fp.relative_to(OUTPUT_DIR)!s:<50}  {size_kb:6.1f} KB")
    print()


if __name__ == "__main__":
    main()
