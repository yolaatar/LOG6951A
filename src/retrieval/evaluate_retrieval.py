# evaluate_retrieval.py — évaluation comparée cosinus vs MMR (Tâche 2)
# usage : python src/retrieval/evaluate_retrieval.py
#         python src/retrieval/evaluate_retrieval.py --export
#         python src/retrieval/evaluate_retrieval.py --param-sweep
#         python src/retrieval/evaluate_retrieval.py --k 5 --lambda-mult 0.3 --export

import sys
import argparse
from datetime import datetime, timezone
from itertools import combinations, product as iproduct
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from ingestion.indexer import load_vectorstore
from retrieval.cosine_retriever import cosine_search_with_scores
from retrieval.mmr_retriever import mmr_search
from retrieval.eval_queries import EVAL_QUERIES
from config import RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA


# ── métriques simples ───────────────────────────────────────────────────────

def _source_label(doc: Document) -> str:
    """Nom court de la source (filename ou URL tronquée)."""
    src = doc.metadata.get("source", "?")
    if src.startswith("http"):
        return src
    return Path(src).name


def _jaccard(a: str, b: str) -> float:
    """Similarité Jaccard sur les mots (tokens) — heuristique de redondance."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def redundancy(docs: List[Document], threshold: float = 0.5) -> Tuple[int, int]:
    """Retourne (nb paires redondantes, nb paires total)."""
    pairs = list(combinations(docs, 2))
    if not pairs:
        return 0, 0
    n_redundant = sum(
        1 for d1, d2 in pairs
        if _jaccard(d1.page_content, d2.page_content) >= threshold
    )
    return n_redundant, len(pairs)


def compute_metrics(docs: List[Document]) -> Dict[str, Any]:
    sources = {_source_label(d) for d in docs}
    types = {d.metadata.get("type_document", "?") for d in docs}
    avg_len = round(sum(len(d.page_content) for d in docs) / len(docs)) if docs else 0
    r_count, r_total = redundancy(docs)
    return {
        "n_docs": len(docs),
        "distinct_sources": len(sources),
        "source_labels": sorted(sources),
        "distinct_types": len(types),
        "type_labels": sorted(types),
        "avg_length": avg_len,
        "redundant_pairs": r_count,
        "total_pairs": r_total,
    }


def _metrics_line(m: Dict) -> str:
    return (
        f"sources={m['distinct_sources']} {m['source_labels']}, "
        f"types={m['distinct_types']} {m['type_labels']}, "
        f"longueur moy.={m['avg_length']} car., "
        f"redondance={m['redundant_pairs']}/{m['total_pairs']} paires"
    )


def _observation(m_cos: Dict, m_mmr: Dict) -> str:
    parts = []
    delta_src = m_mmr["distinct_sources"] - m_cos["distinct_sources"]
    delta_red = m_mmr["redundant_pairs"] - m_cos["redundant_pairs"]

    if delta_src > 0:
        parts.append(f"MMR couvre {delta_src} source(s) de plus → meilleure diversité ✓")
    elif delta_src < 0:
        parts.append("Cosinus couvre plus de sources (inhabituel)")
    else:
        parts.append("Même couverture de sources pour les deux stratégies")

    if delta_red < 0:
        parts.append(f"MMR réduit la redondance de {abs(delta_red)} paire(s) ✓")
    elif delta_red > 0:
        parts.append("MMR introduit plus de redondance (augmenter λ si indésirable)")

    return " | ".join(parts)


# ── affichage console ───────────────────────────────────────────────────────

def _truncate(text: str, n: int = 100) -> str:
    t = text.replace("\n", " ").strip()
    return t[:n] + "…" if len(t) > n else t


def print_query_results(
    results_cos: List[Tuple[Document, float]],
    results_mmr: List[Document],
    q_info: Dict,
    k: int,
    fetch_k: int,
    lam: float,
) -> None:
    sep = "═" * 72
    thin = "─" * 50
    print(f"\n{sep}")
    print(f"  Requête {q_info['id']}/5 — {q_info['category']}")
    print(f"  \"{q_info['query']}\"")
    print(f"{sep}")

    # cosinus
    docs_cos = [d for d, _ in results_cos]
    print(f"\n  COSINUS  (k={k})")
    print(f"  {thin}")
    for i, (doc, score) in enumerate(results_cos):
        src = _source_label(doc)
        doc_type = doc.metadata.get("type_document", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(f"  [{i+1}]  {src:<24}  {doc_type:<10}  score={score:.3f}  id={chunk_id}")
        print(f"        \"{_truncate(doc.page_content, 88)}\"")
    m_cos = compute_metrics(docs_cos)
    print(f"\n  → {_metrics_line(m_cos)}")

    # MMR
    print(f"\n  MMR      (k={k}, fetch_k={fetch_k}, λ={lam})")
    print(f"  {thin}")
    for i, doc in enumerate(results_mmr):
        src = _source_label(doc)
        doc_type = doc.metadata.get("type_document", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(f"  [{i+1}]  {src:<24}  {doc_type:<10}  score=n/a    id={chunk_id}")
        print(f"        \"{_truncate(doc.page_content, 88)}\"")
    m_mmr = compute_metrics(results_mmr)
    print(f"\n  → {_metrics_line(m_mmr)}")

    print(f"\n  OBSERVATION : {_observation(m_cos, m_mmr)}")


def print_summary(all_metrics: List[Dict]) -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print("  SYNTHÈSE — moyennes sur les 5 requêtes")
    print(f"{sep}")
    print(f"  {'Stratégie':<12}  {'Sources':>8}  {'Types':>6}  {'Longueur':>10}  {'Redondance':>12}")
    print(f"  {'':12}  {'distinctes':>8}  {'':>6}  {'moy.(car.)':>10}  {'paires':>12}")
    print(f"  {'─'*62}")
    for label, key in [("Cosinus", "cosine"), ("MMR", "mmr")]:
        ml = [m[key] for m in all_metrics]
        n = len(ml)
        print(
            f"  {label:<12}  "
            f"{sum(m['distinct_sources'] for m in ml)/n:>8.1f}  "
            f"{sum(m['distinct_types'] for m in ml)/n:>6.1f}  "
            f"{sum(m['avg_length'] for m in ml)/n:>10.0f}  "
            f"{sum(m['redundant_pairs'] for m in ml)/n:>12.1f}"
        )
    print(f"{sep}\n")


# ── balayage de paramètres ──────────────────────────────────────────────────

def run_param_sweep(vectorstore, queries: List[Dict]) -> None:
    K_VALUES = [3, 5]
    FETCH_K_VALUES = [8, 12]
    LAMBDA_VALUES = [0.3, 0.5, 0.7]
    total = len(K_VALUES) * len(FETCH_K_VALUES) * len(LAMBDA_VALUES)

    print(f"\n{'═' * 72}")
    print(f"  BALAYAGE DE PARAMÈTRES MMR  ({total} combinaisons × {len(queries)} requêtes)")
    print(f"{'═' * 72}")
    print(f"  {'k':>3}  {'fetch_k':>7}  {'λ':>5}  │  {'sources':>7}  {'types':>6}  {'longueur':>9}  {'redond.':>8}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*5}  ┼  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*8}")

    sweep_rows = []
    for k, fetch_k, lam in iproduct(K_VALUES, FETCH_K_VALUES, LAMBDA_VALUES):
        src_list, typ_list, len_list, red_list = [], [], [], []
        for q in queries:
            docs = mmr_search(vectorstore, q["query"], k=k, fetch_k=fetch_k, lambda_mult=lam)
            m = compute_metrics(docs)
            src_list.append(m["distinct_sources"])
            typ_list.append(m["distinct_types"])
            len_list.append(m["avg_length"])
            red_list.append(m["redundant_pairs"])
        n = len(queries)
        avg_src = sum(src_list) / n
        avg_typ = sum(typ_list) / n
        avg_len = sum(len_list) / n
        avg_red = sum(red_list) / n
        sweep_rows.append((k, fetch_k, lam, avg_src, avg_typ, avg_len, avg_red))
        print(f"  {k:>3}  {fetch_k:>7}  {lam:>5.1f}  │  {avg_src:>7.1f}  {avg_typ:>6.1f}  {avg_len:>9.0f}  {avg_red:>8.1f}")

    # recommandation : maximiser sources distinctes - redondance
    best = max(sweep_rows, key=lambda r: r[3] - r[6])
    print(
        f"\n  Recommandation : k={best[0]}, fetch_k={best[1]}, λ={best[2]:.1f}"
        f"  →  sources moy.={best[3]:.1f}, redondance moy.={best[6]:.1f}"
    )
    print(f"  (critère : maximiser sources_distinctes − redondance)\n")


# ── export markdown ─────────────────────────────────────────────────────────

def _md_row(cells: List) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _md_table(headers: List[str], rows: List[List]) -> str:
    return "\n".join([
        _md_row(headers),
        _md_row(["---"] * len(headers)),
        *[_md_row(r) for r in rows],
    ])


def build_markdown(
    all_results: List[Dict],
    k: int,
    fetch_k: int,
    lam: float,
    corpus_count: int,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Évaluation du Retrieval — ResearchPal",
        "",
        f"**Date** : {now}  ",
        f"**Corpus** : {corpus_count} chunks indexés  ",
        "**Modèle d'embeddings** : sentence-transformers/all-MiniLM-L6-v2",
        "",
        "## Paramètres d'évaluation",
        "",
        _md_table(
            ["Paramètre", "Valeur", "Description"],
            [
                ["k", k, "Chunks retournés par requête"],
                ["fetch_k (MMR)", fetch_k, "Candidats initiaux avant sélection MMR"],
                ["λ (MMR)", lam, "0 = diversité max · 1 = pertinence max"],
            ],
        ),
        "",
        "---",
        "",
    ]

    all_cos_m, all_mmr_m = [], []

    for entry in all_results:
        q = entry["query_info"]
        results_cos: List[Tuple[Document, float]] = entry["cosine"]
        docs_mmr: List[Document] = entry["mmr"]
        docs_cos = [d for d, _ in results_cos]
        m_cos = compute_metrics(docs_cos)
        m_mmr = compute_metrics(docs_mmr)
        all_cos_m.append(m_cos)
        all_mmr_m.append(m_mmr)

        lines += [
            f"## Requête {q['id']} — {q['category']}",
            "",
            f"> **{q['query']}**",
            "",
            f"*Intérêt pour la comparaison* : {q['rationale']}",
            "",
            "### Résultats Cosinus",
            "",
            _md_table(
                ["#", "Source", "Type", "chunk_id", "Score", "Extrait (100 car.)"],
                [
                    [i+1, _source_label(d), d.metadata.get("type_document","?"),
                     d.metadata.get("chunk_id","?"), f"{s:.3f}",
                     f'"{_truncate(d.page_content, 100)}"']
                    for i, (d, s) in enumerate(results_cos)
                ],
            ),
            "",
            f"**Métriques** : {_metrics_line(m_cos)}",
            "",
            f"### Résultats MMR (fetch_k={fetch_k}, λ={lam})",
            "",
            _md_table(
                ["#", "Source", "Type", "chunk_id", "Score", "Extrait (100 car.)"],
                [
                    [i+1, _source_label(d), d.metadata.get("type_document","?"),
                     d.metadata.get("chunk_id","?"), "n/a",
                     f'"{_truncate(d.page_content, 100)}"']
                    for i, d in enumerate(docs_mmr)
                ],
            ),
            "",
            f"**Métriques** : {_metrics_line(m_mmr)}",
            "",
            "### Observation",
            "",
            _observation(m_cos, m_mmr),
            "",
            "---",
            "",
        ]

    # synthèse
    n = len(all_results)
    lines += [
        "## Synthèse globale",
        "",
        _md_table(
            ["Stratégie", "Sources distinctes (moy.)", "Types (moy.)", "Longueur moy. (car.)", "Redondance (paires moy.)"],
            [
                ["Cosinus",
                 f"{sum(m['distinct_sources'] for m in all_cos_m)/n:.1f}",
                 f"{sum(m['distinct_types'] for m in all_cos_m)/n:.1f}",
                 f"{sum(m['avg_length'] for m in all_cos_m)/n:.0f}",
                 f"{sum(m['redundant_pairs'] for m in all_cos_m)/n:.1f}"],
                ["MMR",
                 f"{sum(m['distinct_sources'] for m in all_mmr_m)/n:.1f}",
                 f"{sum(m['distinct_types'] for m in all_mmr_m)/n:.1f}",
                 f"{sum(m['avg_length'] for m in all_mmr_m)/n:.0f}",
                 f"{sum(m['redundant_pairs'] for m in all_mmr_m)/n:.1f}"],
            ],
        ),
        "",
    ]

    return "\n".join(lines)


# ── point d'entrée ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResearchPal — Évaluation comparée cosinus vs MMR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python src/retrieval/evaluate_retrieval.py\n"
            "  python src/retrieval/evaluate_retrieval.py --export\n"
            "  python src/retrieval/evaluate_retrieval.py --k 5 --lambda-mult 0.3 --export\n"
            "  python src/retrieval/evaluate_retrieval.py --param-sweep\n"
        ),
    )
    parser.add_argument("--k", type=int, default=RETRIEVAL_TOP_K,
                        help=f"Top-k chunks retournés (défaut : {RETRIEVAL_TOP_K})")
    parser.add_argument("--fetch-k", type=int, default=MMR_FETCH_K,
                        help=f"Candidats initiaux MMR (défaut : {MMR_FETCH_K})")
    parser.add_argument("--lambda-mult", type=float, default=MMR_LAMBDA,
                        help=f"λ MMR — 0=diversité, 1=pertinence (défaut : {MMR_LAMBDA})")
    parser.add_argument("--export", nargs="?", const="outputs/retrieval_eval.md",
                        metavar="CHEMIN",
                        help="Export Markdown (défaut : outputs/retrieval_eval.md)")
    parser.add_argument("--param-sweep", action="store_true",
                        help="Lance le balayage de paramètres MMR")
    args = parser.parse_args()

    print("\nResearchPal — Évaluation Retrieval\n")

    print("Chargement de la base ChromaDB...")
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as e:
        print(f"\n[ERREUR] {e}")
        sys.exit(1)

    corpus_count = vectorstore._collection.count()
    print(f"  → {corpus_count} chunks indexés | {len(EVAL_QUERIES)} requêtes de test\n")

    # évaluation principale sur les 5 requêtes
    all_results = []
    all_metrics = []

    for q_info in EVAL_QUERIES:
        results_cos = cosine_search_with_scores(vectorstore, q_info["query"], k=args.k)
        results_mmr = mmr_search(
            vectorstore, q_info["query"],
            k=args.k, fetch_k=args.fetch_k, lambda_mult=args.lambda_mult,
        )
        print_query_results(results_cos, results_mmr, q_info, args.k, args.fetch_k, args.lambda_mult)

        docs_cos = [d for d, _ in results_cos]
        all_results.append({"query_info": q_info, "cosine": results_cos, "mmr": results_mmr})
        all_metrics.append({
            "cosine": compute_metrics(docs_cos),
            "mmr": compute_metrics(results_mmr),
        })

    print_summary(all_metrics)

    # balayage paramètres
    if args.param_sweep:
        run_param_sweep(vectorstore, EVAL_QUERIES)

    # export markdown
    if args.export:
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = build_markdown(all_results, args.k, args.fetch_k, args.lambda_mult, corpus_count)
        out_path.write_text(md, encoding="utf-8")
        print(f"[EXPORT] Rapport écrit dans : {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
