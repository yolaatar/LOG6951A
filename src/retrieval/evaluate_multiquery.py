# evaluate_multiquery.py — comparaison brute vs multi-query + RRF (T4)
#
# Métriques calculées pour chaque requête :
#   - nb_chunks          : nombre de chunks récupérés
#   - nb_sources         : sources de fichiers distinctes
#   - nb_types           : types de documents distincts
#   - variants_generated : variantes générées par le LLM (multi-query seulement)
#
# Usage :
#   python src/retrieval/evaluate_multiquery.py
#   python src/retrieval/evaluate_multiquery.py --export

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.eval_queries import EVAL_QUERIES
from retrieval.cosine_retriever import cosine_search_with_scores
from retrieval.multiquery import multiquery_retrieve
from ingestion.indexer import load_vectorstore
from config import RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA
from rag.chain import get_llm


# ── calcul des métriques ─────────────────────────────────────────────────────

def compute_metrics(docs, variants=None) -> dict:
    sources = set()
    doc_types = set()
    for d in docs:
        src = d.metadata.get("source", "?")
        sources.add(Path(src).name if not src.startswith("http") else src)
        doc_types.add(d.metadata.get("doc_type", "?"))
    return {
        "nb_chunks": len(docs),
        "nb_sources": len(sources),
        "nb_types": len(doc_types),
        "sources": sorted(sources),
        "variants": variants or [],
    }


# ── affichage console ────────────────────────────────────────────────────────

def print_comparison(entry: dict, brute_m: dict, multi_m: dict) -> None:
    SEP = "─" * 70
    print(f"\n{'═' * 70}")
    print(f"  Q{entry['id']} [{entry['category']}]")
    print(f"  {entry['query']}")
    print(f"{'─' * 70}")

    print("  BRUTE (cosine pur)")
    print(f"    chunks : {brute_m['nb_chunks']}  |  sources : {brute_m['nb_sources']}  |  types : {brute_m['nb_types']}")
    print(f"    fichiers : {', '.join(brute_m['sources'])}")

    print("  MULTI-QUERY + RRF (cosine)")
    variants = multi_m["variants"]
    if variants:
        print(f"    variantes ({len(variants)}) :")
        for v in variants:
            print(f"      • {v}")
    else:
        print("    variantes : (aucune — fallback brut)")
    print(f"    chunks : {multi_m['nb_chunks']}  |  sources : {multi_m['nb_sources']}  |  types : {multi_m['nb_types']}")
    print(f"    fichiers : {', '.join(multi_m['sources'])}")

    delta_src = multi_m["nb_sources"] - brute_m["nb_sources"]
    sign = "+" if delta_src >= 0 else ""
    print(f"  Δ sources : {sign}{delta_src}")
    print(f"{'═' * 70}")


# ── export Markdown ──────────────────────────────────────────────────────────

def build_markdown(results: list) -> str:
    lines = [
        "# Évaluation Multi-Query (T4)",
        "",
        "Comparaison **Brute cosine** vs **Multi-Query + RRF** sur 5 requêtes annotées.",
        "",
        "| # | Catégorie | Brute chunks | Brute sources | MQ chunks | MQ sources | Δ sources |",
        "|---|-----------|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        b, m = r["brute"], r["multi"]
        delta = m["nb_sources"] - b["nb_sources"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {r['id']} | {r['category']} "
            f"| {b['nb_chunks']} | {b['nb_sources']} "
            f"| {m['nb_chunks']} | {m['nb_sources']} "
            f"| {sign}{delta} |"
        )

    lines += ["", "## Détail par requête", ""]
    for r in results:
        b, m = r["brute"], r["multi"]
        lines += [
            f"### Q{r['id']} — {r['category']}",
            "",
            f"> {r['query']}",
            "",
            "**Brute (cosine)**",
            f"- Chunks : {b['nb_chunks']}",
            f"- Sources : {', '.join(b['sources'])}",
            "",
            "**Multi-Query + RRF**",
        ]
        if m["variants"]:
            lines.append(f"- Variantes générées ({len(m['variants'])}) :")
            for v in m["variants"]:
                lines.append(f"  - {v}")
        else:
            lines.append("- Variantes : fallback brut")
        lines += [
            f"- Chunks : {m['nb_chunks']}",
            f"- Sources : {', '.join(m['sources'])}",
            "",
        ]
    return "\n".join(lines)


# ── point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation Multi-Query T4")
    parser.add_argument("--export", action="store_true", help="Exporter outputs/multiquery_eval.md")
    parser.add_argument("--queries", type=str, default=None, help="IDs de requêtes (ex: 1,2,3)")
    args = parser.parse_args()

    print("Chargement du vectorstore et du LLM...")
    vectorstore = load_vectorstore()
    llm = get_llm()

    selected_ids = None
    if args.queries:
        selected_ids = {int(q) for q in args.queries.split(",")}

    results = []
    for entry in EVAL_QUERIES:
        if selected_ids and entry["id"] not in selected_ids:
            continue

        question = entry["query"]
        print(f"\n[Q{entry['id']}] {question[:60]}...")

        # brute cosine
        brute_docs = [d for d, _ in cosine_search_with_scores(vectorstore, question, k=RETRIEVAL_TOP_K)]
        brute_m = compute_metrics(brute_docs)

        # multi-query + RRF
        multi_docs, variants = multiquery_retrieve(
            vectorstore, question, llm,
            k=RETRIEVAL_TOP_K, fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA,
            strategy="cosine",
        )
        multi_m = compute_metrics(multi_docs, variants)

        record = {
            "id": entry["id"],
            "category": entry["category"],
            "query": question,
            "brute": brute_m,
            "multi": multi_m,
        }
        results.append(record)
        print_comparison(entry, brute_m, multi_m)

    # résumé global
    avg_delta = sum(r["multi"]["nb_sources"] - r["brute"]["nb_sources"] for r in results) / max(len(results), 1)
    print(f"\n{'═' * 70}")
    print(f"  Résumé : Δ moyen sources = {avg_delta:+.2f} (multi-query vs brute)")
    print(f"{'═' * 70}\n")

    if args.export:
        out_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "multiquery_eval.md"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(build_markdown(results), encoding="utf-8")
        print(f"Rapport exporté : {out_path}")


if __name__ == "__main__":
    main()
