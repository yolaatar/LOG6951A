# test_retrieval.py — vérifie que le retrieval fonctionne sur la base indexée
# prérequis : avoir lancé run_ingestion.py d'abord
# usage : python src/retrieval/test_retrieval.py [--query "..."] [--top-k 4]

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.indexer import load_vectorstore
from retrieval.cosine_retriever import get_cosine_retriever
from retrieval.mmr_retriever import get_mmr_retriever
from config import RETRIEVAL_TOP_K


DEFAULT_QUERIES = [
    "Qu'est-ce que le RAG et comment fonctionne-t-il ?",
    "Comment fonctionne ChromaDB pour le stockage vectoriel ?",
    "Quelles sont les stratégies de retrieval disponibles dans LangChain ?",
]


def display_results(query: str, results: list, strategy: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Stratégie : {strategy}")
    print(f"  Requête   : \"{query}\"")
    print(f"  Résultats : {len(results)} chunk(s)")
    print(f"{'─' * 60}")

    for i, doc in enumerate(results):
        meta = doc.metadata
        source = Path(meta.get("source", "?")).name
        if meta.get("source", "").startswith("http"):
            source = meta.get("source", "?")

        preview = doc.page_content[:150].replace("\n", " ")

        print(f"\n  Chunk {i + 1}")
        print(f"    Source      : {source}")
        print(f"    Type        : {meta.get('type_document', '?')}")
        print(f"    chunk_id    : {meta.get('chunk_id', '?')}")
        print(f"    chunk_index : {meta.get('chunk_index', '?')}")
        print(f"    Contenu     : \"{preview}...\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchPal — Test du retrieval")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--strategy", choices=["cosine", "mmr", "both"], default="both")
    args = parser.parse_args()

    print("\nResearchPal — Test du retrieval\n")

    print("Chargement de la base ChromaDB...")
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as e:
        print(f"\n[ERREUR] {e}")
        sys.exit(1)

    count = vectorstore._collection.count()
    print(f"  → {count} chunk(s) indexé(s).")

    cosine_retriever = get_cosine_retriever(vectorstore, k=args.top_k)
    mmr_retriever = get_mmr_retriever(vectorstore, k=args.top_k)

    queries = [args.query] if args.query else DEFAULT_QUERIES
    print(f"\nTest avec {len(queries)} requête(s), top_k={args.top_k}\n")

    for query in queries:
        print(f"\n{'═' * 60}")
        print(f"  REQUÊTE : \"{query}\"")
        print(f"{'═' * 60}")

        if args.strategy in {"cosine", "both"}:
            try:
                results_cosine = cosine_retriever.invoke(query)
                display_results(query, results_cosine, "Cosinus (similarité)")
            except Exception as exc:
                print(f"\n  [ERREUR cosinus] {exc}")

        if args.strategy in {"mmr", "both"}:
            try:
                results_mmr = mmr_retriever.invoke(query)
                display_results(query, results_mmr, "MMR (diversité)")
            except Exception as exc:
                print(f"\n  [ERREUR MMR] {exc}")

        if args.strategy == "both":
            try:
                sources_cosine = {
                    Path(d.metadata.get("source", "?")).name
                    if not d.metadata.get("source", "").startswith("http") else "web"
                    for d in results_cosine
                }
                sources_mmr = {
                    Path(d.metadata.get("source", "?")).name
                    if not d.metadata.get("source", "").startswith("http") else "web"
                    for d in results_mmr
                }
                print(f"\n  Sources cosinus : {sources_cosine}")
                print(f"  Sources MMR     : {sources_mmr}")
                if sources_cosine != sources_mmr:
                    print("  ✓ MMR a diversifié les sources par rapport au cosinus.")
            except NameError:
                pass

    print(f"\n{'═' * 60}")
    print("\n[SUCCESS] Test terminé.\n")


if __name__ == "__main__":
    main()
