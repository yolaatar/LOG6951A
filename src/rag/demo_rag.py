# demo_rag.py — démonstration console du pipeline RAG (T3 / T4)
#
# Modes :
#   python src/rag/demo_rag.py              → scénario automatique 3 tours
#   python src/rag/demo_rag.py --interactive → mode Q&A interactif
#   python src/rag/demo_rag.py --multiquery  → active MultiQuery + RRF

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chain import build_rag_pipeline


# ── utilitaires d'affichage ──────────────────────────────────────────────────

SEP = "─" * 70


def print_result(result, turn_num: int = 0) -> None:
    if turn_num:
        print(f"\n{'═' * 70}")
        print(f"  TOUR {turn_num}  |  stratégie : {result.strategy}")
        print(f"{'═' * 70}")

    print(f"\nQuestion : {result.question}")

    if result.query_variants:
        print(f"\nVariantes générées ({len(result.query_variants)}) :")
        for v in result.query_variants:
            print(f"  • {v}")

    print(f"\nDocuments récupérés ({len(result.retrieved_documents)}) :")
    for i, doc in enumerate(result.retrieved_documents, 1):
        src = doc.metadata.get("source", "?")
        label = src if src.startswith("http") else Path(src).name
        dtype = doc.metadata.get("doc_type", "?")
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] ({dtype}) {label} — {snippet}…")

    print(f"\nSources distinctes : {', '.join(result.sources) or '(aucune)'}")

    print(f"\n{SEP}")
    print("Réponse :")
    print(SEP)
    print(result.answer)
    print(SEP)


# ── scénario automatique 3 tours ─────────────────────────────────────────────

AUTO_QUESTIONS = [
    "What is Retrieval-Augmented Generation and how does it work?",
    "How does MMR retrieval differ from cosine similarity search?",
    "What are the limitations of this approach for enterprise applications?",
]


def run_auto(pipeline, use_multiquery: bool) -> None:
    print("\n" + "═" * 70)
    print("  DÉMO AUTOMATIQUE — 3 tours de conversation")
    if use_multiquery:
        print("  Mode : MultiQuery + RRF (cosine de base)")
    print("═" * 70)

    for i, question in enumerate(AUTO_QUESTIONS, 1):
        result = pipeline.answer(
            question,
            strategy="cosine",
            use_multiquery=use_multiquery,
        )
        print_result(result, turn_num=i)

    print("\n[mémoire]")
    pipeline.memory.print_summary()


# ── mode interactif ──────────────────────────────────────────────────────────

def run_interactive(pipeline, use_multiquery: bool) -> None:
    strategy = "cosine"
    print("\n" + "═" * 70)
    print("  ResearchPal — mode interactif")
    print("  Commandes : 'quit'|'exit' pour quitter, 'reset' pour effacer la mémoire")
    print("  Préfixe de stratégie : 'mmr:' pour forcer MMR")
    print("═" * 70 + "\n")

    while True:
        try:
            user_input = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Au revoir.")
            break
        if user_input.lower() == "reset":
            pipeline.reset_memory()
            print(">>> Mémoire effacée.")
            continue

        strat = strategy
        question = user_input
        if user_input.lower().startswith("mmr:"):
            strat = "mmr"
            question = user_input[4:].strip()

        result = pipeline.answer(question, strategy=strat, use_multiquery=use_multiquery)
        print_result(result)


# ── point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Démo RAG ResearchPal")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Lance le mode Q&A interactif",
    )
    parser.add_argument(
        "--multiquery", "-m",
        action="store_true",
        help="Active MultiQuery + RRF",
    )
    args = parser.parse_args()

    print("Initialisation du pipeline RAG...")
    pipeline = build_rag_pipeline()

    if args.interactive:
        run_interactive(pipeline, use_multiquery=args.multiquery)
    else:
        run_auto(pipeline, use_multiquery=args.multiquery)


if __name__ == "__main__":
    main()
