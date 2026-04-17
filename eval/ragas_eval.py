# eval/ragas_eval.py — évaluation qualité RAGAS (T5)
#
# Métriques calculées (minimum requis par l'énoncé) :
#   - faithfulness      : les affirmations de la réponse sont-elles soutenues par le contexte ?
#   - answer_relevancy  : la réponse répond-elle à la question posée ?
#
# Dataset : 15 paires Q/R de référence (data/eval_dataset.json)
#   - ≥ 10 paires corpus
#   - ≥ 3 paires adversariales
#   - ≥ 2 paires multi-hop
#
# LLM utilisé : Mistral 7B via Ollama (même choix que le pipeline)
# Embeddings  : all-MiniLM-L6-v2 (même choix que le pipeline)
#
# Usage :
#   cd ResearchPal
#   python eval/ragas_eval.py
#   → résultats dans eval/ragas_results.json

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR, OLLAMA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL_NAME

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "eval_dataset.json"
RESULTS_PATH = EVAL_DIR / "ragas_results.json"


# ── Génération des réponses via le pipeline agentique ────────────────────────

def generate_answers(dataset: list) -> list:
    """Exécute le pipeline agentique sur chaque question et collecte les réponses
    et contextes récupérés pour l'évaluation RAGAS."""
    from agent.graph import run_agent

    records = []
    for item in dataset:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        qtype = item["type"]

        print(f"  [{qid:02d}/{len(dataset)}] ({qtype}) {question[:60]}…")

        try:
            result = run_agent(question, thread_id=f"eval_{qid}")
            answer = result.get("generation", "")
            docs = result.get("relevant_docs") or result.get("documents") or []
            contexts = [doc.page_content for doc in docs] if docs else [""]
            tool_used = result.get("tool_used", "corpus")
            retry_count = result.get("retry_count", 0)
        except Exception as exc:
            print(f"    ⚠ Erreur pipeline : {exc}")
            answer = ""
            contexts = [""]
            tool_used = "error"
            retry_count = 0

        records.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "tool_used": tool_used,
            "retry_count": retry_count,
        })

    return records


# ── Calcul des métriques RAGAS ────────────────────────────────────────────────

def run_ragas(records: list) -> dict:
    """Calcule faithfulness + answer_relevancy via RAGAS 0.4.x configuré avec Ollama.

    RAGAS 0.4.x utilise InstructorLLM via l'endpoint OpenAI-compatible d'Ollama,
    et HuggingFaceEmbeddings natif pour les embeddings.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from ragas import evaluate
        # Utiliser les classes old-style (héritent de Metric, compatibles avec evaluate())
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
    except ImportError as e:
        print(f"RAGAS non installé : {e}")
        print("  → pip install ragas datasets")
        return {}

    import warnings
    warnings.filterwarnings("ignore")
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    print("\n[ragas] Configuration LLM et embeddings locaux…")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    faithfulness_metric = Faithfulness()
    faithfulness_metric.llm = ragas_llm

    answer_rel_metric = AnswerRelevancy()
    answer_rel_metric.llm = ragas_llm
    answer_rel_metric.embeddings = ragas_emb

    eval_data = {
        "question":     [r["question"]     for r in records],
        "answer":       [r["answer"]       for r in records],
        "contexts":     [r["contexts"]     for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    }

    dataset = Dataset.from_dict(eval_data)

    print("[ragas] Calcul des métriques (faithfulness + answer_relevancy)…")
    results = evaluate(
        dataset,
        metrics=[faithfulness_metric, answer_rel_metric],
    )

    scores = results.to_pandas().to_dict(orient="records")
    aggregate = {
        "faithfulness_mean":     results["faithfulness"],
        "answer_relevancy_mean": results["answer_relevancy"],
    }

    return {"aggregate": aggregate, "per_question": scores}


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ResearchPal v2 — Évaluation RAGAS (T5)")
    print("=" * 60)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n[1/3] Génération des réponses ({len(dataset)} questions)…")
    records = generate_answers(dataset)

    # Sauvegarde intermédiaire des réponses générées
    answers_path = EVAL_DIR / "generated_answers.json"
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  → Réponses sauvegardées : {answers_path}")

    print("\n[2/3] Calcul des métriques RAGAS…")
    ragas_scores = run_ragas(records)

    print("\n[3/3] Sauvegarde des résultats…")
    output = {
        "ragas": ragas_scores,
        "dataset_stats": {
            "total": len(dataset),
            "corpus": sum(1 for r in dataset if r["type"] == "corpus"),
            "multi_hop": sum(1 for r in dataset if r["type"] == "multi_hop"),
            "adversarial": sum(1 for r in dataset if r["type"] == "adversarial"),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  → Résultats : {RESULTS_PATH}")

    if ragas_scores.get("aggregate"):
        agg = ragas_scores["aggregate"]
        print("\n  ┌─ Résultats RAGAS ────────────────────────────────┐")
        print(f"  │  faithfulness      : {agg.get('faithfulness_mean', 'N/A'):.3f}")
        print(f"  │  answer_relevancy  : {agg.get('answer_relevancy_mean', 'N/A'):.3f}")
        print("  └──────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
