# eval/llm_judge.py — évaluateur LLM-as-judge maison (T5)
#
# Critère évalué : qualité des citations de sources
#   (non couvert par RAGAS faithfulness/answer_relevancy)
#
# Sous-critères :
#   1. précision_citations (0-3) : les [N] dans la réponse désignent-ils les bons docs ?
#   2. complétude_citations (0-3) : toutes les affirmations clés sont-elles citées ?
#   3. honnêteté_limites (0-4)   : la section Limites est-elle appropriée et honnête ?
#
# Barème total : /10
# Format de sortie : JSON structuré (requis par l'énoncé)
#
# Le prompt d'évaluation complet est inclus dans JUDGE_PROMPT ci-dessous (annexe).
#
# Usage :
#   cd ResearchPal
#   python eval/llm_judge.py
#   → résultats dans eval/judge_results.json

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import OLLAMA_MODEL, OLLAMA_BASE_URL

EVAL_DIR = Path(__file__).resolve().parent
ANSWERS_PATH = EVAL_DIR / "generated_answers.json"
RESULTS_PATH = EVAL_DIR / "judge_results.json"


# ── Prompt d'évaluation (ANNEXE — requis par l'énoncé T5) ───────────────────
#
# Ce prompt est utilisé tel quel pour chaque paire question/réponse.
# Il impose un format de sortie JSON strict avec barème détaillé.

JUDGE_PROMPT = """\
Tu es un évaluateur rigoureux de systèmes RAG (Retrieval-Augmented Generation).
Ta mission est d'évaluer la qualité des CITATIONS DE SOURCES dans une réponse générée.

---
QUESTION POSÉE :
{question}

RÉPONSE GÉNÉRÉE :
{answer}

DOCUMENTS SOURCE DISPONIBLES :
{contexts}

RÉPONSE DE RÉFÉRENCE :
{ground_truth}
---

Évalue la réponse selon les trois critères suivants :

CRITÈRE 1 — Précision des citations (0-3 points) :
  3 : Toutes les citations [N] dans la réponse renvoient au bon document source.
  2 : La majorité des citations sont correctes, au plus 1 erreur.
  1 : Plusieurs citations incorrectes ou manquantes pour des affirmations clés.
  0 : Aucune citation, ou citations systématiquement incorrectes.
  Note : si la réponse dit honnêtement "aucun document pertinent", donner 3.

CRITÈRE 2 — Complétude des citations (0-3 points) :
  3 : Toutes les affirmations importantes sont ancrées sur une source numérotée.
  2 : La plupart des affirmations sont citées, quelques oublis mineurs.
  1 : Moins de la moitié des affirmations sont citées.
  0 : Aucune affirmation n'est reliée à une source.
  Note : pour les questions hors-corpus avec réponse d'absence, donner 3 si c'est honnête.

CRITÈRE 3 — Honnêteté des limites (0-4 points) :
  4 : La section "Limites / Incertitudes" identifie précisément ce que le contexte
      ne permet pas de confirmer. Aucune affirmation non fondée en réponse.
  3 : Les limites sont signalées mais de manière générique ou incomplète.
  2 : Certaines limites importantes ne sont pas mentionnées, ou la section est absente
      mais la réponse reste globalement honnête.
  1 : Des affirmations non fondées apparaissent dans la réponse sans être signalées.
  0 : Hallucinations claires ou affirmations inventées présentées comme des faits.

---
INSTRUCTIONS :
- Évalue uniquement ce qui est dans la réponse, pas ce qui aurait pu être mieux dit.
- Pour les questions adversariales (hors corpus), une réponse honnête d'incapacité
  est la MEILLEURE réponse possible (score maximal).
- Retourne UNIQUEMENT le JSON ci-dessous, sans texte avant ni après.

FORMAT DE SORTIE JSON OBLIGATOIRE :
{{
  "précision_citations": <entier 0-3>,
  "complétude_citations": <entier 0-3>,
  "honnêteté_limites": <entier 0-4>,
  "score_total": <entier 0-10>,
  "justification": "<explication concise en 1-3 phrases>"
}}
"""


# ── Évaluation d'une paire ────────────────────────────────────────────────────

def judge_one(llm, record: dict) -> dict:
    """Évalue une paire question/réponse avec le prompt LLM-as-judge."""
    from langchain_core.messages import HumanMessage

    contexts_str = "\n\n".join(
        f"[{i+1}] {ctx[:400]}" for i, ctx in enumerate(record.get("contexts", [""]))
    )

    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        answer=record.get("answer", ""),
        contexts=contexts_str,
        ground_truth=record.get("ground_truth", ""),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Extraction du JSON (robuste aux éventuels caractères parasites)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
        else:
            raise ValueError("JSON non trouvé dans la réponse du juge")

        # Validation du score_total
        expected_total = (
            scores.get("précision_citations", 0)
            + scores.get("complétude_citations", 0)
            + scores.get("honnêteté_limites", 0)
        )
        scores["score_total"] = expected_total  # recalcule pour cohérence

    except Exception as exc:
        scores = {
            "précision_citations": 0,
            "complétude_citations": 0,
            "honnêteté_limites": 0,
            "score_total": 0,
            "justification": f"Erreur d'évaluation : {exc}",
        }

    return {
        "id": record["id"],
        "type": record["type"],
        "question": record["question"],
        "scores": scores,
        "tool_used": record.get("tool_used", ""),
        "retry_count": record.get("retry_count", 0),
    }


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ResearchPal v2 — LLM-as-judge (T5)")
    print("=" * 60)

    if not ANSWERS_PATH.exists():
        print(f"\nErreur : {ANSWERS_PATH} introuvable.")
        print("  → Lancez d'abord : python eval/ragas_eval.py")
        sys.exit(1)

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    print(f"\n[1/2] Évaluation de {len(records)} réponses par LLM-as-judge…\n")
    results = []
    for record in records:
        qid = record["id"]
        qtype = record["type"]
        print(f"  [{qid:02d}/{len(records)}] ({qtype}) {record['question'][:55]}…")
        result = judge_one(llm, record)
        s = result["scores"]
        print(f"         → citations={s.get('précision_citations',0)}/3  "
              f"complétude={s.get('complétude_citations',0)}/3  "
              f"limites={s.get('honnêteté_limites',0)}/4  "
              f"total={s.get('score_total',0)}/10")
        results.append(result)

    # Agrégats par type de question
    def avg_score(subset):
        if not subset:
            return 0.0
        return sum(r["scores"]["score_total"] for r in subset) / len(subset)

    corpus_res    = [r for r in results if r["type"] == "corpus"]
    multihop_res  = [r for r in results if r["type"] == "multi_hop"]
    adversarial_res = [r for r in results if r["type"] == "adversarial"]

    aggregate = {
        "score_moyen_global": avg_score(results),
        "score_moyen_corpus": avg_score(corpus_res),
        "score_moyen_multi_hop": avg_score(multihop_res),
        "score_moyen_adversarial": avg_score(adversarial_res),
        "precision_citations_moy":   sum(r["scores"].get("précision_citations",0) for r in results) / len(results),
        "complétude_citations_moy":  sum(r["scores"].get("complétude_citations",0) for r in results) / len(results),
        "honnêteté_limites_moy":     sum(r["scores"].get("honnêteté_limites",0) for r in results) / len(results),
    }

    output = {"aggregate": aggregate, "per_question": results}

    print("\n[2/2] Sauvegarde…")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  → Résultats : {RESULTS_PATH}")
    print("\n  ┌─ LLM-as-judge — Résumé ──────────────────────────────┐")
    print(f"  │  Score global     : {aggregate['score_moyen_global']:.2f}/10")
    print(f"  │  Corpus           : {aggregate['score_moyen_corpus']:.2f}/10")
    print(f"  │  Multi-hop        : {aggregate['score_moyen_multi_hop']:.2f}/10")
    print(f"  │  Adversarial      : {aggregate['score_moyen_adversarial']:.2f}/10")
    print("  └──────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
