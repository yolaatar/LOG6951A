# context_mode_eval.py — Comparaison des 4 modes de gestion du contexte (T6)
#
# Évalue les 4 modes sur les mêmes dialogues que multiturn_validation.py afin
# que les résultats soient directement comparables.
#
# Modes évalués :
#   none      — question brute pour le retrieval, historique dans prompt LLM
#   heuristic — enrichissement coréférence + continuité thématique
#   concat    — préfixe les 2 dernières questions in-scope (sans LLM)
#   rewrite   — reformulation autonome via LLM
#
# Métriques par tour :
#   rejected        — question filtrée hors-périmètre
#   n_docs          — documents récupérés
#   cosine_score    — score cosinus de la requête de retrieval effective
#   grounding       — proportion de phrases ancrées dans le contexte
#   halluc_risk     — risque de hallucination détecté par regex
#   hist_ref        — la réponse fait-elle référence à l'historique
#
# Usage : cd src && ../.venv/bin/python evaluation/context_mode_eval.py

import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUT_OF_SCOPE_SCORE_THRESHOLD, DOMAIN_KEYWORDS
from rag.chain import RAGPipeline, RAGResult
from retrieval.cosine_retriever import cosine_search_with_scores


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RÉPERTOIRES DE SORTIE
# ─────────────────────────────────────────────────────────────────────────────

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "rag_eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MODES ET DIALOGUES
# ─────────────────────────────────────────────────────────────────────────────

MODES: Dict[str, Dict] = {
    "none": {
        "label": "Aucun (baseline)",
        "use_heuristic_context": False,
        "use_concat_context":    False,
        "use_query_rewriting":   False,
    },
    "heuristic": {
        "label": "Heuristiques",
        "use_heuristic_context": True,
        "use_concat_context":    False,
        "use_query_rewriting":   False,
    },
    "concat": {
        "label": "Concaténation",
        "use_heuristic_context": False,
        "use_concat_context":    True,
        "use_query_rewriting":   False,
    },
    "rewrite": {
        "label": "Réécriture LLM",
        "use_heuristic_context": False,
        "use_concat_context":    False,
        "use_query_rewriting":   True,
    },
}

# Mêmes dialogues que multiturn_validation.py pour cohérence des résultats
DIALOGUES = [
    {
        "id": "VAL1",
        "name": "Dialogue implicite — pronoms anaphoriques (3 tours)",
        "description": (
            "Tour 2 : 'ses' (pronom possessif). Tour 3 : 'ces problèmes' "
            "(démonstratif). Sans gestion du contexte, les deux seraient rejetés."
        ),
        "turns": [
            "Qu'est-ce que RAG (Retrieval-Augmented Generation) et comment fonctionne-t-il ?",
            "Quelles sont ses principales limites par rapport aux LLMs classiques ?",
            "Comment MMR aide-t-il à résoudre certains de ces problèmes de redondance ?",
        ],
        "expected_refusal_turns": [],   # aucun tour ne doit être rejeté
    },
    {
        "id": "VAL2",
        "name": "Dialogue court — questions sous le seuil de longueur (3 tours)",
        "description": (
            "Tours 2 et 3 courts (< 9 mots). Sans enrichissement, leur score "
            "cosinus seul est insuffisant."
        ),
        "turns": [
            "Explique-moi la stratégie MMR dans un pipeline RAG.",
            "Quels sont ses paramètres clés ?",
            "Et comment choisir la valeur de lambda ?",
        ],
        "expected_refusal_turns": [],
    },
    {
        "id": "VAL3",
        "name": "Dialogue synthèse — référence au tour 1 depuis le tour 3",
        "description": (
            "Tour 3 : 'ta première réponse'. Teste la robustesse de la mémoire "
            "glissante sur 3 tours."
        ),
        "turns": [
            "Qu'est-ce que le chunking et pourquoi est-il important dans un pipeline RAG ?",
            "Quelle est la différence entre chunking fixe et chunking récursif ?",
            "En résumé, quelle stratégie recommandes-tu d'après ta première réponse ?",
        ],
        "expected_refusal_turns": [],
    },
    {
        "id": "VAL4",
        "name": "Robustesse — hors-périmètre résiste après historique valide",
        "description": (
            "Tour 3 clairement hors-périmètre : doit être rejeté même après "
            "2 tours valides."
        ),
        "turns": [
            "Comment fonctionne le retrieval par similarité cosinus dans ChromaDB ?",
            "Quels sont les avantages de la persistance locale du vectorstore ?",
            "Quelle est la capitale de la France et quelle est sa population ?",
        ],
        "expected_refusal_turns": [3],  # le tour 3 doit être rejeté
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

_RE_HALLUC = [
    re.compile(r"\b\d{1,3}[,\.]\d+\s*%"),
    re.compile(r"\baccuracy\s*[:=]\s*\d", re.IGNORECASE),
    re.compile(r"\b(MMLU|HellaSwag|ARC|TruthfulQA|benchmark)\b.*\b\d{2}", re.IGNORECASE),
    re.compile(r"\bprécision\s*(de|:)\s*\d", re.IGNORECASE),
    re.compile(r"\bscore\s*(de|:)\s*\d{2,}", re.IGNORECASE),
]
_RE_HIST_REF = re.compile(
    r"(précédemment|mentionné|expliqué|j[''']avais|premier point|première réponse"
    r"|tour\s*1|comme tu|comme je|comme indiqué|dans ta)",
    re.IGNORECASE,
)


def _is_rejected(result: RAGResult) -> bool:
    return len(result.retrieved_documents) == 0 and "hors périmètre" in result.answer


def _halluc_risk(answer: str) -> bool:
    return any(p.search(answer) for p in _RE_HALLUC)


def _hist_ref(answer: str) -> bool:
    return bool(_RE_HIST_REF.search(answer))


def _grounding(answer: str, docs: list) -> float:
    ctx_words: set = set()
    for doc in docs:
        ctx_words.update(w.lower() for w in re.findall(r"\w{4,}", doc.page_content))
    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 20]
    if not sentences or not ctx_words:
        return 0.0
    grounded = sum(
        1 for s in sentences
        if len(set(w.lower() for w in re.findall(r"\w{4,}", s)) & ctx_words) >= 3
    )
    return round(grounded / len(sentences), 3)


def _cosine_score(pipeline: RAGPipeline, query: str) -> Optional[float]:
    try:
        pairs = cosine_search_with_scores(pipeline.vectorstore, query, k=1)
        return round(pairs[0][1], 4) if pairs else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXÉCUTION D'UN DIALOGUE POUR UN MODE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    question: str
    retrieval_query: str
    cosine_score: Optional[float]
    rejected: bool
    n_docs: int
    sources: List[str]
    grounding: float
    halluc_risk: bool
    hist_ref: bool
    answer: str


def run_dialogue_for_mode(
    pipeline: RAGPipeline,
    dialogue: dict,
    mode_kwargs: dict,
    mode_label: str,
) -> List[TurnResult]:
    pipeline.reset_memory()
    results = []

    for i, question in enumerate(dialogue["turns"], 1):
        result: RAGResult = pipeline.answer(question, **mode_kwargs)

        rq    = result.retrieval_query or question
        score = _cosine_score(pipeline, rq)

        tr = TurnResult(
            turn=i,
            question=question,
            retrieval_query=rq,
            cosine_score=score,
            rejected=_is_rejected(result),
            n_docs=len(result.retrieved_documents),
            sources=result.sources,
            grounding=_grounding(result.answer, result.retrieved_documents),
            halluc_risk=_halluc_risk(result.answer),
            hist_ref=_hist_ref(result.answer),
            answer=result.answer,
        )
        results.append(tr)

        status = "REJET" if tr.rejected else "ok"
        score_str = f"{score:.4f}" if score is not None else "  n/a "
        print(f"    [{mode_label:18s}] Tour {i} : {status:5s} | "
              f"score={score_str} | docs={tr.n_docs} | "
              f"ground={tr.grounding:.2f}")

    pipeline.reset_memory()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CALCUL DES MÉTRIQUES AGRÉGÉES PAR MODE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModeSummary:
    mode_id: str
    label: str
    false_rejections: int    # suivis rejetés à tort (VAL1-3)
    correct_rejections: int  # hors-périmètre rejetés à raison (VAL4 tour 3)
    missed_rejections: int   # hors-périmètre non rejetés (VAL4 tour 3 passé)
    total_followup_turns: int
    avg_cosine_score: float
    avg_n_docs: float
    avg_grounding: float
    halluc_count: int
    hist_ref_count: int


def compute_summary(
    mode_id: str,
    label: str,
    all_dial_results: Dict[str, List[TurnResult]],
) -> ModeSummary:
    false_rejections   = 0
    correct_rejections = 0
    missed_rejections  = 0
    total_followup     = 0
    scores, n_docs_list, groundings = [], [], []
    halluc_count = hist_ref_count = 0

    for dial in DIALOGUES:
        turns = all_dial_results[dial["id"]]
        expected_refusals = set(dial.get("expected_refusal_turns", []))

        for tr in turns:
            if tr.cosine_score is not None:
                scores.append(tr.cosine_score)
            n_docs_list.append(tr.n_docs)
            groundings.append(tr.grounding)
            if tr.halluc_risk:
                halluc_count += 1
            if tr.hist_ref:
                hist_ref_count += 1

            if tr.turn in expected_refusals:
                # Tour qui DOIT être rejeté
                if tr.rejected:
                    correct_rejections += 1
                else:
                    missed_rejections += 1
            else:
                # Tour qui ne doit PAS être rejeté (suivi ou question explicite)
                total_followup += 1
                if tr.turn > 1 and tr.rejected:
                    false_rejections += 1

    return ModeSummary(
        mode_id=mode_id,
        label=label,
        false_rejections=false_rejections,
        correct_rejections=correct_rejections,
        missed_rejections=missed_rejections,
        total_followup_turns=total_followup,
        avg_cosine_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
        avg_n_docs=round(sum(n_docs_list) / len(n_docs_list), 2) if n_docs_list else 0.0,
        avg_grounding=round(sum(groundings) / len(groundings), 3) if groundings else 0.0,
        halluc_count=halluc_count,
        hist_ref_count=hist_ref_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    all_results: Dict[str, Dict[str, List[TurnResult]]],
    summaries: Dict[str, ModeSummary],
) -> None:
    p = REPORT_DIR / "context_mode_eval.txt"

    with open(p, "w", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"{'═'*80}\n")
        f.write(f"  Évaluation des modes de gestion du contexte — {ts}\n")
        f.write(f"  Modes : {', '.join(m['label'] for m in MODES.values())}\n")
        f.write(f"{'═'*80}\n\n")

        # ── Détail par dialogue ───────────────────────────────────────────────
        for dial in DIALOGUES:
            did = dial["id"]
            f.write(f"{'═'*80}\n")
            f.write(f"  {did} — {dial['name']}\n")
            f.write(f"  {dial['description']}\n")
            f.write(f"{'═'*80}\n\n")

            for i, question in enumerate(dial["turns"], 1):
                f.write(f"  ── Tour {i} : {question[:80]}\n\n")
                f.write(f"  {'Mode':<22} {'Requête retrieval (60c)':<62} {'Score':>6} {'Docs':>4} {'Rejet':>5} {'Ground':>6}\n")
                f.write(f"  {'─'*22} {'─'*62} {'─'*6} {'─'*4} {'─'*5} {'─'*6}\n")

                for mode_id, mode_cfg in MODES.items():
                    tr = all_results[mode_id][did][i - 1]
                    rq_short = tr.retrieval_query[:60].replace("\n", " ")
                    score_str = f"{tr.cosine_score:.4f}" if tr.cosine_score is not None else "  n/a"
                    f.write(
                        f"  {mode_cfg['label']:<22} {rq_short:<62} "
                        f"{score_str:>6} {tr.n_docs:>4} "
                        f"{'OUI ✗' if tr.rejected else 'non  ':>5} "
                        f"{tr.grounding:>6.3f}\n"
                    )

                f.write("\n")

                # Réponses par mode
                for mode_id, mode_cfg in MODES.items():
                    tr = all_results[mode_id][did][i - 1]
                    excerpt = tr.answer[:400] + "…" if len(tr.answer) > 400 else tr.answer
                    f.write(f"  [{mode_cfg['label']}]\n")
                    for line in excerpt.splitlines():
                        f.write(f"    {line}\n")
                    f.write("\n")

                f.write(f"{'─'*80}\n\n")

        # ── Tableau récapitulatif ─────────────────────────────────────────────
        f.write(f"{'═'*80}\n  TABLEAU RÉCAPITULATIF\n{'═'*80}\n\n")

        header = (
            f"  {'Mode':<22} {'Faux rejets':>11} {'Rejets OK':>9} "
            f"{'Rejets manqués':>15} {'Cosine moy':>10} "
            f"{'Docs moy':>8} {'Ground moy':>10} {'Halluc':>6} {'HistRef':>7}\n"
        )
        f.write(header)
        f.write(f"  {'─'*22} {'─'*11} {'─'*9} {'─'*15} {'─'*10} {'─'*8} {'─'*10} {'─'*6} {'─'*7}\n")

        for mode_id, s in summaries.items():
            f.write(
                f"  {s.label:<22} "
                f"{s.false_rejections:>11} "
                f"{s.correct_rejections:>9} "
                f"{s.missed_rejections:>15} "
                f"{s.avg_cosine_score:>10.4f} "
                f"{s.avg_n_docs:>8.2f} "
                f"{s.avg_grounding:>10.3f} "
                f"{s.halluc_count:>6} "
                f"{s.hist_ref_count:>7}\n"
            )

        f.write(f"\n  Légende :\n")
        f.write(f"    Faux rejets      : suivis correctement posés mais filtrés hors-périmètre\n")
        f.write(f"    Rejets OK        : questions hors-périmètre correctement rejetées\n")
        f.write(f"    Rejets manqués   : questions hors-périmètre non filtrées\n")
        f.write(f"    Cosine moy       : score cosinus moyen de la requête de retrieval\n")
        f.write(f"    Ground moy       : proportion moyenne de phrases ancrées dans le contexte\n")
        f.write(f"    Halluc           : tours avec risque de hallucination détecté\n")
        f.write(f"    HistRef          : tours où la réponse référence l'historique\n")

        # ── Verdict par mode ──────────────────────────────────────────────────
        f.write(f"\n{'═'*80}\n  VERDICT\n{'═'*80}\n\n")

        best_false_rej = min(summaries.values(), key=lambda s: s.false_rejections)
        best_cosine    = max(summaries.values(), key=lambda s: s.avg_cosine_score)
        best_grounding = max(summaries.values(), key=lambda s: s.avg_grounding)

        for mode_id, s in summaries.items():
            badges = []
            if s.false_rejections == best_false_rej.false_rejections:
                badges.append("meilleur taux de faux rejets")
            if s.avg_cosine_score == best_cosine.avg_cosine_score:
                badges.append("meilleur score cosinus")
            if s.avg_grounding == best_grounding.avg_grounding:
                badges.append("meilleur ancrage")
            if s.missed_rejections > 0:
                badges.append(f"⚠ {s.missed_rejections} rejet(s) manqué(s)")

            badge_str = " | ".join(badges) if badges else "—"
            f.write(f"  {s.label:<22} : {badge_str}\n")

    print(f"  [export] context_mode_eval.txt")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXPORT CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(all_results: Dict[str, Dict[str, List[TurnResult]]]) -> None:
    p = REPORT_DIR / "context_mode_eval.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "dialogue", "turn", "question",
            "retrieval_query", "cosine_score",
            "rejected", "n_docs",
            "grounding", "halluc_risk", "hist_ref",
        ])
        writer.writeheader()
        for mode_id in MODES:
            for dial in DIALOGUES:
                for tr in all_results[mode_id][dial["id"]]:
                    writer.writerow({
                        "mode":            mode_id,
                        "dialogue":        dial["id"],
                        "turn":            tr.turn,
                        "question":        tr.question,
                        "retrieval_query": tr.retrieval_query,
                        "cosine_score":    tr.cosine_score if tr.cosine_score is not None else "",
                        "rejected":        tr.rejected,
                        "n_docs":          tr.n_docs,
                        "grounding":       tr.grounding,
                        "halluc_risk":     tr.halluc_risk,
                        "hist_ref":        tr.hist_ref,
                    })
    print(f"  [export] context_mode_eval.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 72)
    print("  Évaluation des modes de gestion du contexte")
    print("═" * 72)

    print("\n[1/3] Initialisation du pipeline...")
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        print(f"  ERREUR : {exc}")
        sys.exit(1)

    # all_results[mode_id][dial_id] = List[TurnResult]
    all_results: Dict[str, Dict[str, List[TurnResult]]] = {m: {} for m in MODES}

    print("\n[2/3] Exécution des dialogues...\n")
    for dial in DIALOGUES:
        print(f"  {dial['id']} — {dial['name']}")
        for mode_id, mode_cfg in MODES.items():
            kwargs = {k: v for k, v in mode_cfg.items() if k != "label"}
            turns = run_dialogue_for_mode(pipeline, dial, kwargs, mode_cfg["label"])
            all_results[mode_id][dial["id"]] = turns
        print()

    print("[3/3] Calcul des métriques et rédaction du rapport...\n")
    summaries = {
        mode_id: compute_summary(mode_id, mode_cfg["label"], all_results[mode_id])
        for mode_id, mode_cfg in MODES.items()
    }

    write_report(all_results, summaries)
    write_csv(all_results)

    # Affichage console du récapitulatif
    print(f"\n  {'Mode':<22} {'Faux rejets':>11} {'Cosine moy':>10} {'Ground moy':>10}")
    print(f"  {'─'*22} {'─'*11} {'─'*10} {'─'*10}")
    for s in summaries.values():
        print(f"  {s.label:<22} {s.false_rejections:>11} "
              f"{s.avg_cosine_score:>10.4f} {s.avg_grounding:>10.3f}")

    print(f"\n  Rapports : {REPORT_DIR}/context_mode_eval.txt")
    print(f"             {REPORT_DIR}/context_mode_eval.csv")


if __name__ == "__main__":
    main()
