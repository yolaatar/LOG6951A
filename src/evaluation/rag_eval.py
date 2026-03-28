# rag_eval.py — Évaluation end-to-end du pipeline RAG (Tâche 3)
#
# Pattern de prompting : Persona + Structured Output  (+ Cognitive Verifier implicite)
# Évalue : citation, structure, ancrage contextuel, risque d'hallucination, gestion de l'historique
#
# Usage :
#   cd src && ../.venv/bin/python evaluation/rag_eval.py

import csv
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OLLAMA_MODEL, RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA
from rag.chain import RAGPipeline, RAGResult
from rag.prompt import SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RÉPERTOIRES DE SORTIE
# ─────────────────────────────────────────────────────────────────────────────

REPORT_DIR     = Path(__file__).resolve().parent.parent.parent / "reports" / "rag_eval"
FIGURES_DIR    = REPORT_DIR / "figures"
TRANSCRIPTS_DIR = REPORT_DIR / "transcripts"

for _d in [REPORT_DIR, FIGURES_DIR, TRANSCRIPTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DÉFINITION DES CAS DE TEST
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    category: str          # "use_case" | "edge_case"
    subcategory: str       # description courte du type de test
    query: str
    strategy: str = "cosine"
    use_multiquery: bool = False
    expected_citation: bool = True
    expected_out_of_scope: bool = False
    notes: str = ""
    # liste de requêtes à exécuter avant ce test (pour les suivis multi-tours)
    setup_queries: List[str] = field(default_factory=list)


# ── Cas d'usage normaux ───────────────────────────────────────────────────────

USE_CASE_QUERIES: List[TestCase] = [
    TestCase(
        id="UC1",
        category="use_case",
        subcategory="Requête factuelle directe",
        query="Qu'est-ce que RAG (Retrieval-Augmented Generation) et pourquoi cette approche "
              "est-elle utilisée dans les systèmes de traitement du langage naturel ?",
        strategy="cosine",
        expected_citation=True,
        notes="Cas nominal : réponse factuelle attendue avec citations et structure complète.",
    ),
    TestCase(
        id="UC2",
        category="use_case",
        subcategory="Synthèse multi-sources",
        query="Quelles sont les principales différences entre la similarité cosinus et MMR "
              "pour la récupération de documents dans un pipeline RAG ?",
        strategy="cosine",
        expected_citation=True,
        notes="Nécessite une synthèse à travers plusieurs chunks — teste la capacité de citation croisée.",
    ),
    TestCase(
        id="UC3",
        category="use_case",
        subcategory="Requête technique orientée recherche",
        query="Comment le chevauchement (overlap) entre les chunks affecte-t-il la qualité "
              "du retrieval dans un pipeline RAG ? Quels paramètres faut-il ajuster ?",
        strategy="mmr",
        expected_citation=True,
        notes="Requête technique précise — test de la stratégie MMR pour la diversité.",
    ),
    TestCase(
        id="UC4",
        category="use_case",
        subcategory="Suivi dépendant de l'historique",
        query="Quelles sont les principales limites de l'approche que tu viens de décrire ?",
        strategy="cosine",
        expected_citation=True,
        notes="Dépend de l'historique (UC1–UC3 déjà dans la mémoire). "
              "Teste la gestion de la coreférence implicite.",
    ),
]

# ── Cas limites / stress-test ─────────────────────────────────────────────────

EDGE_CASE_QUERIES: List[TestCase] = [
    TestCase(
        id="EC1",
        category="edge_case",
        subcategory="Requête ambiguë / sous-spécifiée",
        query="Comment améliorer les résultats ?",
        strategy="cosine",
        expected_citation=True,
        notes="Requête non-spécifique : risque de réponse générique, citation absente, "
              "hallucination sur ce que signifie 'améliorer'.",
    ),
    TestCase(
        id="EC2",
        category="edge_case",
        subcategory="Support de retrieval faible — sujet périphérique",
        query="Quel est le meilleur modèle d'embedding pour des textes en japonais ? "
              "Donne des exemples de modèles spécifiques avec leurs scores.",
        strategy="cosine",
        expected_citation=False,
        expected_out_of_scope=False,  # 'embedding' passe le filtre de domaine
        notes="Risque élevé : le corpus ne contient pas cette info. "
              "Le modèle est tenté de citer des modèles inexistants dans le corpus.",
    ),
    TestCase(
        id="EC3",
        category="edge_case",
        subcategory="Piège lexical — chevauchement trompeur",
        query="Qu'est-ce qu'un chunk dans le contexte du traitement du texte en général ?",
        strategy="cosine",
        expected_citation=True,
        notes="Le mot 'chunk' génère un retrieval dense mais le contexte est spécifique au RAG. "
              "Risque de réponse trop large dépassant les sources.",
    ),
    TestCase(
        id="EC4",
        category="edge_case",
        subcategory="Appât à l'hallucination — chiffres précis absents du corpus",
        query="Quelle est la précision exacte (accuracy) de mistral:7b-instruct sur le "
              "benchmark MMLU ? Donne le score en pourcentage.",
        strategy="cosine",
        expected_citation=False,
        expected_out_of_scope=False,
        notes="Risque maximal d'hallucination : chiffre précis absent du corpus. "
              "Le modèle doit signaler l'absence d'information, pas inventer un score.",
    ),
    TestCase(
        id="EC5",
        category="edge_case",
        subcategory="Suivi multi-tours — référence au tour 1 depuis le tour 3",
        query="Peux-tu développer le premier point que tu avais mentionné dans ta toute "
              "première réponse sur ce sujet ?",
        strategy="cosine",
        expected_citation=True,
        notes="Exécuté après 2 tours de setup sur MMR. "
              "Teste si la mémoire glissante préserve le tour 1 au tour 3.",
        setup_queries=[
            "Qu'est-ce que la stratégie MMR et comment fonctionne-t-elle ?",
            "Quels sont les paramètres clés de MMR et comment les choisir ?",
        ],
    ),
    TestCase(
        id="EC6",
        category="edge_case",
        subcategory="Hors périmètre explicite",
        query="Quelle est la capitale de la France et quelle est sa population actuelle ?",
        strategy="cosine",
        expected_out_of_scope=True,
        expected_citation=False,
        notes="Hors périmètre clair. Attendu : détection automatique, refus de répondre.",
    ),
    TestCase(
        id="EC7",
        category="edge_case",
        subcategory="Retrieval trompeur — glissement sémantique",
        query="Comment fonctionne le mécanisme d'attention dans les transformers "
              "pour produire des représentations vectorielles ?",
        strategy="cosine",
        expected_citation=True,
        notes="'Embedding' et 'vectoriel' déclenchent un retrieval sur des docs RAG, "
              "pas sur l'architecture transformer. Teste la fidélité au contexte récupéré.",
    ),
]

ALL_QUERIES: List[TestCase] = USE_CASE_QUERIES + EDGE_CASE_QUERIES


# ── Dialogues multi-tours ─────────────────────────────────────────────────────

DIALOGUE_FLOWS = [
    {
        "id": "DIAL1",
        "name": "Dialogue normal — pipeline RAG (3 tours)",
        "description": "Dialogue réaliste avec progression thématique cohérente.",
        "turns": [
            "Explique-moi ce qu'est un pipeline RAG et comment il fonctionne de bout en bout.",
            "Comment la stratégie MMR améliore-t-elle les résultats par rapport à la similarité cosinus ?",
            "En tenant compte de tout ce que tu viens d'expliquer, quelle stratégie de retrieval "
            "recommanderais-tu pour un corpus académique avec beaucoup de documents redondants ?",
        ],
        "strategy": "cosine",
    },
    {
        "id": "DIAL2",
        "name": "Dialogue technique — chunking (3 tours)",
        "description": "Suivi technique sur le chunking, avec question dépendant de l'historique au tour 3.",
        "turns": [
            "Quels sont les paramètres importants à configurer dans une stratégie de chunking pour RAG ?",
            "Quelle est la différence entre le chunking fixe et le chunking récursif ?",
            "Sur la base de ce que tu as expliqué, quel impact a précisément le chevauchement "
            "(overlap) sur la qualité des réponses générées par le LLM ?",
        ],
        "strategy": "cosine",
    },
    {
        "id": "DIAL3",
        "name": "Dialogue limite — contexte insuffisant (3 tours)",
        "description": "Dialogue conçu pour provoquer l'insuffisance de contexte et tester la transparence.",
        "turns": [
            "Qu'est-ce que le multi-query retrieval avec Reciprocal Rank Fusion (RRF) ?",
            "Quelle est la valeur exacte du paramètre rrf_k utilisé dans ce projet et pourquoi ?",
            "Quel modèle de benchmark as-tu utilisé pour valider ces résultats ? "
            "Donne les chiffres précis de performance.",
        ],
        "strategy": "cosine",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MÉTRIQUES D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

# Patterns regex
_RE_CITATION    = re.compile(r"\[\d+\]")
_RE_RESPONSE    = re.compile(r"\*\*Réponse\*\*", re.IGNORECASE)
_RE_SOURCES     = re.compile(r"\*\*Sources\*\*", re.IGNORECASE)
_RE_LIMITS      = re.compile(r"\*\*Limites", re.IGNORECASE)
_RE_HIST_REF    = re.compile(
    r"(précédemment|mentionné|expliqué|comme je|comme tu|"
    r"tour précédent|comme indiqué|dans ma réponse|j['']avais)",
    re.IGNORECASE,
)
# Proxy hallucination : affirmations numériques précises
_RE_HALLUC = [
    re.compile(r"\b\d{1,3}[,\.]\d+\s*%"),              # pourcentages précis
    re.compile(r"\baccuracy\s*[:=]\s*\d", re.IGNORECASE),
    re.compile(r"\bprécision\s*(de|:)\s*\d", re.IGNORECASE),
    re.compile(r"\bscore\s*(de|:)\s*\d{2,3}[,\.]?\d*", re.IGNORECASE),
    re.compile(r"\b(MMLU|HellaSwag|ARC|TruthfulQA)\b.*\b\d{2}"),
]


def _grounding_score(answer: str, docs: List) -> float:
    """
    Proxy d'ancrage lexical.
    Fraction de phrases de la réponse qui partagent ≥ 3 mots avec le contexte récupéré.
    Proxy imparfait (lexical uniquement) ; explicitement indiqué comme tel.
    """
    if not docs:
        return 0.0
    context_words: set = set()
    for doc in docs:
        context_words.update(w.lower() for w in re.findall(r"\w{4,}", doc.page_content))

    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    grounded = sum(
        1 for s in sentences
        if len(set(w.lower() for w in re.findall(r"\w{4,}", s)) & context_words) >= 3
    )
    return round(grounded / len(sentences), 3)


def evaluate_result(result: RAGResult, tc: TestCase) -> Dict[str, Any]:
    ans = result.answer

    citation_present    = bool(_RE_CITATION.search(ans))
    response_section    = bool(_RE_RESPONSE.search(ans))
    sources_section     = bool(_RE_SOURCES.search(ans))
    limits_section      = bool(_RE_LIMITS.search(ans))
    structure_ok        = response_section and sources_section and limits_section
    grounding           = _grounding_score(ans, result.retrieved_documents)
    hallucination_risk  = any(p.search(ans) for p in _RE_HALLUC)
    out_of_scope        = len(result.retrieved_documents) == 0
    history_ref         = bool(_RE_HIST_REF.search(ans))

    # Verdict hiérarchique
    if out_of_scope and tc.expected_out_of_scope:
        verdict = "hors périmètre ✓"
    elif out_of_scope and not tc.expected_out_of_scope:
        verdict = "retrieval vide"
    elif hallucination_risk:
        verdict = "risque hallucination"
    elif not structure_ok:
        verdict = "structure incomplète"
    elif not citation_present and tc.expected_citation:
        verdict = "citation manquante"
    elif grounding < 0.25 and len(result.retrieved_documents) > 0:
        verdict = "ancrage faible"
    else:
        verdict = "succès"

    return {
        "id":                 tc.id,
        "category":           tc.category,
        "subcategory":        tc.subcategory,
        "query_short":        tc.query[:75] + "…" if len(tc.query) > 75 else tc.query,
        "strategy":           result.strategy,
        "n_docs":             len(result.retrieved_documents),
        "unique_sources":     len(result.sources),
        "citation_present":   citation_present,
        "structure_ok":       structure_ok,
        "grounding_score":    grounding,
        "hallucination_risk": hallucination_risk,
        "out_of_scope":       out_of_scope,
        "history_ref":        history_ref,
        "verdict":            verdict,
        "answer_len":         len(ans),
        "notes":              tc.notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXÉCUTION DES TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    pipeline: RAGPipeline, tc: TestCase
) -> Tuple[RAGResult, Dict[str, Any]]:
    print(f"    [{tc.id}] {tc.subcategory[:55]}")
    result  = pipeline.answer(tc.query, strategy=tc.strategy,
                              use_multiquery=tc.use_multiquery)
    metrics = evaluate_result(result, tc)
    return result, metrics


def run_all_single_tests(
    pipeline: RAGPipeline,
) -> Tuple[List[Tuple[RAGResult, Dict]], List[Tuple[RAGResult, Dict]]]:
    """Exécute les requêtes individuelles. Retourne (uc_results, ec_results)."""

    # ── Cas d'usage : mémoire persistante sur UC1→UC4 ────────────────────────
    print("  Cas d'usage (mémoire partagée UC1→UC4) :")
    pipeline.reset_memory()
    uc_results = []
    for tc in USE_CASE_QUERIES:
        res, met = run_single(pipeline, tc)
        uc_results.append((res, met))

    # ── Cas limites : chaque test isolé, sauf EC5 avec setup ─────────────────
    print("  Cas limites (sessions isolées) :")
    ec_results = []
    for tc in EDGE_CASE_QUERIES:
        pipeline.reset_memory()
        # injecter les tours de setup si nécessaire
        for setup_q in tc.setup_queries:
            print(f"      [setup] {setup_q[:60]}")
            pipeline.answer(setup_q)
        res, met = run_single(pipeline, tc)
        ec_results.append((res, met))

    pipeline.reset_memory()
    return uc_results, ec_results


def run_dialogue_flow(pipeline: RAGPipeline, flow: dict) -> List[Dict]:
    """Exécute un flux de dialogue multi-tours et retourne les métriques par tour."""
    print(f"  {flow['id']} : {flow['name']}")
    pipeline.reset_memory()
    turns_data = []

    for i, q in enumerate(flow["turns"], 1):
        print(f"    Tour {i}: {q[:65]}…")
        history_before = len(pipeline.memory)
        result = pipeline.answer(q, strategy=flow["strategy"])
        ans = result.answer

        turns_data.append({
            "flow_id":         flow["id"],
            "flow_name":       flow["name"],
            "turn":            i,
            "question":        q,
            "answer":          ans,
            "sources":         result.sources,
            "n_docs":          len(result.retrieved_documents),
            "history_before":  history_before,
            "citation_present": bool(_RE_CITATION.search(ans)),
            "structure_ok":    all([
                bool(_RE_RESPONSE.search(ans)),
                bool(_RE_SOURCES.search(ans)),
                bool(_RE_LIMITS.search(ans)),
            ]),
            "history_ref":     bool(_RE_HIST_REF.search(ans)),
            "grounding":       _grounding_score(ans, result.retrieved_documents),
        })

    pipeline.reset_memory()
    return turns_data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FIGURES (toutes en français)
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "use_case":           "#2980b9",
    "edge_case":          "#e74c3c",
    "succès":             "#27ae60",
    "hors périmètre ✓":  "#7f8c8d",
    "retrieval vide":     "#95a5a6",
    "risque hallucination": "#c0392b",
    "structure incomplète": "#8e44ad",
    "citation manquante": "#e67e22",
    "ancrage faible":     "#d35400",
}


def _save(fig: plt.Figure, name: str) -> None:
    p = FIGURES_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {name}")


def fig_verdict_breakdown(metrics: List[Dict]) -> None:
    uc = [m for m in metrics if m["category"] == "use_case"]
    ec = [m for m in metrics if m["category"] == "edge_case"]

    all_verdicts = sorted(set(m["verdict"] for m in metrics))
    uc_cnt = Counter(m["verdict"] for m in uc)
    ec_cnt = Counter(m["verdict"] for m in ec)
    x = np.arange(len(all_verdicts))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, [uc_cnt.get(v, 0) for v in all_verdicts], w,
                label="Cas d'usage", color="#2980b9", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + w/2, [ec_cnt.get(v, 0) for v in all_verdicts], w,
                label="Cas limites",  color="#e74c3c", alpha=0.88, edgecolor="white")

    for bars, cnt in [(b1, uc_cnt), (b2, ec_cnt)]:
        for b, v in zip(bars, all_verdicts):
            val = cnt.get(v, 0)
            if val > 0:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                        str(val), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(all_verdicts, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Nombre de cas", fontsize=11)
    ax.set_title("Distribution des verdicts par catégorie de requête\n"
                 "Cas d'usage (bleu) vs Cas limites (rouge)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "verdict_breakdown.png")


def fig_compliance_rates(metrics: List[Dict]) -> None:
    uc = [m for m in metrics if m["category"] == "use_case"]
    ec = [m for m in metrics if m["category"] == "edge_case"]

    def pct(group, key):
        if not group:
            return 0.0
        return sum(1 for m in group if m[key]) / len(group) * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    cats = ["Cas d'usage", "Cas limites"]

    panels = [
        ("Citation présente [N]",          "citation_present"),
        ("Structure en 3 sections",         "structure_ok"),
        ("Ancrage contextuel ≥ 0.25",       None),
    ]

    for ax, (title, key) in zip(axes, panels):
        if key is not None:
            vals = [pct(uc, key), pct(ec, key)]
        else:
            vals = [
                sum(1 for m in uc if m["grounding_score"] >= 0.25) / max(len(uc), 1) * 100,
                sum(1 for m in ec if m["grounding_score"] >= 0.25) / max(len(ec), 1) * 100,
            ]
        bars = ax.bar(cats, vals, color=["#2980b9", "#e74c3c"],
                      alpha=0.88, edgecolor="white", width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 120)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Taux (%)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Taux de conformité par catégorie de requête",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "compliance_rates.png")


def fig_grounding_scores(metrics: List[Dict]) -> None:
    from matplotlib.patches import Patch

    ids    = [m["id"] for m in metrics]
    scores = [m["grounding_score"] for m in metrics]
    colors = ["#2980b9" if m["category"] == "use_case" else "#e74c3c"
              for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    bars = ax.bar(ids, scores, color=colors, alpha=0.88, edgecolor="white")
    ax.axhline(0.25, color="gold", linestyle="--", linewidth=1.8,
               label="Seuil acceptable (0.25)")

    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    legend_elems = [
        Patch(facecolor="#2980b9", alpha=0.88, label="Cas d'usage"),
        Patch(facecolor="#e74c3c", alpha=0.88, label="Cas limites"),
        plt.Line2D([0], [0], color="gold", linestyle="--", lw=1.8, label="Seuil 0.25"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Identifiant de la requête", fontsize=10)
    ax.set_ylabel("Score d'ancrage (proxy lexical)", fontsize=10)
    ax.set_title("Score d'ancrage contextuel par requête\n"
                 "↑ plus élevé = réponse mieux ancrée dans les documents récupérés\n"
                 "(proxy : fraction de phrases partageant ≥ 3 mots avec le contexte)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "grounding_scores.png")


def fig_hallucination_flags(metrics: List[Dict]) -> None:
    from matplotlib.patches import Patch

    ids   = [m["id"] for m in metrics]
    risks = [m["hallucination_risk"] for m in metrics]
    colors = ["#c0392b" if r else "#27ae60" for r in risks]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.bar(ids, [1] * len(ids), color=colors, alpha=0.85, edgecolor="white")

    for i, (id_, risk) in enumerate(zip(ids, risks)):
        ax.text(i, 0.5, "Risque" if risk else "OK",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

    legend_elems = [
        Patch(facecolor="#27ae60", alpha=0.85, label="Aucun risque détecté"),
        Patch(facecolor="#c0392b", alpha=0.85, label="Risque d'hallucination"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="upper right")
    ax.set_yticks([])
    ax.set_xlabel("Identifiant de la requête", fontsize=10)
    ax.set_title("Risque d'hallucination par requête\n"
                 "(proxy : présence d'affirmations numériques précises non ancrées — "
                 "pourcentages, scores benchmark, etc.)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "hallucination_flags.png")


def fig_dialogue_quality(all_turns: List[Dict]) -> None:
    by_flow = defaultdict(list)
    for t in all_turns:
        by_flow[t["flow_id"]].append(t)

    flow_ids = sorted(by_flow.keys())
    n = len(flow_ids)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fid in zip(axes, flow_ids):
        turns = sorted(by_flow[fid], key=lambda t: t["turn"])
        labels = [f"Tour {t['turn']}" for t in turns]
        x = np.arange(len(turns))
        w = 0.25

        ax.bar(x - w,  [int(t["citation_present"]) for t in turns], w,
               label="Citation [N]",     color="#2980b9", alpha=0.88)
        ax.bar(x,      [int(t["structure_ok"])      for t in turns], w,
               label="Structure 3 sec.", color="#27ae60", alpha=0.88)
        ax.bar(x + w,  [int(t["history_ref"])       for t in turns], w,
               label="Réf. historique",  color="#e67e22", alpha=0.88)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non", "Oui"], fontsize=9)
        ax.set_ylim(0, 1.35)
        ax.set_title(f"{fid}\n{turns[0]['flow_name'][:40]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Qualité des dialogues multi-tours\n"
                 "Citation, structure 3 sections et référence à l'historique par tour",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "dialogue_quality.png")


def fig_summary_overview(metrics: List[Dict]) -> None:
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.42)

    # ── Panel 1 : répartition des verdicts (camembert) ──
    ax1 = fig.add_subplot(gs[0, 0])
    vc  = Counter(m["verdict"] for m in metrics)
    labels = list(vc.keys())
    sizes  = list(vc.values())
    colors_pie = [_C.get(l, "#95a5a6") for l in labels]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=None, colors=colors_pie, autopct="%1.0f%%",
        startangle=140, textprops={"fontsize": 8},
    )
    ax1.legend(wedges, labels, loc="lower center", fontsize=7,
               bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax1.set_title("Répartition des verdicts", fontsize=10, fontweight="bold")

    # ── Panel 2 : distribution des scores d'ancrage ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist([m["grounding_score"] for m in metrics], bins=8, range=(0, 1),
             color="#2980b9", alpha=0.85, edgecolor="white")
    ax2.axvline(0.25, color="gold", linestyle="--", lw=1.8, label="Seuil 0.25")
    ax2.set_xlabel("Score d'ancrage", fontsize=9)
    ax2.set_ylabel("Nb de requêtes", fontsize=9)
    ax2.set_title("Distribution des scores d'ancrage", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3 : citation présente par catégorie ──
    ax3 = fig.add_subplot(gs[1, 0])
    cats = ["Cas d'usage", "Cas limites"]
    uc   = [m for m in metrics if m["category"] == "use_case"]
    ec   = [m for m in metrics if m["category"] == "edge_case"]
    uc_c  = sum(1 for m in uc if     m["citation_present"])
    ec_c  = sum(1 for m in ec if     m["citation_present"])
    uc_nc = sum(1 for m in uc if not m["citation_present"])
    ec_nc = sum(1 for m in ec if not m["citation_present"])
    x = np.arange(2)
    ax3.bar(x, [uc_c,  ec_c],  0.42, label="Avec citations",  color="#27ae60", alpha=0.88)
    ax3.bar(x, [uc_nc, ec_nc], 0.42, bottom=[uc_c, ec_c],
            label="Sans citations", color="#c0392b", alpha=0.88)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cats, fontsize=9)
    ax3.set_ylabel("Nb de requêtes", fontsize=9)
    ax3.set_title("Présence de citations\npar catégorie", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4 : nombre de docs récupérés ──
    ax4 = fig.add_subplot(gs[1, 1])
    uc_docs = [m["n_docs"] for m in uc]
    ec_docs = [m["n_docs"] for m in ec]
    ax4.boxplot([uc_docs, ec_docs], tick_labels=cats, patch_artist=True,
                boxprops={"facecolor": "#bdc3c7"},
                medianprops={"color": "#2c3e50", "linewidth": 2})
    ax4.set_ylabel("Nb de documents récupérés", fontsize=9)
    ax4.set_title("Documents récupérés par catégorie", fontsize=10, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)

    n_uc = len(uc)
    n_ec = len(ec)
    fig.suptitle(
        f"Vue d'ensemble — Évaluation du pipeline RAG end-to-end  "
        f"({len(metrics)} requêtes : {n_uc} cas d'usage + {n_ec} cas limites)",
        fontsize=12, fontweight="bold",
    )
    _save(fig, "summary_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _src_label(doc) -> str:
    src = doc.metadata.get("source", "?")
    return src if src.startswith("http") else Path(src).name


def export_csv(metrics: List[Dict]) -> None:
    if not metrics:
        return
    p = REPORT_DIR / "metrics.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print("  [export] metrics.csv")


def export_individual_transcripts(
    results: List[Tuple[RAGResult, Dict]],
    test_cases: List[TestCase],
) -> None:
    for (result, met), tc in zip(results, test_cases):
        p = TRANSCRIPTS_DIR / f"{tc.id.lower()}_transcript.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"{'═'*72}\n")
            f.write(f"  {tc.id} — {tc.subcategory}\n")
            f.write(f"  Catégorie : {tc.category}  |  Stratégie : {result.strategy}\n")
            f.write(f"{'═'*72}\n\n")
            f.write(f"  Question :\n    {tc.query}\n\n")
            f.write(f"  Notes : {tc.notes}\n\n")
            f.write(f"{'─'*72}\n")
            f.write(f"  Métriques d'évaluation\n")
            f.write(f"    Docs récupérés  : {met['n_docs']}\n")
            f.write(f"    Sources uniques : {met['unique_sources']}\n")
            f.write(f"    Citation [N]    : {'Oui' if met['citation_present'] else 'Non'}\n")
            f.write(f"    Structure 3 sec.: {'Oui' if met['structure_ok'] else 'Non'}\n")
            f.write(f"    Ancrage         : {met['grounding_score']:.3f}  (proxy lexical)\n")
            f.write(f"    Risque halluc.  : {'Oui' if met['hallucination_risk'] else 'Non'}\n")
            f.write(f"    Hors périmètre  : {'Oui' if met['out_of_scope'] else 'Non'}\n")
            f.write(f"    Verdict         : {met['verdict']}\n\n")
            f.write(f"{'─'*72}\n")
            f.write(f"  Documents récupérés ({met['n_docs']})\n")
            for i, doc in enumerate(result.retrieved_documents, 1):
                f.write(f"    [{i}] {_src_label(doc)}  "
                        f"(chunk_id={doc.metadata.get('chunk_id', '?')})\n")
                excerpt = textwrap.fill(
                    doc.page_content[:250], width=68,
                    initial_indent="        ", subsequent_indent="        ",
                )
                f.write(f"{excerpt}\n\n")
            f.write(f"{'─'*72}\n")
            f.write(f"  Réponse générée\n\n")
            for line in result.answer.splitlines():
                f.write(f"  {line}\n")
    print(f"  [export] {len(results)} transcripts individuels")


def export_dialogue_transcripts(all_turns: List[Dict]) -> None:
    by_flow: Dict[str, List] = defaultdict(list)
    for t in all_turns:
        by_flow[t["flow_id"]].append(t)

    for fid, turns in sorted(by_flow.items()):
        turns = sorted(turns, key=lambda t: t["turn"])
        p = TRANSCRIPTS_DIR / f"{fid.lower()}_dialogue.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"{'═'*72}\n")
            f.write(f"  {fid} — {turns[0]['flow_name']}\n")
            f.write(f"{'═'*72}\n\n")
            for t in turns:
                f.write(f"  ── Tour {t['turn']}\n")
                f.write(f"  Utilisateur : {t['question']}\n\n")
                f.write(f"  Métriques :\n")
                f.write(
                    f"    Citation : {'Oui' if t['citation_present'] else 'Non'}  |  "
                    f"Structure : {'OK' if t['structure_ok'] else 'NOK'}  |  "
                    f"Docs : {t['n_docs']}  |  "
                    f"Ancrage : {t['grounding']:.2f}  |  "
                    f"Réf. hist. : {'Oui' if t['history_ref'] else 'Non'}\n\n"
                )
                f.write("  ResearchPal :\n")
                for line in t["answer"].splitlines():
                    f.write(f"    {line}\n")
                f.write(f"\n{'─'*72}\n\n")
    print(f"  [export] {len(by_flow)} transcripts de dialogue")


def export_markdown(
    metrics: List[Dict],
    all_turns: List[Dict],
    timestamp: str,
) -> None:
    p = REPORT_DIR / "summary.md"

    def pct(group, key):
        if not group:
            return 0.0
        return sum(1 for m in group if m[key]) / len(group) * 100

    uc = [m for m in metrics if m["category"] == "use_case"]
    ec = [m for m in metrics if m["category"] == "edge_case"]

    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Évaluation du pipeline RAG end-to-end — Tâche 3\n\n")
        f.write(f"*Généré le {timestamp}*\n\n")

        f.write("## Paramètres du pipeline\n\n")
        f.write("| Paramètre | Valeur |\n|---|---|\n")
        f.write(f"| Modèle LLM | `{OLLAMA_MODEL}` |\n")
        f.write(f"| Top-K | `{RETRIEVAL_TOP_K}` |\n")
        f.write(f"| fetch\\_k (MMR) | `{MMR_FETCH_K}` |\n")
        f.write(f"| λ (MMR) | `{MMR_LAMBDA}` |\n")
        f.write(f"| Pattern de prompting | **Persona + Structured Output** |\n")
        f.write(f"| Mémoire conversationnelle | Fenêtre glissante (5 tours max) |\n")
        f.write(f"| Détection hors périmètre | Score cosinus < 0.1 + filtre mots-clés |\n\n")

        f.write("## Résumé des résultats\n\n")
        f.write("| Métrique | Cas d'usage | Cas limites | Total |\n|---|---|---|---|\n")
        for label, key in [
            ("Citation présente", "citation_present"),
            ("Structure complète", "structure_ok"),
            ("Hallucination détectée", "hallucination_risk"),
            ("Hors périmètre détecté", "out_of_scope"),
        ]:
            f.write(
                f"| {label} | {pct(uc, key):.0f}% | {pct(ec, key):.0f}% "
                f"| {pct(metrics, key):.0f}% |\n"
            )
        avg_gs = lambda g: np.mean([m["grounding_score"] for m in g]) if g else 0
        f.write(
            f"| Ancrage moyen | {avg_gs(uc):.3f} | {avg_gs(ec):.3f} "
            f"| {avg_gs(metrics):.3f} |\n\n"
        )

        f.write("## Tableau détaillé par requête\n\n")
        f.write("| ID | Catégorie | Sous-type | Cit. | Struct. | Ancrage | Verdict |\n")
        f.write("|---|---|---|:---:|:---:|---:|---|\n")
        for m in metrics:
            f.write(
                f"| {m['id']} | {m['category']} | {m['subcategory'][:35]} | "
                f"{'✓' if m['citation_present'] else '✗'} | "
                f"{'✓' if m['structure_ok'] else '✗'} | "
                f"{m['grounding_score']:.2f} | {m['verdict']} |\n"
            )

        f.write("\n## Analyse des dialogues multi-tours\n\n")
        by_flow: Dict[str, List] = defaultdict(list)
        for t in all_turns:
            by_flow[t["flow_id"]].append(t)
        for fid, turns in sorted(by_flow.items()):
            turns = sorted(turns, key=lambda x: x["turn"])
            f.write(f"### {fid} — {turns[0]['flow_name']}\n\n")
            f.write("| Tour | Citation | Structure | Réf. hist. | Docs | Ancrage |\n")
            f.write("|:---:|:---:|:---:|:---:|:---:|---:|\n")
            for t in turns:
                f.write(
                    f"| {t['turn']} "
                    f"| {'✓' if t['citation_present'] else '✗'} "
                    f"| {'✓' if t['structure_ok'] else '✗'} "
                    f"| {'✓' if t['history_ref'] else '–'} "
                    f"| {t['n_docs']} "
                    f"| {t['grounding']:.2f} |\n"
                )
            f.write("\n")

        f.write("## Prompt système (Persona + Structured Output)\n\n```\n")
        f.write(SYSTEM_PROMPT)
        f.write("\n```\n\n")

        f.write("## Figures générées\n\n")
        for fp in sorted(FIGURES_DIR.glob("*.png")):
            f.write(f"- `figures/{fp.name}`\n")

    print("  [export] summary.md")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 68)
    print("  ResearchPal — Évaluation end-to-end Tâche 3")
    print("═" * 68)

    # ── 1. Initialisation ────────────────────────────────────────────────────
    print("\n[1/5] Initialisation du pipeline RAG...")
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        print(f"\n  ERREUR : {exc}")
        print("  Vérifiez qu'Ollama est lancé : ollama serve")
        sys.exit(1)

    # ── 2. Requêtes individuelles ────────────────────────────────────────────
    print("\n[2/5] Requêtes individuelles...")
    uc_results, ec_results = run_all_single_tests(pipeline)
    all_results = uc_results + ec_results
    all_metrics = [m for _, m in all_results]

    # ── 3. Dialogues multi-tours ─────────────────────────────────────────────
    print("\n[3/5] Dialogues multi-tours...")
    all_turns: List[Dict] = []
    for flow in DIALOGUE_FLOWS:
        all_turns.extend(run_dialogue_flow(pipeline, flow))

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("\n[4/5] Génération des figures...")
    fig_verdict_breakdown(all_metrics)
    fig_compliance_rates(all_metrics)
    fig_grounding_scores(all_metrics)
    fig_hallucination_flags(all_metrics)
    fig_dialogue_quality(all_turns)
    fig_summary_overview(all_metrics)

    # ── 5. Exports ───────────────────────────────────────────────────────────
    print("\n[5/5] Export des résultats...")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    export_csv(all_metrics)
    export_individual_transcripts(all_results, ALL_QUERIES)
    export_dialogue_transcripts(all_turns)
    export_markdown(all_metrics, all_turns, ts)

    # ── Résumé console ───────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  ANALYSE COMPLÈTE")
    print("═" * 68)

    n_tot     = len(all_metrics)
    n_success = sum(1 for m in all_metrics if m["verdict"] == "succès")
    n_uc_ok   = sum(1 for m in uc_results if m[1]["verdict"] == "succès")
    n_ec_ok   = sum(1 for m in ec_results if m[1]["verdict"] == "succès")

    print(f"\n  Total        : {n_tot} requêtes")
    print(f"  Succès total : {n_success}/{n_tot}  ({n_success/n_tot*100:.0f}%)")
    print(f"  Cas d'usage  : {n_uc_ok}/{len(uc_results)} réussis")
    print(f"  Cas limites  : {n_ec_ok}/{len(ec_results)} réussis")
    print(f"\n  Tous les résultats dans : {REPORT_DIR}")

    all_files = sorted(REPORT_DIR.rglob("*"))
    for fp in all_files:
        if fp.is_file():
            rel  = fp.relative_to(REPORT_DIR)
            size = fp.stat().st_size
            print(f"    {str(rel):<52} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
