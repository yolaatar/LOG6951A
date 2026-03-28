# before_after_eval.py — Comparaison avant/après pour 2 améliorations du prompt (Tâche 3)
#
# Amélioration A : Cognitive Verifier — testé sur EC4 (appât hallucination) et EC7 (retrieval trompeur)
# Amélioration B : Calibrage de l'historique — testé sur UC4 (suivi) et EC5 (mémoire tour 1)
#
# Usage : cd src && ../.venv/bin/python evaluation/before_after_eval.py

import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate

from rag.chain import RAGPipeline
from rag.prompt import SYSTEM_PROMPT, format_context

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "rag_eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — VARIANTES DE PROMPT ET D'HISTORIQUE
# ─────────────────────────────────────────────────────────────────────────────

# ── Règle Cognitive Verifier ─────────────────────────────────────────────────
_CV_RULE = """\
Vérification cognitive (à effectuer AVANT de rédiger **Réponse**) :
  Pour chaque affirmation factuelle que tu t'apprêtes à écrire, pose-toi la question :
  « Est-ce que cette information figure explicitement dans un extrait numéroté du contexte ? »
  → OUI : écris-la dans **Réponse** avec la citation [N].
  → NON : écris-la dans **Limites / Incertitudes**, JAMAIS dans **Réponse**.
  Ne génère AUCUN chiffre, score de benchmark, pourcentage, date ou nom de modèle
  qui ne figure pas mot pour mot dans le contexte fourni.

"""

SYSTEM_PROMPT_CV = SYSTEM_PROMPT.replace(
    "Règles absolues :\n1.",
    "Règles absolues :\n1.",
).replace(
    "2. Tu n'inventes aucune information.",
    "2. Tu n'inventes aucune information.",
)
# Insertion après la règle 2 (avant la règle 3)
SYSTEM_PROMPT_CV = SYSTEM_PROMPT_CV.replace(
    "3. Chaque affirmation importante doit être rattachée à une source numérotée [N].",
    _CV_RULE
    + "3. Chaque affirmation importante doit être rattachée à une source numérotée [N].",
)


# ── Formatage de l'historique — baseline (troncature brute) ──────────────────

def format_history_baseline(history: List[Tuple[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Historique de la conversation :"]
    for i, (q, a) in enumerate(history, 1):
        lines.append(f"Tour {i}")
        lines.append(f"  Utilisateur : {q}")
        a_short = a[:300] + "…" if len(a) > 300 else a
        lines.append(f"  Assistant   : {a_short}")
    return "\n".join(lines) + "\n\n"


# ── Formatage de l'historique — calibré (extraction de la section Réponse) ──

def _extract_answer_core(answer: str, max_chars: int = 280) -> str:
    """
    Extrait uniquement le corps de la section **Réponse** (sans headers ni citations).
    Si la structure 3-sections est absente, revient à la troncature brute.
    """
    m = re.search(
        r"\*\*Réponse\*\*\s*\n(.*?)(?:\n\*\*Sources\*\*|\n\*\*Limites|\Z)",
        answer, re.DOTALL | re.IGNORECASE,
    )
    if m:
        core = m.group(1).strip()
        core = re.sub(r"\[\d+\]", "", core).strip()   # supprimer [N]
        core = re.sub(r"\s+", " ", core)
        return (core[:max_chars] + "…") if len(core) > max_chars else core
    return (answer[:300] + "…") if len(answer) > 300 else answer


def format_history_calibrated(history: List[Tuple[str, str]]) -> str:
    """
    Résumé structuré : pour chaque tour, seul le corps de la réponse (section Réponse)
    est conservé, sans les headers, sources et limites qui gaspillent la fenêtre de contexte.
    """
    if not history:
        return ""
    lines = ["Historique de la conversation (résumé) :"]
    for i, (q, a) in enumerate(history, 1):
        lines.append(f"Tour {i}")
        lines.append(f"  Utilisateur : {q}")
        core = _extract_answer_core(a)
        lines.append(f"  Réponse principale : {core}")
    return "\n".join(lines) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MOTEUR D'EXÉCUTION PARAMÉTRABLE
# ─────────────────────────────────────────────────────────────────────────────

def run_with_config(
    pipeline: RAGPipeline,
    question: str,
    system_prompt: str,
    history_formatter,
    record_in_memory: bool = True,
) -> Dict:
    """
    Exécute une requête avec un prompt et un formateur d'historique arbitraires,
    sans modifier les fichiers source du pipeline.
    """
    docs, _ = pipeline._retrieve(question, "cosine", False)
    context = format_context(docs)
    history_pairs = pipeline.memory.format_history_for_prompt()
    history_block = history_formatter(history_pairs)

    system = system_prompt.replace("{history_block}", history_block)
    prompt_tpl = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    chain = prompt_tpl | pipeline.llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)

    if record_in_memory:
        srcs = [Path(d.metadata.get("source", "?")).name for d in docs]
        pipeline.memory.add_turn(question, answer, sources=srcs)

    return {"answer": answer, "docs": docs}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MÉTRIQUES D'ÉVALUATION CIBLÉES
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
    r"|tour\s*1|comme tu|comme je|comme indiqué)",
    re.IGNORECASE,
)


def _halluc_risk(answer: str) -> bool:
    return any(p.search(answer) for p in _RE_HALLUC)


def _limits_acknowledges_absence(answer: str) -> bool:
    """Section Limites indique-t-elle explicitement l'absence d'info dans le corpus ?"""
    m = re.search(r"\*\*Limites.*", answer, re.DOTALL | re.IGNORECASE)
    if not m:
        return False
    txt = m.group(0).lower()
    return any(k in txt for k in [
        "corpus", "contexte", "extrait", "information",
        "pas dans", "absent", "ne figure", "ne contient",
        "insuffisant", "manquant", "fourni", "disponible",
    ])


def _history_referenced(answer: str) -> bool:
    return bool(_RE_HIST_REF.search(answer))


def _first_response_point_found(answer: str, expected_fragment: str) -> bool:
    """
    Vérifie si la réponse mentionne une idée proche du fragment attendu du tour 1.
    Proxy : présence d'au moins 2 mots-clés du fragment dans la réponse.
    """
    keywords = [w for w in re.findall(r"\w{5,}", expected_fragment.lower())]
    found = sum(1 for k in keywords if k in answer.lower())
    return found >= 2


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PROTOCOLE DE COMPARAISON
# ─────────────────────────────────────────────────────────────────────────────

def _divider(title: str = "") -> str:
    return f"\n{'─'*72}\n  {title}\n{'─'*72}" if title else "\n" + "─" * 72


def _format_answer_block(label: str, answer: str, metrics: dict) -> str:
    lines = [f"\n  [{label}]"]
    lines.append(f"  Métriques :")
    lines.append(f"    Risque hallucination   : {'OUI ⚠' if metrics['halluc'] else 'non'}")
    lines.append(f"    Limites explicites     : {'OUI ✓' if metrics['limits_ok'] else 'non'}")
    lines.append(f"    Référence historique   : {'OUI ✓' if metrics['hist_ref'] else 'non'}")
    lines.append(f"    Ancrage contextuel     : {metrics['grounding']:.3f}")
    lines.append("")
    lines.append("  Réponse (extrait 600 chars) :")
    excerpt = answer[:600] + "…" if len(answer) > 600 else answer
    for line in excerpt.splitlines():
        lines.append(f"    {line}")
    return "\n".join(lines)


# ── Test A : Cognitive Verifier (EC4 + EC7) ───────────────────────────────────

def test_cognitive_verifier(pipeline: RAGPipeline) -> Dict:
    """Compare BASELINE vs CV sur EC4 et EC7. Retourne un dict de résultats."""
    results = {}

    queries = {
        "EC4": (
            "Quelle est la précision exacte (accuracy) de mistral:7b-instruct sur le "
            "benchmark MMLU ? Donne le score en pourcentage.",
            "halluc_reduced",   # critère de succès attendu
        ),
        "EC7": (
            "Comment fonctionne le mécanisme d'attention dans les transformers "
            "pour produire des représentations vectorielles ?",
            "limits_more_explicit",
        ),
    }

    for qid, (question, criterion) in queries.items():
        for variant, (sys_prompt, hist_fmt, label) in {
            "baseline": (SYSTEM_PROMPT,    format_history_baseline,  "Baseline"),
            "cv":       (SYSTEM_PROMPT_CV, format_history_baseline,  "Cognitive Verifier"),
        }.items():
            pipeline.reset_memory()
            out = run_with_config(pipeline, question, sys_prompt, hist_fmt,
                                  record_in_memory=False)
            ans  = out["answer"]
            docs = out["docs"]
            met  = {
                "halluc":    _halluc_risk(ans),
                "limits_ok": _limits_acknowledges_absence(ans),
                "hist_ref":  _history_referenced(ans),
                "grounding": _grounding(ans, docs),
            }
            results.setdefault(qid, {})[variant] = {
                "label":    label,
                "answer":   ans,
                "docs":     docs,
                "metrics":  met,
                "criterion": criterion,
            }
        pipeline.reset_memory()

    return results


# ── Test B : Calibrage historique (UC4 + EC5) ────────────────────────────────

_UC_SETUP_QUERIES = [
    "Qu'est-ce que RAG (Retrieval-Augmented Generation) et pourquoi est-il utilisé ?",
    "Quelles sont les différences entre cosinus similarity et MMR dans un pipeline RAG ?",
    "Comment le chevauchement entre les chunks affecte-t-il la qualité du retrieval ?",
]

_EC5_SETUP_QUERIES = [
    "Qu'est-ce que la stratégie MMR et comment fonctionne-t-elle ?",
    "Quels sont les paramètres clés de MMR et comment les choisir ?",
]

# Contenu attendu du tour 1 pour EC5 (fragment de référence)
_EC5_TURN1_EXPECTED = "MMR Maximal Marginal Relevance diversité pertinence"


def test_history_calibration(pipeline: RAGPipeline) -> Dict:
    """Compare BASELINE vs CALIBRATED sur UC4 (après 3 tours) et EC5 (après 2 tours setup)."""
    results = {}

    configs = {
        "baseline":   (SYSTEM_PROMPT, format_history_baseline,  "Baseline"),
        "calibrated": (SYSTEM_PROMPT, format_history_calibrated, "Historique calibré"),
    }

    # ── UC4 ──────────────────────────────────────────────────────────────────
    uc4_question = "Quelles sont les principales limites de l'approche que tu viens de décrire ?"

    for variant, (sys_prompt, hist_fmt, label) in configs.items():
        pipeline.reset_memory()
        # Setup : injecter UC1–UC3 dans la mémoire avec le formateur correspondant
        for setup_q in _UC_SETUP_QUERIES:
            out = run_with_config(pipeline, setup_q, sys_prompt, hist_fmt,
                                  record_in_memory=True)
        # Tour 4 : la requête de suivi
        out4 = run_with_config(pipeline, uc4_question, sys_prompt, hist_fmt,
                               record_in_memory=False)
        ans  = out4["answer"]
        docs = out4["docs"]
        met  = {
            "halluc":    _halluc_risk(ans),
            "limits_ok": _limits_acknowledges_absence(ans),
            "hist_ref":  _history_referenced(ans),
            "grounding": _grounding(ans, docs),
        }
        # Vérifier la présence du mot-clé "RAG" et "limite" dans la réponse
        rag_mentioned = bool(re.search(r"\bRAG\b", ans, re.IGNORECASE))
        results.setdefault("UC4", {})[variant] = {
            "label":       label,
            "answer":      ans,
            "docs":        docs,
            "metrics":     met,
            "rag_mentioned": rag_mentioned,
            "criterion":   "hist_ref_and_rag",
        }

    # ── EC5 ──────────────────────────────────────────────────────────────────
    ec5_question = (
        "Peux-tu développer le premier point que tu avais mentionné dans ta toute "
        "première réponse sur ce sujet ?"
    )

    for variant, (sys_prompt, hist_fmt, label) in configs.items():
        pipeline.reset_memory()
        for setup_q in _EC5_SETUP_QUERIES:
            run_with_config(pipeline, setup_q, sys_prompt, hist_fmt,
                            record_in_memory=True)
        out = run_with_config(pipeline, ec5_question, sys_prompt, hist_fmt,
                              record_in_memory=False)
        ans  = out["answer"]
        docs = out["docs"]
        met  = {
            "halluc":    _halluc_risk(ans),
            "limits_ok": _limits_acknowledges_absence(ans),
            "hist_ref":  _history_referenced(ans),
            "grounding": _grounding(ans, docs),
        }
        first_point_found = _first_response_point_found(ans, _EC5_TURN1_EXPECTED)
        results.setdefault("EC5", {})[variant] = {
            "label":           label,
            "answer":          ans,
            "docs":            docs,
            "metrics":         met,
            "first_point":     first_point_found,
            "criterion":       "first_point_referenced",
        }

    pipeline.reset_memory()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RAPPORT ET DÉCISION
# ─────────────────────────────────────────────────────────────────────────────

def _improvement_verdict(qid: str, baseline: dict, improved: dict) -> Tuple[bool, str]:
    """
    Retourne (amélioration: bool, explication: str) selon des critères spécifiques à chaque test.
    """
    bm = baseline["metrics"]
    im = improved["metrics"]

    if qid == "EC4":
        improved_flag = not im["halluc"] or im["limits_ok"]
        baseline_flag = bm["halluc"] or not bm["limits_ok"]
        ok = improved_flag and (not bm["halluc"] == False or im["limits_ok"] > bm["limits_ok"])
        expl = (
            f"Hallucination : {bm['halluc']} → {im['halluc']}  |  "
            f"Limites explicites : {bm['limits_ok']} → {im['limits_ok']}"
        )
        return (not im["halluc"] and im["limits_ok"]) or (not bm["halluc"] and not im["halluc"]), expl

    elif qid == "EC7":
        improved_grounding = im["grounding"] > bm["grounding"] + 0.02
        improved_limits    = im["limits_ok"] and not bm["limits_ok"]
        expl = (
            f"Ancrage : {bm['grounding']:.3f} → {im['grounding']:.3f}  |  "
            f"Limites explicites : {bm['limits_ok']} → {im['limits_ok']}"
        )
        return improved_grounding or improved_limits, expl

    elif qid == "UC4":
        improved_href = im["hist_ref"] and not bm["hist_ref"]
        improved_rag  = improved["rag_mentioned"] and not baseline.get("rag_mentioned", False)
        expl = (
            f"Réf. historique : {bm['hist_ref']} → {im['hist_ref']}  |  "
            f"RAG mentionné : {baseline.get('rag_mentioned', False)} → {improved.get('rag_mentioned', False)}"
        )
        return improved_href or improved_rag, expl

    elif qid == "EC5":
        improved_fp   = improved.get("first_point", False) and not baseline.get("first_point", False)
        improved_href = im["hist_ref"] and not bm["hist_ref"]
        expl = (
            f"Réf. historique : {bm['hist_ref']} → {im['hist_ref']}  |  "
            f"1er point trouvé : {baseline.get('first_point', False)} → {improved.get('first_point', False)}"
        )
        return improved_fp or improved_href, expl

    return False, "critère non défini"


def write_comparison_report(
    cv_results: Dict,
    hist_results: Dict,
) -> Tuple[bool, bool]:
    """Écrit le rapport de comparaison. Retourne (keep_cv, keep_hist_calibration)."""

    p = REPORT_DIR / "before_after_comparison.txt"
    keep_cv   = False
    keep_hist = False

    with open(p, "w", encoding="utf-8") as f:
        ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"{'═'*72}\n")
        f.write(f"  Comparaison Avant/Après — Tâche 3 — {ts}\n")
        f.write(f"{'═'*72}\n")

        # ── Amélioration A : Cognitive Verifier ──────────────────────────────
        f.write(f"\n{'═'*72}\n")
        f.write(f"  AMÉLIORATION A : Cognitive Verifier\n")
        f.write(f"  Requêtes testées : EC4 (appât hallucination), EC7 (retrieval trompeur)\n")
        f.write(f"{'═'*72}\n")

        cv_improved_count = 0
        for qid in ["EC4", "EC7"]:
            baseline = cv_results[qid]["baseline"]
            improved = cv_results[qid]["cv"]
            improved_flag, expl = _improvement_verdict(qid, baseline, improved)
            if improved_flag:
                cv_improved_count += 1

            f.write(_divider(f"{qid} — Critère : {baseline['criterion']}"))
            f.write(f"\n  {expl}")
            f.write(f"\n  Verdict : {'AMÉLIORATION ✓' if improved_flag else 'PAS D AMÉLIORATION ✗'}\n")
            f.write(_format_answer_block("BASELINE", baseline["answer"], baseline["metrics"]))
            f.write(_format_answer_block("COGNITIVE VERIFIER", improved["answer"], improved["metrics"]))
            f.write("\n")

        keep_cv = cv_improved_count >= 1
        f.write(f"\n  DÉCISION Cognitive Verifier : {'CONSERVER ✓' if keep_cv else 'RETIRER ✗'}")
        f.write(f"  ({cv_improved_count}/2 requêtes améliorées)\n")

        # ── Amélioration B : Calibrage historique ────────────────────────────
        f.write(f"\n{'═'*72}\n")
        f.write(f"  AMÉLIORATION B : Calibrage de l'historique\n")
        f.write(f"  Requêtes testées : UC4 (suivi après 3 tours), EC5 (référence tour 1)\n")
        f.write(f"{'═'*72}\n")

        hist_improved_count = 0
        for qid in ["UC4", "EC5"]:
            baseline = hist_results[qid]["baseline"]
            improved = hist_results[qid]["calibrated"]
            improved_flag, expl = _improvement_verdict(qid, baseline, improved)
            if improved_flag:
                hist_improved_count += 1

            f.write(_divider(f"{qid} — Critère : {baseline['criterion']}"))
            f.write(f"\n  {expl}")
            f.write(f"\n  Verdict : {'AMÉLIORATION ✓' if improved_flag else 'PAS D AMÉLIORATION ✗'}\n")
            f.write(_format_answer_block("BASELINE", baseline["answer"], baseline["metrics"]))
            f.write(_format_answer_block("CALIBRÉ", improved["answer"], improved["metrics"]))
            f.write("\n")

        keep_hist = hist_improved_count >= 1
        f.write(f"\n  DÉCISION Calibrage historique : {'CONSERVER ✓' if keep_hist else 'RETIRER ✗'}")
        f.write(f"  ({hist_improved_count}/2 requêtes améliorées)\n")

        f.write(f"\n{'═'*72}\n")
        f.write(f"  RÉSUMÉ DES DÉCISIONS\n")
        f.write(f"{'═'*72}\n")
        f.write(f"  Cognitive Verifier         : {'CONSERVÉ — appliqué dans prompt.py' if keep_cv else 'RETIRÉ'}\n")
        f.write(f"  Calibrage historique       : {'CONSERVÉ — appliqué dans prompt.py' if keep_hist else 'RETIRÉ'}\n")

    print(f"  [export] before_after_comparison.txt")
    return keep_cv, keep_hist


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> Tuple[bool, bool]:
    print("═" * 68)
    print("  Comparaison Avant/Après — Tâche 3")
    print("═" * 68)

    print("\n[1/4] Initialisation du pipeline...")
    try:
        pipeline = RAGPipeline()
    except RuntimeError as exc:
        print(f"  ERREUR : {exc}")
        import sys; sys.exit(1)

    print("\n[2/4] Test Amélioration A — Cognitive Verifier (EC4, EC7)...")
    cv_results = test_cognitive_verifier(pipeline)

    print("\n[3/4] Test Amélioration B — Calibrage historique (UC4, EC5)...")
    hist_results = test_history_calibration(pipeline)

    print("\n[4/4] Rédaction du rapport de comparaison...")
    keep_cv, keep_hist = write_comparison_report(cv_results, hist_results)

    print(f"\n  Cognitive Verifier    : {'CONSERVER' if keep_cv   else 'RETIRER'}")
    print(f"  Calibrage historique  : {'CONSERVER' if keep_hist  else 'RETIRER'}")

    return keep_cv, keep_hist


if __name__ == "__main__":
    keep_cv, keep_hist = main()
    # Exporter les décisions pour le script suivant
    decisions_path = REPORT_DIR / "_decisions.txt"
    decisions_path.write_text(f"keep_cv={keep_cv}\nkeep_hist={keep_hist}\n")
