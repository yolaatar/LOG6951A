# memory_v2/episodic.py — mémoire long terme épisodique (T3, Option B)
#
# Stratégie retenue : Option B — Mémoire épisodique légère
#
# Principe : stocker 3 à 5 résolutions de requêtes complexes réussies dans un
# fichier JSON persistant (data/episodic_memory.json), et les injecter comme
# exemples few-shot dans le system prompt au démarrage de chaque session.
#
# Critères de sélection d'une résolution "digne d'être mémorisée" :
#   - Réponse substantielle (> 200 caractères)
#   - Au moins 2 sources citées
#   - L'agent a utilisé le corpus (tool_used = "corpus")
#   - Aucun cycle de correction n'a été nécessaire (retry_count = 0) → qualité
#
# Justification du choix (Option B vs A vs C) :
#   ResearchPal est un assistant de RECHERCHE. Les résolutions complexes (multi-
#   sources, bien citées) sont la forme de savoir la plus utile à réinjecter.
#   Option A (cache sémantique) nécessite une infrastructure d'embedding en plus.
#   Option C (préférences) est moins pertinente pour un outil académique mono-usage.
#   Option B n'ajoute aucune infrastructure : juste un fichier JSON.
#
# Limites observées :
#   - Pas de déduplication sémantique : si 5 questions similaires sont mémorisées,
#     les exemples few-shot deviennent redondants.
#   - La mémoire n'évolue pas si le corpus change (risque de staleness).
#   - Capacité fixe : les exemples les plus anciens sont supprimés si MAX_EXAMPLES
#     est atteint (fenêtre glissante FIFO).

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR


MAX_EXAMPLES = 5
MEMORY_FILE = DATA_DIR / "episodic_memory.json"

# Critères de mémorisation
MIN_ANSWER_LEN = 200
MIN_SOURCES = 2


# ── Structure d'un épisode ───────────────────────────────────────────────────

def _make_episode(question: str, answer: str, sources: List[str]) -> dict:
    return {
        "question":   question,
        "answer":     answer[:600],   # tronqué pour éviter les prompts trop longs
        "sources":    sources[:5],
        "timestamp":  datetime.now().isoformat(),
    }


# ── Persistance ──────────────────────────────────────────────────────────────

def _load() -> List[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(episodes: List[dict]) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)


# ── API publique ─────────────────────────────────────────────────────────────

def maybe_store_episode(
    question: str,
    answer: str,
    sources: List[str],
    tool_used: str = "corpus",
    retry_count: int = 0,
) -> bool:
    """Stocke l'épisode si les critères de qualité sont réunis.

    Retourne True si l'épisode a été stocké, False sinon.
    """
    # Critères de sélection
    if tool_used != "corpus":
        return False
    if retry_count > 0:        # une résolution sans cycle = meilleure qualité
        return False
    if len(answer) < MIN_ANSWER_LEN:
        return False
    if len(sources) < MIN_SOURCES:
        return False

    episodes = _load()

    # Éviter les doublons stricts sur la question
    existing_questions = {e["question"] for e in episodes}
    if question in existing_questions:
        return False

    episodes.append(_make_episode(question, answer, sources))

    # Fenêtre glissante FIFO : garder les MAX_EXAMPLES plus récents
    if len(episodes) > MAX_EXAMPLES:
        episodes = episodes[-MAX_EXAMPLES:]

    _save(episodes)
    return True


def load_episodes() -> List[dict]:
    """Charge tous les épisodes mémorisés."""
    return _load()


def format_few_shot_block(episodes: Optional[List[dict]] = None) -> str:
    """Formate les épisodes comme bloc few-shot pour injection dans le system prompt.

    Retourne une chaîne vide si aucun épisode disponible.
    """
    if episodes is None:
        episodes = _load()
    if not episodes:
        return ""

    lines = [
        "\n\n--- Exemples de résolutions passées (few-shot) ---"
    ]
    for i, ep in enumerate(episodes, 1):
        q = ep.get("question", "")
        a = ep.get("answer", "")
        srcs = ", ".join(ep.get("sources", []))
        lines.append(f"\nExemple {i} :")
        lines.append(f"  Question : {q}")
        lines.append(f"  Réponse  : {a[:300]}{'…' if len(a) > 300 else ''}")
        lines.append(f"  Sources  : {srcs}")
    lines.append("--- Fin des exemples ---\n")
    return "\n".join(lines)


def clear_episodes() -> None:
    """Efface tous les épisodes (pour les tests)."""
    _save([])
