# memory.py — historique de conversation multi-tours (T3)
# Stocke chaque tour : question + réponse + sources utilisées + had_retrieval
#
# Le champ had_retrieval est utilisé par le filtre out-of-scope de chain.py :
# si un tour récent a eu un vrai retrieval (had_retrieval=True), les questions
# de suivi sont autorisées même si leur score cosinus seul est faible.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)  # noms de fichiers / URLs
    had_retrieval: bool = True  # False si répondu hors-périmètre sans LLM


class ConversationMemory:
    """Fenêtre glissante sur les derniers tours de conversation."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    # ── API principale ───────────────────────────────────────────────────

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        sources: Optional[List[str]] = None,
        had_retrieval: bool = True,
    ) -> None:
        """Enregistre un échange. Troncature automatique après max_turns."""
        self._turns.append(Turn(
            question=user_message,
            answer=assistant_message,
            sources=sources or [],
            had_retrieval=had_retrieval,
        ))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def recent_had_retrieval(self, window: int = 2) -> bool:
        """
        Retourne True si au moins un des `window` derniers tours a eu un vrai retrieval.
        Utilisé pour autoriser les questions de suivi implicites.
        """
        return any(t.had_retrieval for t in self._turns[-window:])

    def last_inscope_turn(self) -> Optional[Turn]:
        """Retourne le dernier tour qui a eu un vrai retrieval, ou None."""
        for t in reversed(self._turns):
            if t.had_retrieval:
                return t
        return None

    def get_history(self) -> List[Turn]:
        """Retourne tous les tours mémorisés."""
        return list(self._turns)

    def clear_history(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)

    # ── Formatage pour le prompt ────────────────────────────────────────────

    def format_history_for_prompt(self) -> List[tuple[str, str]]:
        """
        Retourne l'historique sous forme de liste (question, réponse)
        compatible avec format_history_block() de prompt.py.
        """
        return [(t.question, t.answer) for t in self._turns]

    # ── Affichage console ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Affiche un résumé lisible de la conversation."""
        if not self._turns:
            print("  (aucun historique)")
            return
        for i, turn in enumerate(self._turns, 1):
            print(f"  Tour {i}")
            print(f"    Q : {turn.question}")
            ans_short = turn.answer[:120].replace("\n", " ") + ("…" if len(turn.answer) > 120 else "")
            print(f"    A : {ans_short}")
            if turn.sources:
                src_labels = [
                    Path(s).name if not s.startswith("http") else s
                    for s in turn.sources
                ]
                print(f"    Sources : {', '.join(src_labels)}")
