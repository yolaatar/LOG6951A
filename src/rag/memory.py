# memory.py — historique de conversation multi-tours (T3)
# Stocke chaque tour : question + réponse + sources utilisées

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)  # noms de fichiers / URLs


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
    ) -> None:
        """Enregistre un échange. Troncature automatique après max_turns."""
        self._turns.append(Turn(
            question=user_message,
            answer=assistant_message,
            sources=sources or [],
        ))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

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
