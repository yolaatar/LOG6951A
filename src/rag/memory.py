# memory.py — gestion de l'historique de conversation (fenêtre glissante)

from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ConversationMemory:
    """Stocke les échanges récents pour le contexte multi-tours."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[BaseMessage] = []

    def add_exchange(self, user_msg: str, ai_msg: str) -> None:
        """Ajoute un échange et tronque si nécessaire."""
        self.history.append(HumanMessage(content=user_msg))
        self.history.append(AIMessage(content=ai_msg))

        # garder seulement les max_turns derniers échanges (2 messages par tour)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[BaseMessage]:
        return self.history

    def clear(self) -> None:
        self.history = []
