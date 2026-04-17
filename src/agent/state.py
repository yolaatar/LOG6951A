# agent/state.py — définition de l'état LangGraph (T1)
#
# AgentState contient tous les champs qui transitent entre les nœuds du graphe.
# Champs obligatoires selon l'énoncé : question, documents, generation, retry_count.
# Champs supplémentaires : retrieval_query (reformulée par transform_query),
# relevant_docs (après grading), tool_used (corpus ou web), web_results.

from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    # ── champs obligatoires (énoncé T1) ──────────────────────────────────────
    question: str                        # question originale de l'utilisateur
    documents: List[Document]            # documents récupérés par retrieve
    generation: str                      # réponse générée par le LLM
    retry_count: int                     # nombre de cycles de correction (max 3)

    # ── champs supplémentaires ───────────────────────────────────────────────
    retrieval_query: str                 # requête effective (peut être reformulée)
    relevant_docs: List[Document]        # docs jugés pertinents après grade_documents
    grade_decision: str                  # "sufficient" | "insufficient"
    tool_used: str                       # "corpus" | "web"
    web_results: Optional[str]           # résultats bruts de la recherche web
