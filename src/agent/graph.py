# agent/graph.py — construction du graphe LangGraph Corrective RAG (T1 + T2 + T3)
#
# Architecture du graphe :
#
#   START
#     ↓
#   route_query ──→ choisit "corpus" ou "web" (T2 : sélection dynamique d'outil)
#     │
#     ├── "corpus" → retrieve (search_corpus @tool)
#     │               ↓
#     │            grade_documents (évalue la pertinence — T1 Corrective RAG)
#     │               ↓
#     │    ┌── "sufficient"          → generate → END
#     │    └── "insufficient"
#     │         ├── retry_count < 3  → transform_query → retrieve (CYCLE T1)
#     │         └── retry_count ≥ 3  → generate → END  (garde-fou)
#     │
#     └── "web" → web_search_node (web_search @tool)
#                   ↓
#                generate → END
#
# Mémoire court terme (T3) : SQLite checkpointer — persist entre sessions.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    make_route_query_node,
    make_retrieve_node,
    make_grade_documents_node,
    make_transform_query_node,
    make_web_search_node,
    make_generate_node,
    decide_after_grading,
    decide_after_routing,
)
from config import OLLAMA_MODEL, OLLAMA_BASE_URL


# ── Initialisation LLM (réutilise get_llm du TP1) ────────────────────────────

def _get_llm(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
    from rag.chain import get_llm
    return get_llm(model, base_url)


# ── Construction du graphe ────────────────────────────────────────────────────

def build_agent_graph(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    checkpointer=None,
):
    """Construit et compile le graphe LangGraph Corrective RAG.

    Args:
        model       : modèle Ollama à utiliser.
        base_url    : URL du serveur Ollama.
        checkpointer: checkpointer LangGraph pour la mémoire court terme (T3).
                      Si None, le graphe est compilé sans persistance.

    Returns:
        CompiledGraph prêt à invoquer via .invoke() ou .stream().
    """
    llm = _get_llm(model, base_url)

    # ── Instanciation des nœuds ──────────────────────────────────────────────
    route_query     = make_route_query_node(llm)
    retrieve        = make_retrieve_node()
    grade_documents = make_grade_documents_node(llm)
    transform_query = make_transform_query_node(llm)
    web_search_node = make_web_search_node()
    generate        = make_generate_node(llm)

    # ── Définition du graphe ─────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("route_query",     route_query)
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate",        generate)

    # ── Arêtes ordinaires ────────────────────────────────────────────────────
    workflow.set_entry_point("route_query")
    workflow.add_edge("retrieve",        "grade_documents")
    workflow.add_edge("transform_query", "retrieve")        # ← CYCLE correctif
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate",        END)

    # ── Arêtes conditionnelles ───────────────────────────────────────────────
    # Après route_query : corpus → retrieve  |  web → web_search_node
    workflow.add_conditional_edges(
        "route_query",
        decide_after_routing,
        {
            "retrieve":        "retrieve",
            "web_search_node": "web_search_node",
        },
    )

    # Après grade_documents : sufficient → generate  |  insufficient → transform_query
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate":        "generate",
            "transform_query": "transform_query",
        },
    )

    # ── Compilation ──────────────────────────────────────────────────────────
    return workflow.compile(checkpointer=checkpointer)


# ── Chargement du checkpointer SQLite (T3 — mémoire court terme) ─────────────

def get_checkpointer(db_path: str | None = None):
    """Retourne un checkpointer SQLite pour la persistance inter-sessions.

    Utilise SqliteSaver si disponible (langgraph-checkpoint-sqlite installé),
    sinon bascule sur MemorySaver (mémoire uniquement, non persistant).
    """
    from config import DATA_DIR

    if db_path is None:
        db_path = str(DATA_DIR / "checkpoints" / "agent_state.db")

    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        # Utiliser sqlite3.connect() directement pour éviter le context manager
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn=conn)
    except (ImportError, Exception):
        from langgraph.checkpoint.memory import MemorySaver
        print("[warning] SqliteSaver non disponible → MemorySaver (non persistant)")
        return MemorySaver()


# ── API publique ──────────────────────────────────────────────────────────────

_graph = None
_checkpointer = None


def get_agent_graph(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
    """Retourne le graphe compilé (singleton, avec checkpointer SQLite)."""
    global _graph, _checkpointer
    if _graph is None:
        _checkpointer = get_checkpointer()
        _graph = build_agent_graph(model, base_url, checkpointer=_checkpointer)
        print("  [agent] Graphe LangGraph compilé avec checkpointer SQLite")
    return _graph


def run_agent(
    question: str,
    thread_id: str = "default",
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    """Exécute le graphe agentique pour une question.

    Args:
        question  : question de l'utilisateur.
        thread_id : identifiant de session (pour la mémoire court terme T3).

    Returns:
        dict avec les champs : generation, tool_used, retry_count, documents,
        relevant_docs, retrieval_query.
    """
    graph = get_agent_graph(model, base_url)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "question":      question,
        "retrieval_query": question,
        "documents":     [],
        "relevant_docs": [],
        "generation":    "",
        "retry_count":   0,
        "grade_decision": "",
        "tool_used":     "",
        "web_results":   None,
    }

    result = graph.invoke(initial_state, config=config)
    return result
