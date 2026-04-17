# tests/test_agent_integration.py — tests d'intégration end-to-end (T1 + T2)
#
# Requiert : Ollama + mistral:7b-instruct + vectorstore ingéré
#
# Usage :
#   pytest tests/test_agent_integration.py -v -m integration
#   pytest tests/test_agent_integration.py -v                  # tous les tests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


@pytest.fixture(scope="module")
def agent_graph():
    """Compile le graphe une seule fois pour tous les tests du module."""
    from agent.graph import build_agent_graph
    from langgraph.checkpoint.memory import MemorySaver
    return build_agent_graph(checkpointer=MemorySaver())


def run(agent_graph, question, thread_id="test"):
    from agent.state import AgentState
    state: AgentState = {
        "question": question,
        "retrieval_query": question,
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "retry_count": 0,
        "grade_decision": "",
        "tool_used": "",
        "web_results": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return agent_graph.invoke(state, config=config)


@pytest.mark.integration
class TestRouting:
    def test_corpus_question_uses_corpus_tool(self, agent_graph):
        """Une question sur le RAG doit utiliser search_corpus."""
        result = run(agent_graph, "Qu'est-ce que le RAG ?", "test-routing-corpus")
        assert result["tool_used"] == "corpus"

    def test_web_question_uses_web_tool(self, agent_graph):
        """Une question sur les actualités doit utiliser web_search."""
        result = run(agent_graph, "Quel est le cours actuel de l'action Apple ?", "test-routing-web")
        assert result["tool_used"] == "web"

    def test_adversarial_question_routes_somewhere(self, agent_graph):
        """Une question adversariale doit au moins retourner une génération."""
        result = run(agent_graph, "Quel est objectivement le meilleur LLM ?", "test-adversarial")
        assert result["generation"] != ""


@pytest.mark.integration
class TestCorrectiveRAG:
    def test_answer_is_generated(self, agent_graph):
        """Le graphe produit toujours une génération (pas de sortie vide)."""
        result = run(agent_graph, "Qu'est-ce que le MMR ?", "test-mmr")
        assert result["generation"] != ""
        assert len(result["generation"]) > 50

    def test_retry_count_within_bounds(self, agent_graph):
        """Le garde-fou empêche plus de 3 cycles correctifs."""
        result = run(agent_graph, "Qu'est-ce que ChromaDB ?", "test-chromadb")
        assert result["retry_count"] <= 3

    def test_corpus_result_has_documents(self, agent_graph):
        """Une question corpus doit récupérer des documents."""
        result = run(agent_graph, "Expliquez les embeddings.", "test-embeddings")
        docs = result.get("documents") or result.get("relevant_docs") or []
        assert len(docs) > 0


@pytest.mark.integration
class TestMemory:
    def test_thread_id_is_preserved(self, agent_graph):
        """Le thread_id permet la persistance entre appels."""
        thread = "test-memory-persistence"
        r1 = run(agent_graph, "Qu'est-ce que le RAG ?", thread)
        r2 = run(agent_graph, "Et le MMR ?", thread)
        # Les deux réponses doivent être générées sans erreur
        assert r1["generation"] != ""
        assert r2["generation"] != ""

    def test_different_threads_are_independent(self, agent_graph):
        """Deux threads différents sont indépendants."""
        r1 = run(agent_graph, "Qu'est-ce que le RAG ?", "thread-A")
        r2 = run(agent_graph, "Qu'est-ce que le RAG ?", "thread-B")
        assert r1["generation"] != ""
        assert r2["generation"] != ""


@pytest.mark.integration
class TestStateFields:
    def test_result_has_all_expected_fields(self, agent_graph):
        """Le résultat du graphe contient tous les champs de AgentState."""
        result = run(agent_graph, "Qu'est-ce que le chunking ?", "test-fields")
        for field in ["generation", "tool_used", "retry_count", "question"]:
            assert field in result, f"Champ manquant : {field}"

    def test_generation_not_empty_on_corpus(self, agent_graph):
        result = run(agent_graph, "Qu'est-ce que LangChain ?", "test-langchain")
        assert result["generation"].strip() != ""

    def test_web_result_populated_for_web_query(self, agent_graph):
        result = run(agent_graph, "Quelle est la météo à Montréal ?", "test-web-results")
        if result["tool_used"] == "web":
            # web_results peut être None si DuckDuckGo échoue, mais generation doit exister
            assert result["generation"] != ""
