# tests/test_tools.py — tests des outils @tool (T2)
#
# search_corpus : testé avec le vrai vectorstore (requiert l'ingestion)
# web_search    : testé avec mock (pas de dépendance réseau)
#
# Usage :
#   pytest tests/test_tools.py -v
#   pytest tests/test_tools.py -v -m "not integration"   # sans vectorstore

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


# ── web_search (mocké) ──────────────────────────────────────────────────────

class TestWebSearch:
    def test_returns_formatted_results(self):
        """web_search formate correctement les résultats DuckDuckGo."""
        from agent.tools import web_search

        fake_results = [
            {"title": "Apple Stock", "href": "https://example.com/aapl", "body": "Apple stock price today."},
            {"title": "AAPL NASDAQ", "href": "https://example2.com", "body": "AAPL trading at 200$"},
        ]

        with patch("ddgs.DDGS") as mock_ddgs:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.text.return_value = fake_results
            mock_ddgs.return_value = instance

            result = web_search.invoke({"query": "Apple stock price"})

        assert "[1]" in result
        assert "Apple Stock" in result
        assert "https://example.com/aapl" in result

    def test_returns_message_on_empty_results(self):
        from agent.tools import web_search

        with patch("ddgs.DDGS") as mock_ddgs:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.text.return_value = []
            mock_ddgs.return_value = instance

            result = web_search.invoke({"query": "quelque chose d'introuvable"})

        assert "Aucun résultat" in result

    def test_handles_exception_gracefully(self):
        from agent.tools import web_search

        with patch("ddgs.DDGS") as mock_ddgs:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.text.side_effect = Exception("réseau indisponible")
            mock_ddgs.return_value = instance

            result = web_search.invoke({"query": "test"})

        assert "Erreur" in result


# ── search_corpus (intégration — requiert vectorstore) ─────────────────────

@pytest.mark.integration
class TestSearchCorpus:
    def test_returns_results_for_known_topic(self):
        """search_corpus trouve des résultats pour 'RAG' (dans le corpus)."""
        from agent.tools import search_corpus
        result = search_corpus.invoke({"query": "RAG Retrieval-Augmented Generation"})
        assert "[1]" in result
        assert len(result) > 50

    def test_returns_results_for_mmr(self):
        from agent.tools import search_corpus
        result = search_corpus.invoke({"query": "MMR Maximal Marginal Relevance"})
        assert "[1]" in result
        assert "MMR" in result or "Marginal" in result

    def test_returns_string_for_off_topic(self):
        """search_corpus retourne quand même une string pour une question hors-corpus."""
        from agent.tools import search_corpus
        result = search_corpus.invoke({"query": "cours de bourse Apple AAPL"})
        assert isinstance(result, str)
