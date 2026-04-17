# tests/test_memory.py — tests unitaires de la mémoire épisodique (T3)
#
# Usage :
#   pytest tests/test_memory.py -v

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


@pytest.fixture(autouse=True)
def tmp_episodic_file(monkeypatch, tmp_path):
    """Redirige la mémoire épisodique vers un fichier temporaire."""
    import memory_v2.episodic as ep
    fake_path = tmp_path / "episodic_memory.json"
    fake_path.write_text("[]")
    monkeypatch.setattr(ep, "MEMORY_FILE", fake_path)
    return fake_path


class TestLoadEpisodes:
    def test_empty_file_returns_empty_list(self):
        from memory_v2.episodic import load_episodes
        assert load_episodes() == []

    def test_loads_existing_episodes(self, tmp_episodic_file):
        from memory_v2.episodic import load_episodes
        data = [{"question": "Q1", "answer": "A1"}]
        tmp_episodic_file.write_text(json.dumps(data))
        episodes = load_episodes()
        assert len(episodes) == 1
        assert episodes[0]["question"] == "Q1"


class TestMaybeStoreEpisode:
    def test_stores_high_quality_episode(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes
        maybe_store_episode(
            question="Qu'est-ce que le RAG ?",
            answer="Le RAG est une architecture qui combine retrieval et génération. " * 10,
            sources=["intro_rag.txt", "langchain_notes.md"],
            tool_used="corpus",
            retry_count=0,
        )
        episodes = load_episodes()
        assert len(episodes) == 1
        assert episodes[0]["question"] == "Qu'est-ce que le RAG ?"

    def test_does_not_store_short_answer(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes
        maybe_store_episode(
            question="Q?",
            answer="Trop court.",
            sources=["a.txt", "b.txt"],
            tool_used="corpus",
            retry_count=0,
        )
        assert load_episodes() == []

    def test_does_not_store_web_results(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes
        maybe_store_episode(
            question="Cours Apple ?",
            answer="Le cours de l'action Apple est disponible sur Boursorama. " * 10,
            sources=["https://boursorama.com", "https://zonebourse.com"],
            tool_used="web",
            retry_count=0,
        )
        assert load_episodes() == []

    def test_does_not_store_with_retries(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes
        maybe_store_episode(
            question="Q avec retry ?",
            answer="Réponse après plusieurs cycles correctifs. " * 10,
            sources=["a.txt", "b.txt"],
            tool_used="corpus",
            retry_count=2,
        )
        assert load_episodes() == []

    def test_does_not_store_insufficient_sources(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes
        maybe_store_episode(
            question="Q ?",
            answer="Réponse longue mais une seule source. " * 10,
            sources=["seule_source.txt"],
            tool_used="corpus",
            retry_count=0,
        )
        assert load_episodes() == []

    def test_respects_max_examples_limit(self):
        from memory_v2.episodic import maybe_store_episode, load_episodes, MAX_EXAMPLES
        for i in range(MAX_EXAMPLES + 3):
            maybe_store_episode(
                question=f"Question {i} ?",
                answer=f"Réponse longue pour la question {i}. " * 10,
                sources=[f"source_{i}_a.txt", f"source_{i}_b.txt"],
                tool_used="corpus",
                retry_count=0,
            )
        episodes = load_episodes()
        assert len(episodes) <= MAX_EXAMPLES


class TestClearEpisodes:
    def test_clear_empties_file(self):
        from memory_v2.episodic import maybe_store_episode, clear_episodes, load_episodes
        maybe_store_episode(
            question="Q ?",
            answer="Longue réponse à conserver en mémoire épisodique. " * 10,
            sources=["a.txt", "b.txt"],
            tool_used="corpus",
            retry_count=0,
        )
        assert len(load_episodes()) == 1
        clear_episodes()
        assert load_episodes() == []


class TestFormatFewShotBlock:
    def test_empty_returns_empty_string(self):
        from memory_v2.episodic import format_few_shot_block
        assert format_few_shot_block() == ""

    def test_formats_episodes_as_block(self):
        from memory_v2.episodic import maybe_store_episode, format_few_shot_block
        maybe_store_episode(
            question="Qu'est-ce que le MMR ?",
            answer="Le MMR combine pertinence et diversité dans la recherche vectorielle. " * 6,
            sources=["intro_rag.txt", "langchain_notes.md"],
            tool_used="corpus",
            retry_count=0,
        )
        block = format_few_shot_block()
        assert "MMR" in block
        assert len(block) > 0
