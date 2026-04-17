# tests/test_graph_logic.py — tests unitaires de la logique du graphe LangGraph
#
# Ces tests ne requièrent PAS Ollama ni ChromaDB (logique pure).
#
# Usage :
#   pytest tests/test_graph_logic.py -v

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.nodes import decide_after_grading, decide_after_routing


# ── decide_after_grading ────────────────────────────────────────────────────

class TestDecideAfterGrading:
    def test_sufficient_goes_to_generate(self):
        state = {"grade_decision": "sufficient", "retry_count": 0}
        assert decide_after_grading(state) == "generate"

    def test_sufficient_even_with_retries(self):
        state = {"grade_decision": "sufficient", "retry_count": 2}
        assert decide_after_grading(state) == "generate"

    def test_insufficient_first_try_goes_to_transform(self):
        state = {"grade_decision": "insufficient", "retry_count": 0}
        assert decide_after_grading(state) == "transform_query"

    def test_insufficient_second_try_goes_to_transform(self):
        state = {"grade_decision": "insufficient", "retry_count": 2}
        assert decide_after_grading(state) == "transform_query"

    def test_guardfou_at_3_retries(self):
        """retry_count >= 3 → garde-fou → generate forcé."""
        state = {"grade_decision": "insufficient", "retry_count": 3}
        assert decide_after_grading(state) == "generate"

    def test_guardfou_above_3_retries(self):
        state = {"grade_decision": "insufficient", "retry_count": 5}
        assert decide_after_grading(state) == "generate"

    def test_missing_grade_decision_defaults_to_insufficient(self):
        state = {"retry_count": 0}
        assert decide_after_grading(state) == "transform_query"

    def test_missing_retry_count_defaults_to_zero(self):
        state = {"grade_decision": "insufficient"}
        assert decide_after_grading(state) == "transform_query"


# ── decide_after_routing ────────────────────────────────────────────────────

class TestDecideAfterRouting:
    def test_web_goes_to_web_search_node(self):
        state = {"tool_used": "web"}
        assert decide_after_routing(state) == "web_search_node"

    def test_corpus_goes_to_retrieve(self):
        state = {"tool_used": "corpus"}
        assert decide_after_routing(state) == "retrieve"

    def test_unknown_tool_defaults_to_retrieve(self):
        state = {"tool_used": "unknown"}
        assert decide_after_routing(state) == "retrieve"

    def test_missing_tool_used_defaults_to_retrieve(self):
        state = {}
        assert decide_after_routing(state) == "retrieve"
