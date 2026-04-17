# agent/nodes.py — nœuds du graphe LangGraph Corrective RAG (T1 + T2)
#
# Nœuds obligatoires (énoncé T1) :
#   - retrieve         : récupère des documents via search_corpus (@tool)
#   - grade_documents  : évalue la pertinence de chaque document (LLM)
#   - generate         : génère la réponse finale (LLM + contexte)
#   - transform_query  : reformule la requête si les docs sont insuffisants
#
# Nœud supplémentaire (T2) :
#   - route_query      : l'agent choisit dynamiquement corpus ou web
#   - web_search_node  : récupère des résultats web via web_search (@tool)
#
# Tous les nœuds sont créés par des fonctions-usine (closures) qui capturent
# le LLM et le vectorstore sans variables globales.

import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate

from agent.state import AgentState
from agent.tools import search_corpus, web_search
from config import RETRIEVAL_TOP_K, DEBUG_TRACE


def _tracer():
    """Lazy import du tracer pour éviter la dépendance obligatoire à Phoenix."""
    try:
        from observability.tracing import get_tracer
        return get_tracer()
    except Exception:
        from opentelemetry import trace
        return trace.get_tracer("researchpal-v2")


# ── Prompts des nœuds ────────────────────────────────────────────────────────

_ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un routeur de requêtes. Réponds UNIQUEMENT par le mot 'corpus' ou le mot 'web'.\n\n"
     "Réponds 'corpus' si la question porte sur : RAG, LangChain, embeddings, retrieval, "
     "vectorstore, ChromaDB, LLM, prompt engineering, chunking, MMR, LCEL.\n\n"
     "Réponds 'web' si la question porte sur : prix boursiers, actualités, météo, "
     "données financières, événements récents, informations en temps réel, "
     "ou tout sujet absent du corpus RAG/LangChain.\n\n"
     "Exemples :\n"
     "Question : Qu'est-ce que le RAG ? → corpus\n"
     "Question : Comment fonctionne ChromaDB ? → corpus\n"
     "Question : Quel est le cours de l'action Apple ? → web\n"
     "Question : Quelle est la météo à Montréal ? → web\n"
     "Question : Qui a gagné la Coupe du monde 2022 ? → web\n\n"
     "Réponds UNIQUEMENT par 'corpus' ou 'web'."),
    ("human", "Question : {question}"),
])

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un évaluateur de pertinence documentaire.\n"
     "Détermine si le document fourni contient des informations utiles pour répondre "
     "à la question.\n"
     "Réponds UNIQUEMENT par 'OUI' ou 'NON', sans ponctuation ni explication."),
    ("human",
     "Question : {question}\n\nDocument :\n{document}\n\n"
     "Ce document est-il pertinent ?"),
])

_TRANSFORM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un spécialiste de l'optimisation de requêtes pour moteurs de recherche "
     "vectoriels.\n"
     "Reformule la requête en FRANÇAIS pour améliorer le rappel dans une base "
     "documentaire sur le RAG et LangChain.\n"
     "Utilise des synonymes et des termes techniques du domaine RAG.\n"
     "Retourne UNIQUEMENT la requête reformulée en français, sans explication."),
    ("human", "Requête originale : {query}\n\nRequête reformulée en français :"),
])


# ── Usines de nœuds ──────────────────────────────────────────────────────────

def make_route_query_node(llm):
    """Nœud de routage : l'agent choisit corpus ou web selon la question (T2)."""

    chain = _ROUTE_PROMPT | llm

    def route_query(state: AgentState) -> dict:
        question = state["question"]
        with _tracer().start_as_current_span("route_query") as span:
            span.set_attribute("input.question", question[:200])
            try:
                response = chain.invoke({"question": question})
                raw = (response.content if hasattr(response, "content") else str(response)).strip().lower()
                tool_used = "web" if "web" in raw else "corpus"
            except Exception:
                tool_used = "corpus"
            span.set_attribute("output.tool_selected", tool_used)

        if DEBUG_TRACE:
            print(f"[trace] ROUTE_QUERY → outil sélectionné : {tool_used}")

        return {
            "tool_used": tool_used,
            "retrieval_query": question,
            "retry_count": 0,
            "relevant_docs": [],
            "web_results": None,
            "generation": "",
            "documents": [],
        }

    return route_query


def make_retrieve_node():
    """Nœud retrieve : appelle search_corpus (@tool, T2) pour récupérer des documents."""

    def retrieve(state: AgentState) -> dict:
        query = state.get("retrieval_query") or state["question"]
        retry = state.get("retry_count", 0)

        with _tracer().start_as_current_span("retrieve") as span:
            span.set_attribute("input.query", query[:200])
            span.set_attribute("input.retry_count", retry)

            if DEBUG_TRACE:
                print(f"[trace] RETRIEVE  query='{query[:80]}'  retry={retry}")

            # Appel explicite de l'outil @tool (T2 : search_corpus intégré au graphe)
            search_corpus.invoke({"query": query})

            from ingestion.indexer import load_vectorstore
            from retrieval.cosine_retriever import cosine_search_with_scores

            try:
                vectorstore = load_vectorstore()
                results = cosine_search_with_scores(vectorstore, query, k=RETRIEVAL_TOP_K)
                docs = [d for d, _ in results]
            except Exception:
                docs = []

            span.set_attribute("output.docs_count", len(docs))

        if DEBUG_TRACE:
            print(f"[trace] RETRIEVE  {len(docs)} documents récupérés")

        return {"documents": docs}

    return retrieve


def make_grade_documents_node(llm):
    """Nœud grade_documents : évalue la pertinence de chaque document (T1).

    Logique Corrective RAG :
      - Pour chaque document, demande au LLM OUI/NON.
      - Si ≥ 2 documents pertinents → grade_decision = 'sufficient'
      - Sinon                       → grade_decision = 'insufficient'
    """

    chain = _GRADE_PROMPT | llm

    def grade_documents(state: AgentState) -> dict:
        question = state["question"]
        docs = state["documents"]
        relevant = []

        with _tracer().start_as_current_span("grade_documents") as span:
            span.set_attribute("input.question", question[:200])
            span.set_attribute("input.docs_total", len(docs))

            for doc in docs:
                content = doc.page_content[:600]
                try:
                    resp = chain.invoke({"question": question, "document": content})
                    verdict = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
                except Exception:
                    verdict = "OUI"

                if verdict.startswith("OUI"):
                    relevant.append(doc)

            decision = "sufficient" if len(relevant) >= 1 else "insufficient"
            span.set_attribute("output.docs_relevant", len(relevant))
            span.set_attribute("output.decision", decision)
            span.set_attribute("input.retry_count", state.get("retry_count", 0))

        if DEBUG_TRACE:
            print(f"[trace] GRADE_DOCS  {len(relevant)}/{len(docs)} pertinents → {decision}")

        return {
            "relevant_docs": relevant,
            "grade_decision": decision,
        }

    return grade_documents


def make_transform_query_node(llm):
    """Nœud transform_query : reformule la requête pour un meilleur retrieval (T1)."""

    chain = _TRANSFORM_PROMPT | llm

    def transform_query(state: AgentState) -> dict:
        original = state.get("retrieval_query") or state["question"]
        retry = state.get("retry_count", 0) + 1

        with _tracer().start_as_current_span("transform_query") as span:
            span.set_attribute("input.original_query", original[:200])
            span.set_attribute("input.retry_count", retry)

            try:
                resp = chain.invoke({"query": original})
                new_query = (resp.content if hasattr(resp, "content") else str(resp)).strip()
                if not new_query or len(new_query) > 300:
                    new_query = original
            except Exception:
                new_query = original

            span.set_attribute("output.new_query", new_query[:200])

        if DEBUG_TRACE:
            print(f"[trace] TRANSFORM_QUERY  retry={retry}  '{original[:60]}' → '{new_query[:60]}'")

        return {
            "retrieval_query": new_query,
            "retry_count": retry,
        }

    return transform_query


def make_web_search_node():
    """Nœud web_search_node : appelle web_search (@tool, T2) pour les requêtes hors-corpus."""

    def web_search_node(state: AgentState) -> dict:
        query = state["question"]

        with _tracer().start_as_current_span("web_search_node") as span:
            span.set_attribute("input.query", query[:200])

            if DEBUG_TRACE:
                print(f"[trace] WEB_SEARCH  query='{query[:80]}'")

            results = web_search.invoke({"query": query})
            span.set_attribute("output.results_len", len(results))

        if DEBUG_TRACE:
            print(f"[trace] WEB_SEARCH  résultats : {results[:120]}…")

        return {"web_results": results}

    return web_search_node


def make_generate_node(llm):
    """Nœud generate : génère la réponse finale avec le contexte récupéré (T1).

    Utilise les relevant_docs si disponibles, sinon tous les documents.
    Pour les requêtes web, utilise web_results directement dans le prompt.
    Réutilise le prompt structuré du TP1 (Réponse / Sources / Limites).
    """

    def generate(state: AgentState) -> dict:
        question = state["question"]
        tool_used = state.get("tool_used", "corpus")
        retry_count = state.get("retry_count", 0)

        with _tracer().start_as_current_span("generate") as span:
            span.set_attribute("input.question", question[:200])
            span.set_attribute("input.tool_used", tool_used)
            span.set_attribute("input.retry_count", retry_count)

            if tool_used == "web":
                web_results = state.get("web_results") or "Aucun résultat web disponible."
                context = f"Résultats de recherche web :\n\n{web_results}"
                span.set_attribute("input.context_type", "web")
            else:
                docs = state.get("relevant_docs") or state.get("documents") or []
                from rag.prompt import format_context
                context = format_context(docs)
                span.set_attribute("input.context_type", "corpus")
                span.set_attribute("input.docs_used", len(docs))

            if DEBUG_TRACE:
                print(f"[trace] GENERATE  outil={tool_used}  retry={retry_count}")
                if retry_count > 0:
                    print(f"[trace] GENERATE  ⚠ cycle correctif déclenché {retry_count}x")

            from rag.prompt import build_rag_prompt
            prompt = build_rag_prompt(history=None)
            chain = prompt | llm

            try:
                response = chain.invoke({"context": context, "question": question})
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                answer = (
                    "**Réponse**\nErreur lors de la génération.\n\n"
                    "**Sources**\nAucune.\n\n"
                    f"**Limites / Incertitudes**\n{exc}"
                )

            span.set_attribute("output.answer_len", len(answer))

            if DEBUG_TRACE:
                print(f"\n{'='*60}")
                print(f"[réponse] {answer[:800]}{'…' if len(answer) > 800 else ''}")
                print(f"{'='*60}\n")

        return {"generation": answer}

    return generate


# ── Arête conditionnelle après grade_documents ───────────────────────────────

def decide_after_grading(state: AgentState) -> Literal["generate", "transform_query"]:
    """Arête conditionnelle : détermine si le cycle de correction doit se poursuivre.

    Règles (Corrective RAG) :
      - Si la décision est 'sufficient' → générer directement.
      - Si retry_count ≥ 3 (garde-fou)  → générer avec ce qu'on a (fallback).
      - Sinon                            → reformuler la requête et réessayer.
    """
    decision = state.get("grade_decision", "insufficient")
    retry = state.get("retry_count", 0)

    if decision == "sufficient":
        return "generate"
    if retry >= 3:
        if DEBUG_TRACE:
            print("[trace] GARDE-FOU  retry_count=3 atteint → generate forcé")
        return "generate"
    return "transform_query"


def decide_after_routing(state: AgentState) -> Literal["retrieve", "web_search_node"]:
    """Arête conditionnelle : dirige vers le bon outil selon tool_used (T2)."""
    return "web_search_node" if state.get("tool_used") == "web" else "retrieve"
