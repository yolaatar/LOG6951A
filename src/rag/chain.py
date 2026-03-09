# chain.py — pipeline RAG end-to-end (T3)
# pattern : Persona + Structured Output + mémoire 3 tours
# usage   : from rag.chain import build_rag_pipeline, answer_question

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
)
from ingestion.indexer import load_vectorstore
from retrieval.cosine_retriever import cosine_search_with_scores
from retrieval.mmr_retriever import mmr_search
from rag.prompt import build_rag_prompt, format_context, format_citations
from rag.memory import ConversationMemory


# ── résultat d'une requête RAG ───────────────────────────────────────────────

@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[str]
    retrieved_documents: List[Document]
    query_variants: List[str] = field(default_factory=list)
    strategy: str = "cosine"


# ── LLM ─────────────────────────────────────────────────────────────────────

def get_llm(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
    """Initialise ChatOllama. Lève RuntimeError si Ollama n'est pas accessible."""
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url)
        llm.invoke("ping")
        return llm
    except ImportError:
        raise RuntimeError("langchain-ollama manquant : pip install langchain-ollama")
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de joindre Ollama sur {base_url}.\n"
            f"  → Lancez : ollama serve\n"
            f"  → Vérifiez le modèle  : ollama list\n"
            f"  Erreur : {exc}"
        )


# ── pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Pipeline RAG complet.  Stratégies : cosine · mmr · multiquery+cosine · multiquery+mmr
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        k: int = RETRIEVAL_TOP_K,
        fetch_k: int = MMR_FETCH_K,
        lambda_mult: float = MMR_LAMBDA,
    ):
        print("  [pipeline] Connexion à Ollama...")
        self.llm = get_llm(model, base_url)
        print(f"  [pipeline] LLM prêt : {model}")

        print("  [pipeline] Chargement du vectorstore...")
        self.vectorstore = load_vectorstore()
        print(f"  [pipeline] Vectorstore : {self.vectorstore._collection.count()} chunks")

        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.memory = ConversationMemory()

    def _retrieve(self, question: str, strategy: str, use_multiquery: bool):
        if use_multiquery:
            from retrieval.multiquery import multiquery_retrieve
            return multiquery_retrieve(
                self.vectorstore, question, self.llm,
                k=self.k, fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult, strategy=strategy,
            )
        if strategy == "mmr":
            docs = mmr_search(
                self.vectorstore, question,
                k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult,
            )
        else:
            docs = [d for d, _ in cosine_search_with_scores(self.vectorstore, question, k=self.k)]
        return docs, []

    def answer(
        self,
        question: str,
        strategy: str = "cosine",
        use_multiquery: bool = False,
    ) -> "RAGResult":
        """Répond à une question avec contexte + historique de 3 tours."""
        docs, variants = self._retrieve(question, strategy, use_multiquery)

        context = format_context(docs)
        history = self.memory.format_history_for_prompt()
        prompt = build_rag_prompt(history=history)
        chain = prompt | self.llm

        response = chain.invoke({"context": context, "question": question})
        answer_text = response.content if hasattr(response, "content") else str(response)

        sources_seen: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "?")
            label = src if src.startswith("http") else Path(src).name
            if label not in sources_seen:
                sources_seen.append(label)

        self.memory.add_turn(question, answer_text, sources=sources_seen)

        strat_label = ("multiquery+" if use_multiquery else "") + strategy
        return RAGResult(
            question=question,
            answer=answer_text,
            sources=sources_seen,
            retrieved_documents=docs,
            query_variants=variants,
            strategy=strat_label,
        )

    def reset_memory(self) -> None:
        self.memory.clear_history()


# ── API publique ─────────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def build_rag_pipeline(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    k: int = RETRIEVAL_TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> RAGPipeline:
    """Crée (et met en cache) le pipeline. À appeler une seule fois au démarrage."""
    global _pipeline
    _pipeline = RAGPipeline(model=model, base_url=base_url, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    return _pipeline


def answer_question(
    question: str,
    strategy: str = "cosine",
    use_multiquery: bool = False,
    pipeline: Optional[RAGPipeline] = None,
) -> RAGResult:
    """Répond via le pipeline RAG (utilise le singleton si pipeline=None)."""
    p = pipeline or _pipeline
    if p is None:
        raise RuntimeError("Pipeline non initialisé. Appelez build_rag_pipeline() d'abord.")
    return p.answer(question, strategy=strategy, use_multiquery=use_multiquery)


# stub de compatibilité
def build_rag_chain(retriever=None, llm=None):
    raise RuntimeError("Utilisez build_rag_pipeline() à la place.")
