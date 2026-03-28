# chain.py — pipeline RAG end-to-end (T3)
# pattern : Persona + Structured Output + Cognitive Verifier + mémoire 3 tours
#
# Architecture de filtrage history-aware (ordre des étapes dans answer()) :
#   1. Enrichissement de la requête  — _build_retrieval_query()
#      Si la question courante est un suivi (pronoms, courte), on lui préfixe
#      la question du dernier tour in-scope pour ancrer le retrieval dans le
#      sujet de conversation.
#   2. Scoring cosinus sur la requête enrichie (pas sur la question brute)
#   3. Décision out-of-scope tenant compte de la continuité conversationnelle :
#      refus UNIQUEMENT si score bas ET pas de domaine ET pas de suivi récent réussi
#   4. Retrieval sur la requête enrichie
#   5. Génération LLM avec la question originale (pas la requête enrichie)
#   6. Stockage en mémoire avec had_retrieval=True/False
#
# usage   : from rag.chain import build_rag_pipeline, answer_question

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    DEBUG_TRACE,
    OUT_OF_SCOPE_SCORE_THRESHOLD,
    DOMAIN_KEYWORDS,
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


# ── Détection de coréférence / question de suivi ──────────────────────────────
#
# Ensemble de tokens français indiquant qu'une question dépend du contexte
# conversationnel précédent : pronoms, déterminants anaphoriques, marqueurs
# de continuité, questions courtes sans ancrage lexical explicite.

_COREF_TOKENS = frozenset({
    # pronoms personnels sujets (quasi-toujours pronominaux, pas articles)
    "il", "elle", "ils", "elles", "lui", "leur", "leurs",
    # déterminants démonstratifs
    "ce", "cet", "cette", "ces",
    "celui", "celle", "ceux", "celles", "celui-ci", "celle-ci",
    # proformes
    "cela", "ça", "ç'",
    # marqueurs de renvoi explicite
    "précédent", "précédente", "précédemment",
    "mentionné", "mentionnée", "décrit", "décrite",
    "expliqué", "expliquée", "indiqué",
    "l'approche", "cette approche", "cet outil", "ce système",
    "cette méthode", "cette technique", "cette architecture",
    "l'algorithme", "cet algorithme",
    # formules de suivi conversationnel
    "tu viens", "tu as mentionné", "tu as décrit", "tu as expliqué",
    "comme tu", "comme je",
    "dans ce cas", "dans cette optique",
    "par rapport à ça", "par rapport à cela",
    "à ce sujet", "sur ce point",
    "développe", "développer", "approfondir",
    "résume", "résumer", "synthétise",
    "compare", "comparer", "vis-à-vis",
    "dans ta réponse", "dans ta première", "dans ton explication",
    "premier point", "deuxième point", "troisième point",
    "première réponse", "tour précédent",
})

# Possessifs traités séparément : anaphoriques seulement s'ils sont dans
# les 5 premiers mots (évite le faux positif "...et sa population" en fin de phrase)
_POSSESSIVE_TOKENS = frozenset({"son", "sa", "ses"})

# Partition de _COREF_TOKENS pour la stratégie de correspondance :
#   - un seul mot  → correspondance exacte sur mot entier (évite "elle" ⊂ "quelle")
#   - plusieurs mots → correspondance sous-chaîne (naturelle pour les locutions)
_COREF_SINGLE = frozenset(t for t in _COREF_TOKENS if " " not in t)
_COREF_MULTI  = frozenset(t for t in _COREF_TOKENS if " " in t)

# Seuil de longueur en mots : questions courtes sont présumées des suivis
_SHORT_QUESTION_THRESHOLD = 9


# ── pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Pipeline RAG complet.  Stratégies : cosine · mmr · multiquery+cosine · multiquery+mmr

    Filtrage hors-périmètre history-aware :
      - La requête de retrieval est enrichie avec le sujet du dernier tour in-scope
        lorsque la question courante est détectée comme un suivi.
      - La décision out-of-scope tient compte de la continuité conversationnelle :
        un suivi après un tour in-scope est autorisé même si son score seul est faible.
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

    # ── helpers de détection ─────────────────────────────────────────────────

    @staticmethod
    def _is_domain_question(question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in DOMAIN_KEYWORDS)

    @staticmethod
    def _is_followup_question(question: str) -> bool:
        """
        Détecte si la question est un suivi conversationnel.

        Trois signaux complémentaires :
          1. Longueur — une question courte (< _SHORT_QUESTION_THRESHOLD mots)
             sans ancrage thématique explicite est probablement un suivi.
          2. Tokens de coréférence — pronoms, déterminants anaphoriques, marqueurs
             de continuité présents dans _COREF_TOKENS, avec correspondance
             exacte sur mot entier pour les tokens mono-mots (évite "elle" ⊂ "quelle")
             et sous-chaîne pour les locutions multi-mots.
          3. Possessifs positionnels — "son/sa/ses" dans les 5 premiers mots
             seulement (évite le faux positif "...et sa population" en fin de phrase).

        "le/la/les/y/en" sont exclus : trop fréquents comme articles/particules
        pour être des marqueurs fiables de coréférence inter-tours.

        Retourne True si l'un des trois signaux est activé.
        """
        words = question.strip().split()
        if len(words) < _SHORT_QUESTION_THRESHOLD:
            return True
        q_lower = question.lower()
        # Mots entiers uniquement pour les tokens mono-mots
        words_set = {w.rstrip("?.,;:!\"'") for w in q_lower.split()}
        if any(token in words_set for token in _COREF_SINGLE):
            return True
        # Sous-chaîne pour les locutions multi-mots (naturel)
        if any(token in q_lower for token in _COREF_MULTI):
            return True
        # Possessifs anaphoriques seulement en position initiale (≤ 5 premiers mots)
        first_words = {w.rstrip("?.,;:!\"'") for w in q_lower.split()[:5]}
        return any(p in first_words for p in _POSSESSIVE_TOKENS)

    def _build_retrieval_query(self, question: str) -> Tuple[str, bool]:
        """
        Construit la requête de retrieval enrichie avec le contexte historique.

        Stratégie d'enrichissement :
          - Si la mémoire est vide  → question brute (pas d'enrichissement).
          - Si la question n'est pas un suivi → question brute.
          - Sinon : préfixer la question du dernier tour in-scope (had_retrieval=True)
            pour ancrer le retrieval dans le sujet de la conversation.

        Le préfixe est tronqué à 100 caractères pour ne pas noyer l'embedding.

        Retourne (requête_enrichie, bool_enrichissement).
        """
        if not self.memory._turns:
            return question, False

        if not self._is_followup_question(question):
            return question, False

        last_turn = self.memory.last_inscope_turn()
        if last_turn is None:
            return question, False

        topic_anchor = last_turn.question[:100].rstrip()
        enriched = f"{topic_anchor} {question}"

        if DEBUG_TRACE:
            print(f"[trace] QUERY_ENRICHMENT: '{question[:60]}' → '{enriched[:80]}'")

        return enriched, True

    # ── retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, retrieval_query: str, strategy: str, use_multiquery: bool):
        if use_multiquery:
            from retrieval.multiquery import multiquery_retrieve
            return multiquery_retrieve(
                self.vectorstore, retrieval_query, self.llm,
                k=self.k, fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult, strategy=strategy,
            )
        if strategy == "mmr":
            docs = mmr_search(
                self.vectorstore, retrieval_query,
                k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult,
            )
        else:
            docs = [d for d, _ in cosine_search_with_scores(
                self.vectorstore, retrieval_query, k=self.k
            )]
        return docs, []

    # ── pipeline principal ───────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        strategy: str = "cosine",
        use_multiquery: bool = False,
    ) -> "RAGResult":
        """
        Répond à une question en 6 étapes history-aware :

          1. Enrichissement de la requête de retrieval avec le contexte historique
          2. Score cosinus sur la requête enrichie (pas la question brute)
          3. Décision out-of-scope : score + domaine + continuité conversationnelle
          4. Retrieval sur la requête enrichie
          5. Génération LLM avec la question originale (pas la requête enrichie)
          6. Stockage en mémoire avec had_retrieval=True/False
        """

        # ── Étape 1 : enrichissement de la requête ───────────────────────────
        retrieval_query, was_enriched = self._build_retrieval_query(question)

        # ── Étape 2 : score de pertinence sur la requête enrichie ─────────────
        best_score = None
        try:
            score_pairs = cosine_search_with_scores(self.vectorstore, retrieval_query, k=1)
            if score_pairs:
                best_score = score_pairs[0][1]
        except Exception:
            best_score = None

        # ── Étape 3 : décision out-of-scope (history-aware) ───────────────────
        #
        # La condition de refus est stricte : tous les critères suivants doivent
        # être réunis simultanément :
        #   a) score bas sur la requête enrichie
        #   b) pas de mot-clé de domaine dans la requête enrichie
        #   c) la question n'est PAS un suivi après un tour récent in-scope
        #
        # Le critère (c) est la clé : un suivi conversationnel après un tour
        # in-scope est autorisé même si son score seul serait insuffisant.
        # Sans (c), UC4 / EC5 et toutes les questions de suivi implicites
        # seraient incorrectement rejetées.

        topic_continuity = was_enriched and self.memory.recent_had_retrieval(window=2)

        out_of_scope = (
            best_score is not None
            and best_score < OUT_OF_SCOPE_SCORE_THRESHOLD
            and not self._is_domain_question(retrieval_query)
            and not topic_continuity
        )

        if DEBUG_TRACE:
            print(f"\n[trace] USER_PROMPT: {question}")
            print(f"[trace] RETRIEVAL_QUERY: {retrieval_query[:100]}")
            if best_score is not None:
                print(f"[trace] BEST_COSINE_SCORE (enriched): {best_score:.4f}")
            print(f"[trace] WAS_ENRICHED={was_enriched}  "
                  f"TOPIC_CONTINUITY={topic_continuity}  "
                  f"OUT_OF_SCOPE={out_of_scope}")

        if out_of_scope:
            answer_text = (
                "**Réponse**\n"
                "Votre question semble hors périmètre du corpus indexé "
                "(RAG/LangChain/docs chargés).\n"
                "Je ne peux pas répondre de manière fiable sans source pertinente.\n\n"
                "**Sources**\n"
                "Aucune source pertinente récupérée.\n\n"
                "**Limites / Incertitudes**\n"
                "Le corpus actuel ne contient pas d'information suffisamment "
                "liée à cette question."
            )
            if DEBUG_TRACE:
                print("[trace] MODEL_INPUT: <skipped - question detected as out-of-scope>")
                print("[trace] MODEL_RESPONSE:")
                print(answer_text)

            # had_retrieval=False : ce tour ne compte pas comme ancrage conversationnel
            self.memory.add_turn(question, answer_text, sources=[], had_retrieval=False)
            return RAGResult(
                question=question,
                answer=answer_text,
                sources=[],
                retrieved_documents=[],
                query_variants=[],
                strategy=("multiquery+" if use_multiquery else "") + strategy,
            )

        # ── Étape 4 : retrieval sur la requête enrichie ───────────────────────
        docs, variants = self._retrieve(retrieval_query, strategy, use_multiquery)

        # ── Étape 5 : génération LLM avec la question originale ───────────────
        # Important : le LLM reçoit la question originale (pronoms inclus),
        # pas la requête enrichie. L'historique injecté dans le prompt fournit
        # le contexte suffisant pour résoudre les anaphores.
        context = format_context(docs)
        history = self.memory.format_history_for_prompt()
        prompt = build_rag_prompt(history=history)
        chain = prompt | self.llm

        if DEBUG_TRACE:
            prompt_value = prompt.format_prompt(context=context, question=question)
            print("[trace] MODEL_INPUT:")
            for msg in prompt_value.to_messages():
                role = getattr(msg, "type", "message")
                print(f"[{role}]\n{msg.content}\n")

        response = chain.invoke({"context": context, "question": question})
        answer_text = response.content if hasattr(response, "content") else str(response)

        if DEBUG_TRACE:
            print("[trace] MODEL_RESPONSE:")
            print(answer_text)

        sources_seen: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "?")
            label = src if src.startswith("http") else Path(src).name
            if label not in sources_seen:
                sources_seen.append(label)

        # ── Étape 6 : stockage en mémoire (had_retrieval=True) ────────────────
        self.memory.add_turn(question, answer_text,
                             sources=sources_seen, had_retrieval=True)

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
