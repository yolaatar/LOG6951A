# agent/tools.py — outils personnalisés @tool (T2)
#
# Deux outils exposés à l'agent :
#   1. search_corpus  — wrapping du retriever ChromaDB du TP1 (OBLIGATOIRE)
#   2. web_search     — recherche DuckDuckGo sans clé API
#
# Chaque description d'outil précise :
#   - quand utiliser l'outil
#   - quand NE PAS l'utiliser
#   - le format de retour attendu

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool


# ── Outil 1 : search_corpus ──────────────────────────────────────────────────

@tool
def search_corpus(query: str) -> str:
    """Recherche dans le corpus documentaire indexé (ChromaDB) les passages les plus
    pertinents pour répondre à une question.

    Utiliser quand :
    - La question porte sur des concepts du domaine RAG / LangChain / embeddings /
      retrieval / LLM / prompt engineering présents dans les documents indexés.
    - La question concerne des documents uploadés par l'utilisateur dans ResearchPal.
    - La question est technique et peut être répondue à partir de la base locale.

    Ne pas utiliser quand :
    - La question porte sur des événements récents, actualités, données temps réel
      (cours de bourse, météo, résultats sportifs, etc.).
    - La question est clairement hors du domaine couvert par le corpus.
    - Un appel précédent à search_corpus a retourné 0 résultats pertinents sur ce sujet.

    Format de retour :
    Chaîne de texte avec les passages numérotés [1], [2], ... chacun précédé de sa
    source (nom de fichier ou URL). Retourne "Aucun résultat pertinent." si le corpus
    ne contient rien de pertinent.
    """
    from ingestion.indexer import load_vectorstore
    from retrieval.cosine_retriever import cosine_search_with_scores
    from config import RETRIEVAL_TOP_K

    try:
        vectorstore = load_vectorstore()
        results = cosine_search_with_scores(vectorstore, query, k=RETRIEVAL_TOP_K)
    except Exception as exc:
        return f"Erreur lors de la recherche dans le corpus : {exc}"

    if not results:
        return "Aucun résultat pertinent dans le corpus."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "source inconnue")
        label = src if src.startswith("http") else Path(src).name
        content = doc.page_content.strip()
        parts.append(f"[{i}] {label} (score={score:.3f})\n{content}")

    return "\n\n".join(parts)


# ── Outil 2 : web_search ────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Effectue une recherche web via DuckDuckGo pour obtenir des informations
    récentes ou absentes du corpus local.

    Utiliser quand :
    - La question porte sur des événements récents, actualités ou données temps réel.
    - La question est clairement hors du domaine documentaire local (ex. : prix,
      météo, résultats sportifs, nouvelles versions de logiciels publiées récemment).
    - search_corpus a retourné des résultats insuffisants ou non pertinents.
    - L'utilisateur demande explicitement une recherche web ou une information externe.

    Ne pas utiliser quand :
    - La question peut être répondue directement à partir du corpus indexé.
    - La question est purement théorique ou conceptuelle (préférer search_corpus).
    - La question porte sur du code ou des configurations présentes dans les documents.

    Format de retour :
    Les 3 premiers résultats web avec titre, URL et extrait textuel. Retourne un
    message d'erreur explicite si la recherche échoue ou renvoie 0 résultats.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Erreur : package duckduckgo-search non installé. Lancez : pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"Erreur lors de la recherche web : {exc}"

    if not raw:
        return "Aucun résultat web trouvé pour cette requête."

    parts = []
    for i, r in enumerate(raw, 1):
        title = r.get("title", "Sans titre")
        url = r.get("href", "")
        body = r.get("body", "")[:300]
        parts.append(f"[{i}] {title}\n    URL : {url}\n    Extrait : {body}")

    return "\n\n".join(parts)


# ── Export ───────────────────────────────────────────────────────────────────

TOOLS = [search_corpus, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
