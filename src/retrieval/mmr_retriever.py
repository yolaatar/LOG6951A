# mmr_retriever.py — retriever MMR (Maximal Marginal Relevance)
# MMR : sélectionne des chunks pertinents ET diversifiés en rejetant les near-duplicates

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA


def get_mmr_retriever(
    vectorstore: Chroma,
    k: int = RETRIEVAL_TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> VectorStoreRetriever:
    """
    Retourne un retriever LangChain basé sur MMR.

    Paramètres :
        vectorstore  : collection Chroma déjà chargée
        k            : nombre de chunks finalement retournés
        fetch_k      : nb de candidats initiaux récupérés par cosinus avant
                       re-classement MMR (fetch_k >> k recommandé)
        lambda_mult  : équilibre pertinence / diversité
                       0.0 = diversité maximale (ignore la pertinence)
                       1.0 = pertinence maximale (équivalent cosinus)
                       0.5 = compromis équilibré (valeur recommandée)

    Algorithme MMR :
        Score_MMR(d) = λ · sim(d, q) - (1-λ) · max_{d'∈S} sim(d, d')
        où S est l'ensemble des chunks déjà sélectionnés.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def mmr_search(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> List[Document]:
    """
    Variante directe (sans wrapper Retriever) — retourne une liste de Documents.

    Utilisé dans le script d'évaluation et le balayage de paramètres.
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
