# cosine_retriever.py — retriever par similarité cosinus

import warnings
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import RETRIEVAL_TOP_K


def get_cosine_retriever(vectorstore: Chroma, k: int = RETRIEVAL_TOP_K) -> VectorStoreRetriever:
    """
    Retourne un retriever LangChain basé sur la similarité cosinus.

    Paramètres :
        vectorstore : collection Chroma déjà chargée
        k           : nombre de chunks à retourner (top-k)

    Score : distance cosinus entre le vecteur requête et chaque chunk.
    Plus le score est proche de 1, plus le chunk est pertinent.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def cosine_search_with_scores(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Variante avec scores : retourne [(Document, score), ...] triés par pertinence.

    Utilisé dans le script d'évaluation pour afficher les scores de similarité.
    Les scores ChromaDB peuvent dépasser [0,1] ; le warning LangChain est supprimé.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)
