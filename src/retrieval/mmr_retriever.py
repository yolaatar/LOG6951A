# mmr_retriever.py — retriever MMR (Maximal Marginal Relevance)
# MMR équilibre pertinence et diversité des résultats

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA


def get_mmr_retriever(
    vectorstore: Chroma,
    k: int = RETRIEVAL_TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> VectorStoreRetriever:
    """Retourne un retriever MMR (diversité contrôlée par lambda_mult)."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )
