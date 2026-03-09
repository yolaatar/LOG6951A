# cosine_retriever.py — retriever par similarité cosinus

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import RETRIEVAL_TOP_K


def get_cosine_retriever(vectorstore: Chroma, k: int = RETRIEVAL_TOP_K) -> VectorStoreRetriever:
    """Retourne un retriever cosinus simple (similarity search)."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
