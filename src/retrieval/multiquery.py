# multiquery.py — MultiQueryRetriever (Tâche 2)
# génère plusieurs variantes de la requête pour améliorer le recall

from langchain_core.vectorstores import VectorStoreRetriever

from config import MULTIQUERY_VARIANTS


def get_multiquery_retriever(vectorstore, llm) -> VectorStoreRetriever:
    # TODO: implémenter quand le LLM sera branché
    raise NotImplementedError("MultiQueryRetriever pas encore implémenté (Tâche 2)")

    # exemple d'implémentation à venir :
    # from langchain.retrievers import MultiQueryRetriever
    # return MultiQueryRetriever.from_llm(
    #     retriever=vectorstore.as_retriever(),
    #     llm=llm,
    #     parser_key="lines",
    # )
