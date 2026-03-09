# chain.py — chaîne RAG LCEL (Tâche 2)
# pipeline : retriever | format_docs | prompt | llm | parser

from langchain_core.runnables import Runnable


def get_llm():
    # TODO: initialiser Ollama ici
    raise NotImplementedError("LLM pas encore branché (Tâche 2)")

    # from langchain_ollama import OllamaLLM
    # from config import OLLAMA_MODEL, OLLAMA_BASE_URL
    # return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def build_rag_chain(retriever, llm) -> Runnable:
    # TODO: assembler la chaîne LCEL ici
    raise NotImplementedError("Chaîne RAG pas encore implémentée (Tâche 2)")

    # from langchain_core.output_parsers import StrOutputParser
    # from langchain_core.runnables import RunnablePassthrough
    # from rag.prompt import build_rag_prompt
    #
    # def format_docs(docs):
    #     return "\n\n".join(d.page_content for d in docs)
    #
    # prompt = build_rag_prompt()
    # return (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
