# prompt.py — template de prompt pour la chaîne RAG

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """Tu es un assistant de recherche expert. Réponds uniquement
en te basant sur les extraits de documents fournis. Si la réponse n'est pas
dans les documents, dis-le clairement. Sois précis et concis.

Contexte :
{context}"""


def build_rag_prompt() -> ChatPromptTemplate:
    """Construit le ChatPromptTemplate utilisé dans la chaîne RAG."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
