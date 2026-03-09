"""
ingestion/__init__.py

Expose les fonctions principales du module d'ingestion.
"""

from .loaders import load_pdf, load_web, load_text, load_markdown, load_document
from .chunking import split_documents
from .indexer import (
    get_embedding_function,
    load_vectorstore,
    index_documents,
    reset_vectorstore,
    get_or_create_vectorstore,  # alias de compat
)

__all__ = [
    "load_pdf",
    "load_web",
    "load_text",
    "load_markdown",
    "load_document",
    "split_documents",
    "get_embedding_function",
    "load_vectorstore",
    "index_documents",
    "reset_vectorstore",
    "get_or_create_vectorstore",
]
