"""
retrieval/__init__.py

Expose les retrievers disponibles dans ResearchPal.
"""

from .cosine_retriever import get_cosine_retriever
from .mmr_retriever import get_mmr_retriever

__all__ = [
    "get_cosine_retriever",
    "get_mmr_retriever",
]
