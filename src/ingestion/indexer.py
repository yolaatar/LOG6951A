# indexer.py — gestion de la base vectorielle ChromaDB

import shutil
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

# singleton — on charge le modèle une seule fois par session
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embeddings (singleton)."""
    global _embeddings
    if _embeddings is None:
        print(f"  [embeddings] Chargement : {EMBEDDING_MODEL_NAME}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # requis pour cosinus
        )
        print("  [embeddings] Modèle prêt.")
    return _embeddings


def load_vectorstore() -> Chroma:
    """Ouvre la collection ChromaDB existante (lecture seule)."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            f"Base ChromaDB introuvable dans : {CHROMA_DIR}\n"
            f"  → Lancez d'abord : python src/ingestion/run_ingestion.py"
        )

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(CHROMA_DIR),
    )


def index_documents(chunks: List[Document]) -> Chroma:
    """Indexe une liste de chunks dans ChromaDB (crée ou complète la collection)."""
    if not chunks:
        raise ValueError("La liste de chunks est vide. Rien à indexer.")

    embeddings = get_embedding_function()

    ids = [
        chunk.metadata.get("chunk_id", str(i))
        for i, chunk in enumerate(chunks)
    ]

    chroma_exists = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

    if not chroma_exists:
        print(f"  [indexer] Création de la collection '{CHROMA_COLLECTION_NAME}'...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
        )
    else:
        print(f"  [indexer] Ajout de {len(chunks)} chunk(s) à la collection existante...")
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        vectorstore.add_documents(documents=chunks, ids=ids)

    count = vectorstore._collection.count()
    print(f"  [indexer] Collection '{CHROMA_COLLECTION_NAME}' : {count} chunk(s) total.")
    print(f"  [indexer] Persisté dans : {CHROMA_DIR}")
    return vectorstore


def reset_vectorstore() -> None:
    """Supprime et recrée la collection ChromaDB (destructif)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"  [reset] Collection supprimée : {CHROMA_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [reset] Dossier recréé : {CHROMA_DIR}")


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
) -> Chroma:
    """Alias : indexe si documents fourni, sinon charge l'existant."""
    if documents:
        return index_documents(documents)
    return load_vectorstore()
