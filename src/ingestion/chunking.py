# chunking.py — découpage des documents en chunks avec métadonnées

import hashlib
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Découpe une liste de Documents en chunks, ajoute chunk_index et chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # attribuer un index + un id déterministe par source
    index_by_source: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = index_by_source[source]
        index_by_source[source] += 1

        # hash tronqué → id stable entre deux ingestions des mêmes fichiers
        raw = f"{source}:{idx}".encode("utf-8")
        chunk_id = hashlib.sha256(raw).hexdigest()[:16]

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = chunk_id

    print(
        f"  [split_documents] {len(documents)} doc(s) → {len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
