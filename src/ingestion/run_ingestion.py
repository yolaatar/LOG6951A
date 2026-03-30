# run_ingestion.py — pipeline complet : chargement → chunking → indexation
# usage : python src/ingestion/run_ingestion.py [--reset]

import sys
import argparse
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loaders import load_text, load_markdown, load_web, load_pdf
from ingestion.chunking import split_documents
from ingestion.indexer import index_documents, reset_vectorstore
from config import RAW_DIR


# sources à ingérer — modifier ici pour ajouter des fichiers locaux
LOCAL_SOURCES = [
    RAW_DIR / "intro_rag.txt",
    RAW_DIR / "langchain_notes.md",
]

WEB_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"


def load_local_sources() -> List:
    """Charge les fichiers de LOCAL_SOURCES, saute ceux qui sont absents."""
    from langchain_core.documents import Document
    all_docs: List[Document] = []

    for path in LOCAL_SOURCES:
        if not path.exists():
            print(f"  [SKIP] Fichier absent : {path.name}")
            continue

        ext = path.suffix.lower()
        if ext == ".txt":
            docs = load_text(path)
        elif ext in {".md", ".markdown"}:
            docs = load_markdown(path)
        elif ext == ".pdf":
            docs = load_pdf(path)
        else:
            print(f"  [SKIP] Format non reconnu : {path.name}")
            continue

        all_docs.extend(docs)

    return all_docs


def load_pdfs_from_raw() -> List:
    """Charge automatiquement tous les PDF présents dans data/raw/."""
    from langchain_core.documents import Document
    pdfs: List[Document] = []

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("  [INFO] Aucun PDF détecté dans data/raw/.")
        return []

    for pdf_path in pdf_files:
        pdfs.extend(load_pdf(pdf_path))

    return pdfs


def load_web_source() -> List:
    """Essaie de charger WEB_URL, retourne [] en cas d'erreur réseau."""
    print(f"  Tentative de chargement : {WEB_URL}")
    try:
        return load_web(WEB_URL)
    except Exception as exc:
        print(f"  [WARN] Source web ignorée : {exc}")
        return []


def print_summary(all_docs: List, chunks: List) -> None:
    from collections import Counter

    print("\n" + "=" * 60)
    print("        RÉSUMÉ DE L'INGESTION")
    print("=" * 60)

    type_counter = Counter(d.metadata.get("type_document", "?") for d in all_docs)
    print(f"\n  Documents chargés  : {len(all_docs)}")
    print(f"  Types de sources   : {dict(type_counter)}")
    print(f"  Chunks produits    : {len(chunks)}")

    print("\n  Détail par source :")
    source_counter = Counter(
        Path(c.metadata.get("source", "?")).name
        if not c.metadata.get("source", "?").startswith("http")
        else c.metadata.get("source", "?")
        for c in chunks
    )
    for source, count in sorted(source_counter.items()):
        print(f"    • {count:3d} chunks ← {source}")

    if chunks:
        first = chunks[0]
        print("\n  Exemple de métadonnées (chunk 0) :")
        for key, val in first.metadata.items():
            print(f"    {key:<20} : {val}")

    print("\n  Aperçu de chunks :")
    for i, chunk in enumerate(chunks[:2]):
        preview = chunk.page_content[:120].replace("\n", " ")
        source_name = (
            Path(chunk.metadata.get("source", "?")).name
            if not chunk.metadata.get("source", "?").startswith("http")
            else "web"
        )
        print(f"\n  [Chunk {i} — {source_name}]")
        print(f"  \"{preview}...\"")

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchPal — Pipeline d'ingestion")
    parser.add_argument("--reset", action="store_true", help="Réinitialise la base Chroma avant l'ingestion.")
    args = parser.parse_args()

    print("\nResearchPal — Ingestion\n")

    if args.reset:
        print("⚠️  Réinitialisation de la base ChromaDB...")
        reset_vectorstore()
        print()

    print("Étape 1 — Chargement des sources\n")
    all_docs = []

    local_docs = load_local_sources()
    all_docs.extend(local_docs)

    # dédupliquer les PDF si déjà listés dans LOCAL_SOURCES
    pdf_docs = load_pdfs_from_raw()
    known_sources = {str(d.metadata.get("source")) for d in local_docs}
    for doc in pdf_docs:
        if str(doc.metadata.get("source")) not in known_sources:
            all_docs.append(doc)

    web_docs = load_web_source()
    all_docs.extend(web_docs)

    if not all_docs:
        print("\n[ERREUR] Aucun document chargé. Vérifiez data/raw/.")
        sys.exit(1)

    types_found = {d.metadata.get("type_document") for d in all_docs}
    print(f"\n  → {len(all_docs)} document(s) chargé(s), types : {types_found}\n")

    print("Étape 2 — Chunking\n")
    chunks = split_documents(all_docs)
    print()

    print("Étape 3 — Indexation ChromaDB\n")
    index_documents(chunks)
    print()

    print_summary(all_docs, chunks)

    print("\n[SUCCESS] Ingestion terminée.")


if __name__ == "__main__":
    main()
