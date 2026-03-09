# loaders.py — fonctions pour charger les différents types de documents

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"  → Vérifiez que le fichier est bien dans data/raw/"
        )
    if not path.is_file():
        raise ValueError(f"Le chemin n'est pas un fichier : {path}")


def load_pdf(file_path: str | Path) -> List[Document]:
    """Charge un PDF page par page."""
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(file_path).resolve()
    _validate_file(path)

    docs = PyPDFLoader(str(path)).load()

    if not docs:
        print(f"  [WARN] Aucun contenu extrait de '{path.name}' (PDF vide ?)")
        return []

    date = _now_iso()
    for doc in docs:
        doc.metadata.update({
            "source": str(path),
            "type_document": "pdf",
            "date_ingestion": date,
        })

    print(f"  [load_pdf] {len(docs)} page(s) ← '{path.name}'")
    return docs


def load_web(url: str) -> List[Document]:
    """Charge le texte principal d'une page web."""
    from langchain_community.document_loaders import WebBaseLoader

    try:
        loader = WebBaseLoader(web_paths=[url], requests_kwargs={"timeout": 10})
        docs = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger '{url}' : {exc}") from exc

    if not docs or not docs[0].page_content.strip():
        raise RuntimeError(f"La page '{url}' est vide ou inaccessible.")

    date = _now_iso()
    for doc in docs:
        # normaliser les espaces multiples issus du HTML
        doc.page_content = " ".join(doc.page_content.split())
        doc.metadata.update({
            "source": url,
            "type_document": "web",
            "date_ingestion": date,
        })

    print(f"  [load_web] {len(docs)} document(s) ← '{url}'")
    return docs


def load_text(file_path: str | Path) -> List[Document]:
    """Charge un fichier .txt comme un seul Document."""
    path = Path(file_path).resolve()
    _validate_file(path)

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        print(f"  [WARN] Fichier vide : '{path.name}'")

    doc = Document(
        page_content=content,
        metadata={
            "source": str(path),
            "type_document": "text",
            "date_ingestion": _now_iso(),
        },
    )
    print(f"  [load_text] 1 document ← '{path.name}' ({len(content)} caractères)")
    return [doc]


def load_markdown(file_path: str | Path) -> List[Document]:
    """Charge un fichier .md comme un seul Document."""
    path = Path(file_path).resolve()
    _validate_file(path)

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        print(f"  [WARN] Fichier vide : '{path.name}'")

    doc = Document(
        page_content=content,
        metadata={
            "source": str(path),
            "type_document": "markdown",
            "date_ingestion": _now_iso(),
        },
    )
    print(f"  [load_markdown] 1 document ← '{path.name}' ({len(content)} caractères)")
    return [doc]


def load_document(source: str) -> List[Document]:
    """Dispatcher : détecte automatiquement le type et appelle le bon loader."""
    if source.startswith(("http://", "https://")):
        return load_web(source)

    path = Path(source)
    ext = path.suffix.lower()

    dispatch = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_markdown,
        ".markdown": load_markdown,
    }

    if ext not in dispatch:
        raise ValueError(
            f"Extension non supportée : '{ext}'. "
            f"Formats acceptés : .pdf, .txt, .md, .markdown, ou URL."
        )

    return dispatch[ext](path)


load_url = load_web
