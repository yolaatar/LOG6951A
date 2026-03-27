# config.py — toutes les constantes du projet sont ici
# modifier ce fichier pour changer les paramètres sans toucher au reste

from pathlib import Path

# chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

# embeddings locaux (pas de compte nécessaire)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM via Ollama — faire `ollama pull mistral:7b-instruct` avant
OLLAMA_MODEL = "mistral:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

# chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# chroma
CHROMA_COLLECTION_NAME = "researchpal_docs"

# retrieval
RETRIEVAL_TOP_K = 4
MMR_FETCH_K = 20       # nb de candidats avant ré-ordonnancement MMR
MMR_LAMBDA = 0.5       # 0 = diversité max, 1 = pertinence max
MULTIQUERY_VARIANTS = 3

# debug / garde-fou hors sujet
DEBUG_TRACE = True
OUT_OF_SCOPE_SCORE_THRESHOLD = 0.1
DOMAIN_KEYWORDS = [
    "rag",
    "retrieval",
    "augment",
    "langchain",
    "vector",
    "chroma",
    "embedding",
    "mmr",
    "cosine",
    "prompt",
    "llm",
    "chunk",
    "document",
    "source",
    "recherche",
]


def print_config() -> None:
    print("=" * 50)
    print("       ResearchPal — Configuration")
    print("=" * 50)
    print(f"  Projet root   : {PROJECT_ROOT}")
    print(f"  Data dir      : {DATA_DIR}")
    print(f"  Chroma dir    : {CHROMA_DIR}")
    print(f"  Embedding     : {EMBEDDING_MODEL_NAME}")
    print(f"  LLM (Ollama)  : {OLLAMA_MODEL}")
    print(f"  Chunk size    : {CHUNK_SIZE}  |  overlap : {CHUNK_OVERLAP}")
    print(f"  Collection    : {CHROMA_COLLECTION_NAME}")
    print(f"  Top-K         : {RETRIEVAL_TOP_K}")
    print("=" * 50)
