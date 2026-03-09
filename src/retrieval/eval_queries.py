# eval_queries.py — jeu de 5 requêtes pour la comparaison cosinus vs MMR
#
# Critères de sélection :
#   - couvrir des types de requêtes différents (factuelle, comparaison, synthèse, précise, limite)
#   - rester réalistes par rapport au corpus actuel (RAG, LangChain, ChromaDB, embeddings)
#   - mettre en évidence les différences de comportement entre cosinus et MMR

EVAL_QUERIES = [
    {
        "id": 1,
        "category": "Définition / question factuelle",
        "query": "What is Retrieval-Augmented Generation and how does it work?",
        # Question centrale du domaine, couverte par plusieurs chunks de sources différentes
        # (intro_rag.txt + Wikipedia). Cosinus devrait rapporter des chunks très proches,
        # MMR devrait diversifier entre les deux sources.
        "rationale": (
            "Question générale couverte par tout le corpus. "
            "Permet de voir si MMR évite les chunks quasi-identiques de la même source."
        ),
    },
    {
        "id": 2,
        "category": "Comparaison entre deux concepts",
        "query": "What is the difference between cosine similarity search and MMR retrieval?",
        # Sujet traité dans langchain_notes.md et potentiellement Wikipedia.
        # Cosinus risque de ramener plusieurs chunks du même fichier ; MMR devrait diversifier.
        "rationale": (
            "Comparaison entre deux stratégies. "
            "Teste si cosinus crée de la redondance quand un seul fichier domine le sujet."
        ),
    },
    {
        "id": 3,
        "category": "Synthèse / question large",
        "query": "How do embeddings and vector databases enable semantic search?",
        # Sujet transversal — embeddings (intro_rag.txt), Chroma (indexer), usage LangChain.
        # Requête large : cosinus peut retourner des chunks hétérogènes ; MMR devrait couvrir
        # plus de sources distinctes grâce à la pénalisation de la redondance.
        "rationale": (
            "Requête large qui touche à plusieurs composants du corpus. "
            "MMR devrait montrer une meilleure couverture des types de documents."
        ),
    },
    {
        "id": 4,
        "category": "Question précise / mot-clé rare",
        "query": "What is chunk_id and how is it computed in the ingestion pipeline?",
        # Terme très spécifique au code du projet (chunking.py / intro_rag.txt).
        # Peu de chunks correspondent → les deux stratégies risquent de retourner
        # des résultats similaires ou de piocher hors-sujet.
        "rationale": (
            "Requête précise sur un concept interne au corpus. "
            "Cas intéressant où fetch_k de MMR peut aider ou non selon la densité du corpus."
        ),
    },
    {
        "id": 5,
        "category": "Cas limite / requête ambiguë",
        "query": "What are the limitations of local language models for enterprise use?",
        # Sujet à peine effleuré dans le corpus (mention d'Ollama, de latence locale).
        # Résultats probablement peu pertinents pour les deux stratégies.
        # Utile pour l'analyse : que retourne-t-on quand le corpus ne répond pas vraiment ?
        "rationale": (
            "Sujet hors-corpus principal. "
            "Permet d'observer le comportement des retrievers quand la requête est mal couverte."
        ),
    },
]
