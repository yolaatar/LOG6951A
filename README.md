# ResearchPal 📚

Assistant de recherche documentaire basé sur une architecture RAG
(**Retrieval-Augmented Generation**) — TP LOG6951A.

**Stack** : Python 3.11 · LangChain · ChromaDB · Sentence Transformers · Ollama · Streamlit

---

## Table des matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Corpus de test — data/raw/](#corpus-de-test--dataraw)
4. [Tâche 1 — Ingestion et indexation](#tâche-1--ingestion-et-indexation)
5. [Tâche 1 — Test du retrieval](#tâche-1--test-du-retrieval)
6. [Lancement de l'interface Streamlit](#lancement-de-linterface-streamlit)
7. [Structure du projet](#structure-du-projet)
8. [État d'avancement](#état-davancement)
9. [Dépendances — notes importantes](#dépendances--notes-importantes)

---

## Prérequis

| Outil | Version minimale | Installation |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| pip | récent | `pip install --upgrade pip` |
| Ollama | dernière | [ollama.com](https://ollama.com/download) |
| Git | — | *(optionnel)* |

---

## Installation

### 1. Accéder au projet

```bash
cd ResearchPal
```

### 2. Créer et activer l'environnement virtuel

```bash
# Créer le venv (Python 3.11 recommandé)
python3.11 -m venv venv

# Activer sur macOS / Linux
source venv/bin/activate

# Activer sur Windows
venv\Scripts\activate
```

> Vérifiez que le bon Python est actif : `python --version`

### 3. Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ L'installation prend 2-5 minutes. `sentence-transformers` et `torch`
> représentent la majorité du volume (~1 GB au total).

### 4. Installer Ollama (pour le pipeline RAG — Tâche 2)

Téléchargement : <https://ollama.com/download>

```bash
# Dans un terminal dédié (à garder ouvert)
ollama serve

# Télécharger le modèle (une seule fois, ~4 GB)
ollama pull mistral:7b-instruct
```

> Ollama n'est pas nécessaire pour la Tâche 1 (ingestion + retrieval).
> Il est requis uniquement pour la génération de réponses (Tâche 2).

### 5. Vérifier l'initialisation

```bash
python src/main.py
```

Sortie attendue :
```
ResearchPal — Démarrage
==================================================
       ResearchPal — Configuration
==================================================
  Embedding     : sentence-transformers/all-MiniLM-L6-v2
  LLM (Ollama)  : mistral:7b-instruct
  Chunk size    : 800  |  overlap : 150
  Collection    : researchpal_docs
==================================================
[SUCCESS] L'environnement ResearchPal est prêt.
```

---

## Corpus de test — data/raw/

Le dossier `data/raw/` contient les documents source à ingérer.

### Fichiers fournis d'office (types text + markdown)

| Fichier | Type | Contenu |
|---|---|---|
| `intro_rag.txt` | `text` | Introduction au RAG, embeddings, ChromaDB |
| `langchain_notes.md` | `markdown` | Notes sur LangChain, LCEL, retrievers |

### Ajouter un PDF (optionnel — type pdf)

Déposez n'importe quel fichier PDF dans `data/raw/` :
```bash
cp mon_article.pdf data/raw/
```
Le script d'ingestion détecte automatiquement tous les `.pdf` présents.

### Source web (type web — nécessite une connexion)

Le script charge également la page Wikipédia sur le RAG comme
exemple de source web. Cette étape est ignorée proprement si vous
n'avez pas de connexion.

> Pour ajouter d'autres URLs, modifiez `WEB_URL` dans
> `src/ingestion/run_ingestion.py`.

### Résumé des types supportés

| Extension / Source | `type_document` | Loader utilisé |
|---|---|---|
| `.pdf` | `pdf` | `PyPDFLoader` |
| URL `http(s)://` | `web` | `WebBaseLoader` |
| `.txt` | `text` | lecture directe |
| `.md` / `.markdown` | `markdown` | lecture directe |

---

## Tâche 1 — Ingestion et indexation

### Lancer l'ingestion

```bash
# Ingestion des sources disponibles
python src/ingestion/run_ingestion.py

# Réinitialiser la base et tout réingérer depuis zéro
python src/ingestion/run_ingestion.py --reset
```

### Sortie attendue

```
ResearchPal — Ingestion

Étape 1 — Chargement des sources

  [load_text]     1 document ← intro_rag.txt (2847 caractères)
  [load_markdown] 1 document ← langchain_notes.md (4156 caractères)
  [load_web]      1 document(s) ← https://en.wikipedia.org/wiki/...

  → 3 document(s) chargé(s), types : {'text', 'markdown', 'web'}

Étape 2 — Chunking

  [split_documents] 3 doc(s) → 27 chunk(s) (size=800, overlap=150)

Étape 3 — Indexation ChromaDB

  [embeddings] Chargement : sentence-transformers/all-MiniLM-L6-v2
  [indexer] Création de la collection 'researchpal_docs'...
  [indexer] Collection 'researchpal_docs' : 27 chunk(s) total.

============================================================
        RÉSUMÉ DE L'INGESTION
============================================================
  Documents chargés  : 3
  Types de sources   : {'text': 1, 'markdown': 1, 'web': 1}
  Chunks produits    : 27

  Détail par source :
    •  10 chunks ← intro_rag.txt
    •  12 chunks ← langchain_notes.md
    •   5 chunks ← https://en.wikipedia.org/...

  Exemple de métadonnées (chunk 0) :
    source               : /chemin/data/raw/intro_rag.txt
    type_document        : text
    date_ingestion       : 2026-03-09T14:32:01.123456+00:00
    chunk_index          : 0
    chunk_id             : a3f1e2b4c5d6e7f8

[SUCCESS] Ingestion terminée.
```

> Le premier lancement télécharge le modèle `all-MiniLM-L6-v2` (~90 MB).
> Les lancements suivants se font en cache.

### Où sont stockées les données ?

```
data/chroma_db/     ← base vectorielle persistante (ignorée par .gitignore)
```

---

## Tâche 1 — Test du retrieval

### Lancer le test

```bash
# Test avec les 3 requêtes prédéfinies, cosinus + MMR
python src/retrieval/test_retrieval.py

# Requête personnalisée, cosinus uniquement, top-5
python src/retrieval/test_retrieval.py --query "comment fonctionne MMR" --top-k 5 --strategy cosine
```

### Options disponibles

| Option | Valeur | Description |
|---|---|---|
| `--query` | texte | Question spécifique (défaut : 3 requêtes prédéfinies) |
| `--top-k` | entier | Nb de chunks retournés (défaut : 4) |
| `--strategy` | `cosine` / `mmr` / `both` | Stratégie à tester (défaut : `both`) |

### Sortie attendue

```
ResearchPal — Test du retrieval
Chargement de la base ChromaDB...
  → Collection chargée : 27 chunk(s) indexé(s).

════════════════════════════════════════════════════════════
  REQUÊTE : "Qu'est-ce que le RAG et comment fonctionne-t-il ?"
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
  Stratégie : Cosinus (similarité)
  Résultats : 4 chunk(s)
────────────────────────────────────────────────────────────

  Chunk 1
    Source      : intro_rag.txt
    Type        : text
    chunk_id    : a3f1e2b4c5d6e7f8
    chunk_index : 0
    Contenu     : "Introduction au RAG (Retrieval-Augmented Generation) ..."

  Sources cosinus : {'intro_rag.txt', 'langchain_notes.md'}
  Sources MMR     : {'intro_rag.txt', 'langchain_notes.md', 'web'}
  ✓ MMR a diversifié les sources par rapport au cosinus.
```

---

## Lancement de l'interface Streamlit

```bash
streamlit run src/ui/app.py
```

Ouvre automatiquement `http://localhost:8501`.

> ⚠️ L'interface affiche des réponses placeholder (Tâche 1).
> La connexion au pipeline RAG réel est prévue en Tâche 2.

---

## Structure du projet

```
ResearchPal/
├── README.md
├── requirements.txt
├── .gitignore
├── rapport.pdf                      ← À ajouter avant le rendu
│
├── data/
│   ├── raw/
│   │   ├── intro_rag.txt            ← Source de test (type: text)   ✅
│   │   ├── langchain_notes.md       ← Source de test (type: markdown) ✅
│   │   └── [vos_fichiers.pdf]       ← Sources PDF optionnelles
│   └── chroma_db/                   ← Base vectorielle (auto-généré) ✅
│
└── src/
    ├── config.py                    ← ⭐ Configuration centrale       ✅
    ├── main.py                      ← Vérification environnement      ✅
    │
    ├── ingestion/
    │   ├── loaders.py               ← load_pdf/web/text/markdown      ✅
    │   ├── chunking.py              ← Split récursif + chunk_id       ✅
    │   ├── indexer.py               ← ChromaDB load/index/reset       ✅
    │   └── run_ingestion.py         ← Pipeline d'ingestion exécutable ✅
    │
    ├── retrieval/
    │   ├── cosine_retriever.py      ← Stratégie cosinus               ✅
    │   ├── mmr_retriever.py         ← Stratégie MMR                   ✅
    │   ├── test_retrieval.py        ← Script de validation retrieval  ✅
    │   └── multiquery.py            ← Placeholder (Tâche 2)
    │
    ├── rag/
    │   ├── prompt.py                ← System prompt                   ✅
    │   ├── chain.py                 ← Pipeline RAG (Tâche 2)
    │   └── memory.py                ← Historique conversationnel      ✅
    │
    └── ui/
        └── app.py                   ← Interface Streamlit (Tâche 3)  ✅ (squelette)
```

---

## État d'avancement

| Composant | Statut | Notes |
|---|---|---|
| Environnement venv | ✅ Fait | Python 3.11 |
| `config.py` | ✅ Fait | Tous les paramètres centralisés |
| `loaders.py` | ✅ Fait | PDF, web, text, markdown |
| `chunking.py` | ✅ Fait | Récursif + `chunk_id` + `chunk_index` |
| `indexer.py` | ✅ Fait | Chroma load/index/reset |
| `run_ingestion.py` | ✅ Fait | Pipeline complet exécutable |
| Retrieval cosinus | ✅ Fait | Via `as_retriever(search_type="similarity")` |
| Retrieval MMR | ✅ Fait | Via `as_retriever(search_type="mmr")` |
| `test_retrieval.py` | ✅ Fait | Teste les 2 stratégies en comparaison |
| Pipeline RAG (chain.py) | ⏳ Tâche 2 | À connecter avec ChatOllama |
| Multi-query retrieval | ⏳ Tâche 2 | Placeholder dans `multiquery.py` |
| Interface Streamlit | ⏳ Tâche 3 | Squelette présent, RAG à brancher |

---

## Dépendances — notes importantes

| Package | Rôle | Note |
|---|---|---|
| `langchain-ollama` | Client ChatOllama | Requiert `ollama serve` actif (Tâche 2) |
| `langchain-huggingface` | Embeddings HuggingFace | Télécharge ~90 MB au premier usage |
| `langchain-chroma` | Intégration ChromaDB | Remplace l'ancien `langchain.vectorstores.Chroma` |
| `chromadb` | Base vectorielle locale | Persistance dans `data/chroma_db/` |
| `sentence-transformers` | Modèles d'embeddings | `all-MiniLM-L6-v2` téléchargé automatiquement |
| `pypdf` | Lecture PDF | Remplace `PyPDF2` (déprécié) |
| `beautifulsoup4` | Scraping web | Utilisé par `WebBaseLoader` |
| `streamlit` | Interface web | Lance avec `streamlit run src/ui/app.py` |

### Modèles Ollama alternatifs (si RAM insuffisante)

```bash
ollama pull phi3:mini          # ~2 GB — très rapide, moins précis
ollama pull gemma2:2b          # ~1.6 GB — bon compromis
```

Mettez à jour `OLLAMA_MODEL` dans `src/config.py` après le pull.

---

*LOG6951A — 2026*

