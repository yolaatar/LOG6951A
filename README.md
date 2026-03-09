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
6. [Tâche 2 — Évaluation retrieval (cosinus vs MMR)](#tâche-2--évaluation-retrieval-cosinus-vs-mmr)
7. [Tâche 3 — Pipeline RAG avec citations et mémoire](#tâche-3--pipeline-rag-avec-citations-et-mémoire)
8. [Tâche 4 — Multi-Query + RRF](#tâche-4--multi-query--rrf)
9. [Tâche 5 — Interface Streamlit](#tâche-5--interface-streamlit)
10. [Structure du projet](#structure-du-projet)
11. [État d'avancement](#état-davancement)
12. [Dépendances — notes importantes](#dépendances--notes-importantes)

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

## Tâche 2 — Évaluation retrieval (cosinus vs MMR)

### Prérequis

La base ChromaDB doit être peuplée (Tâche 1) :
```bash
python src/ingestion/run_ingestion.py
```

### Lancer l'évaluation standard

```bash
# Évaluation console — 5 requêtes × cosinus + MMR
python src/retrieval/evaluate_retrieval.py

# Avec paramètres personnalisés
python src/retrieval/evaluate_retrieval.py --k 5 --fetch-k 12 --lambda-mult 0.3

# Export du rapport Markdown (outputs/retrieval_eval.md)
python src/retrieval/evaluate_retrieval.py --export

# Export vers un chemin spécifique
python src/retrieval/evaluate_retrieval.py --export mon_rapport.md
```

### Options disponibles

| Option | Défaut | Description |
|---|---|---|
| `--k` | 4 | Nombre de chunks retournés par requête |
| `--fetch-k` | 20 | Candidats initiaux pour MMR (avant re-classement) |
| `--lambda-mult` | 0.5 | Équilibre diversité/pertinence MMR (0=diversité, 1=pertinence) |
| `--export [chemin]` | — | Exporte le rapport en Markdown |
| `--param-sweep` | — | Balayage de paramètres MMR |

### Balayage de paramètres MMR

Pour identifier les meilleurs réglages MMR sur le corpus :

```bash
python src/retrieval/evaluate_retrieval.py --param-sweep
```

Teste les combinaisons suivantes et affiche un tableau comparatif :

| Paramètre | Valeurs testées |
|---|---|
| `k` | 3, 5 |
| `fetch_k` | 8, 12 |
| `lambda_mult` | 0.3, 0.5, 0.7 |

### Fichiers produits

| Fichier | Description |
|---|---|
| `outputs/retrieval_eval.md` | Rapport Markdown avec résultats par requête, métriques et observations |

### Métriques calculées

| Métrique | Description |
|---|---|
| **Sources distinctes** | Nombre de fichiers/URLs différents dans le top-k |
| **Types distincts** | Diversité des types de documents (`text`, `markdown`, `web`) |
| **Longueur moyenne** | Taille moyenne des chunks retournés (en caractères) |
| **Paires redondantes** | Paires de chunks avec similarité Jaccard ≥ 0.5 |

### Requêtes de test (src/retrieval/eval_queries.py)

| # | Catégorie | Intérêt |
|---|---|---|
| 1 | Définition / factuelle | Teste la pertinence basique |
| 2 | Comparaison entre concepts | Révèle la redondance intra-source |
| 3 | Synthèse / question large | Montre la couverture multi-source |
| 4 | Question précise / mot-clé rare | Teste les cas de corpus sparse |
| 5 | Cas limite / requête ambiguë | Comportement hors-corpus |

---

## Tâche 3 — Pipeline RAG avec citations et mémoire

### Prérequis

- Base ChromaDB peuplée (`python src/ingestion/run_ingestion.py`)
- Ollama actif avec le modèle chargé :
  ```bash
  ollama serve          # dans un terminal dédié
  ollama pull mistral:7b-instruct
  ```

### Architecture T3

| Composant | Fichier | Rôle |
|---|---|---|
| Prompt | `src/rag/prompt.py` | Pattern Persona + Structured Output (Réponse / Sources / Limites) |
| Mémoire | `src/rag/memory.py` | Historique 3 tours avec sources |
| Pipeline | `src/rag/chain.py` | `RAGPipeline` · `build_rag_pipeline()` · `answer_question()` |

### Démo console (scénario automatique 3 tours)

```bash
# Scénario automatique — 3 questions enchaînées avec mémoire
python src/rag/demo_rag.py

# Mode interactif (Q&A libre)
python src/rag/demo_rag.py --interactive

# Avec MultiQuery + RRF activé
python src/rag/demo_rag.py --multiquery
python src/rag/demo_rag.py --interactive --multiquery
```

Dans le mode interactif, préfixez votre question avec `mmr:` pour forcer la stratégie MMR.

### Format de réponse (Structured Output)

Chaque réponse du LLM suit le format :

```
**Réponse**
<réponse factuelle avec citations [N]>

**Sources**
[1] fichier.txt · [2] autre.md

**Limites / Incertitudes**
<ce qui ne peut pas être affirmé depuis le contexte>
```

---

## Tâche 4 — Multi-Query + RRF

### Principe

1. Le LLM génère N reformulations de la question (défaut : 3)
2. Chaque variante déclenche une recherche independante (cosine ou MMR)
3. Les résultats sont fusionnés avec **Reciprocal Rank Fusion** :
   $\text{RRF}(d) = \sum_q \frac{1}{k + \text{rank}_q(d)}$
4. Déduplication par `chunk_id`, tri par score décroissant

### Évaluation comparée (brute cosine vs multi-query + RRF)

```bash
# Évaluation console sur les 5 requêtes de eval_queries.py
python src/retrieval/evaluate_multiquery.py

# Sélectionner 3 requêtes spécifiques
python src/retrieval/evaluate_multiquery.py --queries 1,2,3

# Exporter le rapport
python src/retrieval/evaluate_multiquery.py --export
```

### Sortie attendue (extrait)

```
══════════════════════════════════════════════════════════════════════
  Q1 [Définition / question factuelle]
  What is Retrieval-Augmented Generation and how does it work?
──────────────────────────────────────────────────────────────────────
  BRUTE (cosine pur)
    chunks : 4  |  sources : 2  |  types : 2
    fichiers : intro_rag.txt, langchain_notes.md
  MULTI-QUERY + RRF (cosine)
    variantes (3) :
      • Explain the concept of Retrieval-Augmented Generation (RAG)
      • How does RAG combine retrieval and generation in NLP?
      • What role does retrieval play in augmented language models?
    chunks : 4  |  sources : 3  |  types : 3
    fichiers : intro_rag.txt, langchain_notes.md, wikipedia_rag
  Δ sources : +1
══════════════════════════════════════════════════════════════════════
```

### Fichier produit

| Fichier | Description |
|---|---|
| `outputs/multiquery_eval.md` | Rapport Markdown avec tableau comparatif et détail par requête |

---

## Tâche 5 — Interface Streamlit

```bash
streamlit run src/ui/app.py
```

Ouvre `http://localhost:8501`.

### Fonctionnalités

| Zone | Description |
|---|---|
| **Sidebar — Stratégie** | 4 modes : Cosine · MMR · MultiQuery+Cosine · MultiQuery+MMR |
| **Sidebar — Documents** | Upload .txt / .md / .pdf → `data/raw/` + bouton re-indexation |
| **Sidebar — Mémoire** | Bouton "Effacer la conversation" (reset historique + mémoire RAG) |
| **Chat** | Affiche l'historique complet avec rôles `user` / `assistant` |
| **Réponse** | Réponse formatée (Structured Output) + expander Sources |
| **MultiQuery** | Expander dédié affichant les variantes générées par le LLM |

### Notes

- Le pipeline est mis en cache par Streamlit (`@st.cache_resource`) — un seul chargement au démarrage.
- Si Ollama n'est pas accessible, un message d'erreur clair s'affiche avec les instructions de démarrage.
- Après upload + re-indexation, rafraîchissez la page (F5) pour recharger le pipeline avec le nouveau corpus.

---

## Lancement de l'interface Streamlit

```bash
streamlit run src/ui/app.py
```

Ouvre automatiquement `http://localhost:8501`.

> Ollama doit être lancé (`ollama serve`) avant de démarrer l'interface.

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
├── outputs/
│   ├── retrieval_eval.md            ← Rapport T2 généré par --export ✅ (auto)
│   └── multiquery_eval.md           ← Rapport T4 généré par --export ✅ (auto)
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
    │   ├── cosine_retriever.py      ← Similarité cosinus              ✅
    │   ├── mmr_retriever.py         ← MMR avec paramètres documentés  ✅
    │   ├── eval_queries.py          ← 5 requêtes annotées             ✅ T2
    │   ├── evaluate_retrieval.py    ← Éval cosinus vs MMR             ✅ T2
    │   ├── multiquery.py            ← Multi-Query + RRF               ✅ T4
    │   ├── evaluate_multiquery.py   ← Éval brute vs multi-query       ✅ T4
    │   └── test_retrieval.py        ← Validation rapide               ✅
    │
    ├── rag/
    │   ├── prompt.py                ← Persona + Structured Output     ✅ T3
    │   ├── memory.py                ← Mémoire 3 tours avec sources    ✅ T3
    │   ├── chain.py                 ← RAGPipeline end-to-end          ✅ T3
    │   └── demo_rag.py              ← Démo console / interactive      ✅ T3
    │
    └── ui/
        └── app.py                   ← Interface Streamlit complète    ✅ T5
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
| Retrieval cosinus | ✅ Fait | `cosine_search_with_scores()` |
| Retrieval MMR | ✅ Fait | Paramètres documentés + `mmr_search()` |
| `eval_queries.py` | ✅ Fait T2 | 5 requêtes annotées par catégorie |
| `evaluate_retrieval.py` | ✅ Fait T2 | Éval comparée + métriques + export Markdown |
| Balayage paramètres MMR | ✅ Fait T2 | `--param-sweep` (12 combinaisons) |
| `test_retrieval.py` | ✅ Fait | Validation rapide des 2 stratégies |
| `prompt.py` | ✅ Fait T3 | Persona + Structured Output + mémoire |
| `memory.py` | ✅ Fait T3 | Historique 3 tours avec sources |
| `chain.py` | ✅ Fait T3 | `RAGPipeline` + `RAGResult` + gestion erreurs |
| `demo_rag.py` | ✅ Fait T3 | Démo console auto + mode interactif |
| `multiquery.py` | ✅ Fait T4 | `generate_query_variants()` + `rrf_fuse()` + fallback |
| `evaluate_multiquery.py` | ✅ Fait T4 | Éval brute vs multi-query + export Markdown |
| Interface Streamlit | ✅ Fait T5 | 4 stratégies, upload documents, historique, sources |

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

