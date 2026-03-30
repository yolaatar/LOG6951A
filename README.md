# ResearchPal

Pipeline RAG (Retrieval-Augmented Generation) conversationnel avec interface Streamlit.
Développé dans le cadre du cours LOG6951A — Polytechnique Montréal.

Le système indexe un corpus de documents (texte, Markdown, PDF, web), puis répond aux questions via un LLM local (Mistral 7B via Ollama) en citant ses sources. Il gère la mémoire conversationnelle sur fenêtre glissante et propose plusieurs stratégies de retrieval.

---

## Prérequis

| Outil | Version minimale | Rôle |
|---|---|---|
| Python | 3.11 | Exécution du projet |
| [Ollama](https://ollama.com) | toute version récente | Inférence LLM locale |
| Modèle Mistral | `mistral:7b-instruct` | LLM utilisé par le pipeline |

Vérifier qu'Ollama est installé et que le modèle est disponible :

```bash
ollama list
# mistral:7b-instruct doit apparaître dans la liste
```

Si le modèle n'est pas présent :

```bash
ollama pull mistral:7b-instruct
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd LOG6951A
```

### 2. Créer et activer l'environnement virtuel

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Le premier `pip install` télécharge le modèle d'embedding `all-MiniLM-L6-v2` (~90 Mo) depuis HuggingFace. Une connexion internet est requise lors de cette étape.

### 4. Vérifier l'environnement

```bash
python src/main.py
```

Cette commande vérifie que les répertoires nécessaires existent et affiche la configuration active. Aucune sortie d'erreur = environnement prêt.

---

## Ingestion du corpus

Avant de lancer l'interface, il faut indexer les documents dans ChromaDB.

Le corpus par défaut se compose de :
- `data/raw/intro_rag.txt` — introduction au RAG
- `data/raw/langchain_notes.md` — notes LangChain
- Page Wikipedia sur le RAG (chargée via URL)

```bash
python src/ingestion/run_ingestion.py
```

Pour réinitialiser la base vectorielle et tout réindexer :

```bash
python src/ingestion/run_ingestion.py --reset
```

La base ChromaDB est persistée dans `data/chroma_db/` (environ 51 chunks indexés).

---

## Lancer l'interface

```bash
streamlit run src/ui/app.py
```

L'interface s'ouvre automatiquement à l'adresse [http://localhost:8501](http://localhost:8501).

**Fonctionnalités disponibles :**
- Sélection de la stratégie de retrieval dans la barre latérale :
  - **Cosine** — recherche par similarité cosinus (rapide)
  - **MMR** — Maximal Marginal Relevance (diversifié, λ=0.5)
  - **MultiQuery + Cosine** — 3 variantes de requête + fusion RRF
  - **MultiQuery + MMR** — combinaison des deux
- Import de documents supplémentaires (`.txt`, `.md`, `.pdf`)
- Réinitialisation de la mémoire de conversation
- Affichage des sources citées pour chaque réponse

---

## Dépannage

### `chromadb` ne démarre pas / erreur SQLite

ChromaDB nécessite une version récente de SQLite. Sur certains systèmes Linux :

```bash
pip install pysqlite3-binary
```

Puis ajouter en tête de `src/config.py` :

```python
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
```

### Ollama ne répond pas

Vérifier que le service Ollama est bien démarré :

```bash
ollama serve   # ou relancer l'application Ollama Desktop
```

### Le modèle d'embedding échoue au premier lancement

Le modèle `all-MiniLM-L6-v2` est téléchargé automatiquement depuis HuggingFace lors du premier appel. Si le téléchargement échoue (réseau), il sera mis en cache dans `~/.cache/huggingface/`. Relancer simplement la commande.

### `ModuleNotFoundError: No module named 'config'`

Les scripts s'exécutent avec `src/` dans le `PYTHONPATH`. Toujours lancer depuis la racine du projet, jamais depuis `src/` :

```bash
# Correct
python src/ingestion/run_ingestion.py

# Incorrect
cd src && python ingestion/run_ingestion.py
```

---

## Structure du projet

```
├── data/
│   ├── raw/              # Corpus source (txt, md, pdf)
│   └── chroma_db/        # Base vectorielle persistante (auto-généré)
├── outputs/              # Rapports d'évaluation générés
├── reports/              # Analyses détaillées (chunking, retrieval, RAG)
└── src/
    ├── config.py         # Configuration centralisée (modèles, seuils, chemins)
    ├── main.py           # Vérification de l'environnement
    ├── ingestion/        # Chargement, découpage, indexation des documents
    ├── retrieval/        # Cosine, MMR, Multi-Query + RRF
    ├── rag/              # Prompt, mémoire, pipeline RAGPipeline
    ├── evaluation/       # Scripts d'évaluation (voir section ci-dessous)
    └── ui/
        └── app.py        # Interface Streamlit (point d'entrée principal)
```

---

## Usage avancé — Scripts d'évaluation

> Ces scripts sont indépendants de l'interface. Ils nécessitent que l'ingestion ait été effectuée au préalable (`python src/ingestion/run_ingestion.py`).

### Analyse du chunking (Tâche 1)

Compare les stratégies fixe, récursive et sémantique. Génère figures et métriques dans `reports/chunking_analysis/`.

```bash
python src/evaluation/chunking_analysis.py
```

### Évaluation du retrieval — cosinus vs MMR (Tâche 2)

Évalue les deux stratégies sur 5 requêtes de test. Génère `reports/retrieval_eval/` et `outputs/retrieval_eval.md`.

```bash
python src/evaluation/retrieval_eval.py
```

### Évaluation Multi-Query + RRF (Tâche 4)

Compare cosinus, MMR, MultiQuery+Cosine et MultiQuery+MMR. Génère `outputs/multiquery_eval.md`.

```bash
python src/retrieval/evaluate_multiquery.py
```

### Évaluation complète du pipeline RAG (Tâches 3/4)

Exécute les cas de test UC1–UC4, EC1–EC7 et 3 dialogues multi-tours. Génère `reports/rag_eval/`.

```bash
python src/evaluation/rag_eval.py
```

### Évaluation des modes de contexte (Tâche 4 — ablation)

Compare les 4 modes d'enrichissement de requête (baseline, heuristique, concaténation, réécriture LLM) sur les dialogues VAL1–VAL4.

```bash
python src/evaluation/context_mode_eval.py
```

> Les fichiers déjà présents dans `outputs/` et `reports/` correspondent aux résultats des expériences documentées dans le rapport.
