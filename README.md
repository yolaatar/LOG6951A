# ResearchPal v2

Pipeline RAG conversationnel **et** agent LangGraph Corrective RAG avec interface Streamlit.
Développé dans le cadre du cours LOG6951A — Polytechnique Montréal (TP1 + TP2).

Le système indexe un corpus de documents (texte, Markdown, PDF, web), puis répond aux questions via un LLM local (Mistral 7B via Ollama) en citant ses sources. En mode Agent (TP2), il sélectionne dynamiquement l'outil de recherche (corpus ou web), applique un cycle Corrective RAG, et conserve une mémoire court et long terme.

---

## Prérequis

| Outil | Version minimale | Rôle |
|---|---|---|
| Python | 3.11 | Exécution du projet |
| [Ollama](https://ollama.com) | toute version récente | Inférence LLM locale |
| Modèle Mistral | `mistral:7b-instruct` | LLM utilisé par le pipeline |

```bash
ollama list           # mistral:7b-instruct doit apparaître
ollama pull mistral:7b-instruct   # si absent
```

---

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

> Le premier `pip install` télécharge `all-MiniLM-L6-v2` (~90 Mo) depuis HuggingFace.

> **Note compatibilité** : `mistralai<2.0.0` est requis (conflit instructor/ragas avec v2).

---

## Ingestion du corpus

```bash
python src/ingestion/run_ingestion.py          # indexation initiale
python src/ingestion/run_ingestion.py --reset  # réinitialiser et tout réindexer
```

Corpus par défaut : `data/raw/intro_rag.txt`, `data/raw/langchain_notes.md`, page Wikipedia RAG (~51 chunks indexés dans `data/chroma_db/`).

---

## Lancer l'interface

```bash
streamlit run src/ui/app.py
```

Ouvre automatiquement [http://localhost:8501](http://localhost:8501).

### Mode Pipeline RAG (TP1)
- Stratégies de retrieval : Cosine, MMR (λ=0.5), MultiQuery+Cosine, MultiQuery+MMR
- Modes de contexte : Aucun, Heuristiques, Concaténation, Réécriture LLM
- Import de documents supplémentaires (`.txt`, `.md`, `.pdf`)
- Affichage des sources par réponse

### Mode Agent LangGraph (TP2)
- Sélection dynamique d'outil : `search_corpus` (ChromaDB) ou `web_search` (DuckDuckGo)
- Cycle Corrective RAG : grading des docs → reformulation → nouveau retrieval (max 3×)
- Mémoire court terme par session (SQLite checkpointer)
- Mémoire épisodique long terme (JSON, ≤5 exemples few-shot)
- Observabilité via Arize Phoenix (port 6006)
- Affichage de l'outil utilisé et du nombre de cycles correctifs

---

## Observabilité Arize Phoenix (TP2 — T4)

Phoenix démarre automatiquement lors de l'initialisation de l'agent. Interface disponible sur [http://localhost:6006](http://localhost:6006).

```python
# Appelé automatiquement dans la session Streamlit TP2
from observability.tracing import setup_tracing
setup_tracing()
```

---

## Évaluation (TP2 — T5)

### RAGAS — faithfulness + answer_relevancy

```bash
python eval/ragas_eval.py
# → eval/ragas_results.json
```

### LLM-as-judge — qualité des citations

```bash
python eval/llm_judge.py
# → eval/judge_results.json
```

Le dataset d'évaluation (`data/eval_dataset.json`) contient 15 paires Q/R :
- 10 questions corpus
- 2 questions multi-hop
- 3 questions adversariales

---

## Scripts d'évaluation TP1

```bash
python src/evaluation/chunking_analysis.py    # compare fixed/recursive/sémantique
python src/evaluation/retrieval_eval.py       # cosinus vs MMR sur 5 requêtes
python src/retrieval/evaluate_multiquery.py   # MultiQuery + RRF
python src/evaluation/rag_eval.py             # UC1–UC4, EC1–EC7, 3 dialogues
python src/evaluation/context_mode_eval.py    # ablation des 4 modes contexte
```

Résultats dans `outputs/` et `reports/`.

---

## Structure du projet

```
ResearchPal/
├── data/
│   ├── raw/                    # Corpus source (txt, md, pdf)
│   ├── chroma_db/              # Base vectorielle persistante (auto-généré)
│   ├── checkpoints/            # SQLite checkpointer LangGraph (TP2)
│   ├── episodic_memory.json    # Mémoire épisodique long terme (TP2)
│   └── eval_dataset.json       # 15 paires Q/R pour l'évaluation (TP2)
├── eval/                       # Scripts et résultats d'évaluation TP2
│   ├── ragas_eval.py
│   ├── llm_judge.py
│   ├── ragas_results.json
│   ├── judge_results.json
│   └── generated_answers.json
├── outputs/                    # Rapports d'évaluation TP1
├── reports/                    # Analyses détaillées TP1
└── src/
    ├── config.py               # Configuration centralisée
    ├── main.py                 # Vérification de l'environnement
    ├── agent/                  # Graphe LangGraph Corrective RAG (TP2)
    │   ├── state.py            # AgentState TypedDict
    │   ├── tools.py            # search_corpus + web_search (@tool)
    │   ├── nodes.py            # Nœuds du graphe + spans OTel
    │   └── graph.py            # Construction, compilation, run_agent()
    ├── memory_v2/
    │   └── episodic.py         # Mémoire épisodique Option B (TP2)
    ├── observability/
    │   └── tracing.py          # Arize Phoenix + OpenTelemetry (TP2)
    ├── ingestion/              # Loaders, chunking, indexation
    ├── retrieval/              # Cosine, MMR, MultiQuery + RRF
    ├── rag/                    # Prompt, mémoire, RAGPipeline
    ├── evaluation/             # Scripts d'évaluation TP1
    └── ui/
        └── app.py              # Interface Streamlit (TP1 + TP2)
```

---

## Dépannage

**`chromadb` / erreur SQLite sur Linux**
```bash
pip install pysqlite3-binary
# Puis en tête de src/config.py :
import sys, pysqlite3; sys.modules["sqlite3"] = pysqlite3
```

**Ollama ne répond pas**
```bash
ollama serve   # ou relancer Ollama Desktop
```

**`ModuleNotFoundError: No module named 'config'`**
Toujours lancer depuis la racine du projet, pas depuis `src/` :
```bash
# Correct
streamlit run src/ui/app.py
python eval/ragas_eval.py
```

**SqliteSaver / checkpointer**
Le checkpointer utilise `sqlite3.connect(db_path, check_same_thread=False)` passé directement à `SqliteSaver(conn=conn)` — ne pas utiliser `from_conn_string()` qui retourne un context manager incompatible.
