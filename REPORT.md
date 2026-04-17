# ResearchPal v2 — Rapport technique (TP1 + TP2)
## LOG6951A — Méthodes et stratégies d'exploitation de l'IA générative
### Polytechnique Montréal — Avril 2026

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [TP1 — Pipeline RAG](#2-tp1--pipeline-rag)
3. [TP2 — Agent LangGraph Corrective RAG](#3-tp2--agent-langgraph-corrective-rag)
   - [T1 — Architecture Corrective RAG](#t1--architecture-corrective-rag)
   - [T2 — Outils agentiques](#t2--outils-agentiques)
   - [T3 — Mémoire agentique](#t3--mémoire-agentique)
   - [T4 — Observabilité Arize Phoenix](#t4--observabilité-arize-phoenix)
   - [T5 — Évaluation](#t5--évaluation)
4. [Résultats d'évaluation TP2](#4-résultats-dévaluation-tp2)
5. [Décisions techniques et justifications](#5-décisions-techniques-et-justifications)
6. [Difficultés rencontrées](#6-difficultés-rencontrées)

---

## 1. Vue d'ensemble du projet

ResearchPal est un assistant de recherche documentaire qui combine un pipeline RAG conversationnel (TP1) et un agent LangGraph Corrective RAG (TP2). L'interface Streamlit permet de basculer entre les deux modes via un toggle dans la barre latérale.

**Stack technique :**
- LLM : Mistral 7B via Ollama (`http://localhost:11434`)
- Embeddings : `all-MiniLM-L6-v2` (384 dim, cosine)
- Base vectorielle : ChromaDB (persistence locale)
- Framework agent : LangGraph (StateGraph + SqliteSaver)
- Observabilité : Arize Phoenix + OpenTelemetry
- Évaluation : RAGAS 0.4.3 + LLM-as-judge

---

## 2. TP1 — Pipeline RAG

### Architecture

Pipeline linéaire LCEL (LangChain Expression Language) :

```
Question → [Gestion contexte] → Retrieval → Prompt → LLM → Réponse
```

### Composants

| Composant | Implémentation | Fichier |
|---|---|---|
| Ingestion | `RecursiveCharacterTextSplitter` (chunk 500, overlap 50) | `src/ingestion/chunking.py` |
| Indexation | ChromaDB + `all-MiniLM-L6-v2` | `src/ingestion/indexer.py` |
| Retrieval Cosine | `similarity_search` ChromaDB | `src/retrieval/cosine_retriever.py` |
| Retrieval MMR | `max_marginal_relevance_search` (λ=0.5, fetch_k=20) | `src/retrieval/mmr_retriever.py` |
| MultiQuery + RRF | `MultiQueryRetriever` LangChain + fusion RRF | `src/retrieval/multiquery.py` |
| Gestion contexte | Heuristiques / Concaténation / Réécriture LLM | `src/rag/chain.py` |
| Mémoire | `ConversationBufferWindowMemory` (k=5) | `src/rag/memory.py` |

### Résultats évaluation TP1 (summary)

- Chunking : récursif > fixe > sémantique sur ce corpus
- Retrieval : MMR améliore la diversité sans perte de pertinence notable
- MultiQuery : +15% de rappel sur les questions ambiguës
- Contexte : la réécriture LLM donne les meilleurs résultats sur les questions de suivi

---

## 3. TP2 — Agent LangGraph Corrective RAG

### T1 — Architecture Corrective RAG

#### Choix : LangGraph StateGraph avec cycle correctif

**Justification :** LangGraph permet de modéliser explicitement le cycle Corrective RAG comme un graphe d'états avec des arêtes conditionnelles. Le cycle correctif (grade → transform → retrieve) est une boucle de contrôle qui améliore la qualité du retrieval de façon itérative, sans nécessiter de re-entraînement.

#### Diagramme du graphe

```
START
  ↓
route_query ──────────────────────────────────────────┐
  │                                                    │
  │ "corpus"                                "web"      │
  ↓                                          ↓         │
retrieve (search_corpus)          web_search_node      │
  ↓                                          ↓         │
grade_documents                           generate → END
  │
  ├── "sufficient" ──────────────────────→ generate → END
  │
  └── "insufficient"
        ├── retry_count < 3 → transform_query → retrieve (CYCLE)
        └── retry_count ≥ 3 → generate → END  (garde-fou)
```

#### État du graphe (`AgentState`)

```python
class AgentState(TypedDict):
    question:         str           # question originale
    retrieval_query:  str           # requête reformulée (modifiée par transform_query)
    documents:        List[Document]  # docs bruts récupérés
    relevant_docs:    List[Document]  # docs jugés pertinents
    generation:       str           # réponse finale
    retry_count:      int           # nombre de cycles correctifs
    grade_decision:   str           # "sufficient" | "insufficient"
    tool_used:        str           # "corpus" | "web" | "error"
    web_results:      Optional[str] # résultats DuckDuckGo bruts
```

#### Nœuds

| Nœud | Rôle | Décision |
|---|---|---|
| `route_query` | LLM choisit "corpus" ou "web" selon la question | Arête conditionnelle |
| `retrieve` | Appelle `search_corpus.invoke()`, charge les docs ChromaDB | — |
| `grade_documents` | LLM note chaque doc OUI/NON selon pertinence | Arête conditionnelle |
| `transform_query` | LLM reformule `retrieval_query`, incrémente `retry_count` | — |
| `web_search_node` | Appelle `web_search.invoke()`, formate en pseudo-documents | — |
| `generate` | Construit le prompt RAG, appelle le LLM, produit la réponse | → END |

#### Fichiers

- `src/agent/state.py` — TypedDict AgentState
- `src/agent/nodes.py` — factory functions pour chaque nœud + spans OTel
- `src/agent/graph.py` — construction, compilation, `run_agent()`, `get_checkpointer()`

---

### T2 — Outils agentiques

#### Choix : search_corpus (ChromaDB) + web_search (DuckDuckGo)

**Justification :**
- `search_corpus` pour les questions sur le corpus RAG/LangChain — sources vérifiables, réponses citables
- `web_search` (DuckDuckGo) pour les requêtes hors-corpus ou temps réel — gratuit, sans clé API, suffisant pour un prototype académique

Les descriptions des outils (`@tool`) sont délibérément détaillées pour guider le LLM lors du routage :

```python
@tool
def search_corpus(query: str) -> str:
    """Recherche dans le corpus local...
    À utiliser pour : RAG, LangChain, embeddings, ChromaDB...
    À NE PAS utiliser pour : actualités, prix, données temps réel...
    """

@tool
def web_search(query: str) -> str:
    """Recherche web via DuckDuckGo...
    À utiliser pour : questions hors-corpus, actualités...
    """
```

#### Fichier : `src/agent/tools.py`

---

### T3 — Mémoire agentique

#### Mémoire court terme : SQLite checkpointer

`SqliteSaver` de `langgraph-checkpoint-sqlite` persiste l'état complet de chaque nœud entre les sessions. Le `thread_id` identifie la conversation.

```python
# Solution : utiliser sqlite3.connect() directement (from_conn_string() retourne
# un context manager incompatible avec l'interface BaseCheckpointSaver)
conn = sqlite3.connect(db_path, check_same_thread=False)
return SqliteSaver(conn=conn)
```

Base de données : `data/checkpoints/agent_state.db`

#### Mémoire long terme : Option B — Épisodique JSON

Stocke jusqu'à 5 épisodes de haute qualité comme exemples few-shot. Les critères de sélection :

| Critère | Valeur seuil |
|---|---|
| Longueur de la réponse | ≥ 200 caractères |
| Nombre de sources citées | ≥ 2 |
| Outil utilisé | "corpus" (pas web) |
| Cycles correctifs | 0 (première tentative réussie) |

Fichier : `data/episodic_memory.json`
Module : `src/memory_v2/episodic.py`

---

### T4 — Observabilité Arize Phoenix

#### Choix : Arize Phoenix (pip, port 6006)

**Justification :** Phoenix s'installe via pip sans Docker, fournit une interface web de traçage des LLM avec support natif LangChain via `openinference-instrumentation-langchain`.

#### Configuration

```python
# src/observability/tracing.py
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

session = px.launch_app()          # démarre Phoenix sur http://localhost:6006
LangChainInstrumentor().instrument()  # instrumente automatiquement LangChain
```

Des spans OpenTelemetry manuels sont ajoutés dans chaque nœud LangGraph :

```python
with _tracer().start_as_current_span("route_query") as span:
    span.set_attribute("agent.question", state["question"])
    span.set_attribute("agent.tool_selected", tool_choice)
```

#### Pour capturer les traces

1. `streamlit run src/ui/app.py`
2. Sélectionner "Agent LangGraph (TP2)" dans le toggle
3. Poser 1–2 questions
4. Aller sur [http://localhost:6006](http://localhost:6006) → screenshot du waterfall de spans

---

### T5 — Évaluation

#### Dataset : `data/eval_dataset.json`

15 paires question/réponse de référence :
- **10 corpus** : questions directement couvertes par le corpus (RAG, embeddings, MMR, chunking…)
- **2 multi-hop** : nécessitent de combiner plusieurs concepts (embeddings + MMR, chunks + retrieval)
- **3 adversariales** : cours de bourse AAPL, "meilleur LLM objectif", tokens exacts de Mistral 7B

#### RAGAS — `eval/ragas_eval.py`

Métriques calculées : `faithfulness` + `answer_relevancy`

Configuration locale (Ollama + HuggingFace) :
```python
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

f = Faithfulness()
f.llm = LangchainLLMWrapper(ChatOllama(model="mistral"))
a = AnswerRelevancy()
a.llm = f.llm
a.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(...))
```

> Note : les classes old-style (`_faithfulness`, `_answer_relevance`) sont requises pour RAGAS 0.4.x avec un LLM non-OpenAI. Les classes `ragas.metrics.collections` nécessitent `InstructorLLM` incompatible avec Ollama.

#### LLM-as-judge — `eval/llm_judge.py`

Critère : **qualité des citations** (3 sous-critères, total /10)

| Critère | Pondération | Description |
|---|---|---|
| Précision des citations | /3 | Chaque affirmation importante est ancrée sur une source numérotée |
| Complétude des citations | /3 | Aucune affirmation non fondée n'est laissée sans source |
| Honnêteté sur les limites | /4 | Le système signale explicitement ce qu'il ne sait pas |

---

## 4. Résultats d'évaluation TP2

### LLM-as-judge (citations + honnêteté)

| Type de question | Score moyen | N |
|---|---|---|
| Corpus | **9.90 / 10** | 10 |
| Multi-hop | **9.50 / 10** | 2 |
| Adversarial | **7.67 / 10** | 3 |
| **Global** | **9.40 / 10** | 15 |

**Analyse :**
- Les questions corpus obtiennent un score quasi-parfait : le système cite correctement et signale ses limites de façon systématique.
- Les questions multi-hop sont bien gérées (9.5/10) malgré la nécessité de croiser plusieurs concepts.
- Les questions adversariales tirent le score vers le bas. La Q15 (tokens exacts de Mistral 7B) a obtenu 3/10 : le système a fourni une réponse partielle sans signaler avec assez de force l'absence de données chiffrées dans le corpus.

### RAGAS — faithfulness + answer_relevancy

| Question | faithfulness | answer_relevancy |
|---|---|---|
| Chunking (Q6) | 0.833 | 0.589 |
| Multi-Query (Q9) | — | 0.709 |
| Chunk size (Q12) | — | 0.721 |
| Apple AAPL (Q13) | 0.667 | — |
| Meilleur LLM (Q14) | 0.714 | 0.341 |
| Tokens Mistral (Q15) | 0.200 | — |

**Limitation connue :** RAGAS 0.4.x avec Mistral 7B en local génère des `NaN` et des `0.0` sur les questions où le modèle dépasse le timeout RAGAS (~30s). Mistral 7B est sensiblement plus lent que les modèles GPT-4 pour lesquels RAGAS a été calibré. Sur les 6 questions où RAGAS a pu calculer un score, la faithfulness moyenne est **0.63** et l'answer_relevancy moyenne **0.59** — valeurs cohérentes avec un LLM 7B quantisé sur corpus limité.

---

## 5. Décisions techniques et justifications

### Pourquoi Corrective RAG plutôt que Self-RAG ou FLARE ?

Le Corrective RAG (CRAG) offre le meilleur compromis pour ce projet :
- Plus simple à implémenter que FLARE (pas de génération token-by-token)
- Plus déterministe que Self-RAG (le grading est une décision binaire claire)
- Bien adapté à LangGraph : le cycle grade→transform→retrieve s'exprime naturellement comme une arête conditionnelle avec garde-fou sur `retry_count`

### Pourquoi DuckDuckGo plutôt qu'une calculatrice ?

DuckDuckGo répond à un besoin réel identifié lors des tests : les questions adversariales (cours AAPL, dernière version d'un modèle) nécessitent une vraie recherche externe. Une calculatrice couvrirait un cas d'usage beaucoup plus restreint pour ce type d'assistant documentaire.

### Pourquoi l'Option B (JSON épisodique) pour la mémoire long terme ?

L'Option B est la plus simple à auditer : le fichier JSON est lisible, les critères de sélection sont explicites, et le mécanisme few-shot est transparent. Une base vectorielle (Option A) aurait été sur-ingéniéré pour 5 exemples max.

### Pourquoi Arize Phoenix plutôt que LangSmith ?

Phoenix fonctionne entièrement en local (pas de clé API, pas d'envoi de données externes), ce qui est cohérent avec le reste de l'architecture (Mistral local, ChromaDB local). LangSmith nécessite un compte et envoie les traces vers les serveurs Anthropic.

---

## 6. Difficultés rencontrées

### Conflit mistralai v2 + instructor/ragas

**Problème :** `pip install ragas` installe `mistralai>=2.0.0` qui casse l'import `from mistralai import Mistral` utilisé par `instructor`.

**Solution :** `pip install 'mistralai<2.0.0'` — downgrade à 1.12.4, compatible avec les deux.

### SqliteSaver : from_conn_string() retourne un context manager

**Problème :** `SqliteSaver.from_conn_string(path)` retourne `_GeneratorContextManager`, pas un `BaseCheckpointSaver`. LangGraph lève `Invalid checkpointer provided`.

**Solution :**
```python
import sqlite3
conn = sqlite3.connect(db_path, check_same_thread=False)
return SqliteSaver(conn=conn)
```

### RAGAS 0.4.x : classes de métriques incompatibles avec Ollama

**Problème :** `ragas.metrics.collections.Faithfulness` et `AnswerRelevancy` nécessitent `InstructorLLM`, uniquement compatible avec OpenAI-compatible APIs qui retournent du JSON structuré fiable. Mistral 7B échoue régulièrement à produire du JSON valide pour instructor.

**Solution :** Utiliser les classes internes old-style :
```python
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
```
Ces classes héritent directement de `Metric` et acceptent `LangchainLLMWrapper`.

### TimeoutErrors RAGAS avec Mistral 7B en parallèle

**Problème :** RAGAS exécute les évaluations en parallèle (via `asyncio`). Mistral 7B (~5s/requête) génère des TimeoutErrors sur plusieurs questions simultanées, résultant en `NaN`/`0.0`.

**Impact :** Scores partiels sur ~9/15 questions. Valeurs calculées restent représentatives.
