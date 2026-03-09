# Évaluation du Retrieval — ResearchPal

**Date** : 2026-03-09 18:30 UTC  
**Corpus** : 51 chunks indexés  
**Modèle d'embeddings** : sentence-transformers/all-MiniLM-L6-v2

## Paramètres d'évaluation

| Paramètre | Valeur | Description |
| --- | --- | --- |
| k | 4 | Chunks retournés par requête |
| fetch_k (MMR) | 20 | Candidats initiaux avant sélection MMR |
| λ (MMR) | 0.5 | 0 = diversité max · 1 = pertinence max |

---

## Requête 1 — Définition / question factuelle

> **What is Retrieval-Augmented Generation and how does it work?**

*Intérêt pour la comparaison* : Question générale couverte par tout le corpus. Permet de voir si MMR évite les chunks quasi-identiques de la même source.

### Résultats Cosinus

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | c35750a770014863 | 0.505 | ". This approach provides the LLM with key information early in the prompt, encouraging it to priorit…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | ed809bac79395462 | 0.473 | ". According to the MIT Technology Review, these issues occur because RAG systems may misinterpret th…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | ca2792d372a04155 | 0.427 | "encyclopedia Type of information retrieval using LLMs Retrieval-augmented generation (RAG) is a tech…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 7d00bc7a3299e40b | 0.373 | ". International Conference on Neural Information Processing Systems. Red Hook, NY, USA: Curran Assoc…" |

**Métriques** : sources=1 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation'], types=1 ['web'], longueur moy.=567 car., redondance=0/6 paires

### Résultats MMR (fetch_k=20, λ=0.5)

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | c35750a770014863 | n/a | ". This approach provides the LLM with key information early in the prompt, encouraging it to priorit…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | ed809bac79395462 | n/a | ". According to the MIT Technology Review, these issues occur because RAG systems may misinterpret th…" |
| 3 | intro_rag.txt | text | 91552633e4e3bd00 | n/a | "Introduction au RAG (Retrieval-Augmented Generation)  Qu'est-ce que le RAG ? Le Retrieval-Augmented …" |
| 4 | intro_rag.txt | text | a0f1c0a0795db77e | n/a | "Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les documents pertinents :…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt'], types=2 ['text', 'web'], longueur moy.=650 car., redondance=0/6 paires

### Observation

MMR couvre 1 source(s) de plus → meilleure diversité ✓

---

## Requête 2 — Comparaison entre deux concepts

> **What is the difference between cosine similarity search and MMR retrieval?**

*Intérêt pour la comparaison* : Comparaison entre deux stratégies. Teste si cosinus crée de la redondance quand un seul fichier domine le sujet.

### Résultats Cosinus

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | intro_rag.txt | text | a0f1c0a0795db77e | 0.378 | "Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les documents pertinents :…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | e1b833ae7dad93bd | 0.263 | ". Some retrieval methods combine sparse representations, such as SPLADE, with query expansion strate…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | a1adb0fa63e553d7 | 0.248 | ". Sparse vectors, which encode the identity of a word, are typically dictionary-length and contain m…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 07e8370ce3b2b2e7 | 0.229 | ". RAG can be used on unstructured (usually text), semi-structured, or structured data (for example k…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt'], types=2 ['text', 'web'], longueur moy.=604 car., redondance=0/6 paires

### Résultats MMR (fetch_k=20, λ=0.5)

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | intro_rag.txt | text | a0f1c0a0795db77e | n/a | "Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les documents pertinents :…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | e1b833ae7dad93bd | n/a | ". Some retrieval methods combine sparse representations, such as SPLADE, with query expansion strate…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | a1adb0fa63e553d7 | n/a | ". Sparse vectors, which encode the identity of a word, are typically dictionary-length and contain m…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 45ef3ad1cf25cffd | n/a | ". RAG improves large language models (LLMs) by incorporating information retrieval before generating…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt'], types=2 ['text', 'web'], longueur moy.=660 car., redondance=0/6 paires

### Observation

Même couverture de sources pour les deux stratégies

---

## Requête 3 — Synthèse / question large

> **How do embeddings and vector databases enable semantic search?**

*Intérêt pour la comparaison* : Requête large qui touche à plusieurs composants du corpus. MMR devrait montrer une meilleure couverture des types de documents.

### Résultats Cosinus

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 07e8370ce3b2b2e7 | 0.411 | ". RAG can be used on unstructured (usually text), semi-structured, or structured data (for example k…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 42abd0713717f579 | 0.317 | ". Hybrid search[edit] Sometimes vector database searches can miss key facts needed to answer a user'…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | e1b833ae7dad93bd | 0.300 | ". Some retrieval methods combine sparse representations, such as SPLADE, with query expansion strate…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | bdf6366fc31309ce | 0.300 | ". Newer implementations (as of 2023[update]) can also incorporate specific augmentation modules with…" |

**Métriques** : sources=1 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation'], types=1 ['web'], longueur moy.=661 car., redondance=0/6 paires

### Résultats MMR (fetch_k=20, λ=0.5)

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 07e8370ce3b2b2e7 | n/a | ". RAG can be used on unstructured (usually text), semi-structured, or structured data (for example k…" |
| 2 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | a1adb0fa63e553d7 | n/a | ". Sparse vectors, which encode the identity of a word, are typically dictionary-length and contain m…" |
| 3 | intro_rag.txt | text | bf88ed8ece84bd8d | n/a | "Les embeddings Les embeddings sont des représentations vectorielles denses des textes qui capturent …" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 1dac993553820973 | n/a | ". pp. 39–48. doi:10.1145/3397271.3401075. ISBN 978-1-4503-8016-4. ^ Wang, Yup; Conroy, John M.; Moli…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt'], types=2 ['text', 'web'], longueur moy.=607 car., redondance=0/6 paires

### Observation

MMR couvre 1 source(s) de plus → meilleure diversité ✓

---

## Requête 4 — Question précise / mot-clé rare

> **What is chunk_id and how is it computed in the ingestion pipeline?**

*Intérêt pour la comparaison* : Requête précise sur un concept interne au corpus. Cas intéressant où fetch_k de MMR peut aider ou non selon la densité du corpus.

### Résultats Cosinus

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | intro_rag.txt | text | d4bd2f799c42caa6 | 0.205 | "Architecture typique d'un système RAG Un pipeline RAG complet comprend les étapes suivantes :  1. In…" |
| 2 | langchain_notes.md | markdown | d3aa5c6656ff1284 | 0.082 | "### Text Splitters Les `TextSplitter` découpent les documents en chunks avant leur vectorisation. Le…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 10da5c1ad422961b | 0.052 | ". The redesigned language model is shown here. It has been reported that Retro is not reproducible, …" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | bdf6366fc31309ce | -0.068 | ". Newer implementations (as of 2023[update]) can also incorporate specific augmentation modules with…" |

**Métriques** : sources=3 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt', 'langchain_notes.md'], types=3 ['markdown', 'text', 'web'], longueur moy.=772 car., redondance=0/6 paires

### Résultats MMR (fetch_k=20, λ=0.5)

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | intro_rag.txt | text | d4bd2f799c42caa6 | n/a | "Architecture typique d'un système RAG Un pipeline RAG complet comprend les étapes suivantes :  1. In…" |
| 2 | langchain_notes.md | markdown | 3b48d88f5fe4ba1c | n/a | "## Ollama et les LLM locaux  `ChatOllama` (via `langchain-ollama`) est le client officiel pour les m…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | c2e3504c7850f1bf | n/a | ". For example, LLMs can generate misinformation even when pulling from factually correct sources if …" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | a1adb0fa63e553d7 | n/a | ". Sparse vectors, which encode the identity of a word, are typically dictionary-length and contain m…" |

**Métriques** : sources=3 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'intro_rag.txt', 'langchain_notes.md'], types=3 ['markdown', 'text', 'web'], longueur moy.=739 car., redondance=0/6 paires

### Observation

Même couverture de sources pour les deux stratégies

---

## Requête 5 — Cas limite / requête ambiguë

> **What are the limitations of local language models for enterprise use?**

*Intérêt pour la comparaison* : Sujet hors-corpus principal. Permet d'observer le comportement des retrievers quand la requête est mal couverte.

### Résultats Cosinus

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 5f2319be59165078 | 0.091 | ". pp. 8371–8384. arXiv:2301.12652. doi:10.18653/v1/2024.naacl-long.463. Retrieved 16 March 2025. ^ R…" |
| 2 | langchain_notes.md | markdown | c30500fa4c205146 | 0.085 | "# Notes sur LangChain pour le projet ResearchPal  ## Qu'est-ce que LangChain ?  LangChain est un fra…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | f609018d443da254 | 0.080 | ". "Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study". Proceedi…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 103733412631d07e | 0.029 | ". Black-lettered boxes show data being changed, and blue lettering shows the algorithm performing th…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'langchain_notes.md'], types=2 ['markdown', 'web'], longueur moy.=600 car., redondance=0/6 paires

### Résultats MMR (fetch_k=20, λ=0.5)

| # | Source | Type | chunk_id | Score | Extrait (100 car.) |
| --- | --- | --- | --- | --- | --- |
| 1 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 5f2319be59165078 | n/a | ". pp. 8371–8384. arXiv:2301.12652. doi:10.18653/v1/2024.naacl-long.463. Retrieved 16 March 2025. ^ R…" |
| 2 | langchain_notes.md | markdown | c30500fa4c205146 | n/a | "# Notes sur LangChain pour le projet ResearchPal  ## Qu'est-ce que LangChain ?  LangChain est un fra…" |
| 3 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | 9c7f47231a0b646e | n/a | "recognition Whisper Facial recognition AlphaFold Text-to-image models Aurora DALL-E Firefly Flux GPT…" |
| 4 | https://en.wikipedia.org/wiki/Retrieval-augmented_generation | web | ca8420e58189edb9 | n/a | ". These documents supplement information from the LLM's pre-existing training data.[2] This allows L…" |

**Métriques** : sources=2 ['https://en.wikipedia.org/wiki/Retrieval-augmented_generation', 'langchain_notes.md'], types=2 ['markdown', 'web'], longueur moy.=612 car., redondance=0/6 paires

### Observation

Même couverture de sources pour les deux stratégies

---

## Synthèse globale

| Stratégie | Sources distinctes (moy.) | Types (moy.) | Longueur moy. (car.) | Redondance (paires moy.) |
| --- | --- | --- | --- | --- |
| Cosinus | 1.8 | 1.8 | 641 | 0.0 |
| MMR | 2.2 | 2.2 | 654 | 0.0 |
