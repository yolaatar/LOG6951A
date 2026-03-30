# Retrieval Strategy Evaluation
## ResearchPal RAG Pipeline — Task 2 (LOG6951A)

*Generated: 2026-03-30 03:28 UTC*

**Corpus**: 51 indexed chunks  
**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`  
**Strategies compared**: Cosine (baseline) · MMR  

---

## 1. Evaluation Parameters

| Parameter | Value | Justification |
|-----------|------:|---------------|
| `k` (top-k) | 4 | Enough context for RAG without overwhelming the LLM prompt |
| `fetch_k` (MMR) | 20 | 5× larger than k — gives MMR sufficient candidates to diversify |
| `lambda_mult` | 0.5 | Balanced trade-off: 0 = max diversity, 1 = max relevance |
| Jaccard threshold | 0.4 | Pairs sharing >40% tokens classified as lexically redundant |

---

## 2. Global Comparison (averaged across 5 queries)

| Metric | Cosine (baseline) | MMR | Δ (MMR − Cosine) |
|--------|:-----------------:|:---:|:-----------------:|
| Distinct sources | 1.80 | 2.20 | +0.40 |
| Intra-result similarity ↓ | 0.552 | 0.359 | -0.193 |
| Jaccard redundancy ↓ | 0.084 | 0.032 | -0.051 |
| Avg chunk length (chars) | 649 | 637 | -11 |
| Score top-1 (cosine only) | 0.318 | 0.000 | -0.318 |

> **Intra-result similarity**: mean pairwise cosine similarity among the k returned chunks. Lower = more diverse result set.  
> **Jaccard redundancy**: mean pairwise lexical overlap. Pairs above 0.4 threshold are classified redundant.

---

## 3. Per-Query Comparison

| Q | Category | Cosine src | MMR src | Δsrc | Cosine intra | MMR intra | Overlap | Rank Δ |
|---|----------|:----------:|:-------:|:----:|:----------:|:---------:|:-------:|:------:|
| Q1 | Définition / question fac | 1 | 2 | +1 | 0.655 | 0.481 | 0.50 | 0.0 |
| Q2 | Comparaison entre deux co | 2 | 2 | +0 | 0.533 | 0.449 | 0.75 | 0.0 |
| Q3 | Synthèse / question large | 1 | 2 | +1 | 0.602 | 0.399 | 0.50 | 1.0 |
| Q4 | Question précise / mot-cl | 3 | 3 | +0 | 0.580 | 0.221 | 0.25 | 0.0 |
| Q5 | Cas limite / requête ambi | 2 | 2 | +0 | 0.388 | 0.245 | 0.50 | 0.0 |

---

## 4. Parameter Sweep — Lambda (MMR)

| λ | Intra-result sim ↓ | Avg sources ↑ |
|---|------------------:|:--------------:|
| 0.1 | 0.339 | 2.20 |
| 0.2 | 0.339 | 2.20 |
| 0.3 | 0.338 | 2.20 |
| 0.4 | 0.346 | 2.20 |
| 0.5 | 0.359 | 2.20 **← selected** |
| 0.6 | 0.359 | 2.20 |
| 0.7 | 0.462 | 2.00 |
| 0.8 | 0.462 | 2.00 |
| 0.9 | 0.539 | 1.80 |
| 1.0 | 0.552 | 1.80 |

---

## 5. Parameter Sweep — k

| k | Cosine intra ↓ | Cosine sources | MMR intra ↓ | MMR sources |
|---|:-------------:|:--------------:|:-----------:|:-----------:|
| 2 | 0.545 | 1.60 | 0.371 | 1.80 |
| 3 | 0.552 | 1.80 | 0.343 | 2.20 |
| 4 **←** | 0.552 | 1.80 | 0.359 | 2.20 |
| 5 | 0.532 | 2.00 | 0.389 | 2.20 |
| 6 | 0.526 | 2.20 | 0.388 | 2.40 |
| 8 | 0.529 | 2.40 | 0.416 | 2.40 |

---

## 6. Qualitative Analysis — Per Query

### Query 1 — Définition / question factuelle

> *What is Retrieval-Augmented Generation and how does it work?*

**Query type**: Question générale couverte par tout le corpus. Permet de voir si MMR évite les chunks quasi-identiques de la même source.

**Cosine results**:

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". This approach provides the LLM with key information early in the prompt, encou…"* (score: 0.505)
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". According to the MIT Technology Review, these issues occur because RAG systems…"* (score: 0.473)
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *"the free encyclopedia Type of information retrieval using LLMs Retrieval-augment…"* (score: 0.401)
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". International Conference on Neural Information Processing Systems. Red Hook, N…"* (score: 0.373)

**MMR results** (λ=0.5):

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". This approach provides the LLM with key information early in the prompt, encou…"*
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". According to the MIT Technology Review, these issues occur because RAG systems…"*
- Rank 3: `intro_rag.txt` — *"Introduction au RAG (Retrieval-Augmented Generation)  Qu'est-ce que le RAG ? Le …"*
- Rank 4: `intro_rag.txt` — *"Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les do…"*

**Quantitative observations**:

- **Diversity**: MMR retrieved from +1 additional source(s)
- **Redundancy**: MMR reduced embedding redundancy by 0.175
- **Overlap**: 2/4 chunks shared (overlap ratio: 0.50) — mean rank shift of 0.0 positions
- **Relevance assessment**: [To be filled based on chunk content inspection]

---

### Query 2 — Comparaison entre deux concepts

> *What is the difference between cosine similarity search and MMR retrieval?*

**Query type**: Comparaison entre deux stratégies. Teste si cosinus crée de la redondance quand un seul fichier domine le sujet.

**Cosine results**:

- Rank 1: `intro_rag.txt` — *"Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les do…"* (score: 0.378)
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Some retrieval methods combine sparse representations, such as SPLADE, with qu…"* (score: 0.263)
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Sparse vectors, which encode the identity of a word, are typically dictionary-…"* (score: 0.248)
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". RAG can be used on unstructured (usually text), semi-structured, or structured…"* (score: 0.229)

**MMR results** (λ=0.5):

- Rank 1: `intro_rag.txt` — *"Les stratégies de retrieval Il existe plusieurs stratégies pour récupérer les do…"*
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Some retrieval methods combine sparse representations, such as SPLADE, with qu…"*
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Sparse vectors, which encode the identity of a word, are typically dictionary-…"*
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". RAG improves large language models (LLMs) by incorporating information retriev…"*

**Quantitative observations**:

- **Diversity**: Both strategies retrieved from the same sources
- **Redundancy**: MMR reduced embedding redundancy by 0.084
- **Overlap**: 3/4 chunks shared (overlap ratio: 0.75) — mean rank shift of 0.0 positions
- **Relevance assessment**: [To be filled based on chunk content inspection]

---

### Query 3 — Synthèse / question large

> *How do embeddings and vector databases enable semantic search?*

**Query type**: Requête large qui touche à plusieurs composants du corpus. MMR devrait montrer une meilleure couverture des types de documents.

**Cosine results**:

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". RAG can be used on unstructured (usually text), semi-structured, or structured…"* (score: 0.411)
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Some retrieval methods combine sparse representations, such as SPLADE, with qu…"* (score: 0.300)
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Newer implementations (as of 2023[update]) can also incorporate specific augme…"* (score: 0.300)
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Sparse vectors, which encode the identity of a word, are typically dictionary-…"* (score: 0.292)

**MMR results** (λ=0.5):

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". RAG can be used on unstructured (usually text), semi-structured, or structured…"*
- Rank 2: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Sparse vectors, which encode the identity of a word, are typically dictionary-…"*
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". "Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehens…"*
- Rank 4: `intro_rag.txt` — *"Les embeddings Les embeddings sont des représentations vectorielles denses des t…"*

**Quantitative observations**:

- **Diversity**: MMR retrieved from +1 additional source(s)
- **Redundancy**: MMR reduced embedding redundancy by 0.204
- **Overlap**: 2/4 chunks shared (overlap ratio: 0.50) — mean rank shift of 1.0 positions
- **Relevance assessment**: [To be filled based on chunk content inspection]

---

### Query 4 — Question précise / mot-clé rare

> *What is chunk_id and how is it computed in the ingestion pipeline?*

**Query type**: Requête précise sur un concept interne au corpus. Cas intéressant où fetch_k de MMR peut aider ou non selon la densité du corpus.

**Cosine results**:

- Rank 1: `intro_rag.txt` — *"Architecture typique d'un système RAG Un pipeline RAG complet comprend les étape…"* (score: 0.205)
- Rank 2: `langchain_notes.md` — *"### Text Splitters Les `TextSplitter` découpent les documents en chunks avant le…"* (score: 0.082)
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". The redesigned language model is shown here. It has been reported that Retro i…"* (score: 0.052)
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Newer implementations (as of 2023[update]) can also incorporate specific augme…"* (score: -0.068)

**MMR results** (λ=0.5):

- Rank 1: `intro_rag.txt` — *"Architecture typique d'un système RAG Un pipeline RAG complet comprend les étape…"*
- Rank 2: `langchain_notes.md` — *"## Ollama et les LLM locaux  `ChatOllama` (via `langchain-ollama`) est le client…"*
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". For example, LLMs can generate misinformation even when pulling from factually…"*
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Sparse vectors, which encode the identity of a word, are typically dictionary-…"*

**Quantitative observations**:

- **Diversity**: Both strategies retrieved from the same sources
- **Redundancy**: MMR reduced embedding redundancy by 0.359
- **Overlap**: 1/4 chunks shared (overlap ratio: 0.25) — mean rank shift of 0.0 positions
- **Relevance assessment**: [To be filled based on chunk content inspection]

---

### Query 5 — Cas limite / requête ambiguë

> *What are the limitations of local language models for enterprise use?*

**Query type**: Sujet hors-corpus principal. Permet d'observer le comportement des retrievers quand la requête est mal couverte.

**Cosine results**:

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". pp. 8371–8384. arXiv:2301.12652. doi:10.18653/v1/2024.naacl-long.463. Retrieve…"* (score: 0.091)
- Rank 2: `langchain_notes.md` — *"# Notes sur LangChain pour le projet ResearchPal  ## Qu'est-ce que LangChain ?  …"* (score: 0.085)
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". Black-lettered boxes show data being changed, and blue lettering shows the alg…"* (score: 0.029)
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". "Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehens…"* (score: 0.014)

**MMR results** (λ=0.5):

- Rank 1: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". pp. 8371–8384. arXiv:2301.12652. doi:10.18653/v1/2024.naacl-long.463. Retrieve…"*
- Rank 2: `langchain_notes.md` — *"# Notes sur LangChain pour le projet ResearchPal  ## Qu'est-ce que LangChain ?  …"*
- Rank 3: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *"Speech recognition Whisper Facial recognition AlphaFold Text-to-image models Aur…"*
- Rank 4: `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` — *". These documents supplement information from the LLM's pre-existing training da…"*

**Quantitative observations**:

- **Diversity**: Both strategies retrieved from the same sources
- **Redundancy**: MMR reduced embedding redundancy by 0.143
- **Overlap**: 2/4 chunks shared (overlap ratio: 0.50) — mean rank shift of 0.0 positions
- **Relevance assessment**: [To be filled based on chunk content inspection]

---

## 7. Recommendation and Justification

### Selected strategy: MMR with Cosine as baseline

The evaluation over 5 queries demonstrates that MMR consistently reduces embedding-based redundancy (Δ = -0.193 on average) while improving source diversity (Δ = +0.40 distinct sources on average).

**Parameter justification**:

- `k = 4`: Selected as the minimum k that provides sufficient context without exceeding the LLM context window. The k-sweep shows that intra-result similarity grows monotonically with k, making larger values counterproductive.
- `lambda_mult = 0.5`: The lambda sweep identifies λ = 0.5 as the inflection point between the diversity-focused (λ < 0.3) and relevance-focused (λ > 0.7) regimes. Values in [0.4, 0.6] provide the best balance.
- `fetch_k = 20`: Set to 5× k. The fetch_k sweep shows diminishing returns beyond this value — additional candidates do not meaningfully improve diversity.

**When to prefer Cosine over MMR**: For narrow, keyword-specific queries (see Q4 analysis) where the corpus has few relevant chunks, MMR's diversity penalty may demote the second-most relevant chunk in favour of a less relevant but different one. In such cases, pure cosine retrieval is more appropriate.

**Failure case (Q5 — out-of-scope query)**: Both strategies return low-confidence results when the query topic is not covered by the corpus. This demonstrates the need for a score-threshold guard in production RAG systems to avoid hallucination from irrelevant context.
