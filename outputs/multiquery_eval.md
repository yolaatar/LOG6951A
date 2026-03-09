# Évaluation Multi-Query (T4)

Comparaison **Brute cosine** vs **Multi-Query + RRF** sur 5 requêtes annotées.

| # | Catégorie | Brute chunks | Brute sources | MQ chunks | MQ sources | Δ sources |
|---|-----------|:---:|:---:|:---:|:---:|:---:|
| 1 | Définition / question factuelle | 4 | 1 | 4 | 1 | +0 |
| 2 | Comparaison entre deux concepts | 4 | 2 | 4 | 2 | +0 |
| 3 | Synthèse / question large | 4 | 1 | 4 | 1 | +0 |
| 4 | Question précise / mot-clé rare | 4 | 3 | 4 | 3 | +0 |
| 5 | Cas limite / requête ambiguë | 4 | 2 | 4 | 2 | +0 |

## Détail par requête

### Q1 — Définition / question factuelle

> What is Retrieval-Augmented Generation and how does it work?

**Brute (cosine)**
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation

**Multi-Query + RRF**
- Variantes générées (3) :
  - Can you explain Retrieval-Augmented Generation and its functioning?
  - What does Retrieval-Augmented Generation entail and how does it operate?
  - How does Retrieval-Augmented Generation function, and what exactly is it?
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation

### Q2 — Comparaison entre deux concepts

> What is the difference between cosine similarity search and MMR retrieval?

**Brute (cosine)**
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, intro_rag.txt

**Multi-Query + RRF**
- Variantes générées (3) :
  - Could you explain the distinction between cosine similarity search and Maximal Marginal Relevance (MMR) retrieval?
  - What sets cosine similarity search apart from Maximal Marginal Relevance (MMR) retrieval?
  - Can you elucidate the dissimilarities between cosine similarity search and Maximal Marginal Relevance (MMR) retrieval methods?
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, intro_rag.txt

### Q3 — Synthèse / question large

> How do embeddings and vector databases enable semantic search?

**Brute (cosine)**
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation

**Multi-Query + RRF**
- Variantes générées (3) :
  - In what way do embeddings and vector databases facilitate semantic search?
  - How are semantic searches made possible by the use of embeddings and vector databases?
  - What role do embeddings and vector databases play in the process of semantic search?
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation

### Q4 — Question précise / mot-clé rare

> What is chunk_id and how is it computed in the ingestion pipeline?

**Brute (cosine)**
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, intro_rag.txt, langchain_notes.md

**Multi-Query + RRF**
- Variantes générées (3) :
  - Could you explain what chunk_id is and how it gets calculated in the data ingestion process?
  - Can you describe the role of chunk_id in the data pipeline and provide details on its computation?
  - What is the purpose of chunk_id within the ingestion pipeline and how is it determined during the data processing?
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, intro_rag.txt, langchain_notes.md

### Q5 — Cas limite / requête ambiguë

> What are the limitations of local language models for enterprise use?

**Brute (cosine)**
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, langchain_notes.md

**Multi-Query + RRF**
- Variantes générées (3) :
  - What are the drawbacks of local language models when applied in an enterprise setting?
  - What restrictions do local language models face in enterprise usage?
  - In what ways are local language models constrained for enterprise applications?
- Chunks : 4
- Sources : https://en.wikipedia.org/wiki/Retrieval-augmented_generation, langchain_notes.md
