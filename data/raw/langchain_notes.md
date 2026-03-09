# Notes sur LangChain pour le projet ResearchPal

## Qu'est-ce que LangChain ?

LangChain est un framework Python (et JavaScript) conçu pour faciliter le développement
d'applications basées sur des modèles de langage (LLM). Il fournit des abstractions et
des outils pour composer des chaînes de traitement complexes de façon modulaire et lisible.

## Composants principaux

### Document et DocumentLoader
Un `Document` LangChain est la brique de base : il contient `page_content` (le texte)
et `metadata` (un dictionnaire de métadonnées). Les `DocumentLoaders` encapsulent la
logique de chargement depuis différentes sources : PDF, web, bases de données, etc.

### Text Splitters
Les `TextSplitter` découpent les documents en chunks avant leur vectorisation.
Le `RecursiveCharacterTextSplitter` est le plus recommandé pour le texte générique :
il essaie de couper aux séparateurs naturels (paragraphes, lignes, phrases) avant
de couper au niveau des caractères.

Paramètres clés :
- `chunk_size` : taille maximale d'un chunk en caractères
- `chunk_overlap` : nombre de caractères partagés entre deux chunks adjacents
- `separators` : liste ordonnée des séparateurs à tester

### Embeddings
LangChain propose des interfaces unifiées pour de nombreux modèles d'embeddings :
- `HuggingFaceEmbeddings` (via `langchain-huggingface`) : modèles locaux gratuits
- `OpenAIEmbeddings` : modèle payant d'OpenAI
- `OllamaEmbeddings` : embeddings via l'API Ollama locale

### VectorStores
Les `VectorStore` stockent et interrogent des vecteurs. LangChain supporte Chroma,
FAISS, Pinecone, Weaviate, et de nombreux autres. L'interface est unifiée :
`as_retriever()` retourne un `Retriever` compatible avec toutes les chaînes.

## LCEL — LangChain Expression Language

La LCEL est la syntaxe privilégiée pour composer des chaînes depuis LangChain v0.2.
Elle utilise l'opérateur `|` pour chaîner les composants :

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "Qu'est-ce que le RAG ?"})
```

Les avantages de LCEL :
- Streaming natif
- Support async intégré
- Inspection et débogage simplifiés via LangSmith
- Composition modulaire et testable

## ChatPromptTemplate

Un `ChatPromptTemplate` structure l'entrée envoyée au LLM :
- `SystemMessage` : définit le comportement et le rôle de l'assistant
- `HumanMessage` : message de l'utilisateur
- `AIMessage` : réponse précédente du modèle (pour l'historique)
- `MessagesPlaceholder` : injecte une liste de messages dynamiquement (utile pour l'historique)

## Retrievers

Un `Retriever` est une interface qui, étant donné une requête textuelle, retourne
une liste de `Document` pertinents. Les deux modes principaux dans ChromaDB :

- `search_type="similarity"` : similarité cosinus pure
- `search_type="mmr"` : Maximal Marginal Relevance (pertinence + diversité)

Le `MultiQueryRetriever` encapsule un retriever existant et génère automatiquement
plusieurs variantes de la question pour maximiser le rappel.

## Gestion de la mémoire conversationnelle

Pour maintenir le contexte entre les tours de conversation, LangChain propose :
- `ChatMessageHistory` : liste simple de messages en mémoire
- `ConversationBufferMemory` : garde l'historique complet
- `ConversationSummaryMemory` : résume l'historique pour économiser les tokens

Dans un pipeline RAG avec LCEL, l'historique est passé via `MessagesPlaceholder`
dans le prompt template, ce qui donne un contrôle total sur ce qui est transmis au LLM.

## Ollama et les LLM locaux

`ChatOllama` (via `langchain-ollama`) est le client officiel pour les modèles Ollama.
Configuration minimale :

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral:7b-instruct", base_url="http://localhost:11434")
```

Avant utilisation : `ollama serve` doit être actif et le modèle doit avoir été téléchargé
avec `ollama pull mistral:7b-instruct`.

## Bonnes pratiques

- Toujours normaliser les embeddings pour utiliser la similarité cosinus correctement
- Séparer les modules (loaders, chunking, retrieval, RAG) pour faciliter les tests
- Stocker les métadonnées utiles dès l'ingestion pour pouvoir filtrer au retrieval
- Utiliser des `chunk_id` déterministes pour éviter les doublons lors des réingestions
