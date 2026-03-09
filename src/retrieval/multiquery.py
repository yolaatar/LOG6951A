# multiquery.py — Multi-Query Retrieval avec RRF (T4)
#
# Stratégie :
#   1. Générer N reformulations de la requête via le LLM
#   2. Récupérer des chunks pour chaque variante (cosine ou MMR)
#   3. Fusionner avec Reciprocal Rank Fusion (RRF)
#   4. Dédupliquer par chunk_id
#   5. Fallback sur la requête brute seule si le LLM échoue

from collections import defaultdict
from typing import List, Tuple

from langchain_core.documents import Document

from config import MULTIQUERY_VARIANTS, RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA


# ── génération des variantes ─────────────────────────────────────────────────

_VARIANT_PROMPT = """\
Génère {n} reformulations différentes de la question suivante.
Objectif : améliorer le recall en couvrant des formulations alternatives.
Retourne UNIQUEMENT les {n} questions reformulées, une par ligne, sans numérotation.

Question originale : {question}"""


def generate_query_variants(question: str, llm, n: int = MULTIQUERY_VARIANTS) -> List[str]:
    """
    Demande au LLM n reformulations de la question.
    Retourne une liste vide en cas d'erreur (déclenchera le fallback).
    """
    prompt = _VARIANT_PROMPT.format(n=n, question=question)
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        variants = [
            line.strip().lstrip("-•123456789). ").strip()
            for line in text.strip().splitlines()
            if line.strip()
        ]
        variants = [v for v in variants if len(v) > 10][:n]
        return variants
    except Exception as exc:
        print(f"  [multiquery] Erreur LLM lors de la génération des variantes : {exc}")
        return []


# ── RRF ─────────────────────────────────────────────────────────────────────

def rrf_fuse(
    results_by_query: List[List[Document]],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion.

    Score(d) = Σ_q  1 / (rrf_k + rank_q(d))

    Déduplique par chunk_id, retourne la liste triée par score décroissant.
    """
    scores: dict = defaultdict(float)
    doc_by_id: dict = {}

    for results in results_by_query:
        for rank, doc in enumerate(results, 1):
            chunk_id = doc.metadata.get("chunk_id") or doc.page_content[:40]
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            doc_by_id[chunk_id] = doc

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [doc_by_id[cid] for cid in sorted_ids]


# ── pipeline multi-query complet ─────────────────────────────────────────────

def multiquery_retrieve(
    vectorstore,
    question: str,
    llm,
    k: int = RETRIEVAL_TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    strategy: str = "cosine",
) -> Tuple[List[Document], List[str]]:
    """
    Pipeline Multi-Query avec RRF.

    Retourne (docs_fusionnés[:k], variantes_utilisées).
    Fallback sur requête brute si le LLM ne génère aucune variante.
    """
    from retrieval.cosine_retriever import cosine_search_with_scores
    from retrieval.mmr_retriever import mmr_search

    print(f"  [multiquery] Génération de {MULTIQUERY_VARIANTS} variantes...")
    variants = generate_query_variants(question, llm, n=MULTIQUERY_VARIANTS)

    if not variants:
        print("  [multiquery] Fallback : requête brute uniquement")

    all_queries = [question] + variants
    print(f"  [multiquery] {len(all_queries)} requête(s) : 1 brute + {len(variants)} variante(s)")

    results_by_query: List[List[Document]] = []
    for q in all_queries:
        if strategy == "mmr":
            docs = mmr_search(vectorstore, q, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        else:
            docs = [d for d, _ in cosine_search_with_scores(vectorstore, q, k=k)]
        results_by_query.append(docs)

    fused = rrf_fuse(results_by_query)
    return fused[:k], variants


# stub de compatibilité
def get_multiquery_retriever(vectorstore, llm):
    raise RuntimeError("Utilisez multiquery_retrieve() à la place.")
