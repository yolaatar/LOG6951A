# app.py — interface Streamlit ResearchPal v2 (TP1 + TP2)
# usage : streamlit run src/ui/app.py

import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RETRIEVAL_TOP_K, MMR_FETCH_K, MMR_LAMBDA
from rag.chain import build_rag_pipeline, RAGPipeline

DATA_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

STRATEGIES = {
    "Cosine (rapide)":   ("cosine", False),
    "MMR (diversifié)":  ("mmr",    False),
    "MultiQuery + Cosine": ("cosine", True),
    "MultiQuery + MMR":    ("mmr",   True),
}


# ── singletons Streamlit ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du pipeline RAG (TP1)...")
def get_pipeline() -> RAGPipeline:
    return build_rag_pipeline(k=RETRIEVAL_TOP_K, fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA)


@st.cache_resource(show_spinner="Chargement du pipeline agentique (TP2)...")
def get_agent():
    from agent.graph import get_agent_graph
    return get_agent_graph()


# ── état de session ──────────────────────────────────────────────────────────

if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "thread_id"      not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ResearchPal v2")
    st.markdown("---")

    # ── mode : TP1 ou TP2 ────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        options=["Pipeline RAG (TP1)", "Agent LangGraph (TP2)"],
        index=0,
        help=(
            "**Pipeline RAG (TP1)** : pipeline linéaire, plusieurs stratégies "
            "de retrieval et de gestion du contexte.\n\n"
            "**Agent LangGraph (TP2)** : state machine Corrective RAG avec "
            "sélection dynamique d'outils (corpus ou web) et mémoire agentique."
        ),
    )
    use_agent = (mode == "Agent LangGraph (TP2)")

    st.markdown("---")

    if not use_agent:
        # ── options TP1 ──────────────────────────────────────────────────────
        st.subheader("Stratégie de retrieval")
        strategy_label = st.radio("Mode", options=list(STRATEGIES.keys()), index=0)
        strategy, use_multiquery = STRATEGIES[strategy_label]

        st.markdown("---")

        st.subheader("Gestion du contexte")
        context_mode = st.radio(
            "Mode",
            options=["Aucun", "Heuristiques", "Concaténation", "Réécriture (LLM)"],
            index=0,
            help=(
                "**Aucun** : question brute pour le retrieval.\n\n"
                "**Heuristiques** : détection de coréférence + enrichissement.\n\n"
                "**Concaténation** : préfixe les 2 dernières questions in-scope.\n\n"
                "**Réécriture (LLM)** : reformule la question (+1 appel LLM)."
            ),
        )
        use_heuristic_context = (context_mode == "Heuristiques")
        use_concat_context    = (context_mode == "Concaténation")
        use_query_rewriting   = (context_mode == "Réécriture (LLM)")
    else:
        # ── info mode agentique ──────────────────────────────────────────────
        st.info(
            "🤖 **Mode Agent actif**\n\n"
            "L'agent sélectionne automatiquement l'outil selon votre question :\n"
            "- **search_corpus** pour les questions liées au corpus RAG/LangChain\n"
            "- **web_search** pour les requêtes hors-corpus ou temps réel\n\n"
            "Le cycle Corrective RAG regrade les docs jusqu'à 3× si nécessaire."
        )
        st.caption(f"Thread ID : `{st.session_state.thread_id[:8]}…`")

    st.markdown("---")

    # ── ajout de documents ────────────────────────────────────────────────────
    st.subheader("Ajouter des documents")
    uploaded = st.file_uploader(
        "Fichiers (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        saved = []
        for f in uploaded:
            dest = DATA_RAW / f.name
            dest.write_bytes(f.read())
            saved.append(f.name)
        st.success(f"Sauvegardé : {', '.join(saved)}")

        if st.button("Ré-indexer les documents"):
            with st.spinner("Ingestion en cours..."):
                run_script = Path(__file__).resolve().parent.parent / "ingestion" / "run_ingestion.py"
                result = subprocess.run(
                    [sys.executable, str(run_script), "--reset"],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    st.cache_resource.clear()
                    st.success("Vectorstore mis à jour — rechargez (F5).")
                else:
                    st.error(f"Erreur d'ingestion :\n{result.stderr[:500]}")

    st.markdown("---")

    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = __import__("uuid").uuid4().__str__()
        try:
            get_pipeline().reset_memory()
        except Exception:
            pass
        st.rerun()


# ── titre principal ──────────────────────────────────────────────────────────

st.title("ResearchPal — Assistant RAG")
badge = "🤖 Agent LangGraph (TP2)" if use_agent else "🔍 Pipeline RAG (TP1)"
st.caption(badge)

# ── chargement du composant actif ────────────────────────────────────────────

pipeline = None
agent    = None

if use_agent:
    try:
        agent = get_agent()
    except RuntimeError as e:
        st.error(f"**Erreur agent** : {e}")
        st.info("Vérifiez qu'Ollama est lancé (`ollama serve`).")
else:
    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        st.error(f"**Erreur pipeline** : {e}")
        st.info("Vérifiez qu'Ollama est lancé (`ollama serve`).")

# ── historique ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
        if msg.get("agent_meta"):
            meta = msg["agent_meta"]
            with st.expander(f"Détails agent — outil: {meta.get('tool_used','?')} | cycles: {meta.get('retry_count',0)}"):
                st.caption(f"Requête effective : `{meta.get('retrieval_query','')}`")

# ── nouvelle question ─────────────────────────────────────────────────────────

if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── mode agent (TP2) ─────────────────────────────────────────────────
        if use_agent:
            if agent is None:
                st.error("Agent non disponible. Vérifiez Ollama.")
            else:
                with st.spinner("Agent en cours de raisonnement..."):
                    try:
                        from agent.state import AgentState
                        from memory_v2.episodic import maybe_store_episode

                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        initial: AgentState = {
                            "question":       prompt,
                            "retrieval_query": prompt,
                            "documents":      [],
                            "relevant_docs":  [],
                            "generation":     "",
                            "retry_count":    0,
                            "grade_decision": "",
                            "tool_used":      "",
                            "web_results":    None,
                        }
                        result = agent.invoke(initial, config=config)

                        answer      = result.get("generation", "Aucune réponse générée.")
                        tool_used   = result.get("tool_used", "?")
                        retry_count = result.get("retry_count", 0)
                        docs        = result.get("relevant_docs") or result.get("documents") or []
                        sources     = list({
                            (s if s.startswith("http") else Path(s).name)
                            for doc in docs
                            for s in [doc.metadata.get("source", "")]
                            if s
                        })
                        retrieval_query = result.get("retrieval_query", "")

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for s in sources:
                                    st.markdown(f"- {s}")

                        with st.expander(
                            f"Détails agent — outil: **{tool_used}** | "
                            f"cycles correctifs: **{retry_count}**"
                        ):
                            st.caption(f"Requête effective : `{retrieval_query}`")
                            if retry_count > 0:
                                st.warning(f"⚠ Cycle Corrective RAG déclenché {retry_count} fois.")

                        # Mémoire épisodique : stocker si critères réunis (T3)
                        maybe_store_episode(
                            question=prompt,
                            answer=answer,
                            sources=sources,
                            tool_used=tool_used,
                            retry_count=retry_count,
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "agent_meta": {
                                "tool_used":       tool_used,
                                "retry_count":     retry_count,
                                "retrieval_query": retrieval_query,
                            },
                        })

                    except Exception as exc:
                        err_msg = f"Erreur agent : {exc}"
                        st.error(err_msg)
                        st.session_state.messages.append({
                            "role": "assistant", "content": err_msg, "sources": [],
                        })

        # ── mode pipeline TP1 ────────────────────────────────────────────────
        else:
            if pipeline is None:
                st.error("Pipeline non disponible. Vérifiez Ollama.")
            else:
                with st.spinner("Recherche et génération..."):
                    try:
                        result = pipeline.answer(
                            prompt,
                            strategy=strategy,
                            use_multiquery=use_multiquery,
                            use_heuristic_context=use_heuristic_context,
                            use_concat_context=use_concat_context,
                            use_query_rewriting=use_query_rewriting,
                        )
                        st.markdown(result.answer)
                        if result.sources:
                            with st.expander(f"Sources ({len(result.sources)})"):
                                for s in result.sources:
                                    st.markdown(f"- {s}")
                        if result.query_variants:
                            with st.expander(f"Variantes MultiQuery ({len(result.query_variants)})"):
                                for v in result.query_variants:
                                    st.markdown(f"- {v}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        })
                    except Exception as exc:
                        err_msg = f"Erreur lors de la génération : {exc}"
                        st.error(err_msg)
                        st.session_state.messages.append({
                            "role": "assistant", "content": err_msg, "sources": [],
                        })
