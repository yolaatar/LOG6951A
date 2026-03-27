# app.py — interface Streamlit de ResearchPal (T5)
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
    "Cosine (rapide)": ("cosine", False),
    "MMR (diversifié)": ("mmr", False),
    "MultiQuery + Cosine": ("cosine", True),
    "MultiQuery + MMR": ("mmr", True),
}


# ── pipeline (singleton Streamlit) ──────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du pipeline RAG...")
def get_pipeline() -> RAGPipeline:
    return build_rag_pipeline(k=RETRIEVAL_TOP_K, fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA)


# ── état de session ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"|"assistant", "content": str, "sources": []}]

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ResearchPal")
    st.markdown("---")

    st.subheader("Stratégie de retrieval")
    strategy_label = st.radio(
        "Mode",
        options=list(STRATEGIES.keys()),
        index=0,
    )
    strategy, use_multiquery = STRATEGIES[strategy_label]

    st.markdown("---")

    # ── ajout de documents ────────────────────────────────────────────────
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
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    st.cache_resource.clear()
                    st.success("Vectorstore mis à jour — rechargez le pipeline (F5).")
                else:
                    st.error(f"Erreur d'ingestion :\n{result.stderr[:500]}")

    st.markdown("---")

    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        try:
            get_pipeline().reset_memory()
        except Exception:
            pass
        st.rerun()

    st.markdown("---")
    st.caption(f"Stratégie : **{strategy_label}**")


# ── titre principal ──────────────────────────────────────────────────────────

st.title("ResearchPal — Assistant RAG")
st.caption("Posez vos questions sur les documents indexés.")

# ── chargement du pipeline (avec gestion d'erreur) ───────────────────────────

pipeline = None
try:
    pipeline = get_pipeline()
except RuntimeError as e:
    st.error(f"**Erreur pipeline** : {e}")
    st.info("Vérifiez qu'Ollama est lancé (`ollama serve`) et que le modèle est disponible.")

# ── historique ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# ── nouvelle question ─────────────────────────────────────────────────────────

if prompt := st.chat_input("Votre question..."):
    print("\n[trace-ui] USER_PROMPT:")
    print(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if pipeline is None:
            st.error("Pipeline non disponible. Vérifiez Ollama.")
        else:
            with st.spinner("Recherche et génération..."):
                try:
                    result = pipeline.answer(
                        prompt,
                        strategy=strategy,
                        use_multiquery=use_multiquery,
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
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })
