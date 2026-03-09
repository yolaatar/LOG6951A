# app.py — interface Streamlit de ResearchPal
# usage : streamlit run src/ui/app.py

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RETRIEVAL_TOP_K

# TODO: brancher la chaîne RAG quand elle sera prête (Tâche 2)
# from ingestion.indexer import load_vectorstore
# from retrieval.cosine_retriever import get_cosine_retriever
# from retrieval.mmr_retriever import get_mmr_retriever
# from rag.chain import build_rag_chain, get_llm
# from rag.memory import ConversationMemory


# initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "cosine"


# sidebar
with st.sidebar:
    st.title("ResearchPal")
    st.markdown("---")

    st.subheader("Paramètres de retrieval")
    retrieval_mode = st.radio(
        "Mode de retrieval",
        options=["cosine", "mmr"],
        index=0 if st.session_state.retrieval_mode == "cosine" else 1,
    )
    st.session_state.retrieval_mode = retrieval_mode

    top_k = st.slider("Nombre de documents (top-k)", min_value=1, max_value=10, value=RETRIEVAL_TOP_K)

    st.markdown("---")
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Mode actif : **{retrieval_mode}**  |  Top-k : **{top_k}**")


# interface principale
st.title("ResearchPal — Assistant RAG")
st.caption("Posez vos questions sur les documents indexés.")

# afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# traitement d'une nouvelle question
if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            # TODO: remplacer par la vraie chaîne RAG
            response = (
                f"**[Placeholder]** Votre question : *{prompt}*\n\n"
                f"Mode : {st.session_state.retrieval_mode} | Top-k : {top_k}\n\n"
                "La chaîne RAG sera branchée à la Tâche 2."
            )
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
