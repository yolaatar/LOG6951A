# prompt.py — prompt RAG avec patterns Persona + Structured Output + Cognitive Verifier (T3)
#
# Patterns appliqués :
#   - Persona : ancre le LLM dans un rôle d'assistant factuel et prudent,
#     ce qui réduit les hallucinations et force la transparence sur les limites.
#   - Structured Output : impose une structure fixe (Réponse / Sources / Limites)
#     qui rend les réponses directement exploitables dans l'interface et le rapport.
#   - Cognitive Verifier : avant chaque affirmation factuelle, le modèle doit vérifier
#     si l'information est explicitement présente dans le contexte ; sinon → Limites.
#     Ajouté après comparaison avant/après : amélioration sur EC4 (Limites explicites)
#     et EC7 (Limites explicites). Calibrage historique testé et non conservé.

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ── System prompt principal ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Tu es ResearchPal, un assistant de recherche documentaire factuel et rigoureux.
Ton rôle est d'aider l'utilisateur à comprendre et exploiter des documents indexés.

Règles absolues :
1. Tu réponds UNIQUEMENT à partir des extraits fournis dans le contexte ci-dessous.
2. Tu n'inventes aucune information. Si le contexte est insuffisant, tu le signales explicitement.
3. Vérification cognitive (Cognitive Verifier) — avant de rédiger **Réponse** :
   Pour chaque affirmation factuelle, pose-toi la question :
   « Est-ce que cette information figure explicitement dans un extrait numéroté ? »
   → OUI : écris-la dans **Réponse** avec la citation [N].
   → NON : écris-la dans **Limites / Incertitudes**, JAMAIS dans **Réponse**.
   Ne génère AUCUN chiffre, score, pourcentage, date ou nom propre absent du contexte.
4. Chaque affirmation importante doit être rattachée à une source numérotée [N].
5. Tu structures TOUJOURS ta réponse en trois sections :

---
**Réponse**
<ta réponse argumentée avec citations [N]>

**Sources**
<liste numérotée des sources citées>

**Limites / Incertitudes**
<ce que le contexte ne permet pas de confirmer, ou ce qui manque>
---

{history_block}
Contexte récupéré :
{context}
"""


# ── Utilitaires de formatage ────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    """
    Formate une liste de Documents en bloc de contexte numéroté.
    Chaque chunk est présenté avec sa source et son type.
    """
    if not docs:
        return "Aucun document récupéré."

    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "source inconnue")
        if src.startswith("http"):
            src_label = src
        else:
            src_label = Path(src).name
        doc_type = doc.metadata.get("type_document", "?")
        content = doc.page_content.strip()
        parts.append(f"[{i}] ({doc_type}) {src_label}\n{content}")

    return "\n\n".join(parts)


def format_citations(docs: List[Document]) -> List[str]:
    """
    Retourne une liste de citations numérotées :
    ["[1] intro_rag.txt", "[2] https://...", ...]
    Utilisé dans la section Sources de la réponse finale.
    """
    citations = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "source inconnue")
        if src.startswith("http"):
            label = src
        else:
            label = Path(src).name
        chunk_id = doc.metadata.get("chunk_id", "?")
        citations.append(f"[{i}] {label}  (chunk_id: {chunk_id})")
    return citations


def format_history_block(history: List[Tuple[str, str]]) -> str:
    """
    Formate l'historique de conversation pour injection dans le prompt.
    history : liste de (question_utilisateur, réponse_assistant)
    """
    if not history:
        return ""
    lines = ["Historique de la conversation :"]
    for i, (q, a) in enumerate(history, 1):
        lines.append(f"Tour {i}")
        lines.append(f"  Utilisateur : {q}")
        # tronquer les réponses longues dans le prompt (garder les 300 premiers chars)
        a_short = a[:300] + "…" if len(a) > 300 else a
        lines.append(f"  Assistant   : {a_short}")
    return "\n".join(lines) + "\n\n"


# ── Construction du prompt ──────────────────────────────────────────────────

def build_rag_prompt(
    history: List[Tuple[str, str]] | None = None,
) -> ChatPromptTemplate:
    """
    Construit le ChatPromptTemplate avec historique optionnel.
    history : liste de (question, réponse) des tours précédents.
    """
    history_block = format_history_block(history or [])
    system = SYSTEM_PROMPT.replace("{history_block}", history_block)

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
