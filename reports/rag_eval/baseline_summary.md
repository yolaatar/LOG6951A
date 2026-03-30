# Évaluation du pipeline RAG — Mode Baseline (sans enrichissement de contexte)
## ResearchPal · Tâche 4 — LOG6951A

*Généré le 2026-03-29 — Source : `context_mode_eval.txt` (mode `Aucun`)*

---

## 1. Objectif et périmètre

Ce rapport évalue le comportement du pipeline RAG en mode **baseline pur** :

| Paramètre de contexte | Valeur |
|---|---|
| `use_heuristic_context` | `False` |
| `use_concat_context` | `False` |
| `use_query_rewriting` | `False` |

Dans ce mode, la requête de retrieval = question brute de l'utilisateur.
L'historique des 3 derniers tours est néanmoins injecté dans le prompt LLM (côté génération),
mais **aucun enrichissement n'est appliqué côté retrieval**.

> Objectif : identifier les cas d'échec structurels qui justifient les techniques
> d'optimisation de requête évaluées en Tâche 4.

---

## 2. Corpus et configuration

| Paramètre | Valeur |
|---|---|
| Modèle LLM | `mistral:7b-instruct` (Ollama local) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384 dim) |
| Vectorstore | ChromaDB, 51 chunks |
| `k` | 4 |
| Seuil hors-périmètre | `score < 0.1` ET pas de mot-clé de domaine |
| Dialogues testés | 4 dialogues × 3 tours = 12 tours au total |

---

## 3. Résultats par dialogue

### VAL1 — Pronoms anaphoriques (3 tours)

> Tour 2 : pronom `ses`. Tour 3 : démonstratif `ces problèmes`.

| Tour | Question | Score cosinus | Docs | Rejeté | Grounding | Verdict |
|------|----------|:-------------:|:----:|:------:|:---------:|---------|
| 1 | "Qu'est-ce que RAG et comment fonctionne-t-il ?" | 0.6647 | 4 | non | 0.714 | ✓ Succès |
| 2 | "Quelles sont ses principales limites par rapport aux LLMs classiques ?" | 0.0245 | 4 | non | 0.714 | ⚠ Partiel |
| 3 | "Comment MMR aide-t-il à résoudre certains de ces problèmes de redondance ?" | 0.0912 | 4 | non | 0.571 | ⚠ Partiel |

**Analyse :**
Le tour 2 n'est pas rejeté car `"llm"` est un mot-clé de domaine (`_is_domain_question` = True).
Cependant, le score cosinus chute à **0.0245** (−95 % par rapport au tour 1) : la requête anaphorique
`"Quelles sont ses principales limites"` n'ancre pas correctement le retrieval dans le sujet RAG.
Les 4 documents récupérés sont partiellement hors-sujet — la réponse est correcte seulement
parce que `"llm"` aligne fortuitement vers des chunks pertinents.
Au tour 3, `"ces problèmes"` sans sujet explicite produit un score de 0.0912 (proche du seuil).

---

### VAL2 — Questions courtes sous le seuil de longueur (3 tours) — ★ CAS CRITIQUE

> Tours 2 et 3 : < 9 mots, aucun mot-clé de domaine.

| Tour | Question | Score cosinus | Docs | Rejeté | Grounding | Verdict |
|------|----------|:-------------:|:----:|:------:|:---------:|---------|
| 1 | "Explique-moi la stratégie MMR dans un pipeline RAG." | 0.2259 | 4 | non | 0.625 | ✓ Succès |
| **2** | **"Quels sont ses paramètres clés ?"** | **0.0240** | **0** | **OUI ✗** | **0.000** | **✗ Échec** |
| **3** | **"Et comment choisir la valeur de lambda ?"** | **0.0958** | **0** | **OUI ✗** | **0.000** | **✗ Échec** |

**Analyse — question de suivi (tour 2) :**
`"Quels sont ses paramètres clés ?"` est la question de suivi emblématique du mode baseline.
Le pronom `"ses"` est anaphorique (réfère à MMR), mais sans enrichissement de la requête :

1. Score cosinus = **0.0240** — en dessous du seuil 0.1
2. Aucun mot-clé de domaine dans la question brute → `_is_domain_question` = False
3. `topic_continuity` = False (heuristique désactivée)
4. → **Rejet** : `"Votre question semble hors périmètre du corpus indexé"`

La réponse est fonctionnellement fausse : le pipeline répond "hors périmètre" à une question
parfaitement légitime sur MMR, dont la réponse est explicitement présente dans le corpus.
Le même problème se répète au tour 3 avec `"Et comment choisir la valeur de lambda ?"`.

**Réponse produite par le baseline (tour 2) :**
```
**Réponse**
Votre question semble hors périmètre du corpus indexé (RAG/LangChain/docs chargés).
Je ne peux pas répondre de manière fiable sans source pertinente.

**Sources**
Aucune source pertinente récupérée.

**Limites / Incertitudes**
Le corpus actuel ne contient pas d'information suffisamment liée à cette question.
```

---

### VAL3 — Référence au tour 1 depuis le tour 3 (3 tours)

| Tour | Question | Score cosinus | Docs | Rejeté | Grounding | Verdict |
|------|----------|:-------------:|:----:|:------:|:---------:|---------|
| 1 | "Qu'est-ce que le chunking et pourquoi est-il important dans un pipeline RAG ?" | 0.3183 | 4 | non | 0.778 | ✓ Succès |
| 2 | "Quelle est la différence entre chunking fixe et chunking récursif ?" | 0.2925 | 4 | non | 0.625 | ✓ Succès |
| 3 | "En résumé, quelle stratégie recommandes-tu d'après ta première réponse ?" | 0.1454 | 4 | non | 0.800 | ⚠ Partiel |

**Analyse :**
Les tours 1 et 2 sont des questions autonomes — le baseline réussit correctement.
Le tour 3 contient une référence explicite (`"ta première réponse"`), mais ici le score cosinus
reste au-dessus du seuil (0.1454) grâce aux termes techniques du corpus présents dans la
question. La réponse produite (grounding=0.800) parle de RAG en général plutôt que de répondre
spécifiquement à "quelle stratégie recommandes-tu" en référence au chunking récursif — dérive
thématique due à un mauvais ancrage de la requête de retrieval.

---

### VAL4 — Robustesse hors-périmètre (3 tours)

| Tour | Question | Score cosinus | Docs | Rejeté | Grounding | Verdict |
|------|----------|:-------------:|:----:|:------:|:---------:|---------|
| 1 | "Comment fonctionne le retrieval par similarité cosinus dans ChromaDB ?" | 0.3987 | 4 | non | 0.750 | ✓ Succès |
| 2 | "Quels sont les avantages de la persistance locale du vectorstore ?" | 0.1598 | 4 | non | 0.727 | ⚠ Partiel |
| **3** | **"Quelle est la capitale de la France et quelle est sa population ?"** | **−0.0464** | **0** | **OUI ✓** | **0.000** | **✓ Rejet correct** |

**Analyse :**
Le rejet hors-périmètre au tour 3 fonctionne correctement même en mode baseline : la question
ne contient aucun mot-clé de domaine et le score est négatif. Le tour 2 est techniquement répondu
mais la réponse dérive sur RAG en général plutôt que sur la persistance du vectorstore — symptôme
d'un score cosinus limite (0.1598) qui récupère des chunks génériques.

---

## 4. Synthèse des métriques baseline

| Métrique | Valeur |
|---|---|
| Tours évalués | 12 |
| **Faux rejets** (suivis rejetés à tort) | **2** (VAL2 T2, VAL2 T3) |
| Rejets corrects (hors-périmètre) | 1 (VAL4 T3) |
| Rejets manqués | 0 |
| Score cosinus moyen | 0.1995 |
| Docs récupérés moyen | 3.0 / 4 |
| **Grounding moyen** | **0.525** |
| Risque hallucination détecté | 0 |
| Références à l'historique | 0 |

---

## 5. Cas de succès

| ID | Question | Raison du succès |
|----|----------|-----------------|
| VAL1 T1 | "Qu'est-ce que RAG ?" | Question autonome et explicite — score 0.66 |
| VAL2 T1 | "Explique-moi la stratégie MMR dans un pipeline RAG." | Termes techniques explicites → bon ancrage |
| VAL3 T1 | "Qu'est-ce que le chunking ?" | Question autonome, score 0.32 |
| VAL3 T2 | "Différence entre chunking fixe et récursif ?" | Question autonome avec termes explicites |
| VAL4 T1 | "Retrieval cosinus dans ChromaDB ?" | Termes techniques dans la question |
| VAL4 T3 | "Capitale de la France ?" | Rejet hors-périmètre correct |

---

## 6. Cas d'échec

| ID | Question | Score | Symptôme | Cause racine |
|----|----------|:-----:|----------|--------------|
| **VAL2 T2** | "Quels sont ses paramètres clés ?" | 0.0240 | **Rejet abusif** — 0 doc récupéré | Pronom `ses` non résolu → requête anaphorique sans ancrage |
| **VAL2 T3** | "Et comment choisir la valeur de lambda ?" | 0.0958 | **Rejet abusif** — 0 doc récupéré | Question courte (6 mots), aucun mot-clé de domaine |
| VAL1 T2 | "Quelles sont ses principales limites ?" | 0.0245 | Retrieval dégradé — score limite | `ses` non résolu, passage grâce à `"llm"` (mot-clé) |
| VAL1 T3 | "Comment MMR aide à résoudre ces problèmes ?" | 0.0912 | Grounding dégradé (0.571) | `ces problèmes` non résolu → chunks partiellement hors-sujet |
| VAL3 T3 | "Quelle stratégie d'après ta première réponse ?" | 0.1454 | Dérive thématique | Requête mal ancrée → retrieval sur RAG général |
| VAL4 T2 | "Avantages de la persistance du vectorstore ?" | 0.1598 | Réponse sur RAG général | Score limite → chunks génériques, pas vectorstore-spécifiques |

---

## 7. Diagnostic

### 7.1 Cause structurelle des échecs

Le mode baseline souffre d'un **désalignement systématique entre la requête de retrieval et l'intention conversationnelle** :

```
Question brute : "Quels sont ses paramètres clés ?"
                      ↑
          Pronom anaphorique non résolu

Embedding du pronom "ses" → vecteur générique
Score cosinus résultant : 0.0240 < seuil 0.1
→ Rejet ou récupération hors-sujet
```

L'injection de l'historique dans le prompt LLM (côté génération) **ne compense pas** l'absence
d'enrichissement côté retrieval : même si le LLM saurait résoudre l'anaphore, il ne reçoit
aucun document pertinent pour y répondre.

### 7.2 Condition de déclenchement des échecs

Un tour de suivi échoue en mode baseline si **toutes** les conditions suivantes sont réunies :

1. La question contient une anaphore ou est courte (< 9 mots)
2. Aucun mot-clé de domaine n'est présent dans la question brute
3. Le score cosinus de la requête brute est < 0.1

→ Ces conditions sont remplies par **toute question de suivi naturelle** dans un dialogue.

### 7.3 Taux de faux rejets

Sur les 8 tours de suivi évalués (tours 2 et 3, hors tours attendus rejetés) :
- **2 faux rejets soit 25 %** des questions de suivi incorrectement rejetées
- Les 6 autres passent, mais avec des scores dégradés (grounding moyen des tours de suivi : 0.486 vs 0.716 pour les tours 1)

---

## 8. Implications pour la Tâche 4

Ces résultats justifient l'évaluation de trois techniques d'enrichissement de la requête :

| Technique | Problème ciblé | Attente |
|---|---|---|
| **Heuristiques** (coréférence + continuité) | Faux rejets VAL2 T2/T3 | Préfixe la question du dernier tour in-scope → score ↑ |
| **Concaténation** | Score bas sur suivis | Préfixe les 2 dernières questions → ancrage vectoriel |
| **Réécriture LLM** | Anaphores complexes | Reformulation autonome → question explicite |

Le baseline constitue la référence basse (**grounding moyen = 0.525**, **2 faux rejets**) que
les modes enrichis doivent dépasser.
