# observability/tracing.py — instrumentation Arize Phoenix (T4)
#
# Outil retenu : Arize Phoenix
#   - Installation : pip install arize-phoenix openinference-instrumentation-langchain
#   - Aucun compte cloud ni clé API requis
#   - Interface locale sur http://localhost:6006
#   - Pas de Docker (contrairement à Langfuse)
#
# Ce module expose deux fonctions :
#   - setup_tracing()  : à appeler UNE FOIS au démarrage de l'application.
#   - get_tracer()     : retourne le tracer OpenTelemetry pour les spans manuels.
#
# Spans nommés par nœud LangGraph (requis par l'énoncé T4) :
#   - "retrieve"         : durée + nb docs récupérés
#   - "grade_documents"  : durée + nb docs pertinents / total
#   - "generate"         : durée + nb tokens estimés
#   - "transform_query"  : durée + requête reformulée
#   - "route_query"      : durée + outil sélectionné
#   - "web_search_node"  : durée + nb résultats
#
# Métriques visibles dans Phoenix (requis par l'énoncé T4) :
#   - latence totale de la requête
#   - latence par nœud (via spans nommés)
#   - tokens consommés par appel LLM (via LangChain instrumentation)
#   - nombre d'itérations du cycle correctif (attribut retry_count)
#   - temps d'exécution des appels d'outils (via spans des tools)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tracer = None
_phoenix_session = None


def setup_tracing(project_name: str = "researchpal-v2") -> bool:
    """Configure Arize Phoenix et instrumente LangChain/LangGraph.

    Retourne True si l'instrumentation a réussi, False si Phoenix n'est pas installé.
    L'application fonctionne normalement même sans observabilité.
    """
    global _tracer, _phoenix_session

    try:
        import phoenix as px
    except ImportError:
        print("[observability] arize-phoenix non installé → tracing désactivé")
        print("  → pip install arize-phoenix openinference-instrumentation-langchain")
        return False

    try:
        # Lance le serveur Phoenix local (http://localhost:6006)
        _phoenix_session = px.launch_app()
        print(f"[observability] Phoenix démarré → {_phoenix_session.url}")
    except Exception as exc:
        print(f"[observability] Impossible de démarrer Phoenix : {exc}")
        print("  → Vérifiez qu'aucun autre processus n'occupe le port 6006")
        return False

    # Configuration OpenTelemetry → Phoenix OTLP endpoint
    try:
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry import trace

        phoenix_endpoint = "http://localhost:6006/v1/traces"
        exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
        provider = trace_sdk.TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer(project_name)

    except ImportError:
        # Fallback : utilise le tracer global sans export explicite
        from opentelemetry import trace
        _tracer = trace.get_tracer(project_name)

    # Instrumentation automatique de LangChain (capture LLM calls + token counts)
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        print("[observability] LangChain instrumenté (LLM calls + tokens)")
    except ImportError:
        print("[observability] openinference-instrumentation-langchain non installé")
        print("  → pip install openinference-instrumentation-langchain")

    print(f"[observability] Projet Phoenix : '{project_name}'")
    print(f"[observability] Interface : http://localhost:6006")
    return True


def get_tracer():
    """Retourne le tracer OpenTelemetry pour les spans manuels dans les nœuds.

    Usage dans un nœud LangGraph :
        from observability.tracing import get_tracer
        tracer = get_tracer()

        def retrieve_node(state):
            with tracer.start_as_current_span("retrieve") as span:
                span.set_attribute("input.query", state["retrieval_query"])
                # ... logique de retrieval ...
                span.set_attribute("output.docs_count", len(docs))
            return {"documents": docs}
    """
    global _tracer
    if _tracer is None:
        # Tracer no-op si setup_tracing() n'a pas été appelé
        from opentelemetry import trace
        _tracer = trace.get_tracer("researchpal-v2")
    return _tracer


def get_phoenix_url() -> str:
    """Retourne l'URL de l'interface Phoenix si active."""
    if _phoenix_session is not None:
        return str(_phoenix_session.url)
    return "http://localhost:6006"
