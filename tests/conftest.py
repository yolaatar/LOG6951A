# tests/conftest.py — configuration pytest globale

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests nécessitant Ollama + vectorstore (lents)"
    )
