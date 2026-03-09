# main.py — vérification que l'environnement est bien configuré
# usage : python src/main.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, RAW_DIR, CHROMA_DIR, print_config


def check_directories() -> None:
    for directory in [DATA_DIR, RAW_DIR, CHROMA_DIR]:
        if not directory.exists():
            print(f"  [CREATION] {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  [OK] {directory}")


def main() -> None:
    print("\nResearchPal — Démarrage\n")
    print_config()

    print("\nVérification des dossiers :")
    check_directories()

    print("\n[SUCCESS] L'environnement ResearchPal est prêt.")
    print("  • Ajouter des documents dans data/raw/")
    print("  • Lancer l'ingestion : python src/ingestion/run_ingestion.py")
    print("  • Interface : streamlit run src/ui/app.py\n")


if __name__ == "__main__":
    main()
