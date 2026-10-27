"""Point d'entrée de l'application (fenêtre). Utilisé par PyInstaller."""
import sys

from retranscription.interface import main

if __name__ == "__main__":
    sys.exit(main())
