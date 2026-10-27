"""Paramètres par défaut (tous modifiables en ligne de commande)."""

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


def dossier_donnees_utilisateur() -> Path:
    """Dossier propre à l'utilisateur pour les modèles et les réglages de l'application."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Retranscription"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Retranscription"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "retranscription"


# Dossier des modèles :
#  - variable d'environnement RETRANSCRIPTION_MODELES si elle existe ;
#  - application installée (PyInstaller) : dossier de données de l'utilisateur ;
#  - sinon : dossier « modeles » à côté du code (usage avec Spyder ou en ligne de commande).
if os.environ.get("RETRANSCRIPTION_MODELES"):
    MODELES = Path(os.environ["RETRANSCRIPTION_MODELES"])
elif getattr(sys, "frozen", False):
    MODELES = dossier_donnees_utilisateur() / "modeles"
else:
    MODELES = Path(__file__).resolve().parent.parent / "modeles"


@dataclass
class Parametres:
    # modèles Vosk (dossiers décompressés)
    modele: str = str(MODELES / "vosk-model-fr-0.22")            # reconnaissance de la parole
    modele_leger: str = str(MODELES / "vosk-model-small-fr-0.22")  # décodage court des fenêtres x-vecteur
    modele_spk: str = str(MODELES / "vosk-model-spk-0.4")         # extracteur de x-vecteurs
    empreinte: str = "vosk"          # "vosk" (x-vecteurs) ou "mfcc" (méthode classique)

    # micros
    reference: Optional[str] = None  # fichier dont le début = temps 0 (défaut : le premier)
    max_decalage_s: float = 1800.0   # écart maximal entre les démarrages des caméras
    poids_canaux: float = 0.0        # part du profil d'énergie des micros dans le regroupement

    # locuteurs
    nb_locuteurs: Optional[int] = None   # recommandé : nombre de personnes présentes
    nb_max: int = 20
    fenetre_s: float = 1.5
    pas_s: float = 0.75
    seuil_appariement: float = 0.7   # cosinus minimal pour reconnaître un locuteur du registre
    #                                  (à calibrer sur vos données : commande « calibrer »)
    marge_appariement: float = 0.1
    duree_min_registre_s: float = 10.0

    # segments
    pause_max_s: float = 2.0

    # indicateurs de validation
    seuil_conf_mot: float = 0.5
    seuil_conf_segment: float = 0.6
    part_mots_faibles: float = 0.3
    seuil_marge: float = 0.05
    n_extraits: int = 10
    duree_extrait_s: float = 120.0

    # fonctionnement
    duree_max_s: Optional[float] = None   # essai : ne traiter que le début (ex. 600 s)
    reprendre: bool = True           # réutiliser la transcription Vosk déjà faite
    garder_pistes: bool = False

    def to_dict(self):
        return asdict(self)
