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


# seuil d'appariement par défaut selon les empreintes : sur les extraits AMI,
# deux personnes différentes ne dépassent pas 0,32 avec WeSpeaker ; les MFCC
# et les x-vecteurs Vosk donnent des similarités plus élevées
SEUILS_APPARIEMENT = {"wespeaker-resnet34": 0.5}
SEUIL_APPARIEMENT_AUTRES = 0.7


def seuil_appariement(p: "Parametres", methode: str) -> float:
    if p.seuil_appariement is not None:
        return float(p.seuil_appariement)
    return SEUILS_APPARIEMENT.get(methode, SEUIL_APPARIEMENT_AUTRES)


@dataclass
class Parametres:
    # reconnaissance de la parole
    moteur: str = "wav2vec2"         # "wav2vec2" (recommandé) ou "vosk"
    modele_w2v: str = str(MODELES / "wav2vec2-fr")               # 32 bits
    modele_w2v_leger: str = str(MODELES / "wav2vec2-fr-int8")    # 8 bits, plus rapide
    petit_modele: bool = False       # ordinateur peu puissant
    # modèles Vosk (dossiers décompressés)
    modele: str = str(MODELES / "vosk-model-fr-0.22")            # reconnaissance de la parole
    modele_leger: str = str(MODELES / "vosk-model-small-fr-0.22")  # décodage court des fenêtres x-vecteur
    modele_spk: str = str(MODELES / "vosk-model-spk-0.4")         # extracteur de x-vecteurs

    # locuteurs
    diarisation: str = "neurale"     # "neurale" (segmentation + empreintes) ou "fenetres"
    modele_segmentation: str = str(MODELES / "segmentation-3.0.onnx")
    modele_wespeaker: str = str(MODELES / "wespeaker-resnet34.onnx")
    empreinte: str = "wespeaker"     # méthode « fenetres » : "wespeaker", "vosk" ou "mfcc"
    seuil_voix: float = 0.75         # distance cosinus de coupure si le nombre de voix est inconnu
    tour_min_s: float = 1.0          # tour de parole minimal (lissage)

    # micros
    reference: Optional[str] = None  # fichier dont le début = temps 0 (défaut : le premier)
    max_decalage_s: float = 1800.0   # écart maximal entre les démarrages des caméras
    poids_canaux: float = 0.0        # part du profil d'énergie des micros dans le regroupement

    # locuteurs
    nb_locuteurs: Optional[int] = None   # recommandé : nombre de personnes présentes
    nb_max: int = 20
    fenetre_s: float = 1.5
    pas_s: float = 0.75
    seuil_appariement: Optional[float] = None   # cosinus minimal pour reconnaître un locuteur
    #   du registre ; None = valeur par défaut de la méthode d'empreinte (SEUILS_APPARIEMENT) ;
    #   à calibrer sur vos données : commande « calibrer »
    marge_appariement: float = 0.1
    duree_min_registre_s: float = 10.0

    # segments
    pause_max_s: float = 2.0

    # indicateurs de validation
    seuil_conf_mot: float = 0.5
    seuil_conf_segment: float = 0.6
    part_mots_faibles: float = 0.3
    seuil_marge: float = 0.05        # méthode « fenetres » (écart de cosinus)
    seuil_marge_activite: float = 0.3   # méthode « neurale » (écart d'activité)
    n_extraits: int = 10
    duree_extrait_s: float = 120.0

    # fonctionnement
    duree_max_s: Optional[float] = None   # essai : ne traiter que le début (ex. 600 s)
    reprendre: bool = True           # réutiliser la reconnaissance déjà faite
    garder_pistes: bool = False

    def to_dict(self):
        return asdict(self)
