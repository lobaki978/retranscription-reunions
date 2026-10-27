# -*- coding: utf-8 -*-
"""
Lancer la retranscription depuis Spyder.

Mode d'emploi
-------------
1. Ouvrez ce fichier dans Spyder.
2. Modifiez la cellule « RÉGLAGES » (chemins, session, nombre de locuteurs).
3. Exécutez UNE cellule à la fois : cliquez dedans, puis Ctrl+Entrée.
   Une cellule commence par une ligne « # %% » (dièse, espace, deux pour-cent).
   Ne lancez pas tout le fichier avec F5 : les étapes 2 à 4 se font après
   des vérifications à la main.

La première fois seulement : cellule 0 (installation et modèles).
"""

# %% RÉGLAGES — à modifier pour chaque session
# Le « r » devant les guillemets est nécessaire pour les chemins Windows.

PROJET = r"C:\Users\VOTRE_NOM\Documents\retranscription_reunions"  # dossier de ce fichier

VIDEOS = r"C:\Users\VOTRE_NOM\Videos\Deliberation\S01"   # toutes les vidéos de la session S01
SESSION = "S01"                 # identifiant court, sans espace
DATE = "2026-03-12"             # date de la session (AAAA-MM-JJ)
NB_LOCUTEURS = 12               # personnes qui parlent (participants + animateurs) ; None si inconnu
REFERENCE = None                # ex. "camera_salle.mp4" : la vidéo que vous utilisez dans ELAN (temps 0)
                                # None = la première vidéo par ordre alphabétique

SORTIES = r"C:\Users\VOTRE_NOM\Documents\Deliberation\sorties"    # résultats (hors OneDrive de préférence)
REGISTRE = r"C:\Users\VOTRE_NOM\Documents\Deliberation\registre"  # empreintes vocales : à garder à part

SEUIL_APPARIEMENT = None        # None = 0,5 (empreintes WeSpeaker) ; à ajuster après la cellule 4
MOTEUR = "wav2vec2"             # reconnaissance : "wav2vec2" (plus juste) ou "vosk" (plus rapide)
PETIT_MODELE = False            # True : modèle allégé (ordinateur de moins de 8 Go de mémoire)
ESSAI = False                   # True : ne traiter que les 10 premières minutes (premier essai)

import sys
if PROJET not in sys.path:
    sys.path.insert(0, PROJET)  # pour que Python trouve le dossier « retranscription »


# %% 0. INSTALLATION — une seule fois (quelques minutes, ~1,2 Go de modèles)
# Exécutez d'abord la cellule RÉGLAGES.
import subprocess
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                       str(Path(PROJET) / "requirements.txt")])

from retranscription.audio import verifier_ffmpeg
from retranscription.config import MODELES
from retranscription.telecharger import necessaires, telecharger
from retranscription.utils import configurer_journal

configurer_journal()
verifier_ffmpeg()          # message d'aide si ffmpeg manque
telecharger(MODELES, necessaires(MOTEUR, PETIT_MODELE))   # dans le dossier « modeles » du projet
print("Installation terminée.")


# %% 1. TRANSCRIRE LA SESSION — long : plusieurs heures pour 6 h de vidéo
from pathlib import Path

from retranscription.config import Parametres
from retranscription.pipeline import transcrire_session
from retranscription.utils import configurer_journal

configurer_journal()
parametres = Parametres(nb_locuteurs=NB_LOCUTEURS, reference=REFERENCE,
                        seuil_appariement=SEUIL_APPARIEMENT, moteur=MOTEUR, petit_modele=PETIT_MODELE,
                        duree_max_s=600.0 if ESSAI else None)
transcrire_session(Path(VIDEOS), SESSION, DATE, Path(SORTIES), Path(REGISTRE), parametres)
print(f"Terminé. Ouvrez {Path(SORTIES) / SESSION / 'validation' / (SESSION + '_synthese.txt')}")


# %% 2. APRÈS CORRECTION DES NUMÉROS — si vous avez modifié <SESSION>_correspondances.csv
from pathlib import Path

from retranscription.pipeline import appliquer
from retranscription.utils import configurer_journal, lire_json
from retranscription.config import Parametres

configurer_journal()
infos = lire_json(Path(SORTIES) / SESSION / "_travail" / "session.json")
appliquer(Path(SORTIES), SESSION, Path(REGISTRE), Parametres(**infos["parametres"]))


# %% 3. MESURER LA FIABILITÉ — après correction des extraits dans ELAN
from pathlib import Path

from retranscription.pipeline import evaluer
from retranscription.utils import configurer_journal

configurer_journal()
resultat = evaluer(Path(SORTIES), SESSION)
print(resultat)


# %% 4. CALIBRER LE SEUIL — après avoir vérifié 2 ou 3 sessions
from pathlib import Path

from retranscription.pipeline import calibrer
from retranscription.utils import configurer_journal

configurer_journal()
calibrage = calibrer(Path(SORTIES), Path(REGISTRE))
print("Seuil proposé :", calibrage.get("seuil_propose"))
