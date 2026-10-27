"""Reconnaissance de la parole avec Vosk (boîte à outils Kaldi).

Vosk décode le signal avec un modèle acoustique (réseau TDNN), un lexique
et un modèle de langue n-gramme, par recherche dans un graphe pondéré
(WFST). Chaque mot produit appartient au lexique : le système ne génère
pas de texte libre. Pour chaque mot, Vosk fournit son début, sa fin et
une confiance (probabilité a posteriori issue du treillis de décodage).

Références : Povey et al. (2011) ; Mohri, Pereira & Riley (2002).
"""

import json
import sys
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .utils import TAUX, hms, log


@dataclass
class Mot:
    debut: float
    fin: float
    mot: str
    conf: float
    enonce: int = 0          # rang de l'énoncé Vosk (découpage aux pauses)
    groupe: int = -1         # groupe de voix local (rempli par la diarisation)

    def to_dict(self):
        return asdict(self)


def analyser_resultat(texte_json: str, decalage: float, enonce: int) -> List[Mot]:
    """Convertit un résultat JSON de Vosk en liste de mots horodatés."""
    res = json.loads(texte_json)
    mots = []
    for m in res.get("result", []):
        mot = m.get("word", "").strip()
        if not mot:
            continue
        mots.append(Mot(debut=round(decalage + float(m["start"]), 3),
                        fin=round(decalage + float(m["end"]), 3),
                        mot=mot, conf=round(float(m.get("conf", 1.0)), 4),
                        enonce=enonce))
    return mots


def chemin_compatible(chemin) -> str:
    """Windows : Kaldi lit mal les chemins avec accents ; on passe par le nom court (8.3)."""
    chemin = str(chemin)
    if sys.platform != "win32" or chemin.isascii():
        return chemin
    try:
        import ctypes
        tampon = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(chemin, tampon, 1024):
            return tampon.value
    except Exception:
        pass
    return chemin


def charger_modele(chemin: str):
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)
    if not Path(chemin).exists():
        raise FileNotFoundError(
            f"Modèle Vosk introuvable : {chemin}\n"
            "Lancez d'abord : python -m retranscription telecharger-modeles")
    log.info("Chargement du modèle Vosk %s…", Path(chemin).name)
    return Model(chemin_compatible(chemin))


def transcrire_wav(chemin_wav: Path, modele, bloc_s: float = 0.5) -> List[Mot]:
    """Transcrit un WAV mono 16 kHz en continu (pas de découpage préalable)."""
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(modele, TAUX)
    rec.SetWords(True)
    mots: List[Mot] = []
    n_enonce = 0
    t0 = time.time()
    with wave.open(str(chemin_wav), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != TAUX or wf.getsampwidth() != 2:
            raise ValueError("Le WAV doit être mono, 16 kHz, 16 bits.")
        total = wf.getnframes()
        lus = 0
        prochain_jalon = 0.05
        n_bloc = int(TAUX * bloc_s)
        while True:
            donnees = wf.readframes(n_bloc)
            if not donnees:
                break
            debut_bloc = lus / TAUX
            lus += len(donnees) // 2
            if rec.AcceptWaveform(donnees):
                mots += analyser_resultat(rec.Result(), 0.0, n_enonce)
                n_enonce += 1
            if lus / max(total, 1) >= prochain_jalon:
                ecoule = time.time() - t0
                log.info("  Vosk : %3.0f %% (%s traités, %.0f min écoulées)",
                         100 * lus / total, hms(debut_bloc, ms=False), ecoule / 60)
                prochain_jalon += 0.05
        mots += analyser_resultat(rec.FinalResult(), 0.0, n_enonce)
    log.info("Vosk : %d mots reconnus.", len(mots))
    return mots
