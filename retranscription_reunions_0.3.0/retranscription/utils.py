"""Fonctions communes : constantes, formatage du temps, journalisation, JSON."""

import json
import logging
import sys
from pathlib import Path

TAUX = 16000          # fréquence d'échantillonnage de travail (Hz), attendue par les modèles
OCTETS = 2            # PCM 16 bits

log = logging.getLogger("retranscription")


def configurer_journal(verbeux: bool = False) -> None:
    niveau = logging.DEBUG if verbeux else logging.INFO
    logging.basicConfig(
        level=niveau,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def hms(secondes: float, ms: bool = True) -> str:
    """12.345 -> '00:00:12.345'."""
    if secondes is None:
        return ""
    signe = "-" if secondes < 0 else ""
    s = abs(secondes)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    reste = s - 3600 * h - 60 * m
    if ms:
        return f"{signe}{h:02d}:{m:02d}:{reste:06.3f}"
    return f"{signe}{h:02d}:{m:02d}:{int(reste):02d}"


def ecrire_json(chemin: Path, donnees) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(chemin.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=1)
    tmp.replace(chemin)


def lire_json(chemin: Path):
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)
