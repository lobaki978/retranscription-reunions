"""Extraction des pistes audio des vidéos et lecture rapide des signaux.

Chaque canal de chaque flux audio de chaque fichier devient une « piste »
mono, 16 kHz, PCM 16 bits, stockée en binaire brut (.raw) pour être lue
par np.memmap sans charger 6 h d'audio en mémoire.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

from .utils import TAUX, log

def _outil(nom: str) -> str:
    """Chemin de ffmpeg / ffprobe : PATH, puis environnement conda (Anaconda/Spyder)."""
    trouve = shutil.which(nom)
    if trouve:
        return trouve
    for dossier in (Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "bin",
                    Path(os.environ.get("RETRANSCRIPTION_FFMPEG", "")) if os.environ.get("RETRANSCRIPTION_FFMPEG") else None):
        if dossier is None:
            continue
        for candidat in (dossier / f"{nom}.exe", dossier / nom):
            if candidat.exists():
                return str(candidat)
    return ""


def verifier_ffmpeg() -> None:
    """Rien à vérifier si PyAV est installé (il contient FFmpeg)."""
    if _pyav():
        return
    manquants = [n for n in ("ffmpeg", "ffprobe") if not _outil(n)]
    if manquants:
        raise RuntimeError(
            "ffmpeg introuvable (" + ", ".join(manquants) + ").\n"
            "Installation sous Windows, au choix :\n"
            "  - dans l'invite Anaconda (Anaconda Prompt) : conda install -c conda-forge ffmpeg\n"
            "  - ou dans un terminal : winget install Gyan.FFmpeg  (puis redémarrer Spyder)\n"
            "Ou indiquer le dossier contenant ffmpeg.exe dans la variable RETRANSCRIPTION_FFMPEG.")


EXTENSIONS_MEDIA = {
    ".mp4", ".mov", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm", ".m4v", ".wmv",
    ".mpg", ".mpeg", ".ts", ".3gp", ".flv", ".vob",
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
}


@dataclass
class Piste:
    id: str             # ex. "camera2_f0_c1"
    fichier: str        # fichier média d'origine
    flux: int           # rang du flux audio dans le fichier
    canal: int          # rang du canal dans le flux
    chemin: str         # fichier .raw extrait
    n: int              # nombre d'échantillons
    horloge: str        # pistes d'un même fichier = même horloge

    @property
    def duree(self) -> float:
        return self.n / TAUX

    def signal(self) -> np.memmap:
        return np.memmap(self.chemin, dtype="<i2", mode="r")

    def to_dict(self):
        return asdict(self)


def lister_medias(dossier: Path) -> List[Path]:
    dossier = Path(dossier)
    if dossier.is_file():
        return [dossier]
    fichiers = sorted(
        p for p in dossier.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS_MEDIA and not p.name.startswith(".")
    )
    return fichiers


def _pyav():
    """PyAV (bibliothèques FFmpeg incluses) s'il est installé, sinon None."""
    try:
        import av
        av.logging.set_level(av.logging.ERROR)
        return av
    except Exception:
        return None


def moteur_audio() -> str:
    return "pyav" if _pyav() else "ffmpeg"


def sonder(fichier: Path) -> dict:
    """Flux audio, nombre de canaux, durée."""
    av = _pyav()
    if av:
        with av.open(str(fichier)) as c:
            flux = [s for s in c.streams if s.type == "audio"]
            return {"flux_audio": [{"canaux": int(s.codec_context.channels), "taux": int(s.rate or 0),
                                    "codec": s.codec_context.name} for s in flux],
                    "duree": float(c.duration / 1e6) if c.duration else 0.0}
    cmd = [_outil("ffprobe"), "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(fichier)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe a échoué sur {fichier} : {res.stderr.strip()}")
    info = json.loads(res.stdout)
    flux = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    return {
        "flux_audio": [{"canaux": int(s.get("channels", 1)),
                        "taux": int(s.get("sample_rate", 0) or 0),
                        "codec": s.get("codec_name", "")} for s in flux],
        "duree": float(info.get("format", {}).get("duration", 0) or 0),
    }


def _extraire_pyav(fichier: Path, sorties_par_flux: dict) -> None:
    """Un seul passage dans le fichier : tous les flux, chaque canal dans son .raw.

    Si le son commence après le début du fichier (décalage fréquent dans les
    MP4/MOV), on ajoute le silence correspondant : le temps 0 reste celui
    de la vidéo.
    """
    av = _pyav()
    with av.open(str(fichier)) as c:
        flux = [s for s in c.streams if s.type == "audio"]
        origine = (c.start_time / 1e6) if c.start_time not in (None, 0) else 0.0
        sorties = {rf: [open(p, "wb") for p in chemins] for rf, chemins in sorties_par_flux.items()}
        # format entrelacé (« s16 ») : PyAV ne sait pas découper en plans les trames
        # de plus de 8 canaux (enregistreurs multipistes : Zoom F8, H6…)
        convertisseurs = {rf: av.AudioResampler(format="s16", layout=flux[rf].layout, rate=TAUX)
                          for rf in sorties_par_flux}
        rang = {flux[rf].index: rf for rf in sorties_par_flux}
        premier = {rf: True for rf in sorties_par_flux}

        def ecrire(rf, trames):
            for t in trames:
                n_c = len(sorties[rf])
                x = np.frombuffer(bytes(t.planes[0]), dtype="<i2")[: t.samples * n_c]
                x = x.reshape(-1, n_c)
                for k, f in enumerate(sorties[rf]):
                    f.write(np.ascontiguousarray(x[:, k]).tobytes())

        try:
            for paquet in c.demux(*[flux[rf] for rf in sorties_par_flux]):
                rf = rang.get(paquet.stream.index)
                if rf is None:
                    continue
                try:
                    trames = paquet.decode()
                except av.error.InvalidDataError:
                    continue                      # paquet abîmé : on l'ignore
                for trame in trames:
                    if premier[rf]:
                        premier[rf] = False
                        retard = (trame.time - origine) if trame.time is not None else 0.0
                        if 0 < retard < 10:
                            silence = np.zeros(int(round(retard * TAUX)), dtype="<i2").tobytes()
                            for f in sorties[rf]:
                                f.write(silence)
                    ecrire(rf, convertisseurs[rf].resample(trame))
            for rf, conv in convertisseurs.items():
                ecrire(rf, conv.resample(None))
        finally:
            for fichiers in sorties.values():
                for f in fichiers:
                    f.close()


def _extraire_ffmpeg(fichier: Path, rang_flux: int, n_canaux: int, sorties: List[Path]) -> None:
    """Variante avec le programme ffmpeg (si PyAV n'est pas installé)."""
    filtres = ";".join(
        f"[0:a:{rang_flux}]pan=mono|c0=c{c},aresample={TAUX}[o{c}]" for c in range(n_canaux)
    )
    cmd = [_outil("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(fichier),
           "-filter_complex", filtres]
    for c, sortie in enumerate(sorties):
        cmd += ["-map", f"[o{c}]", "-acodec", "pcm_s16le", "-f", "s16le", str(sortie)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué sur {fichier} : {res.stderr.strip()}")


def _canaux_identiques(a: np.ndarray, b: np.ndarray, seuil: float = 0.98) -> bool:
    """Deux canaux d'un même fichier sont-ils des copies (stéréo d'un seul micro) ?"""
    n = min(len(a), len(b))
    if n < TAUX:
        return False
    correlations = []
    for debut in np.linspace(0, n - 10 * TAUX, 6).astype(int) if n > 10 * TAUX else [0]:
        x = a[debut:debut + 10 * TAUX].astype(np.float32)
        y = b[debut:debut + 10 * TAUX].astype(np.float32)
        if x.std() < 1 or y.std() < 1:
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return bool(correlations) and float(np.median(correlations)) > seuil


def extraire_pistes(fichiers: List[Path], dossier_travail: Path) -> List[Piste]:
    dossier_travail = Path(dossier_travail)
    dossier_travail.mkdir(parents=True, exist_ok=True)
    pistes: List[Piste] = []
    for fichier in fichiers:
        info = sonder(fichier)
        if not info["flux_audio"]:
            log.warning("Aucun flux audio dans %s : fichier ignoré.", fichier.name)
            continue
        base = fichier.stem.replace(" ", "_")
        sorties_par_flux = {rf: [dossier_travail / f"{base}_f{rf}_c{c}.raw" for c in range(flux["canaux"])]
                            for rf, flux in enumerate(info["flux_audio"])}
        log.info("Extraction du son de %s (%s)…", fichier.name,
                 ", ".join(f"flux {rf} : {len(v)} canal(aux)" for rf, v in sorties_par_flux.items()))
        if _pyav():
            _extraire_pyav(fichier, sorties_par_flux)
        else:
            for rf, chemins in sorties_par_flux.items():
                _extraire_ffmpeg(fichier, rf, len(chemins), chemins)
        for rf, chemins in sorties_par_flux.items():
            gardees: List[Piste] = []
            for c, sortie in enumerate(chemins):
                n = sortie.stat().st_size // 2
                p = Piste(id=f"{base}_f{rf}_c{c}", fichier=str(fichier), flux=rf, canal=c,
                          chemin=str(sortie), n=int(n), horloge=str(fichier))
                if n < TAUX:
                    log.warning("Piste %s vide ou trop courte : ignorée.", p.id)
                    sortie.unlink(missing_ok=True)
                    continue
                if any(_canaux_identiques(q.signal(), p.signal()) for q in gardees):
                    log.info("Piste %s identique à un autre canal du même flux : ignorée.", p.id)
                    sortie.unlink(missing_ok=True)
                    continue
                gardees.append(p)
            pistes.extend(gardees)
    if not pistes:
        raise RuntimeError("Aucune piste audio exploitable.")
    log.info("%d piste(s) retenue(s) : %s", len(pistes), ", ".join(p.id for p in pistes))
    return pistes


def lire(signal: np.ndarray, debut: int, fin: int) -> np.ndarray:
    """Lit [debut, fin) en échantillons ; complète par des zéros hors limites."""
    n = fin - debut
    sortie = np.zeros(n, dtype=np.int16)
    a, b = max(debut, 0), min(fin, len(signal))
    if b > a:
        sortie[a - debut:b - debut] = signal[a:b]
    return sortie


# --- WAV -----------------------------------------------------------------

class EcrivainWav:
    """Écrit un WAV mono 16 kHz par blocs successifs."""

    def __init__(self, chemin: Path):
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        self._w = wave.open(str(chemin), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(TAUX)

    def ecrire(self, bloc: np.ndarray) -> None:
        self._w.writeframes(np.asarray(bloc, dtype="<i2").tobytes())

    def fermer(self) -> None:
        self._w.close()


def wav_memmap(chemin: Path) -> np.memmap:
    """Accès direct aux échantillons d'un WAV PCM 16 bits mono."""
    with open(chemin, "rb") as f:
        tete = f.read(12)
        if tete[:4] != b"RIFF" or tete[8:12] != b"WAVE":
            raise ValueError(f"{chemin} n'est pas un WAV.")
        while True:
            entete = f.read(8)
            if len(entete) < 8:
                raise ValueError(f"Bloc 'data' introuvable dans {chemin}.")
            nom, taille = entete[:4], struct.unpack("<I", entete[4:])[0]
            if nom == b"data":
                debut = f.tell()
                break
            f.seek(taille + (taille % 2), 1)
    return np.memmap(chemin, dtype="<i2", mode="r", offset=debut, shape=(taille // 2,))


def ecrire_extrait_wav(signal: np.ndarray, debut_s: float, fin_s: float, chemin: Path) -> None:
    w = EcrivainWav(chemin)
    w.ecrire(lire(signal, int(round(debut_s * TAUX)), int(round(fin_s * TAUX))))
    w.fermer()
