"""Empreintes vocales par fenêtre glissante.

Deux méthodes :

* « vosk » (par défaut) : x-vecteurs (Snyder et al., 2018) calculés par le
  modèle de locuteur de Vosk. Réseau discriminatif entraîné à reconnaître
  des locuteurs ; on garde sa couche intermédiaire. Il ne produit ni texte
  ni son.
* « mfcc » : méthode classique sans réseau de neurones. Moyenne et
  écart-type de 19 coefficients cepstraux MFCC (Davis & Mermelstein, 1980)
  par fenêtre. Moins discriminante ; utile comme point de comparaison.

Les fenêtres (1,5 s, pas de 0,75 s) ne couvrent que les zones où Vosk a
reconnu des mots : le silence et les bruits seuls ne sont pas modélisés.
Les x-vecteurs de fenêtre restent en mémoire et ne sont jamais écrits.
"""

import json
from typing import List, Tuple

import numpy as np
from scipy.fft import dct, rfft

from .asr import Mot
from .audio import lire
from .utils import TAUX, log


def regions_parole(mots: List[Mot], ecart_max: float = 0.5) -> List[Tuple[float, float]]:
    regions = []
    for m in sorted(mots, key=lambda m: m.debut):
        if regions and m.debut - regions[-1][1] <= ecart_max:
            regions[-1][1] = max(regions[-1][1], m.fin)
        else:
            regions.append([m.debut, m.fin])
    return [(a, b) for a, b in regions]


def fenetres(regions, duree: float = 1.5, pas: float = 0.75,
             duree_min: float = 0.8) -> np.ndarray:
    """Fenêtres [début, fin] dans les zones de parole."""
    sortie = []
    for a, b in regions:
        L = b - a
        if L < duree_min:
            continue
        if L <= duree:
            sortie.append((a, b))
            continue
        t = a
        while t + duree <= b + 1e-6:
            sortie.append((t, t + duree))
            t += pas
        if sortie[-1][1] < b - pas / 2:
            sortie.append((b - duree, b))
    return np.array(sortie, dtype=np.float64).reshape(-1, 2)


# ----------------------------------------------------------------------
class ExtracteurVosk:
    """x-vecteurs Vosk : un décodage court par fenêtre avec le modèle de locuteur."""

    nom = "vosk-xvecteur"

    def __init__(self, chemin_spk: str, modele_asr):
        from vosk import KaldiRecognizer, SpkModel
        from .asr import chemin_compatible
        self.rec = KaldiRecognizer(modele_asr, TAUX, SpkModel(chemin_compatible(chemin_spk)))

    def _un(self, x: np.ndarray):
        self.rec.Reset()
        vecteurs, poids = [], []
        if self.rec.AcceptWaveform(x.tobytes()):
            r = json.loads(self.rec.Result())
            if "spk" in r:
                vecteurs.append(r["spk"]); poids.append(r.get("spk_frames", 1))
        r = json.loads(self.rec.FinalResult())
        if "spk" in r:
            vecteurs.append(r["spk"]); poids.append(r.get("spk_frames", 1))
        if not vecteurs:
            return None
        v = np.average(np.array(vecteurs, dtype=np.float32), axis=0, weights=poids)
        return v

    def calculer(self, signal: np.ndarray, fen: np.ndarray):
        vecteurs, valides = [], []
        n = len(fen)
        jalon = 0.1
        for i, (a, b) in enumerate(fen):
            x = lire(signal, int(a * TAUX), int(b * TAUX))
            v = self._un(x)
            if v is not None and np.all(np.isfinite(v)):
                vecteurs.append(v)
                valides.append(i)
            if (i + 1) / n >= jalon:
                log.info("  x-vecteurs : %3.0f %% (%d/%d fenêtres)", 100 * (i + 1) / n, i + 1, n)
                jalon += 0.1
        if not vecteurs:
            raise RuntimeError("Aucun x-vecteur calculé : vérifiez le modèle de locuteur.")
        return np.vstack(vecteurs), np.array(valides, dtype=int)


# ----------------------------------------------------------------------
def _banc_mel(n_fft: int, n_filtres: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filtres + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / TAUX).astype(int)
    banc = np.zeros((n_filtres, n_fft // 2 + 1))
    for i in range(1, n_filtres + 1):
        g, c, d = bins[i - 1], bins[i], bins[i + 1]
        for j in range(g, c):
            banc[i - 1, j] = (j - g) / max(c - g, 1)
        for j in range(c, d):
            banc[i - 1, j] = (d - j) / max(d - c, 1)
    return banc


class ExtracteurMFCC:
    """Méthode classique : statistiques de MFCC par fenêtre."""

    nom = "mfcc-statistiques"

    def __init__(self, n_mfcc: int = 19, n_filtres: int = 40):
        self.n_fft = 512
        self.lt, self.pas = int(0.025 * TAUX), int(0.010 * TAUX)
        self.hamming = np.hamming(self.lt).astype(np.float32)
        self.banc = _banc_mel(self.n_fft, n_filtres, 60, 7600).astype(np.float32)
        self.n_mfcc = n_mfcc

    def mfcc(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])
        n = 1 + max(0, (len(x) - self.lt) // self.pas)
        idx = np.arange(self.lt)[None, :] + self.pas * np.arange(n)[:, None]
        trames = x[idx] * self.hamming
        spectre = np.abs(rfft(trames, n=self.n_fft)) ** 2
        energie = np.log(spectre @ self.banc.T + 1e-6)
        return dct(energie, type=2, norm="ortho", axis=1)[:, 1:self.n_mfcc + 1]

    def calculer(self, signal: np.ndarray, fen: np.ndarray):
        vecteurs = []
        for a, b in fen:
            c = self.mfcc(lire(signal, int(a * TAUX), int(b * TAUX)))
            vecteurs.append(np.concatenate([c.mean(axis=0), c.std(axis=0)]))
        X = np.vstack(vecteurs).astype(np.float32)
        # normalisation de chaque dimension sur la session
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        return X, np.arange(len(fen))
