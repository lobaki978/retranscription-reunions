"""Alignement temporel des pistes provenant de fichiers différents.

Chaque caméra a sa propre horloge : elle démarre à un instant différent
(décalage) et son horloge audio n'a pas exactement la même fréquence
(dérive, quelques dizaines de ppm, soit jusqu'à ~1 s sur 6 h).

Modèle, pour une piste k et un temps de session t (secondes) :
    temps_dans_la_piste_k = a_k + (1 + d_k) * t

Méthode : intercorrélation des enveloppes d'énergie (log-énergie par trame
de 10 ms, tendance retirée), d'abord sur toute la durée (décalage global),
puis sur des fenêtres réparties le long de l'enregistrement (décalages
locaux). Une régression linéaire robuste sur les décalages locaux donne
a_k et d_k. Les pistes d'un même fichier partagent la même horloge.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, correlation_lags

from .audio import Piste
from .utils import TAUX, log

PAS_ENV = 0.01                       # trame d'enveloppe : 10 ms
ECH_TRAME = int(TAUX * PAS_ENV)      # 160 échantillons


@dataclass
class Alignement:
    piste: str
    a: float                 # décalage (s)
    d: float                 # dérive (sans unité ; ppm = d * 1e6)
    debut: float             # début de la piste en temps de session (s)
    fin: float               # fin de la piste en temps de session (s)
    n_points: int = 0        # fenêtres locales retenues
    r_moyen: float = 1.0     # corrélation moyenne des fenêtres retenues
    residu_ms: float = 0.0   # écart-type des résidus de la régression
    methode: str = "reference"

    def ech(self, t: float) -> int:
        """Indice d'échantillon dans la piste pour le temps de session t."""
        return int(round((self.a + (1.0 + self.d) * t) * TAUX))

    def to_dict(self):
        dct = asdict(self)
        dct["derive_ppm"] = self.d * 1e6
        return dct


def enveloppe(signal: np.ndarray, bloc_s: float = 60.0) -> np.ndarray:
    """Log-énergie par trame de 10 ms, calculée par blocs."""
    n_trames = len(signal) // ECH_TRAME
    env = np.empty(n_trames, dtype=np.float32)
    pas = int(bloc_s / PAS_ENV)
    for i in range(0, n_trames, pas):
        j = min(i + pas, n_trames)
        x = np.asarray(signal[i * ECH_TRAME:j * ECH_TRAME], dtype=np.float32)
        x = x.reshape(j - i, ECH_TRAME)
        env[i:j] = 10.0 * np.log10(np.mean(x * x, axis=1) + 1.0)
    return env


def _detendre(env: np.ndarray) -> np.ndarray:
    """Retire la tendance lente (2 s) : on garde les attaques et les pauses."""
    e = env - uniform_filter1d(env, size=200, mode="nearest")
    e = e - e.mean()
    s = e.std()
    return e / s if s > 0 else e


def _pic(c: np.ndarray, i: int) -> float:
    """Interpolation parabolique autour du maximum (précision sous-trame)."""
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return i + 0.5 * (y0 - y2) / den
    return float(i)


def decalage_global(env_ref: np.ndarray, env_k: np.ndarray, max_s: float) -> (float, float):
    x, y = _detendre(env_k), _detendre(env_ref)
    c = correlate(x, y, mode="full", method="fft")
    lags = correlation_lags(len(x), len(y), mode="full")
    # normalisation par le nombre de trames en recouvrement
    recouv = np.minimum(len(y), len(x) - lags) - np.maximum(0, -lags)
    recouv = np.maximum(recouv, 1)
    c = c / recouv
    masque = (np.abs(lags) * PAS_ENV <= max_s) & (recouv * PAS_ENV >= 60)
    if not masque.any():
        raise RuntimeError("Recouvrement insuffisant entre les pistes pour les aligner.")
    c_m = np.where(masque, c, -np.inf)
    i = int(np.argmax(c_m))
    lag = _pic(c_m, i) + lags[0]
    return lag * PAS_ENV, float(c[i])


def decalages_locaux(env_ref: np.ndarray, env_k: np.ndarray, a0: float,
                     espacement_s: float = 600.0, fenetre_s: float = 120.0,
                     recherche_s: float = 3.0) -> List[tuple]:
    """Décalage local mesuré sur des fenêtres réparties sur la session."""
    x_ref, x_k = _detendre(env_ref), _detendre(env_k)
    w = int(fenetre_s / PAS_ENV)
    r = int(recherche_s / PAS_ENV)
    l0 = int(round(a0 / PAS_ENV))
    points = []
    debut_possible = max(0, -l0 + r)
    fin_possible = min(len(x_ref) - w, len(x_k) - w - l0 - r)
    if fin_possible <= debut_possible:
        return points
    centres = np.arange(debut_possible, fin_possible + 1, int(espacement_s / PAS_ENV))
    if len(centres) < 3:
        centres = np.linspace(debut_possible, fin_possible, 3).astype(int)
    for i0 in centres:
        seg_ref = x_ref[i0:i0 + w]
        j0 = i0 + l0 - r
        seg_k = x_k[j0:j0 + w + 2 * r]
        if len(seg_k) < w + 2 * r or seg_ref.std() == 0:
            continue
        # coefficient de Pearson pour chacun des 2r+1 décalages candidats
        ref_c = (seg_ref - seg_ref.mean()).astype(np.float64)
        x = seg_k.astype(np.float64)
        cumul = np.concatenate([[0.0], np.cumsum(x)])
        cumul2 = np.concatenate([[0.0], np.cumsum(x * x)])
        somme = cumul[w:] - cumul[:-w]
        somme2 = cumul2[w:] - cumul2[:-w]
        var_k = np.maximum(somme2 - somme ** 2 / w, 1e-9)
        num = correlate(x, ref_c, mode="valid", method="fft")
        rho = num / np.sqrt(var_k * np.sum(ref_c ** 2))
        k = int(np.argmax(rho))
        lag = (_pic(rho, k) - r + l0) * PAS_ENV
        t_ref = (i0 + w / 2) * PAS_ENV
        points.append((t_ref, lag, float(rho[k])))
    return points


def regression_robuste(points: List[tuple], r_min: float) -> dict:
    pts = [p for p in points if p[2] >= r_min]
    if len(pts) < 2:
        return {}
    t = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    r = np.array([p[2] for p in pts])
    garde = np.ones(len(t), dtype=bool)
    for _ in range(3):
        if garde.sum() < 2:
            break
        pente, ordonnee = np.polyfit(t[garde], y[garde], 1)
        res = y - (ordonnee + pente * t)
        mad = np.median(np.abs(res[garde] - np.median(res[garde]))) * 1.4826
        nouveau = np.abs(res) <= max(3 * mad, 0.02)
        if (nouveau == garde).all():
            break
        garde = nouveau
    if garde.sum() < 2:
        return {}
    pente, ordonnee = np.polyfit(t[garde], y[garde], 1)
    res = y[garde] - (ordonnee + pente * t[garde])
    return {"a": float(ordonnee), "d": float(pente), "n": int(garde.sum()),
            "r": float(r[garde].mean()), "residu_ms": float(res.std() * 1000)}


def aligner(pistes: List[Piste], reference: str = None, max_decalage_s: float = 1800.0,
            r_min: float = 0.3) -> Dict[str, Alignement]:
    """Aligne toutes les pistes sur l'horloge du fichier de référence.

    Le temps 0 de la session est le début du fichier de référence : les
    temps produits sont donc directement ceux de cette vidéo dans ELAN.
    """
    horloges: Dict[str, List[Piste]] = {}
    for p in pistes:
        horloges.setdefault(p.horloge, []).append(p)
    if reference is None:
        reference = pistes[0].horloge
    ref_cand = [h for h in horloges if h == reference or Path(h).name == Path(reference).name]
    if not ref_cand:
        raise ValueError(f"Fichier de référence introuvable parmi les pistes : {reference}")
    h_ref = ref_cand[0]
    p_ref = horloges[h_ref][0]
    env_ref = None

    alignements: Dict[str, Alignement] = {}
    for h, groupe in horloges.items():
        if h == h_ref:
            a, d, info = 0.0, 0.0, {"methode": "reference", "n": 0, "r": 1.0, "residu_ms": 0.0}
        else:
            if env_ref is None:
                env_ref = enveloppe(p_ref.signal())
            env_k = enveloppe(groupe[0].signal())
            a0, c0 = decalage_global(env_ref, env_k, max_decalage_s)
            pts = decalages_locaux(env_ref, env_k, a0)
            reg = regression_robuste(pts, r_min)
            if reg and abs(reg["d"]) < 5e-4:
                a, d = reg["a"], reg["d"]
                info = {"methode": "regression", "n": reg["n"], "r": reg["r"],
                        "residu_ms": reg["residu_ms"]}
            else:
                a, d = a0, 0.0
                info = {"methode": "global_seul", "n": 0, "r": c0, "residu_ms": float("nan")}
                log.warning("Dérive non estimable pour %s : décalage global seul.", h)
            log.info("Alignement %s : décalage %.3f s, dérive %.1f ppm (%s, %d fenêtres, r=%.2f)",
                     Path(h).name, a, d * 1e6, info["methode"], info["n"], info["r"])
        for p in groupe:
            debut = -a / (1 + d)
            fin = (p.duree - a) / (1 + d)
            alignements[p.id] = Alignement(piste=p.id, a=a, d=d, debut=debut, fin=fin,
                                           n_points=info["n"], r_moyen=info["r"],
                                           residu_ms=info["residu_ms"], methode=info["methode"])
    return alignements
