"""Sélection du meilleur micro et construction du signal composite.

Pour chaque trame de 100 ms (temps de session), on mesure le rapport
signal/bruit (RSB) de chaque piste : énergie de la trame moins le plancher
de bruit de la piste (10e centile). Le RSB neutralise les différences de
gain entre micros.

Par blocs de 1 s, on retient la piste au RSB médian le plus élevé, avec
une hystérésis de 3 dB pour éviter les bascules incessantes. Chaque
bascule est déplacée vers la trame la plus silencieuse voisine (± 0,3 s),
puis lissée par un fondu de 10 ms. Le signal composite sert à la
reconnaissance de la parole et à la diarisation. Avec une seule piste, il est identique à cette piste.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .alignement import Alignement
from .audio import EcrivainWav, Piste, lire
from .utils import TAUX, log

TRAME = 0.1                       # s
ECH = int(TAUX * TRAME)           # 1600 échantillons


class Micros:
    def __init__(self, pistes: List[Piste], alignements: Dict[str, Alignement]):
        self.pistes = pistes
        self.al = [alignements[p.id] for p in pistes]
        self.signaux = [p.signal() for p in pistes]
        self.duree = max(0.0, max(a.fin for a in self.al))
        self.n_trames = int(np.ceil(self.duree / TRAME))
        self.db = None          # (n_trames, n_pistes) énergie en dB, NaN hors couverture
        self.rsb = None         # (n_trames, n_pistes)
        self.gains = None       # gain linéaire par piste (niveau de parole harmonisé)
        self.selection: List[Tuple[float, float, int]] = []

    def limiter(self, duree_max: float) -> None:
        """Ne traiter que le début de la session (essai)."""
        self.duree = min(self.duree, duree_max)
        self.n_trames = int(np.ceil(self.duree / TRAME))

    # ------------------------------------------------------------------
    def mesurer(self, bloc_s: float = 60.0) -> None:
        n_p = len(self.pistes)
        db = np.full((self.n_trames, n_p), np.nan, dtype=np.float32)
        tpb = int(round(bloc_s / TRAME))
        for k, (sig, al) in enumerate(zip(self.signaux, self.al)):
            for j0 in range(0, self.n_trames, tpb):
                j1 = min(j0 + tpb, self.n_trames)
                i0 = al.ech(j0 * TRAME)
                x = lire(sig, i0, i0 + (j1 - j0) * ECH).astype(np.float32)
                e = 10 * np.log10(np.mean(x.reshape(j1 - j0, ECH) ** 2, axis=1) + 1.0)
                db[j0:j1, k] = e
            # hors couverture de la piste
            t = np.arange(self.n_trames) * TRAME
            db[(t < al.debut) | (t + TRAME > al.fin), k] = np.nan
        self.db = db
        plancher = np.nanpercentile(db, 10, axis=0)
        self.rsb = db - plancher[None, :]
        # niveau de parole : 95e centile ; on vise -20 dBFS (≈ 70 dB en échelle int16²)
        parole = np.nanpercentile(db, 95, axis=0)
        cible = 10 * np.log10((32768 * 10 ** (-20 / 20)) ** 2)
        self.gains = np.clip(10 ** ((cible - parole) / 20), 0.25, 8.0)
        for p, f, g in zip(self.pistes, plancher, self.gains):
            log.debug("Piste %s : plancher %.1f dB, gain %.2f", p.id, f, g)

    # ------------------------------------------------------------------
    def choisir(self, bloc_s: float = 1.0, hysteresis_db: float = 3.0,
                recherche_s: float = 0.3) -> None:
        n_p = len(self.pistes)
        if n_p == 1:
            self.selection = [(0.0, self.duree, 0)]
            return
        tpb = int(round(bloc_s / TRAME))
        n_blocs = int(np.ceil(self.n_trames / tpb))
        choix = np.zeros(n_blocs, dtype=int)
        courant = None
        for b in range(n_blocs):
            bloc = self.rsb[b * tpb:(b + 1) * tpb]
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(np.isnan(bloc), -np.inf, bloc), axis=0)
            med = np.where(np.isfinite(med), med, -np.inf)
            meilleur = int(np.argmax(med))
            if courant is None or not np.isfinite(med[courant]):
                courant = meilleur
            elif med[meilleur] > med[courant] + hysteresis_db:
                courant = meilleur
            choix[b] = courant
        # segments + déplacement des bascules vers un creux d'énergie
        selection = []
        debut = 0.0
        r = int(round(recherche_s / TRAME))
        for b in range(1, n_blocs):
            if choix[b] != choix[b - 1]:
                j = b * tpb
                lo, hi = max(0, j - r), min(self.n_trames, j + r + 1)
                pair = self.db[lo:hi][:, [choix[b - 1], choix[b]]]
                energie = np.nansum(np.nan_to_num(pair, nan=200.0), axis=1)
                j_bas = lo + int(np.argmin(energie))
                t = j_bas * TRAME
                if t > debut:
                    selection.append((debut, t, int(choix[b - 1])))
                    debut = t
        selection.append((debut, self.duree, int(choix[-1])))
        # fusion des segments contigus sur la même piste
        fusion = [selection[0]]
        for s in selection[1:]:
            if s[2] == fusion[-1][2]:
                fusion[-1] = (fusion[-1][0], s[1], s[2])
            else:
                fusion.append(s)
        self.selection = fusion
        parts = {}
        for d, f, k in fusion:
            parts[k] = parts.get(k, 0.0) + (f - d)
        for k, v in sorted(parts.items()):
            log.info("Micro %s retenu %.0f %% du temps", self.pistes[k].id, 100 * v / max(self.duree, 1e-9))

    # ------------------------------------------------------------------
    def ecrire_composite(self, chemin, fondu_s: float = 0.01) -> None:
        w = EcrivainWav(chemin)
        nf = int(TAUX * fondu_s)
        rampe = np.linspace(0, 1, nf, endpoint=False, dtype=np.float32) if nf else None
        queue = None   # derniers nf échantillons du segment précédent (float32)
        bloc = TAUX   # 1 s : la dérive d'horloge reste négligeable dans un bloc
        for d, f, k in self.selection:
            sig, al, g = self.signaux[k], self.al[k], float(self.gains[k])
            i_deb, i_fin = int(round(d * TAUX)), int(round(f * TAUX))
            pos = i_deb
            premier = True
            while pos < i_fin:
                fin_bloc = min(pos + bloc, i_fin)
                j0 = al.ech(pos / TAUX)
                if premier and queue is not None and nf:
                    # relire nf échantillons avant la bascule : les deux micros
                    # sont fondus sur le même intervalle de temps
                    x = lire(sig, j0 - nf, j0 + (fin_bloc - pos)).astype(np.float32) * g
                    x[:nf] = queue * (1 - rampe) + x[:nf] * rampe
                    queue = None
                else:
                    x = lire(sig, j0, j0 + (fin_bloc - pos)).astype(np.float32) * g
                premier = False
                if nf and fin_bloc == i_fin and len(x) >= nf:
                    # garder la queue : elle sera fondue avec le début du segment suivant
                    w.ecrire(np.clip(x[:-nf], -32768, 32767).astype(np.int16))
                    queue = x[-nf:]
                else:
                    w.ecrire(np.clip(x, -32768, 32767).astype(np.int16))
                pos = fin_bloc
        if queue is not None:
            w.ecrire(np.clip(queue, -32768, 32767).astype(np.int16))
        w.fermer()

    # ------------------------------------------------------------------
    def profil(self, t0: float, t1: float) -> np.ndarray:
        """RSB moyen de chaque piste sur [t0, t1] : indice de la place du locuteur."""
        j0, j1 = int(t0 / TRAME), max(int(t0 / TRAME) + 1, int(np.ceil(t1 / TRAME)))
        bloc = self.rsb[j0:j1]
        with np.errstate(all="ignore"):
            v = np.nanmean(bloc, axis=0)
        return np.nan_to_num(v, nan=0.0)

    def piste_a(self, t: float) -> int:
        for d, f, k in self.selection:
            if d <= t < f:
                return k
        return self.selection[-1][2]

    def resume(self) -> List[dict]:
        sortie = []
        for k, (p, al) in enumerate(zip(self.pistes, self.al)):
            part = sum(f - d for d, f, kk in self.selection if kk == k)
            sortie.append({
                "piste": p.id, "fichier": Path(p.fichier).name,
                "debut_session_s": round(al.debut, 3), "fin_session_s": round(al.fin, 3),
                "decalage_s": round(al.a, 4), "derive_ppm": round(al.d * 1e6, 2),
                "methode": al.methode, "fenetres": al.n_points,
                "correlation": round(al.r_moyen, 3),
                "residu_ms": None if al.residu_ms != al.residu_ms else round(al.residu_ms, 1),
                "gain": round(float(self.gains[k]), 3) if self.gains is not None else None,
                "part_retenue": round(part / max(self.duree, 1e-9), 4),
            })
        return sortie
