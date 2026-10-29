"""Diarisation neuronale : segmentation locale + empreintes + regroupement.

Principe de la chaîne pyannote 3 (Bredin, 2023 ; Plaquet & Bredin, 2023),
réécrite ici avec onnxruntime :

1. Segmentation locale. Le modèle « segmentation-3.0 » analyse des
   fenêtres de 10 s. Toutes les 17 ms environ, il indique quels locuteurs
   (au plus 3 dans la fenêtre, au plus 2 simultanés) parlent. C'est un
   classifieur : il ne produit ni texte ni son.
2. Empreinte de chaque locuteur local, calculée sur ses seuls moments de
   parole (sans chevauchement) avec le réseau ResNet34 de WeSpeaker
   (Wang et al., 2023).
3. Regroupement de toutes ces empreintes (CAH + affinage, nombre de voix
   fourni ou estimé). Dans une même fenêtre, deux locuteurs locaux
   reçoivent toujours deux voix différentes (affectation hongroise).
4. Reconstruction : chaque trame reçoit le nombre de locuteurs estimé
   par la segmentation, en prenant les voix les plus actives. Les tours
   trop courts sont ensuite supprimés (durées minimales de parole et de
   silence).

Le chevauchement de parole est donc détecté : une trame peut avoir deux
locuteurs actifs.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .diarisation import classer, classer_centroide, normer
from .utils import TAUX, log


def _session(chemin: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(chemin), sess_options=opts, providers=["CPUExecutionProvider"])


# ----------------------------------------------------------------------
class Segmenteur:
    """Modèle pyannote segmentation-3.0 au format ONNX."""

    def __init__(self, chemin: str):
        self.sess = _session(chemin)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.fenetre = int(meta.get("window_size", 160000))
        self.champ_pas = int(meta.get("receptive_field_shift", 270))
        self.champ_taille = int(meta.get("receptive_field_size", 991))
        self.n_locaux = int(meta.get("num_speakers", 3))
        n_classes = int(meta.get("num_classes", 7))
        # table « classe du powerset -> locuteurs actifs »
        table = np.zeros((n_classes, self.n_locaux), dtype=np.int8)
        k = 1
        for i in range(self.n_locaux):
            table[k, i] = 1
            k += 1
        for i in range(self.n_locaux):
            for j in range(i + 1, self.n_locaux):
                if k < n_classes:
                    table[k, i] = table[k, j] = 1
                    k += 1
        self.table = table
        self.entree = self.sess.get_inputs()[0].name

    def predire(self, blocs: np.ndarray) -> np.ndarray:
        """blocs : (N, fenetre) float32 dans [-1, 1] -> (N, trames, n_locaux) en 0/1."""
        y = self.sess.run(None, {self.entree: blocs[:, None, :].astype(np.float32)})[0]
        return self.table[np.argmax(y, axis=-1)]


class ExtracteurWeSpeaker:
    """Empreintes vocales ResNet34 (WeSpeaker, entraîné sur VoxCeleb)."""

    nom = "wespeaker-resnet34"

    def __init__(self, chemin: str):
        from .fbank import fbank
        self._fbank = fbank
        self.sess = _session(chemin)
        self.entree = self.sess.get_inputs()[0].name

    def empreinte(self, x_int16: np.ndarray) -> Optional[np.ndarray]:
        f = self._fbank(x_int16)
        if len(f) < 20:
            return None
        return self.sess.run(None, {self.entree: f[None]})[0][0].astype(np.float32)

    def calculer(self, signal: np.ndarray, fen: np.ndarray):
        """Interface commune avec les autres extracteurs (fenêtres fixes)."""
        from .audio import lire
        vecteurs, valides = [], []
        for i, (a, b) in enumerate(fen):
            v = self.empreinte(lire(signal, int(a * TAUX), int(b * TAUX)).astype(np.float32))
            if v is not None:
                vecteurs.append(v)
                valides.append(i)
        return np.vstack(vecteurs), np.array(valides, dtype=int)


# ----------------------------------------------------------------------
@dataclass
class ResultatDiarisation:
    tours: List[tuple]                 # (groupe, début, fin)
    activite: np.ndarray               # (trames, k) activité moyenne de chaque voix
    actifs: np.ndarray                 # (trames, k) 0/1 après reconstruction et lissage
    pas_trame: float                   # s
    decalage: float                    # temps de la trame 0 (s)
    centres_bruts: np.ndarray          # (k, d) moyenne des empreintes normées, par voix
    k: int
    silhouette: float
    silhouettes_k: dict
    n_empreintes: int
    durees: List[float]

    def trame(self, t: float) -> int:
        return int(np.clip(round((t - self.decalage) / self.pas_trame), 0, len(self.actifs) - 1))


@dataclass
class Locaux:
    """Résultat des étapes 1 et 2 (segmentation locale, empreintes) : réutilisable
    pour essayer plusieurs regroupements sans tout recalculer."""
    lab: np.ndarray            # (C, T, L) activité des locuteurs locaux
    debuts: list               # début de chaque fenêtre (échantillons)
    cles: list                 # (fenêtre, locuteur local) de chaque empreinte
    Xn: np.ndarray             # empreintes normées
    zones: list                # intervalles (s) de parole de chaque empreinte
    duree: float
    champ_pas: int
    champ_taille: int


def diariser(signal: np.ndarray, duree: float, segmenteur: Segmenteur,
             extracteur: ExtracteurWeSpeaker, k: Optional[int] = None, k_max: int = 20,
             pas_s: Optional[float] = None, duree_min_on: float = 0.3,
             duree_min_off: float = 0.5, lot: int = 16, centrer: bool = False,
             affiner: bool = False, seuil_distance: float = 0.75,
             regroupement: str = "moyen", taille_min_voix: int = 6) -> ResultatDiarisation:
    """regroupement : « moyen » (lien moyen, distance cosinus, seuil_distance ; petits
    groupes rattachés si taille_min_voix > 1) ou « pyannote31 » (lien centroïde)."""
    loc = empreintes_locales(signal, duree, segmenteur, extracteur, pas_s=pas_s, lot=lot)
    return regrouper(loc, k=k, k_max=k_max, duree_min_on=duree_min_on, duree_min_off=duree_min_off,
                     centrer=centrer, affiner=affiner, seuil_distance=seuil_distance,
                     regroupement=regroupement, taille_min_voix=taille_min_voix)


def empreintes_locales(signal: np.ndarray, duree: float, segmenteur: Segmenteur,
                       extracteur: ExtracteurWeSpeaker, pas_s: Optional[float] = None,
                       lot: int = 16) -> Locaux:
    F = segmenteur.fenetre
    if pas_s is None:
        pas_s = 1.0 if duree <= 3600 else 2.0
    pas = int(pas_s * TAUX)
    n_ech = int(duree * TAUX)
    debuts = list(range(0, max(n_ech - F, 0) + 1, pas))
    if not debuts or debuts[-1] + F < n_ech:
        debuts.append(max(n_ech - F, 0))

    # 1. segmentation locale ---------------------------------------------
    etiquettes = []
    jalon = 0.1
    for i in range(0, len(debuts), lot):
        blocs = np.zeros((len(debuts[i:i + lot]), F), dtype=np.float32)
        for j, d in enumerate(debuts[i:i + lot]):
            x = np.asarray(signal[d:d + F], dtype=np.float32) / 32768.0
            blocs[j, :len(x)] = x
        etiquettes.append(segmenteur.predire(blocs))
        if (i + lot) / len(debuts) >= jalon:
            log.info("  segmentation : %3.0f %%", 100 * min(1, (i + lot) / len(debuts)))
            jalon += 0.1
    lab = np.concatenate(etiquettes)                       # (C, T, L)
    C, T, L = lab.shape
    ech_par_trame = F / T

    # 2. empreinte de chaque locuteur local ------------------------------
    seul = lab * (lab.sum(axis=2, keepdims=True) == 1)
    cles, vecs, zones = [], [], []
    jalon = 0.1
    for c, d in enumerate(debuts):
        for j in range(L):
            masque = seul[c, :, j].astype(bool)
            if masque.sum() < 20:                         # < 0,34 s seul : on prend aussi les chevauchements
                masque = lab[c, :, j].astype(bool)
            if masque.sum() < 20:
                continue
            idx = np.flatnonzero(masque)
            # morceaux contigus -> échantillons
            coupures = np.flatnonzero(np.diff(idx) > 1)
            morceaux = np.split(idx, coupures + 1)
            parties, intervalles = [], []
            for m in morceaux:
                a = d + int(m[0] * ech_par_trame)
                b = d + int((m[-1] + 1) * ech_par_trame)
                parties.append(np.asarray(signal[a:min(b, n_ech)], dtype=np.float32))
                intervalles.append((a / TAUX, min(b, n_ech) / TAUX))
            x = np.concatenate(parties) if parties else np.zeros(0, np.float32)
            v = extracteur.empreinte(x)
            if v is not None and np.all(np.isfinite(v)):
                cles.append((c, j))
                vecs.append(v)
                zones.append(intervalles)
        if (c + 1) / C >= jalon:
            log.info("  empreintes vocales : %3.0f %%", 100 * (c + 1) / C)
            jalon += 0.1
    if len(vecs) < 2:
        raise RuntimeError("Trop peu de parole détectée pour distinguer des locuteurs.")
    Xn = normer(np.vstack(vecs))
    return Locaux(lab, debuts, cles, Xn, zones, duree, segmenteur.champ_pas, segmenteur.champ_taille)


def regrouper(loc: Locaux, k: Optional[int] = None, k_max: int = 20, duree_min_on: float = 0.3,
              duree_min_off: float = 0.5, centrer: bool = False, affiner: bool = False,
              seuil_distance: float = 0.75, regroupement: str = "moyen",
              taille_min_voix: int = 6, profils: Optional[np.ndarray] = None,
              poids_canaux: float = 0.0) -> ResultatDiarisation:
    """profils : (empreintes × micros) RSB de chaque micro pendant la parole de chaque
    empreinte ; avec poids_canaux > 0, ils s'ajoutent aux empreintes vocales pour le
    regroupement (position de la personne par rapport aux micros)."""
    lab, debuts, cles, Xn, duree = loc.lab, loc.debuts, loc.cles, loc.Xn, loc.duree
    C, T, L = lab.shape
    Z = normer(Xn - Xn.mean(axis=0)) if centrer else Xn
    if profils is not None and poids_canaux > 0 and profils.shape[1] > 1:
        P = profils - profils.mean(axis=0)
        P = normer(P / (P.std(axis=0) + 1e-6))
        Z = np.hstack([Z * np.sqrt(1 - poids_canaux), P * np.sqrt(poids_canaux)]).astype(np.float32)

    # 3. regroupement -----------------------------------------------------
    # chaque empreinte locale résume plusieurs secondes de parole : un groupe
    # d'une seule empreinte peut être un vrai locuteur qui parle peu
    if regroupement == "pyannote31":
        res = classer_centroide(Z, k=k, k_max=k_max)
    else:
        taille_min = 1 if len(Z) <= 1000 else max(3, int(0.0005 * len(Z)))
        # petits groupes rattachés (pyannote 3.1) ; seuil réduit pour un enregistrement court
        tmv = int(min(taille_min_voix, max(1, round(0.1 * len(Z)))))
        res = classer(Z, k=k, k_max=k_max, affiner=affiner, taille_min=taille_min,
                      seuil_distance=seuil_distance, taille_min_seuil=tmv)
    K = res.k
    affect = -np.ones((C, L), dtype=int)
    par_bloc = {}
    for n, (c, j) in enumerate(cles):
        par_bloc.setdefault(c, []).append((j, n))
    for c, liste in par_bloc.items():
        S = np.array([Z[n] @ res.centres.T for _, n in liste])       # (locaux, K)
        li, co = linear_sum_assignment(-S)
        for a, b in zip(li, co):
            affect[c, liste[a][0]] = b
    etiq_finales = np.array([affect[c, j] for c, j in cles])

    # 4. reconstruction ----------------------------------------------------
    pas_trame = loc.champ_pas / TAUX
    decalage = loc.champ_taille / TAUX / 2
    G = int(duree / pas_trame) + 1
    activite = np.zeros((G, K), dtype=np.float32)
    nombre = np.zeros(G, dtype=np.float32)
    couverture = np.zeros(G, dtype=np.float32)
    for c, d in enumerate(debuts):
        g0 = int(round(d / loc.champ_pas))
        g1 = min(G, g0 + T)
        n = g1 - g0
        if n <= 0:
            continue
        couverture[g0:g1] += 1
        nombre[g0:g1] += lab[c, :n].sum(axis=1)
        for j in range(L):
            s = affect[c, j]
            if s >= 0:
                activite[g0:g1, s] += lab[c, :n, j]
    ok = couverture > 0
    activite[ok] /= couverture[ok, None]
    nombre = np.rint(np.where(ok, nombre / np.maximum(couverture, 1), 0)).astype(int)
    ordre = np.argsort(-activite, axis=1)
    actifs = np.zeros((G, K), dtype=np.int8)
    for kk in range(min(K, int(nombre.max(initial=0)))):
        lignes = np.flatnonzero(nombre > kk)
        actifs[lignes, ordre[lignes, kk]] = 1
    # lissage : silences courts comblés, tours trop courts supprimés
    n_off = int(round(duree_min_off / pas_trame))
    n_on = int(round(duree_min_on / pas_trame))
    for s in range(K):
        a = actifs[:, s]
        _combler(a, 0, n_off)
        _combler(a, 1, n_on)
    tours = []
    for s in range(K):
        a = np.concatenate([[0], actifs[:, s], [0]])
        d = np.flatnonzero(np.diff(a) == 1)
        f = np.flatnonzero(np.diff(a) == -1)
        for i0, i1 in zip(d, f):
            tours.append((s, round(i0 * pas_trame + decalage, 3), round(i1 * pas_trame + decalage, 3)))
    tours.sort(key=lambda z: z[1])
    centres_bruts = np.vstack([Xn[etiq_finales == s].mean(axis=0) if np.any(etiq_finales == s)
                               else np.zeros(Xn.shape[1]) for s in range(K)])
    durees = [float(actifs[:, s].sum() * pas_trame) for s in range(K)]
    log.info("Diarisation : %d voix, %d tours, %d empreintes locales.", K, len(tours), len(Xn))
    return ResultatDiarisation(tours, activite, actifs, pas_trame, decalage, centres_bruts, K,
                               res.silhouette, res.silhouettes_k, len(Xn), durees)


def _combler(a: np.ndarray, valeur: int, n_max: int) -> None:
    """Remplace les plages de `valeur` plus courtes que n_max trames (hors bords)."""
    if n_max <= 0:
        return
    x = np.concatenate([[1 - valeur], a, [1 - valeur]])
    chg = np.flatnonzero(np.diff(x != valeur))
    for i0, i1 in zip(chg[::2], chg[1::2]):
        if 0 < i0 and i1 < len(a) and (i1 - i0) < n_max:
            a[i0:i1] = 1 - valeur


def attribuer_mots_tours(mots, res: ResultatDiarisation, portee_s: float = 2.0) -> np.ndarray:
    """Groupe de chaque mot d'après la diarisation : [groupe, part d'activité, marge].

    La « part d'activité » est la proportion des fenêtres d'analyse qui
    attribuent la trame à cette voix ; la marge est l'écart avec la voix
    suivante. Une marge faible signale une attribution incertaine ou une
    parole superposée.
    """
    sortie = np.full((len(mots), 3), np.nan)
    debuts = np.array([t[1] for t in res.tours]) if res.tours else np.zeros(0)
    for i, m in enumerate(mots):
        t = 0.5 * (m.debut + m.fin)
        g = res.trame(t)
        act = res.activite[g]
        actifs = np.flatnonzero(res.actifs[g])
        if len(actifs):
            s = int(actifs[np.argmax(act[actifs])])
        else:
            if not len(debuts):
                continue
            # tour le plus proche
            dist = [max(0.0, a - t, t - b) for _, a, b in res.tours]
            j = int(np.argmin(dist))
            if dist[j] > portee_s:
                continue
            s = res.tours[j][0]
        tri = np.sort(act)
        marge = float(tri[-1] - tri[-2]) if len(tri) > 1 else 1.0
        sortie[i] = (s, float(act[s]), marge)
    return sortie
