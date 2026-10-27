"""Regroupement des fenêtres par voix (diarisation) et attribution des mots.

1. Les empreintes sont centrées sur la moyenne de la session puis
   normées (score cosinus).
2. Classification : au-delà de 2000 fenêtres, sur-segmentation par
   k-moyennes en micro-groupes très homogènes ; puis classification
   ascendante hiérarchique (lien moyen, distance cosinus), la CAH utilisée
   en diarisation par x-vecteurs (Sell et al., 2018) ; enfin affinage des
   centres par k-moyennes sphériques.
3. Nombre de locuteurs : fixé par l'utilisateur (recommandé), sinon estimé
   par l'écart propre normalisé (Park et al., 2020). Le coefficient de
   silhouette (Rousseeuw, 1987) est calculé pour information.
4. Chaque mot reçoit le groupe le plus proche des fenêtres qui le couvrent.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .asr import Mot
from .utils import log


def normer(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def preparer(X: np.ndarray, profils: Optional[np.ndarray] = None, poids_canaux: float = 0.0):
    Z = normer(X - X.mean(axis=0))
    if profils is not None and poids_canaux > 0:
        P = profils - profils.mean(axis=0)
        P = P / (P.std(axis=0) + 1e-6)
        P = normer(P)
        Z = np.hstack([Z * np.sqrt(1 - poids_canaux), P * np.sqrt(poids_canaux)])
    return Z.astype(np.float32)


@dataclass
class ResultatClassification:
    etiquettes: np.ndarray        # groupe local par fenêtre valide (0..k-1)
    centres: np.ndarray           # (k, d) centres normés dans l'espace Z
    similarites: np.ndarray       # (n, k) cosinus fenêtre / centre
    k: int
    silhouette: float
    silhouettes_k: dict           # k -> silhouette (si k choisi automatiquement)


def _silhouette(Z, lab, rng, n_max=4000):
    if len(set(lab)) < 2:
        return -1.0
    if len(Z) > n_max:
        idx = rng.choice(len(Z), n_max, replace=False)
        Z, lab = Z[idx], lab[idx]
        if len(set(lab)) < 2:
            return -1.0
    if len(set(lab)) >= len(Z):          # un groupe par point : silhouette non définie
        return -1.0
    return float(silhouette_score(Z, lab, metric="cosine"))


def nombre_par_ecart_propre(C: np.ndarray, k_max: int = 20) -> int:
    """Nombre de locuteurs par l'écart propre maximal normalisé (NME).

    Matrice d'affinité cosinus élaguée (p plus proches voisins par ligne,
    symétrisée), laplacien L = D - A, nombre de groupes = rang du plus
    grand écart entre valeurs propres successives (von Luxburg, 2007).
    Le seuil d'élagage p est choisi en minimisant p / écart normalisé
    (Park, Han, Kumar & Narayanan, 2020).
    """
    n = len(C)
    A = C @ C.T
    np.fill_diagonal(A, 0)
    ordre = np.argsort(-A, axis=1)
    meilleur = None
    for frac in np.linspace(0.02, 0.3, 15):
        p = max(2, int(frac * n))
        B = np.zeros_like(A)
        np.put_along_axis(B, ordre[:, :p], 1.0, axis=1)
        B = 0.5 * (B + B.T)
        lam = np.sort(np.linalg.eigvalsh(np.diag(B.sum(axis=1)) - B))
        ecarts = np.diff(lam[:k_max + 1])
        if len(ecarts) < 2:
            continue
        kk = int(np.argmax(ecarts[1:]) + 2)          # au moins 2 locuteurs
        g = ecarts[kk - 1] / (lam[-1] + 1e-9)
        r = p / (g + 1e-9)
        if meilleur is None or r < meilleur[0]:
            meilleur = (r, kk)
    return meilleur[1] if meilleur else 2


def _couper(Z, arbre, micro_lab, k, taille_min, iterations=0):
    """Coupe de l'arbre en k groupes substantiels.

    Le lien moyen isole parfois quelques éléments atypiques (toux, rire,
    bruit) en groupes minuscules, au détriment d'un vrai locuteur. Si la
    coupe en k groupes produit un groupe de moins de `taille_min`
    éléments, on coupe l'arbre plus bas jusqu'à obtenir k groupes
    substantiels ; seuls les éléments des petits groupes rejoignent alors
    le centre le plus proche. Les autres gardent leur groupe : une
    réaffectation globale (k-moyennes) dégrade le résultat sur des
    enregistrements réels (essais sur des extraits de réunions AMI).
    """
    lab = fcluster(arbre, k, criterion="maxclust")[micro_lab] - 1
    tailles = np.bincount(lab)
    if len(tailles) >= k and tailles.min() >= taille_min:
        return lab
    for kk in range(k, k + 50):
        lab = fcluster(arbre, kk, criterion="maxclust")[micro_lab] - 1
        tailles = np.bincount(lab)
        if (tailles >= taille_min).sum() >= k or kk >= len(np.unique(micro_lab)):
            break
    gardes = np.argsort(tailles)[::-1][:k]
    centres = normer(np.vstack([Z[lab == g].mean(axis=0) for g in gardes]))
    nouveau = np.argmax(Z @ centres.T, axis=1)
    rang = {g: i for i, g in enumerate(gardes)}
    lab = np.array([rang[g] if g in rang else nouveau[i] for i, g in enumerate(lab)])
    for _ in range(iterations):
        centres = normer(np.vstack([Z[lab == g].mean(axis=0) if np.any(lab == g) else centres[g]
                                    for g in range(len(centres))]))
        nouveau = np.argmax(Z @ centres.T, axis=1)
        if np.array_equal(nouveau, lab):
            break
        lab = nouveau
    return lab


def classer(Z: np.ndarray, k: Optional[int] = None, k_max: int = 20,
            graine: int = 0, taille_min: Optional[int] = None,
            affiner: bool = False, seuil_distance: Optional[float] = None) -> ResultatClassification:
    rng = np.random.default_rng(graine)
    n = len(Z)
    if n < 2:
        raise RuntimeError("Trop peu de parole pour distinguer des locuteurs.")
    if n > 2000:
        n_micro = int(min(400, n // 5))
        km = MiniBatchKMeans(n_clusters=n_micro, random_state=graine, n_init=3,
                             batch_size=4096).fit(Z)
        micro_lab = km.labels_
        C = normer(km.cluster_centers_)
        utilises = np.unique(micro_lab)
        remap = -np.ones(n_micro, dtype=int)
        remap[utilises] = np.arange(len(utilises))
        micro_lab, C = remap[micro_lab], C[utilises]
    else:
        micro_lab, C = np.arange(n), Z
    arbre = linkage(C, method="average", metric="cosine")
    if taille_min is None:
        # peu d'empreintes (enregistrement court) : un locuteur peut n'en avoir que 2 ou 3
        taille_min = max(5, int(0.002 * n)) if n > 1000 else 2
    sil_k = {}
    if k:
        lab = _couper(Z, arbre, micro_lab, int(min(k, len(C))), taille_min, 20 if affiner else 0)
    elif seuil_distance:
        lab = fcluster(arbre, seuil_distance, criterion="distance")[micro_lab] - 1
        log.info("Nombre de voix estimé par seuil de distance (%.2f) : %d", seuil_distance, len(set(lab)))
    else:
        k_eff = nombre_par_ecart_propre(C, k_max=int(min(k_max, len(C) - 1)))
        lab = _couper(Z, arbre, micro_lab, k_eff, taille_min, 20 if affiner else 0)
        for kk in range(max(2, k_eff - 2), k_eff + 3):     # pour information
            sil_k[kk] = round(_silhouette(Z, _couper(Z, arbre, micro_lab, kk, taille_min), rng), 4)
        log.info("Nombre de voix estimé par l'écart propre normalisé : %d", k_eff)
    # renuméroter par ordre de première apparition
    ordre, vus = {}, 0
    for g in lab:
        if g not in ordre:
            ordre[g] = vus; vus += 1
    lab = np.array([ordre[g] for g in lab], dtype=int)
    k_eff = vus
    centres = normer(np.vstack([Z[lab == g].mean(axis=0) for g in range(k_eff)]))
    S = Z @ centres.T
    sil = _silhouette(Z, lab, rng)
    return ResultatClassification(lab, centres, S, k_eff, sil, sil_k)


def viterbi(S: np.ndarray, debuts: np.ndarray, fins: np.ndarray, temperature: float = 0.05,
            p_rester: float = 0.98, p_rester_pause: float = 0.7, pause_s: float = 1.5) -> np.ndarray:
    """Lissage temporel des groupes de fenêtres (modèle de Markov caché).

    États : les voix. Émission : log-vraisemblance proportionnelle au cosinus
    entre la fenêtre et le centre de la voix (S / température). Transition :
    on reste sur la même voix avec la probabilité p_rester ; après une pause
    de plus de `pause_s`, un changement de locuteur devient plus probable
    (p_rester_pause). Le chemin le plus probable est trouvé par l'algorithme
    de Viterbi (Viterbi, 1967). C'est le principe de la resegmentation des
    systèmes de diarisation par x-vecteurs (Landini et al., 2022).
    """
    n, K = S.shape
    if n == 0 or K < 2:
        return np.zeros(n, dtype=int)
    ordre = np.argsort(debuts)
    E = S[ordre] / temperature
    d, f = debuts[ordre], fins[ordre]
    score = E[0].copy()
    retour = np.zeros((n, K), dtype=int)
    for t in range(1, n):
        p = p_rester_pause if d[t] - f[t - 1] > pause_s else p_rester
        rester, changer = np.log(p), np.log((1 - p) / (K - 1))
        meilleur = int(np.argmax(score))
        # pour chaque état : rester, ou venir du meilleur autre état
        venir = np.full(K, score[meilleur] + changer)
        second = np.partition(score, -2)[-2] + changer
        venir[meilleur] = second
        garde = score + rester >= venir
        retour[t] = np.where(garde, np.arange(K), meilleur)
        retour[t, meilleur] = meilleur if garde[meilleur] else int(np.argsort(score)[-2])
        score = np.maximum(score + rester, venir) + E[t]
    chemin = np.zeros(n, dtype=int)
    chemin[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        chemin[t - 1] = retour[t, chemin[t]]
    sortie = np.empty(n, dtype=int)
    sortie[ordre] = chemin
    return sortie


def lisser_tours(mots: List[Mot], groupes: np.ndarray, duree_min: float = 1.0) -> np.ndarray:
    """Supprime les tours de parole trop brefs coincés entre deux tours d'une même autre voix.

    Une suite de mots attribuée à une voix, d'une durée totale inférieure à
    `duree_min`, entourée de mots d'une même autre voix, prend cette voix.
    Répété jusqu'à stabilité. Les groupes négatifs (mots non attribués) sont
    ignorés.
    """
    g = np.array(groupes, dtype=float).copy()
    for _ in range(10):
        change = False
        i = 0
        n = len(mots)
        while i < n:
            if np.isnan(g[i]) or g[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < n and g[j + 1] == g[i]:
                j += 1
            avant = g[i - 1] if i > 0 else np.nan
            apres = g[j + 1] if j + 1 < n else np.nan
            duree = mots[j].fin - mots[i].debut
            if avant == avant and avant == apres and avant != g[i] and duree < duree_min:
                g[i:j + 1] = avant
                change = True
            i = j + 1
        if not change:
            break
    return g


def attribuer_mots(mots: List[Mot], fen: np.ndarray, res: ResultatClassification,
                   portee_s: float = 1.0, lissage: bool = True) -> np.ndarray:
    """Groupe de chaque mot + similarité moyenne et marge (pour la validation).

    Retour : tableau (n_mots, 3) = [groupe, similarité au groupe, marge].
    La marge est l'écart entre le cosinus au centre du groupe retenu et le
    cosinus au centre du deuxième groupe le plus proche. Une marge faible
    signale une attribution incertaine. Avec `lissage`, les groupes des
    fenêtres sont d'abord lissés dans le temps (Viterbi).
    """
    debuts, fins = fen[:, 0], fen[:, 1]
    centres_f = (debuts + fins) / 2
    ordre = np.argsort(debuts)
    debuts_o = debuts[ordre]
    S = res.similarites
    etiq = viterbi(S, debuts, fins) if lissage else np.argmax(S, axis=1)
    sortie = np.full((len(mots), 3), np.nan)
    for i, m in enumerate(mots):
        t = 0.5 * (m.debut + m.fin)
        j_max = np.searchsorted(debuts_o, t, side="right")
        cand = ordre[max(0, j_max - 8):j_max]
        couvrantes = cand[fins[cand] >= t]
        if len(couvrantes) == 0:
            proche = int(np.argmin(np.abs(centres_f - t)))
            if abs(centres_f[proche] - t) > portee_s:
                continue
            couvrantes = np.array([proche])
        s = S[couvrantes].mean(axis=0)
        votes = np.bincount(etiq[couvrantes], minlength=S.shape[1])
        g = int(np.argmax(votes + 1e-3 * s))
        tri = np.sort(s)
        marge = tri[-1] - tri[-2] if len(tri) > 1 else 1.0
        sortie[i] = (g, s[g], marge)
    # mots sans fenêtre : groupe du mot voisin
    for i in range(len(mots)):
        if np.isnan(sortie[i, 0]):
            voisins = [j for j in (i - 1, i + 1) if 0 <= j < len(mots) and not np.isnan(sortie[j, 0])]
            if voisins:
                sortie[i] = (sortie[voisins[0], 0], np.nan, np.nan)
    sortie[:, 0] = lisser_tours(mots, sortie[:, 0], 0.5)
    return sortie
