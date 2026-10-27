"""Registre des voix entre sessions (numérotation stable des locuteurs).

Pour qu'un même participant garde le même numéro d'une session à l'autre,
il faut conserver, entre les sessions, une empreinte moyenne de sa voix :
la moyenne de ses x-vecteurs dans chaque session. C'est une donnée
biométrique au sens du RGPD (art. 4(14)). Le registre est donc :

* rangé dans un dossier séparé des transcriptions ;
* limité aux empreintes moyennes (aucune empreinte par fenêtre) ;
* sans aucun nom : seulement « Speaker NN » ;
* effaçable en une commande à la fin du projet (effacer-registre).

Appariement : cosinus entre les groupes de la nouvelle session et les
locuteurs du registre, puis affectation optimale un-à-un (méthode
hongroise, Kuhn, 1955). Un groupe sous le seuil devient un nouveau
locuteur. Chaque décision est écrite dans un fichier de correspondances
que vous pouvez corriger, puis réappliquer.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import ecrire_json, lire_json, log

AVERTISSEMENT = """REGISTRE DES EMPREINTES VOCALES — DONNÉES BIOMÉTRIQUES (RGPD, art. 4(14) et 9)

Ce dossier contient, pour chaque locuteur pseudonymisé (Speaker NN) et
chaque session, la moyenne de ses x-vecteurs. Ces vecteurs permettent de
reconnaître une voix. Ils ne contiennent ni nom, ni audio, ni texte.

- Ne pas copier ce dossier avec les transcriptions ni le partager.
- Le stocker sur un support chiffré, à accès restreint.
- L'effacer à la fin du projet :  python -m retranscription effacer-registre --confirmer
"""


def nom_speaker(n: int) -> str:
    return f"Speaker {n:02d}"


def numero(speaker: str) -> int:
    m = re.search(r"(\d+)\s*$", str(speaker))
    return int(m.group(1)) if m else 0


def _normer(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class Registre:
    def __init__(self, dossier: Path):
        self.dossier = Path(dossier)
        self.fichier_meta = self.dossier / "registre.json"
        self.fichier_vec = self.dossier / "empreintes.npz"
        self.entrees: List[dict] = []
        self.vecteurs = np.zeros((0, 0), dtype=np.float32)
        if self.fichier_meta.exists():
            self.entrees = lire_json(self.fichier_meta)["entrees"]
            self.vecteurs = np.load(self.fichier_vec)["vecteurs"]

    # ------------------------------------------------------------------
    def sauver(self) -> None:
        self.dossier.mkdir(parents=True, exist_ok=True)
        (self.dossier / "LISEZMOI.txt").write_text(AVERTISSEMENT, encoding="utf-8")
        ecrire_json(self.fichier_meta, {"mis_a_jour": datetime.now().isoformat(timespec="seconds"),
                                        "entrees": self.entrees})
        np.savez(self.fichier_vec, vecteurs=self.vecteurs.astype(np.float32))

    def retirer_session(self, session: str) -> None:
        garde = [i for i, e in enumerate(self.entrees) if e["session"] != session]
        self.entrees = [self.entrees[i] for i in garde]
        for nouvel, e in enumerate(self.entrees):
            e["ligne"] = nouvel
        self.vecteurs = self.vecteurs[garde] if len(self.vecteurs) else self.vecteurs

    def numero_suivant(self) -> int:
        return max([numero(e["speaker"]) for e in self.entrees] + [0]) + 1

    def profils(self, methode: str, exclure_session: str = None):
        """Empreinte moyenne par locuteur (moyenne pondérée par le temps de parole)."""
        acc: Dict[str, list] = {}
        for e in self.entrees:
            if e["methode"] != methode or e["session"] == exclure_session:
                continue
            acc.setdefault(e["speaker"], []).append((self.vecteurs[e["ligne"]], e["duree_s"]))
        noms = sorted(acc, key=numero)
        if not noms:
            return [], None
        P = np.vstack([np.average(np.vstack([v for v, _ in acc[n]]), axis=0,
                                  weights=[max(d, 1e-3) for _, d in acc[n]]) for n in noms])
        return noms, P

    # ------------------------------------------------------------------
    def apparier(self, session: str, centres: np.ndarray, durees: List[float],
                 methode: str, seuil: float, marge_min: float = 0.1,
                 duree_min: float = 10.0) -> List[dict]:
        k = len(centres)
        # les entrées précédentes de cette même session restent candidates :
        # relancer une session conserve ses numéros
        noms, P = self.profils(methode)
        decisions = [None] * k
        S = np.zeros((k, 0))
        if noms:
            tout = np.vstack([_normer(centres), _normer(P)])
            moyenne = tout.mean(axis=0)
            C = _normer(_normer(centres) - moyenne)
            Q = _normer(_normer(P) - moyenne)
            S = C @ Q.T
            lignes, colonnes = linear_sum_assignment(-S)
            for i, j in zip(lignes, colonnes):
                if S[i, j] >= seuil:
                    autres = np.delete(S[i], j)
                    second = float(autres.max()) if len(autres) else float("nan")
                    decisions[i] = {"speaker": noms[j], "similarite": float(S[i, j]),
                                    "second": second}
        prochain = max(self.numero_suivant(), 1)
        for i in range(k):
            if decisions[i] is None:
                meilleur = float(S[i].max()) if S.shape[1] else float("nan")
                decisions[i] = {"speaker": nom_speaker(prochain), "similarite": meilleur,
                                "second": float("nan"), "nouveau": True}
                prochain += 1
        for i, d in enumerate(decisions):
            d["groupe_local"] = i
            d["duree_parole_s"] = round(float(durees[i]), 1)
            sim, sec = d["similarite"], d["second"]
            d["marge"] = sim - sec if (sim == sim and sec == sec) else float("nan")
            motifs = []
            if d.get("nouveau"):
                statut = "nouveau"
                if sim == sim and sim >= seuil - 0.1:
                    motifs.append("proche d'un locuteur existant")
            else:
                statut = "apparié"
                if sim < seuil + 0.1:
                    motifs.append("similarité proche du seuil")
                if d["marge"] == d["marge"] and d["marge"] < marge_min:
                    motifs.append("deux locuteurs proches")
            if durees[i] < duree_min:
                motifs.append("parole brève")
            d["statut"] = statut
            d["a_verifier"] = "oui" if motifs else "non"
            d["motif"] = ", ".join(motifs)
            d.pop("nouveau", None)
        n_app = sum(1 for d in decisions if d["statut"] == "apparié")
        log.info("Registre : %d groupe(s) apparié(s), %d nouveau(x) locuteur(s).", n_app, k - n_app)
        return decisions

    def enregistrer(self, session: str, centres: np.ndarray, durees: List[float],
                    correspondance: Dict[int, str], methode: str) -> None:
        self.retirer_session(session)
        nouvelles = []
        for g, c in enumerate(_normer(centres)):
            self.entrees.append({"session": session, "groupe_local": g,
                                 "speaker": correspondance[g], "duree_s": float(durees[g]),
                                 "methode": methode, "ligne": len(self.entrees)})
            nouvelles.append(c)
        bloc = np.vstack(nouvelles).astype(np.float32)
        self.vecteurs = bloc if self.vecteurs.size == 0 else np.vstack([self.vecteurs, bloc])
        self.sauver()

    def calibrer(self) -> dict:
        """Similarités entre groupes de sessions différentes, selon les numéros actuels.

        À lancer après avoir vérifié (et corrigé) les correspondances de
        quelques sessions. Les paires « même locuteur » et « locuteurs
        différents » donnent deux distributions ; le seuil proposé minimise
        le nombre d'erreurs (fausses fusions + fausses séparations).
        """
        if len(self.entrees) < 2:
            return {"paires": [], "seuil_propose": None}
        V = _normer(self.vecteurs)
        C = _normer(V - V.mean(axis=0))
        S = C @ C.T
        paires = []
        for i, a in enumerate(self.entrees):
            for j in range(i + 1, len(self.entrees)):
                b = self.entrees[j]
                if a["session"] == b["session"] or a["methode"] != b["methode"]:
                    continue
                paires.append({"session_1": a["session"], "groupe_1": a["groupe_local"],
                               "speaker_1": a["speaker"], "session_2": b["session"],
                               "groupe_2": b["groupe_local"], "speaker_2": b["speaker"],
                               "similarite": round(float(S[i, j]), 4),
                               "meme_locuteur": "oui" if a["speaker"] == b["speaker"] else "non"})
        memes = sorted(p["similarite"] for p in paires if p["meme_locuteur"] == "oui")
        diffs = sorted(p["similarite"] for p in paires if p["meme_locuteur"] == "non")
        seuil = None
        if memes and diffs:
            candidats = sorted(set(memes + diffs))
            erreurs = [(sum(m < t for m in memes) + sum(d >= t for d in diffs), t) for t in candidats]
            e_min = min(e for e, _ in erreurs)
            meilleurs = [t for e, t in erreurs if e == e_min]
            # milieu de l'écart entre la valeur retenue et la valeur juste en dessous
            t_haut = meilleurs[-1]
            dessous = [v for v in memes + diffs if v < t_haut]
            t_bas = max(dessous) if dessous else t_haut - 0.05
            seuil = round((t_bas + t_haut) / 2, 3)
        return {"paires": paires, "seuil_propose": seuil,
                "meme_min": memes[0] if memes else None, "meme_mediane": memes[len(memes) // 2] if memes else None,
                "diff_max": diffs[-1] if diffs else None, "diff_mediane": diffs[len(diffs) // 2] if diffs else None,
                "n_memes": len(memes), "n_diffs": len(diffs)}

    def reattribuer(self, session: str, correspondance: Dict[int, str]) -> None:
        for e in self.entrees:
            if e["session"] == session and e["groupe_local"] in correspondance:
                e["speaker"] = correspondance[e["groupe_local"]]
        self.sauver()
