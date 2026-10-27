"""Diarisation neurale sur de vraies réunions (extraits du corpus AMI).

Données : les extraits de 30 s (réunions AMI, anglais, 2 à 4 locuteurs)
livrés avec les tests de pyannote.audio, avec leur annotation de
référence (RTTM). Corpus AMI : Carletta et al. (2005), licence CC BY 4.0.
Les extraits qui contiennent moins de 10 s de parole sont écartés.

Mesures :
  - DER (collier 0,5 s) avec nombre de voix connu, puis estimé ;
  - part des mots bien attribués (mots simulés tous les 0,35 s dans
    les tours de référence, hors chevauchements), avec et sans
    lissage des tours brefs.

Lancement :  python tests/test_diarisation_neurale.py [dossier_cache]
Les extraits et les modèles sont téléchargés une fois dans le cache.
"""

import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))

from scipy.optimize import linear_sum_assignment                       # noqa: E402

from retranscription.asr import Mot                                     # noqa: E402
from retranscription.audio import extraire_pistes                       # noqa: E402
from retranscription.diarisation import lisser_tours                    # noqa: E402
from retranscription.segmentation import (ExtracteurWeSpeaker, Segmenteur,  # noqa: E402
                                          attribuer_mots_tours, diariser)
from retranscription.telecharger import telecharger                     # noqa: E402
from retranscription.utils import configurer_journal                    # noqa: E402
from retranscription.validation import _taux, erreurs_diarisation      # noqa: E402

DONNEES = "https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/tests/data/"
LISTES = ("debug.development.rttm", "debug.test.rttm", "debug.train.rttm")
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "cache_ami"


def recuperer(nom: str) -> Path:
    cible = CACHE / nom
    if not cible.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DONNEES + nom, timeout=60) as r:
            cible.write_bytes(r.read())
    return cible


def lire(nom: str) -> np.ndarray:
    """Les extraits sont en WAV 32 bits flottants : conversion par PyAV."""
    piste = extraire_pistes([recuperer(nom + ".wav")], CACHE / "pistes")[0]
    return np.fromfile(piste.chemin, dtype="<i2")


def faux_mots(ref):
    mots, vrais = [], []
    for s, a, b in ref:
        t = a
        while t + 0.3 <= b:
            m = t + 0.15
            if not any(s2 != s and a2 <= m < b2 for s2, a2, b2 in ref):
                mots.append(Mot(debut=round(t, 3), fin=round(t + 0.3, 3), mot="x", conf=0.9))
                vrais.append(s)
            t += 0.35
    ordre = np.argsort([m.debut for m in mots], kind="stable")
    return [mots[i] for i in ordre], [vrais[i] for i in ordre]


def bien_attribues(vrais, groupes) -> int:
    V = sorted(set(vrais))
    G = sorted({int(g) for g in groupes if g == g})
    if not G:
        return 0
    M = np.array([[sum(1 for v, h in zip(vrais, groupes) if v == vv and h == gg) for gg in G] for vv in V])
    li, co = linear_sum_assignment(-M)
    return int(M[li, co].sum())


def main():
    configurer_journal()
    import logging
    logging.getLogger("retranscription").setLevel(logging.WARNING)
    telecharger(CACHE / "modeles", ["segmentation-3.0.onnx", "wespeaker-resnet34.onnx"])
    seg = Segmenteur(CACHE / "modeles" / "segmentation-3.0.onnx")
    ext = ExtracteurWeSpeaker(CACHE / "modeles" / "wespeaker-resnet34.onnx")

    ref = {}
    for liste in LISTES:
        for ligne in recuperer(liste).read_text().splitlines():
            c = ligne.split()
            ref.setdefault(c[1], []).append((c[7], float(c[3]), float(c[3]) + float(c[4])))

    erreurs = {"connu": np.zeros(4), "estime": np.zeros(4)}
    mots_ok = {(cas, lis): 0 for cas in erreurs for lis in (0.0, 1.0)}
    n_mots = 0
    for nom in sorted(ref):
        r = ref[nom]
        if sum(b - a for _, a, b in r) < 10:
            continue
        try:
            x = lire(nom)
        except urllib.error.HTTPError:          # extrait annoté mais non publié
            continue
        duree = len(x) / 16000
        nk = len({s for s, _, _ in r})
        mots, vrais = faux_mots(r)
        n_mots += len(mots)
        ligne = [f"{nom} ({nk} loc.)"]
        for cas, k in (("connu", nk), ("estime", None)):
            res = diariser(x, duree, seg, ext, k=k, k_max=8)
            e = erreurs_diarisation(r, [(f"S{s}", a, b) for s, a, b in res.tours], 0, duree, collier=0.5)
            erreurs[cas] += [e["total"], e["manquee"], e["fausse_alarme"], e["confusion"]]
            ligne.append(f"{cas} : {res.k} voix, DER {100 * _taux(e):5.1f} %")
            if mots:
                g = attribuer_mots_tours(mots, res, portee_s=float("inf"))[:, 0]
                mots_ok[(cas, 0.0)] += bien_attribues(vrais, g)
                mots_ok[(cas, 1.0)] += bien_attribues(vrais, lisser_tours(mots, g.copy(), 1.0))
        print(" | ".join(ligne))

    print()
    der = {}
    for cas, (tot, manq, fa, conf) in erreurs.items():
        der[cas] = (manq + fa + conf) / tot
        print(f"DER total, nombre de voix {cas} : {100 * der[cas]:.1f} % "
              f"(parole manquée {100 * manq / tot:.1f}, fausses alarmes {100 * fa / tot:.1f}, "
              f"confusion {100 * conf / tot:.1f})")
    for (cas, lis), n in mots_ok.items():
        print(f"Mots bien attribués, nombre de voix {cas}, lissage {lis:.0f} s : {100 * n / n_mots:.1f} %")
    # repères : pyannote 3.1 annonce 18,8 % de DER sur AMI (micro unique, sans collier) ;
    # ici, extraits très courts (30 s) : peu de parole pour chaque voix
    assert der["connu"] < 0.32 and der["estime"] < 0.32, "DER anormalement élevé"
    assert mots_ok[("estime", 1.0)] / n_mots > 0.8, "attribution des mots anormalement basse"
    print("TOUS LES TESTS SONT PASSÉS.")


if __name__ == "__main__":
    main()
