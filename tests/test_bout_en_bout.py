"""Test de bout en bout sur deux sessions simulées (3 caméras, 4 micros).

La reconnaissance de la parole est simulée (les grands modèles ne sont pas
téléchargés dans l'environnement de test) : le « transcripteur »
renvoie les mots de la vérité terrain, avec 5 % de mots remplacés et 2 %
de mots supprimés. Tout le reste tourne réellement : extraction des
pistes, alignement, choix du micro, empreintes MFCC, regroupement,
registre, fichiers de sortie, extraits, évaluation WER/DER.

Lancement :  python tests/test_bout_en_bout.py [dossier_essai]
"""

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))
sys.path.insert(0, str(ICI))

from retranscription.asr import Mot                      # noqa: E402
from retranscription.config import Parametres            # noqa: E402
from retranscription.pipeline import appliquer, evaluer, transcrire_session   # noqa: E402
from retranscription.sorties import ecrire_eaf, lire_csv, lire_eaf           # noqa: E402
from retranscription.utils import configurer_journal, lire_json              # noqa: E402
from simuler import LEXIQUE, simuler_session             # noqa: E402

import tempfile                                          # noqa: E402
ESSAI = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "essai_retranscription"
SORTIE, REGISTRE = ESSAI / "sorties", ESSAI / "registre"


def faux_transcripteur(verite, graine):
    rng = np.random.default_rng(graine)

    def transcrire(_chemin_wav):
        mots = []
        for w in verite:
            u = rng.random()
            if u < 0.02:
                continue                                   # mot oublié
            mot, conf = w["mot"], float(rng.uniform(0.75, 1.0))
            if u < 0.07:
                mot, conf = str(rng.choice(LEXIQUE)), float(rng.uniform(0.2, 0.6))
            mots.append(Mot(debut=w["debut"], fin=w["fin"], mot=mot, conf=round(conf, 3)))
        return mots
    return transcrire


def groupe_par_mot(session):
    """Locuteur attribué à chaque mot de la vérité (par temps du mot)."""
    lignes = lire_csv(SORTIE / session / f"{session}_transcription.csv")
    segs = [(float(l["debut_s"]), float(l["fin_s"]), l["speaker"]) for l in lignes]
    return segs


def attribution(verite, segs):
    res = []
    for w in verite:
        t = (w["debut"] + w["fin"]) / 2
        spk = next((s for a, b, s in segs if a <= t <= b), None)
        res.append((w["locuteur"], spk))
    return res


def corriger_extraits(session, verite, vrai_vers_id):
    """Simule la correction humaine : réécrit chaque extrait depuis la vérité terrain."""
    dossier = SORTIE / session / "validation" / "extraits"
    hyp = lire_json(dossier / "hypotheses.json")
    for e in hyp["extraits"]:
        a, b = e["debut_session"], e["fin_session"]
        # tours de la vérité : mots consécutifs d'un même locuteur
        tours = []
        for w in verite:
            if not (a <= (w["debut"] + w["fin"]) / 2 < b):
                continue
            spk = vrai_vers_id[w["locuteur"]]
            if tours and tours[-1][0] == spk and w["debut"] - tours[-1][2] < 2.0:
                tours[-1][2] = w["fin"]; tours[-1][3].append(w["mot"])
            else:
                tours.append([spk, w["debut"], w["fin"], [w["mot"]]])
        ann = [(s, max(d, a) - a, min(f, b) - a, " ".join(m).capitalize() + ".") for s, d, f, m in tours]
        tiers = sorted({x[0] for x in ann})
        ecrire_eaf(dossier / e["fichier"], ann, tiers, [(dossier / e["fichier"].replace(".eaf", ".wav"), "audio/x-wav")])


def main():
    configurer_journal()
    if ESSAI.exists():
        shutil.rmtree(ESSAI)
    sessions = {"S01": (["A", "B", "C", "D"], 1), "S02": (["B", "C", "D", "E"], 2)}
    verites = {}
    for s, (locs, g) in sessions.items():
        verites[s], attendu = simuler_session(ESSAI / "medias" / s, locs, 600, graine=g)
    # voix synthétiques : méthode par fenêtres et empreintes MFCC (la diarisation
    # neurale est testée sur de vraies voix dans tests/test_diarisation_neurale.py)
    p = Parametres(diarisation="fenetres", empreinte="mfcc", nb_locuteurs=4, n_extraits=4, duree_extrait_s=60)
    p_auto = Parametres(diarisation="fenetres", empreinte="mfcc", nb_locuteurs=None, n_extraits=4,
                        duree_extrait_s=60)

    bilan = {}
    vrai_vers_id = {}

    def mesurer(s):
        att = attribution(verites[s], groupe_par_mot(s))
        par_vrai = {}
        for vrai, spk in att:
            par_vrai.setdefault(str(vrai), Counter())[spk] += 1
        maj = {v: c.most_common(1)[0][0] for v, c in par_vrai.items()}
        # exactitude avec correspondance un-à-un (méthode hongroise)
        from scipy.optimize import linear_sum_assignment
        vrais, ids = sorted(par_vrai), sorted({spk for _, spk in att if spk})
        M = np.array([[par_vrai[v][i] for i in ids] for v in vrais])
        li, co = linear_sum_assignment(-M)
        exact = M[li, co].sum() / len(att)
        assert len(set(maj.values())) == len(maj), f"deux personnes sous le même numéro : {maj}"
        bilan[s] = {"majoritaire": maj, "mots_bien_attribues": round(float(exact), 4)}
        print(f"\n{s} : identifiants {dict(sorted(maj.items()))} ; mots bien attribués {100 * exact:.1f} %")
        return maj

    for s in sessions:
        # S01 : nombre de voix imposé ; S02 : nombre estimé automatiquement
        transcrire_session(ESSAI / "medias" / s, s, f"2026-03-1{s[-1]}", SORTIE, REGISTRE,
                           p if s == "S01" else p_auto,
                           transcripteur=faux_transcripteur(verites[s], 10 + int(s[-1])))
        mesurer(s)

    # 1. cohérence inter-sessions : B, C, D gardent leur numéro
    for v in "BCD":
        assert bilan["S01"]["majoritaire"][v] == bilan["S02"]["majoritaire"][v], f"{v} change de numéro"
    print("OK : B, C, D gardent leur numéro entre S01 et S02.")
    # E n'est pas dans S01. S'il a reçu le numéro de quelqu'un d'autre, l'appariement
    # doit être signalé « à vérifier » ; on simule alors la correction manuelle.
    id_e = bilan["S02"]["majoritaire"]["E"]
    corr = SORTIE / "S02" / "validation" / "S02_correspondances.csv"
    lignes_c = lire_csv(corr)
    if id_e in bilan["S01"]["majoritaire"].values():
        ligne_e = next(l for l in lignes_c if l["speaker"] == id_e)
        print(f"E a reçu {id_e} (similarité {ligne_e['similarite']}) : signalé à vérifier = {ligne_e['a_verifier']} ({ligne_e['motif']})")
        assert ligne_e["a_verifier"] == "oui", "fausse fusion non signalée"
        ligne_e["speaker"] = "Speaker 05"
        with open(corr, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes_c[0].keys()), delimiter=";")
            w.writeheader(); w.writerows(lignes_c)
        appliquer(SORTIE, "S02", REGISTRE, p_auto)
        mesurer("S02")
        print("OK : fausse fusion signalée puis corrigée à la main (Speaker 05).")
    assert bilan["S02"]["majoritaire"]["E"] not in bilan["S01"]["majoritaire"].values()
    for s in sessions:
        for v, spk in bilan[s]["majoritaire"].items():
            vrai_vers_id.setdefault(v, spk)
    from retranscription.pipeline import calibrer
    cal = calibrer(SORTIE, REGISTRE)
    print(f"Calibration : même locuteur min {cal['meme_min']}, différents max {cal['diff_max']}, "
          f"seuil proposé {cal['seuil_propose']}")
    assert cal["diff_max"] < cal["seuil_propose"] <= cal["meme_min"]
    print("OK : le seuil proposé sépare les paires vérifiées.")

    # 2. fichiers attendus
    for s in sessions:
        d = SORTIE / s
        for nom in [f"{s}_transcription.csv", f"{s}_transcription.txt", f"{s}_transcription.eaf",
                    f"{s}_locuteurs.csv", f"{s}_audio.wav",
                    f"validation/{s}_indicateurs_segments.csv", f"validation/{s}_synthese.txt",
                    f"validation/{s}_correspondances.csv", f"validation/{s}_micros.csv",
                    "validation/extraits/hypotheses.json", "validation/extraits/CONSIGNES.txt"]:
            assert (d / nom).exists(), f"manque {nom}"
        assert not (d / "_travail" / "pistes").exists(), "pistes temporaires non effacées"
    print("OK : tous les fichiers de retranscription et de validation sont présents.")

    # 3. ELAN : relecture par pympi (bibliothèque indépendante)
    import pympi
    eaf = pympi.Elan.Eaf(str(SORTIE / "S01" / "S01_transcription.eaf"))
    n_ann = sum(len(eaf.get_annotation_data_for_tier(t)) for t in eaf.get_tier_names())
    print(f"OK : pympi relit le .eaf ({len(eaf.get_tier_names())} pistes, {n_ann} annotations, "
          f"médias : {[m['RELATIVE_MEDIA_URL'] for m in eaf.media_descriptors]})")

    # 4. évaluation : extraits non corrigés détectés, puis corrigés
    res0 = evaluer(SORTIE, "S01")
    assert res0 == {}, "des extraits non corrigés ont été évalués"
    for s in sessions:
        corriger_extraits(s, verites[s], vrai_vers_id)
        bilan[s]["evaluation"] = evaluer(SORTIE, s)
        print(f"{s} : {bilan[s]['evaluation']}")

    # 5. contrôle indépendant du WER et du DER
    from retranscription.validation import normaliser

    def lev(r, h):
        r, h = r.split(), h.split()
        D = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
        D[:, 0], D[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
        return D[-1, -1], len(r)

    def der_trames(ref, hyp, duree, collier=0.25, pas=0.01):
        n = int(duree / pas)
        t = (np.arange(n) + 0.5) * pas
        masque = np.ones(n, bool)
        for _, a, b in ref:
            masque &= ~((np.abs(t - a) < collier) | (np.abs(t - b) < collier))
        R = sorted({x[0] for x in ref}); H = sorted({x[0] for x in hyp})
        Rm = np.array([[any(a <= ti < b for s, a, b in ref if s == r) for ti in t] for r in R])
        Hm = np.array([[any(a <= ti < b for s, a, b in hyp if s == h) for ti in t] for h in H]) if H else np.zeros((0, n), bool)
        from scipy.optimize import linear_sum_assignment
        M = np.array([[np.sum(Rm[i] & Hm[j] & masque) for j in range(len(H))] for i in range(len(R))])
        li, co = linear_sum_assignment(-M) if len(H) else ([], [])
        correct = np.zeros(n)
        for i, j in zip(li, co):
            correct += Rm[i] & Hm[j]
        nr, nh = Rm.sum(0), Hm.sum(0)
        err = (np.maximum(nr, nh) - correct)[masque].sum()
        return err, nr[masque].sum()

    dossier = SORTIE / "S01" / "validation" / "extraits"
    hyp = lire_json(dossier / "hypotheses.json")
    E = N = Ed = Nd = 0
    for e in hyp["extraits"]:
        ref = lire_eaf(dossier / e["fichier"])
        r = normaliser(" ".join(x for *_, x in ref)); h = normaliser(e["texte"])
        err, n = lev(r, h); E += err; N += n
        err_d, n_d = der_trames([(s, a, b) for s, a, b, _ in ref], [tuple(x) for x in e["tours"]],
                                e["fin_session"] - e["debut_session"])
        Ed += err_d; Nd += n_d
    print(f"Contrôle indépendant S01 : WER {100 * E / N:.2f} % (outil : {100 * bilan['S01']['evaluation']['WER']:.2f} %) ; "
          f"DER {100 * Ed / Nd:.2f} % (outil : {100 * bilan['S01']['evaluation']['DER']:.2f} %)")
    assert abs(E / N - bilan["S01"]["evaluation"]["WER"]) < 1e-4
    assert abs(Ed / Nd - bilan["S01"]["evaluation"]["DER"]) < 0.005
    # contrôle par pyannote.metrics (Bredin, 2017), bibliothèque de référence
    from pyannote.core import Annotation, Segment as Seg, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.metrics.identification import IdentificationErrorRate
    m_der, m_ier = DiarizationErrorRate(collar=0.5), IdentificationErrorRate(collar=0.5)
    for e in hyp["extraits"]:
        R, H = Annotation(), Annotation()
        for i, (s, a, b, _) in enumerate(lire_eaf(dossier / e["fichier"])):
            R[Seg(a, b), i] = s
        for i, (s, a, b) in enumerate(e["tours"]):
            if b > a:
                H[Seg(a, b), i] = s
        uem = Timeline([Seg(0, e["fin_session"] - e["debut_session"])])
        m_der(R, H, uem=uem); m_ier(R, H, uem=uem)
    print(f"pyannote.metrics : DER {100 * abs(m_der):.3f} % (outil : {100 * bilan['S01']['evaluation']['DER']:.3f} %) ; "
          f"identification {100 * abs(m_ier):.3f} % (outil : {100 * bilan['S01']['evaluation']['taux_erreur_identification']:.3f} %)")
    assert abs(abs(m_der) - bilan["S01"]["evaluation"]["DER"]) < 1e-4
    assert abs(abs(m_ier) - bilan["S01"]["evaluation"]["taux_erreur_identification"]) < 1e-4

    # 6. métadonnées : le sexe saisi à la main est conservé et repris dans les autres sessions
    chemin = SORTIE / "S01" / "S01_locuteurs.csv"
    lignes = lire_csv(chemin)
    for l in lignes:
        l["sexe"] = "F" if l["speaker"] == vrai_vers_id["B"] else "M"
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
        w.writeheader(); w.writerows(lignes)
    # correction manuelle d'une correspondance dans S02 : on échange deux numéros, puis on rétablit
    corr = SORTIE / "S02" / "validation" / "S02_correspondances.csv"
    original = corr.read_text(encoding="utf-8")
    lc = lire_csv(corr)
    lc[0]["speaker"], lc[1]["speaker"] = lc[1]["speaker"], lc[0]["speaker"]
    with open(corr, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lc[0].keys()), delimiter=";")
        w.writeheader(); w.writerows(lc)
    appliquer(SORTIE, "S02", REGISTRE, p)
    corrige = {l["groupe_local"]: l["speaker"] for l in lire_csv(corr)}
    assert corrige["0"] == lc[0]["speaker"] and corrige["1"] == lc[1]["speaker"]
    reg = lire_json(REGISTRE / "registre.json")
    assert {(e["groupe_local"], e["speaker"]) for e in reg["entrees"] if e["session"] == "S02"} >= \
        {(0, lc[0]["speaker"]), (1, lc[1]["speaker"])}
    corr.write_text(original, encoding="utf-8")
    appliquer(SORTIE, "S02", REGISTRE, p)
    transcrire_session(ESSAI / "medias" / "S01", "S01", "2026-03-11", SORTIE, REGISTRE, p,
                       transcripteur=faux_transcripteur(verites["S01"], 11))
    sexes = {l["speaker"]: l["sexe"] for l in lire_csv(chemin)}
    assert sexes[vrai_vers_id["B"]] == "F", "sexe saisi perdu après relance"
    s2 = {l["speaker"]: l["sexe"] for l in lire_csv(SORTIE / "S02" / "S02_locuteurs.csv")}
    assert s2.get(vrai_vers_id["B"]) == "F", "sexe non repris d'une session à l'autre"
    print("OK : sexe conservé après relance et repris dans S02 ; correspondances corrigées appliquées.")

    # 7. extraits corrigés non écrasés par une relance
    assert evaluer(SORTIE, "S01")["extraits_evalues"] == len(hyp["extraits"])
    print("OK : la relance n'écrase pas les extraits déjà corrigés.")
    print("\nBILAN :", json.dumps({s: {k: v for k, v in b.items() if k != "evaluation"} for s, b in bilan.items()},
                                  ensure_ascii=False))
    print("TOUS LES TESTS SONT PASSÉS.")


if __name__ == "__main__":
    main()
