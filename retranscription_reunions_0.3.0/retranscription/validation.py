"""Validation de la retranscription.

Niveau 1 — indicateurs automatiques (où regarder) : confiance des mots,
marge d'attribution des voix, silhouette du regroupement, qualité de
l'alignement des micros, appariements inter-sessions douteux.

Niveau 2 — mesure de l'erreur sur un échantillon corrigé à la main :
extraits tirés au hasard dans des strates de temps égales (toute la
session est couverte), corrigés dans ELAN, puis comparés à la sortie
automatique :

* WER, taux d'erreur sur les mots = (substitutions + suppressions +
  insertions) / mots de la référence, par alignement de Levenshtein
  (bibliothèque jiwer ; Morris, Maier & Green, 2004).
* DER, taux d'erreur de diarisation = (parole manquée + fausse alarme +
  confusion de locuteur) / durée de parole de la référence, après
  correspondance optimale des étiquettes (définition des campagnes NIST
  Rich Transcription ; Fiscus et al., 2006). Calcul exact par intervalles
  élémentaires ; résultats identiques à pyannote.metrics (Bredin, 2017),
  vérifiés dans les tests.
* Taux d'erreur d'identification : même calcul sans recorrespondance des
  étiquettes. Il mesure si « Speaker 03 » désigne bien la même personne
  que dans les autres sessions.
"""

import random
import re
import unicodedata
from pathlib import Path
from typing import List

import numpy as np

from .audio import ecrire_extrait_wav
from .registre import numero
from .sorties import Segment, ecrire_eaf, lire_eaf, _ecrire_csv, _f
from .utils import ecrire_json, hms, lire_json, log

HESITATIONS = {"euh", "heu", "euhm", "hum", "hm", "hmm", "mh", "mmh", "mm"}

CONSIGNES = """CORRECTION DES EXTRAITS — CONSIGNES

Chaque extrait = un fichier audio (.wav) + un fichier ELAN (.eaf) qui
contient la sortie automatique, une piste par locuteur.

Dans ELAN, pour chaque extrait :
 1. Écoutez l'extrait en entier.
 2. Corrigez le texte des annotations (mots erronés, oubliés, en trop).
 3. Si une phrase est attribuée au mauvais locuteur, déplacez-la sur la
    bonne piste (ou créez la piste « Speaker NN » manquante).
 4. Ajustez les frontières si elles sont fausses de plus d'une demi-seconde.
 5. Ajoutez les passages de parole oubliés ; supprimez ceux qui ne sont
    pas de la parole.
 6. Enregistrez le fichier sous le même nom (Fichier > Enregistrer).

Conventions, pour que la comparaison soit juste :
 - écrire les nombres en lettres (« vingt-trois », pas « 23 ») ;
 - pas d'abréviations ; la ponctuation et les majuscules sont ignorées ;
 - les hésitations (euh, hum…) sont ignorées dans le calcul ;
 - les commentaires entre crochets [rire] ou parenthèses (inaudible) sont ignorés ;
 - parole superposée : annotez chaque locuteur sur sa piste, même si les
   annotations se chevauchent.

Biais à connaître : corriger une sortie automatique (post-édition) tend à
laisser passer des erreurs plausibles. Écoutez avant de lire le texte
proposé. Ne modifiez pas le fichier « hypotheses.json ».

Ensuite :  python -m retranscription evaluer <SESSION>
"""


def _empreinte(annotations: List[tuple]) -> str:
    """Somme de contrôle des annotations : détecte un extrait non corrigé."""
    import hashlib
    brut = "\n".join(f"{t}|{a:.3f}|{b:.3f}|{x}" for t, a, b, x in sorted(annotations))
    return hashlib.sha1(brut.encode("utf-8")).hexdigest()


def normaliser(texte: str, ignorer_hesitations: bool = True) -> str:
    t = unicodedata.normalize("NFC", texte).lower()
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"'", "' ", t)
    t = re.sub(r"[-–—_]", " ", t)
    t = re.sub(r"[^\w' ]", " ", t)
    mots = t.split()
    if ignorer_hesitations:
        mots = [m for m in mots if m not in HESITATIONS]
    return " ".join(mots)


# ----------------------------------------------------------------------
def choisir_extraits(mots: List[dict], duree_session: float, n: int, duree: float,
                     graine: str) -> List[tuple]:
    """Un extrait par strate de temps ; au moins 50 % de parole ; bornes dans des pauses."""
    if duree_session < duree or n <= 0:
        return []
    rng = random.Random(graine)
    parole = sorted((m["debut"], m["fin"]) for m in mots)
    pas = 1.0
    # parole cumulée par pas de 0,1 s : taux de parole d'un intervalle en O(1)
    res = 0.1
    masque = np.zeros(int(np.ceil(duree_session / res)) + 2)
    for d, f in parole:
        masque[int(d / res):int(np.ceil(f / res))] = 1
    cumul = np.concatenate([[0.0], np.cumsum(masque)])
    fins = np.array([f for _, f in parole])
    debuts_suiv = np.array([d for d, _ in parole[1:]] + [np.inf])
    milieux_pauses = ((fins + debuts_suiv) / 2)[(debuts_suiv - fins) >= 0.3]

    def taux_parole(a, b):
        i, j = int(a / res), int(b / res)
        return (cumul[j] - cumul[i]) / max(j - i, 1)

    def vers_pause(t, sens):
        """Déplace t (± 3 s) vers le milieu d'une pause d'au moins 0,3 s."""
        proches = milieux_pauses[np.abs(milieux_pauses - t) <= 3]
        return float(proches[np.argmin(np.abs(proches - t))]) if len(proches) else t

    n = int(min(n, duree_session // duree))
    largeur = duree_session / n
    extraits = []
    for k in range(n):
        a0, a1 = k * largeur, min((k + 1) * largeur, duree_session) - duree
        candidats = [a0 + i * pas for i in range(int(max(0, a1 - a0) / pas) + 1)]
        rng.shuffle(candidats)
        choisi = None
        for c in candidats[:200]:
            if taux_parole(c, c + duree) >= 0.5:
                choisi = c
                break
        if choisi is None:
            continue
        debut = max(0.0, vers_pause(choisi, -1))
        fin = min(duree_session, vers_pause(debut + duree, 1))
        extraits.append((k + 1, round(debut, 3), round(fin, 3)))
    return extraits


def hypothese_extrait(segments: List[Segment], mots: List[dict], correspondance: dict,
                      debut: float, fin: float) -> dict:
    tours = []
    for s in segments:
        if s.fin > debut and s.debut < fin:
            tours.append([s.speaker, round(max(s.debut, debut) - debut, 3),
                          round(min(s.fin, fin) - debut, 3)])
    texte = [m["mot"] for m in mots
             if debut <= (m["debut"] + m["fin"]) / 2 < fin and m.get("groupe", -1) >= 0]
    return {"debut_session": debut, "fin_session": fin, "tours": tours, "texte": " ".join(texte)}


def preparer_extraits(dossier_val: Path, session: str, signal, segments: List[Segment],
                      mots: List[dict], correspondance: dict, duree_session: float,
                      n: int, duree: float, tiers: List[str], recreer: bool = False) -> None:
    dossier = dossier_val / "extraits"
    fichier_h = dossier / "hypotheses.json"
    if fichier_h.exists() and not recreer:
        # extraits déjà tirés (peut-être déjà corrigés) : on ne met à jour que
        # l'hypothèse de référence, jamais les .eaf
        hyp = lire_json(fichier_h)
        for e in hyp["extraits"]:
            e.update(hypothese_extrait(segments, mots, correspondance,
                                       e["debut_session"], e["fin_session"]))
        ecrire_json(fichier_h, hyp)
        return
    extraits = choisir_extraits(mots, duree_session, n, duree, graine=session)
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "CONSIGNES.txt").write_text(CONSIGNES, encoding="utf-8")
    sortie = {"session": session, "graine": session, "extraits": []}
    for k, debut, fin in extraits:
        nom = f"{session}_extrait_{k:02d}"
        ecrire_extrait_wav(signal, debut, fin, dossier / f"{nom}.wav")
        h = hypothese_extrait(segments, mots, correspondance, debut, fin)
        annotations = []
        dans = [m for m in mots if debut <= (m["debut"] + m["fin"]) / 2 < fin]
        for s in segments:
            if s.fin > debut and s.debut < fin:
                txt = " ".join(m["mot"] for m in dans
                               if s.debut <= m["debut"] and m["fin"] <= s.fin
                               and correspondance.get(m.get("groupe", -1)) == s.speaker)
                annotations.append((s.speaker, max(s.debut, debut) - debut,
                                    min(s.fin, fin) - debut, txt))
        ecrire_eaf(dossier / f"{nom}.eaf", annotations, tiers,
                   [(dossier / f"{nom}.wav", "audio/x-wav")])
        h.update({"numero": k, "fichier": f"{nom}.eaf", "debut": hms(debut), "fin": hms(fin),
                  "empreinte_eaf": _empreinte(lire_eaf(dossier / f"{nom}.eaf"))})
        sortie["extraits"].append(h)
    ecrire_json(fichier_h, sortie)
    log.info("%d extrait(s) de validation préparé(s) dans %s", len(extraits), dossier)


# ----------------------------------------------------------------------
def erreurs_diarisation(ref: List[tuple], hyp: List[tuple], debut: float, fin: float,
                        collier: float = 0.5, recorrespondance: bool = True) -> dict:
    """Composantes du DER (définition NIST), en secondes.

    ref, hyp : (locuteur, début, fin). Le collier est la tolérance totale
    autour de chaque frontière de la référence (moitié avant, moitié après).
    Avec recorrespondance, les étiquettes de l'hypothèse sont d'abord
    associées un-à-un à celles de la référence (méthode hongroise) ; sans,
    on obtient le taux d'erreur d'identification.
    """
    from scipy.optimize import linear_sum_assignment
    exclus = [(b - collier / 2, b + collier / 2) for _, a, z in ref for b in (a, z)] if collier > 0 else []
    bornes = {debut, fin}
    for _, a, z in list(ref) + list(hyp):
        bornes.update((a, z))
    for a, z in exclus:
        bornes.update((a, z))
    bornes = sorted(t for t in bornes if debut <= t <= fin)
    intervalles = []
    for t0, t1 in zip(bornes, bornes[1:]):
        if t1 <= t0:
            continue
        m = (t0 + t1) / 2
        if any(a <= m < z for a, z in exclus):
            continue
        R = {s for s, a, z in ref if a <= m < z}
        H = {s for s, a, z in hyp if a <= m < z}
        if R or H:
            intervalles.append((t1 - t0, R, H))
    noms_r = sorted({s for _, R, _ in intervalles for s in R})
    noms_h = sorted({s for _, _, H in intervalles for s in H})
    if recorrespondance and noms_r and noms_h:
        M = np.zeros((len(noms_r), len(noms_h)))
        for d, R, H in intervalles:
            for r in R:
                for h in H:
                    M[noms_r.index(r), noms_h.index(h)] += d
        li, co = linear_sum_assignment(-M)
        corr = {noms_h[j]: noms_r[i] for i, j in zip(li, co) if M[i, j] > 0}
    else:
        corr = {h: h for h in noms_h} if not recorrespondance else {}
    res = {"total": 0.0, "correct": 0.0, "manquee": 0.0, "fausse_alarme": 0.0, "confusion": 0.0}
    for d, R, H in intervalles:
        nr, nh = len(R), len(H)
        bon = len(R & {corr.get(h, ("hyp", h)) for h in H})
        res["total"] += d * nr
        res["correct"] += d * bon
        res["manquee"] += d * max(0, nr - nh)
        res["fausse_alarme"] += d * max(0, nh - nr)
        res["confusion"] += d * (min(nr, nh) - bon)
    return res


def _taux(c: dict) -> float:
    return (c["manquee"] + c["fausse_alarme"] + c["confusion"]) / c["total"] if c["total"] else float("nan")


def evaluer(dossier_val: Path, session: str, collier: float = 0.5,
            ignorer_hesitations: bool = True) -> dict:
    import jiwer

    dossier = dossier_val / "extraits"
    if not (dossier / "hypotheses.json").exists():
        log.error("Pas d'extraits pour la session %s : lancez d'abord la transcription.", session)
        return {}
    hyp = lire_json(dossier / "hypotheses.json")
    lignes, cumul = [], {"S": 0, "D": 0, "I": 0, "N": 0}
    cumul_der = {"total": 0.0, "correct": 0.0, "manquee": 0.0, "fausse_alarme": 0.0, "confusion": 0.0}
    cumul_ier = dict(cumul_der)
    non_corriges = []
    for e in hyp["extraits"]:
        chemin = dossier / e["fichier"]
        if not chemin.exists():
            continue
        ref = lire_eaf(chemin)
        duree = e["fin_session"] - e["debut_session"]
        if _empreinte(ref) == e.get("empreinte_eaf"):
            non_corriges.append(e["fichier"])
            continue
        ref_txt = normaliser(" ".join(x for _, _, _, x in ref), ignorer_hesitations)
        hyp_txt = normaliser(e["texte"], ignorer_hesitations)
        if not ref_txt:
            log.warning("%s : référence vide, extrait ignoré.", e["fichier"])
            continue
        n_ref = len(ref_txt.split())
        if hyp_txt:
            w = jiwer.process_words(ref_txt, hyp_txt)
            S, D, I, wer = w.substitutions, w.deletions, w.insertions, w.wer
        else:
            S, D, I, wer = 0, n_ref, 0, 1.0
        for c, v in (("S", S), ("D", D), ("I", I), ("N", n_ref)):
            cumul[c] += v
        tours_ref = [(t, a, b) for t, a, b, _ in ref if b > a]
        tours_hyp = [(t, a, b) for t, a, b in e["tours"] if b > a]
        der = erreurs_diarisation(tours_ref, tours_hyp, 0.0, duree, collier, True)
        ier = erreurs_diarisation(tours_ref, tours_hyp, 0.0, duree, collier, False)
        for k in cumul_der:
            cumul_der[k] += der[k]
            cumul_ier[k] += ier[k]
        total = der["total"] or 1.0
        lignes.append([session, e["numero"], e["debut"], e["fin"], n_ref, S, D, I,
                       _f(wer, 4), _f(_taux(der), 4),
                       _f(der["manquee"] / total, 4), _f(der["fausse_alarme"] / total, 4),
                       _f(der["confusion"] / total, 4), _f(_taux(ier), 4), _f(der["total"], 2)])
    if non_corriges:
        log.warning("Extraits non corrigés (ignorés) : %s", ", ".join(non_corriges))
    if not lignes:
        log.warning("Aucun extrait corrigé : rien à évaluer.")
        return {}
    wer_global = (cumul["S"] + cumul["D"] + cumul["I"]) / max(cumul["N"], 1)
    resultat = {
        "session": session, "extraits_evalues": len(lignes), "extraits_non_corriges": non_corriges,
        "mots_reference": cumul["N"], "substitutions": cumul["S"], "suppressions": cumul["D"],
        "insertions": cumul["I"], "WER": round(wer_global, 4),
        "DER": round(_taux(cumul_der), 4), "taux_erreur_identification": round(_taux(cumul_ier), 4),
        "collier_s": collier, "hesitations_ignorees": ignorer_hesitations,
    }
    lignes.append([session, "TOTAL", "", "", cumul["N"], cumul["S"], cumul["D"], cumul["I"],
                   resultat["WER"], resultat["DER"], "", "", "", resultat["taux_erreur_identification"], ""])
    _ecrire_csv(dossier_val / f"{session}_evaluation.csv",
                ["session", "extrait", "debut", "fin", "mots_reference", "substitutions",
                 "suppressions", "insertions", "WER", "DER", "part_parole_manquee",
                 "part_fausse_alarme", "part_confusion", "taux_erreur_identification",
                 "parole_reference_s"], lignes)
    ecrire_json(dossier_val / f"{session}_evaluation.json", resultat)
    log.info("WER = %.1f %% ; DER = %.1f %% ; erreur d'identification = %.1f %% (%d extraits)",
             100 * resultat["WER"], 100 * resultat["DER"],
             100 * resultat["taux_erreur_identification"], len(lignes) - 1)
    return resultat


# ----------------------------------------------------------------------
def ecrire_synthese(dossier_val: Path, session: str, infos: dict, segments: List[Segment],
                    mots: List[dict], decisions: List[dict]) -> None:
    confs = [m["conf"] for m in mots if m.get("conf") is not None]
    n_seg_v = sum(1 for s in segments if s.motifs)
    duree_parole = sum(s.duree for s in segments)
    synth = {
        "session": session,
        "duree_session_s": round(infos["duree_session"], 1),
        "duree_parole_segments_s": round(duree_parole, 1),
        "mots": len(mots),
        "confiance_moyenne": round(sum(confs) / len(confs), 4) if confs else None,
        "part_mots_conf_inf_0.5": round(sum(c < 0.5 for c in confs) / len(confs), 4) if confs else None,
        "moteur": infos.get("moteur", "vosk"),
        "modele_reconnaissance": infos.get("modele_reconnaissance", ""),
        "methode_diarisation": infos.get("methode_diarisation", "fenêtres glissantes"),
        "methode_empreinte": infos["methode_empreinte"],
        "fenetres": infos["fenetres"],
        "groupes_de_voix": infos["k"],
        "nombre_voix_impose": infos["k_impose"],
        "silhouette": infos["silhouette"],
        "silhouette_par_k": infos.get("silhouettes_k", {}),
        "segments": len(segments),
        "segments_a_verifier": n_seg_v,
        "groupes_a_verifier": [d["groupe_local"] for d in decisions if d["a_verifier"] == "oui"],
        "micros": infos["micros"],
        "parametres": infos["parametres"],
    }
    ecrire_json(dossier_val / f"{session}_synthese.json", synth)
    L = [f"VALIDATION — session {session}", "=" * 60, "",
         "1. Indicateurs automatiques (où regarder ; ce ne sont pas des taux d'erreur)", ""]
    L.append(f"Durée de la session : {hms(synth['duree_session_s'], ms=False)} ; "
             f"parole segmentée : {hms(duree_parole, ms=False)}")
    if confs:
        L.append(f"Mots reconnus : {len(mots)} ; confiance moyenne {synth['confiance_moyenne']:.2f} ; "
                 f"{100 * synth['part_mots_conf_inf_0.5']:.1f} % des mots sous 0,5")
    neurale = synth["methode_diarisation"].startswith("segmentation")
    L.append(f"Reconnaissance de la parole : {synth['moteur']} ({synth['modele_reconnaissance']})")
    L.append(f"Diarisation : {synth['methode_diarisation']}")
    L.append(f"Empreintes : {synth['methode_empreinte']} sur {synth['fenetres']} "
             + ("locuteurs locaux (fenêtres de 10 s)" if neurale else "fenêtres"))
    estime = " (estimé par seuil de distance)" if neurale else " (estimé par l'écart propre normalisé)"
    L.append(f"Groupes de voix : {synth['groupes_de_voix']}"
             + (" (nombre imposé)" if synth["nombre_voix_impose"] else estime))
    L.append(f"Silhouette du regroupement : {synth['silhouette']:.3f} (de -1 à 1 ; repères de "
             "Kaufman & Rousseeuw, 1990 : > 0,50 structure raisonnable, 0,26-0,50 faible, "
             "<= 0,25 absente)")
    L.append(f"Segments à vérifier : {n_seg_v} / {len(segments)} "
             f"(détail : {session}_indicateurs_segments.csv et piste « à vérifier » dans ELAN)")
    L += ["", "Micros et alignement :"]
    for m in infos["micros"]:
        res = "" if m["residu_ms"] is None else f", résidu {m['residu_ms']} ms"
        L.append(f"  {m['piste']} : décalage {m['decalage_s']} s, dérive {m['derive_ppm']} ppm "
                 f"({m['methode']}, {m['fenetres']} fenêtres, r={m['correlation']}{res}), "
                 f"retenu {100 * m['part_retenue']:.0f} % du temps")
    L += ["", "Correspondances avec le registre inter-sessions :"]
    for d in sorted(decisions, key=lambda d: numero(d["speaker"])):
        sim = "" if d["similarite"] != d["similarite"] else f"cos={d['similarite']:.2f}"
        L.append(f"  groupe {d['groupe_local']:>2} -> {d['speaker']} [{d['statut']}] {sim} "
                 f"{d['duree_parole_s']} s" + (f"  À VÉRIFIER : {d['motif']}" if d["a_verifier"] == "oui" else ""))
    L += ["", f"Pour corriger : modifier la colonne « speaker » de {session}_correspondances.csv,",
          f"puis lancer : python -m retranscription appliquer {session}", "",
          "2. Mesure de l'erreur (WER, DER)", "",
          "Corriger les extraits du dossier extraits/ (voir CONSIGNES.txt), puis :",
          f"  python -m retranscription evaluer {session}"]
    (dossier_val / f"{session}_synthese.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
