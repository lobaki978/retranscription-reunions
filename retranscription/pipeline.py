"""Enchaînement des étapes pour une session."""

import mimetypes
import shutil
from datetime import date as Date
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import sorties, validation
from .alignement import aligner
from .asr import Mot, charger_modele, transcrire_wav
from .audio import extraire_pistes, lister_medias, verifier_ffmpeg, wav_memmap
from .canaux import Micros
from .config import Parametres
from .diarisation import attribuer_mots, classer, normer, preparer
from .empreintes import ExtracteurMFCC, ExtracteurVosk, fenetres, regions_parole
from .registre import Registre, numero
from .utils import ecrire_json, lire_json, log


def _chemins(racine: Path, session: str) -> Dict[str, Path]:
    d = Path(racine) / session
    return {"session": d, "travail": d / "_travail", "validation": d / "validation",
            "audio": d / f"{session}_audio.wav"}


def transcrire_session(dossier_media: Path, session: str, date: Optional[str],
                       racine: Path, dossier_registre: Path, p: Parametres,
                       nouveaux_extraits: bool = False,
                       transcripteur=None, extracteur=None) -> None:
    """transcripteur / extracteur : injectables pour les tests."""
    c = _chemins(racine, session)
    for k in ("session", "travail", "validation"):
        c[k].mkdir(parents=True, exist_ok=True)
    date = date or Date.today().isoformat()
    verifier_ffmpeg()
    # modèles présents ? (vérifié avant les étapes longues)
    requis = []
    if transcripteur is None:
        requis.append(p.modele)
    if extracteur is None and p.empreinte == "vosk":
        requis.append(p.modele_spk)
    manquants = [m for m in requis if not Path(m).exists()]
    if manquants:
        raise FileNotFoundError("Modèle(s) Vosk introuvable(s) : " + ", ".join(manquants)
                                + "\nLancez d'abord : python -m retranscription telecharger-modeles")
    fichiers = lister_medias(dossier_media)
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier audio ou vidéo dans {dossier_media}")
    log.info("Session %s : %d fichier(s) : %s", session, len(fichiers),
             ", ".join(f.name for f in fichiers))

    # 1-3. pistes, alignement, meilleur micro, signal composite --------------
    pistes = extraire_pistes(fichiers, c["travail"] / "pistes")
    reference = p.reference or fichiers[0].name
    alignements = aligner(pistes, reference=reference, max_decalage_s=p.max_decalage_s)
    micros = Micros(pistes, alignements)
    if p.duree_max_s and micros.duree > p.duree_max_s:
        log.info("Essai : seules les %.0f premières minutes sont traitées.", p.duree_max_s / 60)
        micros.limiter(p.duree_max_s)
    micros.mesurer()
    micros.choisir()
    micros.ecrire_composite(c["audio"])
    duree = micros.duree
    log.info("Signal composite : %s (%.1f min)", c["audio"].name, duree / 60)

    # 4. reconnaissance de la parole ------------------------------------------
    cache = c["travail"] / "mots_asr.json"
    taille = c["audio"].stat().st_size
    if (p.reprendre and cache.exists() and lire_json(cache).get("taille_audio") == taille
            and lire_json(cache).get("modele") == Path(p.modele).name):
        mots = [Mot(**m) for m in lire_json(cache)["mots"]]
        log.info("Transcription Vosk existante réutilisée (%d mots).", len(mots))
    else:
        if transcripteur is None:
            modele = charger_modele(p.modele)
            mots = transcrire_wav(c["audio"], modele)
            del modele                      # libère la mémoire du grand modèle
        else:
            mots = transcripteur(c["audio"])
        ecrire_json(cache, {"taille_audio": taille, "modele": Path(p.modele).name,
                            "mots": [m.to_dict() for m in mots]})
    if not mots:
        raise RuntimeError("Aucun mot reconnu : vérifiez le son et le modèle.")

    # 5. empreintes vocales et regroupement -----------------------------------
    signal = wav_memmap(c["audio"])
    fen = fenetres(regions_parole(mots), p.fenetre_s, p.pas_s)
    if extracteur is None:
        if p.empreinte == "mfcc":
            extracteur = ExtracteurMFCC()
        else:
            # petit modèle : il ne sert qu'à faire tourner l'extracteur de x-vecteurs
            leger = p.modele_leger if Path(p.modele_leger).exists() else p.modele
            extracteur = ExtracteurVosk(p.modele_spk, charger_modele(leger))
    log.info("Empreintes vocales (%s) sur %d fenêtres…", extracteur.nom, len(fen))
    X, valides = extracteur.calculer(signal, fen)
    fen = fen[valides]
    profils = None
    if p.poids_canaux > 0 and len(pistes) > 1:
        profils = np.vstack([micros.profil(a, b) for a, b in fen])
    Z = preparer(X, profils, p.poids_canaux)
    res = classer(Z, k=p.nb_locuteurs, k_max=p.nb_max)
    attrib = attribuer_mots(mots, fen, res)
    for m, (g, s, mg) in zip(mots, attrib):
        m.groupe = -1 if g != g else int(g)
    durees = [sum(m.fin - m.debut for m in mots if m.groupe == g) for g in range(res.k)]
    # empreintes moyennes brutes par groupe (pour le registre) ; X est ensuite effacé
    Xn = normer(X)
    centres_bruts = np.vstack([Xn[res.etiquettes == g].mean(axis=0) for g in range(res.k)])
    del X, Xn, Z

    # 6. registre inter-sessions ----------------------------------------------
    registre = Registre(dossier_registre)
    decisions = registre.apparier(session, centres_bruts, durees, extracteur.nom,
                                  p.seuil_appariement, p.marge_appariement, p.duree_min_registre_s)
    correspondance = {d["groupe_local"]: d["speaker"] for d in decisions}
    registre.enregistrer(session, centres_bruts, durees, correspondance, extracteur.nom)
    sorties.ecrire_correspondances(decisions, session,
                                   c["validation"] / f"{session}_correspondances.csv")

    # 7. sauvegarde des résultats intermédiaires (sans empreinte vocale) -------
    mots_d = []
    for m, (g, s, mg) in zip(mots, attrib):
        d = m.to_dict()
        d["sim"] = None if s != s else round(float(s), 4)
        d["marge"] = None if mg != mg else round(float(mg), 4)
        mots_d.append(d)
    ecrire_json(c["travail"] / "mots.json", mots_d)
    medias = [(c["audio"], "audio/x-wav")]
    ref_fichier = next(f for f in fichiers if f.name == Path(reference).name)
    mime = mimetypes.guess_type(ref_fichier.name)[0] or ""
    if mime.startswith("video"):
        medias.insert(0, (ref_fichier, mime))
    infos = {
        "session": session, "date": date, "duree_session": duree,
        "fichiers": [str(f) for f in fichiers], "reference": str(ref_fichier),
        "medias": [[str(a), b] for a, b in medias],
        "methode_empreinte": extracteur.nom, "fenetres": int(len(fen)),
        "k": res.k, "k_impose": bool(p.nb_locuteurs), "silhouette": round(res.silhouette, 4),
        "silhouettes_k": res.silhouettes_k, "micros": micros.resume(),
        "decisions": decisions, "parametres": p.to_dict(),
    }
    ecrire_json(c["travail"] / "session.json", infos)
    sorties._ecrire_csv(c["validation"] / f"{session}_micros.csv",
                        list(infos["micros"][0].keys()),
                        [list(m.values()) for m in infos["micros"]])

    # 8. fichiers de retranscription et de validation --------------------------
    produire(racine, session, p, nouveaux_extraits=nouveaux_extraits)

    # Windows : un fichier ouvert en mémoire (memmap) ne peut pas être effacé ;
    # on libère d'abord toutes les références aux pistes
    del micros, pistes, alignements, signal
    import gc
    gc.collect()
    if not p.garder_pistes:
        shutil.rmtree(c["travail"] / "pistes", ignore_errors=True)
        if (c["travail"] / "pistes").exists():
            log.warning("Pistes temporaires non effacées : supprimez %s à la main.", c["travail"] / "pistes")
    log.info("Session %s terminée. Résultats : %s", session, c["session"])


def lire_correspondances(chemin: Path) -> Dict[int, str]:
    """Lit le fichier de correspondances ; « 5 », « speaker 5 », « Speaker 05 » -> « Speaker 05 »."""
    import re
    from .registre import nom_speaker
    corr = {}
    for ligne in sorties.lire_csv(chemin):
        spk = ligne["speaker"].strip()
        m = re.fullmatch(r"(?i)(speaker)?\s*0*(\d+)", spk)
        if m:
            spk = nom_speaker(int(m.group(2)))
        elif not spk:
            raise ValueError(f"{chemin.name} : speaker vide pour le groupe {ligne['groupe_local']}")
        corr[int(ligne["groupe_local"])] = spk
    return corr


def produire(racine: Path, session: str, p: Parametres, nouveaux_extraits: bool = False) -> None:
    c = _chemins(racine, session)
    infos = lire_json(c["travail"] / "session.json")
    mots = lire_json(c["travail"] / "mots.json")
    correspondance = lire_correspondances(c["validation"] / f"{session}_correspondances.csv")
    decisions = infos["decisions"]
    for d in decisions:
        d["speaker"] = correspondance.get(d["groupe_local"], d["speaker"])
    a_verifier = {d["groupe_local"] for d in decisions if d["a_verifier"] == "oui"}
    segments = sorties.segmenter(mots, correspondance, p.pause_max_s)
    sorties.signaler(segments, p, a_verifier)
    sorties.ecrire_transcription(segments, session, infos["date"], c["session"])
    sorties.ecrire_locuteurs(segments, session, infos["date"], c["session"], Path(racine), decisions)
    sorties.ecrire_eaf_session(segments, c["session"] / f"{session}_transcription.eaf",
                               [tuple(m) for m in infos["medias"]])
    sorties.ecrire_indicateurs(segments, session, c["validation"])
    validation.ecrire_synthese(c["validation"], session, infos, segments, mots, decisions)
    tiers = sorted(set(correspondance.values()), key=numero)
    validation.preparer_extraits(c["validation"], session, wav_memmap(c["audio"]), segments, mots,
                                 correspondance, infos["duree_session"], p.n_extraits,
                                 p.duree_extrait_s, tiers, recreer=nouveaux_extraits)


def appliquer(racine: Path, session: str, dossier_registre: Path, p: Parametres) -> None:
    """Après correction manuelle du fichier de correspondances."""
    c = _chemins(racine, session)
    correspondance = lire_correspondances(c["validation"] / f"{session}_correspondances.csv")
    Registre(dossier_registre).reattribuer(session, correspondance)
    infos = lire_json(c["travail"] / "session.json")
    for d in infos["decisions"]:
        nouveau = correspondance.get(d["groupe_local"], d["speaker"])
        if nouveau != d["speaker"]:
            d["speaker"], d["statut"], d["a_verifier"], d["motif"] = nouveau, "corrigé à la main", "non", ""
    ecrire_json(c["travail"] / "session.json", infos)
    sorties.ecrire_correspondances(infos["decisions"], session,
                                   c["validation"] / f"{session}_correspondances.csv")
    produire(racine, session, p)
    log.info("Correspondances appliquées à la session %s.", session)


def calibrer(racine: Path, dossier_registre: Path) -> dict:
    res = Registre(dossier_registre).calibrer()
    if not res["paires"]:
        log.warning("Il faut au moins deux sessions dans le registre.")
        return res
    entetes = list(res["paires"][0].keys())
    sorties._ecrire_csv(Path(racine) / "calibration_seuil.csv", entetes,
                        [[p[c] for c in entetes] for p in res["paires"]])
    log.info("Paires même locuteur : %d (min %.3f, médiane %.3f)", res["n_memes"],
             res["meme_min"] or float("nan"), res["meme_mediane"] or float("nan"))
    log.info("Paires locuteurs différents : %d (max %.3f, médiane %.3f)", res["n_diffs"],
             res["diff_max"] or float("nan"), res["diff_mediane"] or float("nan"))
    if res["seuil_propose"] is None:
        log.warning("Pas assez de paires des deux types pour proposer un seuil.")
    else:
        log.info("Seuil proposé : --seuil-appariement %.3f  (détail : %s)", res["seuil_propose"],
                 Path(racine) / "calibration_seuil.csv")
    return res


def evaluer(racine: Path, session: str, collier: float = 0.5) -> dict:
    c = _chemins(racine, session)
    return validation.evaluer(c["validation"], session, collier=collier)
