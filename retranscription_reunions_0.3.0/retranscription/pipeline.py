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
from .config import Parametres, seuil_appariement
from .diarisation import attribuer_mots, classer, lisser_tours, normer, preparer
from .empreintes import ExtracteurMFCC, ExtracteurVosk, fenetres, regions_parole
from .registre import Registre, numero
from .utils import ecrire_json, lire_json, log


def modele_reconnaissance(p: Parametres) -> str:
    if p.moteur == "vosk":
        return p.modele_leger if p.petit_modele else p.modele
    return p.modele_w2v_leger if p.petit_modele else p.modele_w2v


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
        requis.append(modele_reconnaissance(p))
    if extracteur is None and p.diarisation == "neurale":
        requis += [p.modele_segmentation, p.modele_wespeaker]
    elif extracteur is None and p.empreinte == "vosk":
        requis.append(p.modele_spk)
    elif extracteur is None and p.empreinte == "wespeaker":
        requis.append(p.modele_wespeaker)
    manquants = [m for m in requis if not Path(m).exists()]
    if manquants:
        raise FileNotFoundError("Modèle(s) introuvable(s) : " + ", ".join(manquants)
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
    nom_modele = modele_reconnaissance(p)
    if (p.reprendre and cache.exists() and lire_json(cache).get("taille_audio") == taille
            and lire_json(cache).get("modele") == Path(nom_modele).name):
        mots = [Mot(**m) for m in lire_json(cache)["mots"]]
        log.info("Reconnaissance existante réutilisée (%d mots).", len(mots))
    else:
        if transcripteur is not None:
            mots = transcripteur(c["audio"])
        elif p.moteur == "vosk":
            modele = charger_modele(nom_modele)
            mots = transcrire_wav(c["audio"], modele)
            del modele                      # libère la mémoire du grand modèle
        else:
            from .asr_w2v import TranscripteurW2V
            moteur = TranscripteurW2V(nom_modele)
            mots = moteur.transcrire_wav(c["audio"])
            del moteur
        ecrire_json(cache, {"taille_audio": taille, "modele": Path(nom_modele).name,
                            "mots": [m.to_dict() for m in mots]})
    if not mots:
        raise RuntimeError("Aucun mot reconnu : vérifiez le son et le modèle.")

    # 5. qui parle quand -------------------------------------------------------
    signal = wav_memmap(c["audio"])
    neurale = (p.diarisation == "neurale" and extracteur is None
               and Path(p.modele_segmentation).exists() and Path(p.modele_wespeaker).exists())
    if neurale:
        from .segmentation import ExtracteurWeSpeaker, Segmenteur, attribuer_mots_tours, diariser
        log.info("Diarisation : segmentation pyannote 3.0 + empreintes WeSpeaker…")
        extracteur = ExtracteurWeSpeaker(p.modele_wespeaker)
        resd = diariser(signal, duree, Segmenteur(p.modele_segmentation), extracteur,
                        k=p.nb_locuteurs, k_max=p.nb_max, seuil_distance=p.seuil_voix)
        attrib = attribuer_mots_tours(mots, resd, portee_s=float("inf"))
        k_voix, centres_bruts, durees = resd.k, resd.centres_bruts, resd.durees
        silhouette, silhouettes_k, n_fen = resd.silhouette, resd.silhouettes_k, resd.n_empreintes
        nom_empreinte = extracteur.nom
        seuil_marge = p.seuil_marge_activite
    else:
        fen = fenetres(regions_parole(mots), p.fenetre_s, p.pas_s)
        if extracteur is None:
            if p.empreinte == "mfcc":
                extracteur = ExtracteurMFCC()
            elif p.empreinte == "wespeaker" and Path(p.modele_wespeaker).exists():
                from .segmentation import ExtracteurWeSpeaker
                extracteur = ExtracteurWeSpeaker(p.modele_wespeaker)
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
        k_voix = res.k
        durees = [sum(m.fin - m.debut for m, g in zip(mots, attrib[:, 0]) if g == kk) for kk in range(res.k)]
        # empreintes moyennes brutes par groupe (pour le registre) ; X est ensuite effacé
        Xn = normer(X)
        centres_bruts = np.vstack([Xn[res.etiquettes == g].mean(axis=0) for g in range(res.k)])
        silhouette, silhouettes_k, n_fen = res.silhouette, res.silhouettes_k, len(fen)
        nom_empreinte = extracteur.nom
        seuil_marge = p.seuil_marge
        del X, Xn, Z
    # tours trop brefs coincés entre deux tours d'une même autre voix
    attrib[:, 0] = lisser_tours(mots, attrib[:, 0], p.tour_min_s)
    for m, (g, s_, mg) in zip(mots, attrib):
        m.groupe = -1 if g != g else int(g)

    # 6. registre inter-sessions ----------------------------------------------
    registre = Registre(dossier_registre)
    seuil = seuil_appariement(p, nom_empreinte)
    log.info("Seuil d'appariement entre sessions : %.2f", seuil)
    decisions = registre.apparier(session, centres_bruts, durees, nom_empreinte,
                                  seuil, p.marge_appariement, p.duree_min_registre_s)
    correspondance = {d["groupe_local"]: d["speaker"] for d in decisions}
    registre.enregistrer(session, centres_bruts, durees, correspondance, nom_empreinte)
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
        "methode_empreinte": nom_empreinte, "fenetres": int(n_fen),
        "methode_diarisation": "segmentation pyannote 3.0 + WeSpeaker" if neurale else "fenêtres glissantes",
        "moteur": p.moteur, "modele_reconnaissance": Path(nom_modele).name, "seuil_marge": seuil_marge,
        "k": k_voix, "k_impose": bool(p.nb_locuteurs), "silhouette": round(silhouette, 4),
        "silhouettes_k": silhouettes_k, "micros": micros.resume(),
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
    p_ind = Parametres(**{**p.to_dict(), "seuil_marge": infos.get("seuil_marge", p.seuil_marge)})
    sorties.signaler(segments, p_ind, a_verifier)
    methode = "Vosk/Kaldi" if infos.get("moteur", "vosk") == "vosk" else "wav2vec 2.0 + CTC"
    sorties.ecrire_transcription(segments, session, infos["date"], c["session"], methode)
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
