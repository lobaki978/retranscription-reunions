"""Interface en ligne de commande.

    python -m retranscription telecharger-modeles
    python -m retranscription transcrire DOSSIER_SESSION --session S01 --date 2026-03-12 --nb-locuteurs 12
    python -m retranscription appliquer S01
    python -m retranscription evaluer S01
    python -m retranscription effacer-registre --confirmer
"""

import argparse
import shutil
import sys
from pathlib import Path

from .config import MODELES, Parametres
from .utils import configurer_journal, log


def _options_communes(sp):
    sp.add_argument("--sortie", default="sorties", help="dossier des résultats (défaut : sorties)")
    sp.add_argument("--registre", default="registre",
                    help="dossier du registre des voix, à garder séparé (défaut : registre)")
    sp.add_argument("-v", "--verbeux", action="store_true")


def main(argv=None):  # noqa: C901
    defaut = Parametres()
    ap = argparse.ArgumentParser(prog="python -m retranscription",
                                 description="Retranscription de réunions sans IA générative.")
    sub = ap.add_subparsers(dest="commande", required=True)

    t = sub.add_parser("telecharger-modeles", help="télécharger les modèles (une seule fois)")
    t.add_argument("--dossier", default=str(MODELES))
    t.add_argument("--moteur", choices=["wav2vec2", "vosk"], default="wav2vec2")
    t.add_argument("--petit", action="store_true", help="modèle allégé (ordinateur peu puissant)")
    t.add_argument("--sans-grand-modele", action="store_true", help=argparse.SUPPRESS)
    t.add_argument("-v", "--verbeux", action="store_true")

    tr = sub.add_parser("transcrire", help="transcrire une session (dossier de vidéos/audio)")
    tr.add_argument("dossier", help="dossier contenant les vidéos ou fichiers audio d'une session")
    tr.add_argument("--session", required=True, help="identifiant court, ex. S01")
    tr.add_argument("--date", help="date de la session (AAAA-MM-JJ)")
    tr.add_argument("--nb-locuteurs", type=int, help="nombre de personnes qui parlent (recommandé)")
    tr.add_argument("--nb-max", type=int, default=defaut.nb_max)
    tr.add_argument("--reference", help="fichier dont le début sert de temps 0 (défaut : le premier)")
    tr.add_argument("--modele", default=defaut.modele)
    tr.add_argument("--modele-leger", default=defaut.modele_leger)
    tr.add_argument("--modele-spk", default=defaut.modele_spk)
    tr.add_argument("--moteur", choices=["wav2vec2", "vosk"], default=defaut.moteur,
                    help="reconnaissance de la parole (défaut : wav2vec2)")
    tr.add_argument("--petit", action="store_true", help="modèle allégé (ordinateur peu puissant)")
    tr.add_argument("--diarisation", choices=["neurale", "fenetres"], default=defaut.diarisation)
    tr.add_argument("--empreinte", choices=["wespeaker", "vosk", "mfcc"], default=defaut.empreinte,
                    help="empreintes de la méthode « fenetres »")
    tr.add_argument("--seuil-voix", type=float, default=defaut.seuil_voix,
                    help="distance de coupure si le nombre de voix est inconnu")
    tr.add_argument("--poids-canaux", type=float, default=defaut.poids_canaux,
                    help="0 à 1 : part de l'énergie des micros dans le regroupement (défaut 0)")
    tr.add_argument("--seuil-appariement", type=float, default=defaut.seuil_appariement)
    tr.add_argument("--pause-max", type=float, default=defaut.pause_max_s)
    tr.add_argument("--extraits", type=int, default=defaut.n_extraits)
    tr.add_argument("--duree-extrait", type=float, default=defaut.duree_extrait_s)
    tr.add_argument("--nouveaux-extraits", action="store_true",
                    help="retirer de nouveaux extraits (écrase ceux existants)")
    tr.add_argument("--refaire-asr", action="store_true", help="ne pas réutiliser la reconnaissance déjà faite")
    tr.add_argument("--garder-pistes", action="store_true")
    tr.add_argument("--essai", action="store_true", help="ne traiter que les 10 premières minutes")
    _options_communes(tr)

    a = sub.add_parser("appliquer", help="appliquer le fichier de correspondances corrigé")
    a.add_argument("session")
    _options_communes(a)

    e = sub.add_parser("evaluer", help="calculer WER et DER sur les extraits corrigés")
    e.add_argument("session")
    e.add_argument("--collier", type=float, default=0.5,
                   help="tolérance totale autour des frontières, en s (défaut 0,5 = ±0,25 s)")
    _options_communes(e)

    cb = sub.add_parser("calibrer", help="proposer un seuil d'appariement à partir des sessions vérifiées")
    _options_communes(cb)

    x = sub.add_parser("effacer-registre", help="effacer les empreintes vocales (fin de projet)")
    x.add_argument("--registre", default="registre")
    x.add_argument("--confirmer", action="store_true")
    x.add_argument("-v", "--verbeux", action="store_true")

    args = ap.parse_args(argv)
    configurer_journal(getattr(args, "verbeux", False))

    if args.commande == "telecharger-modeles":
        from .telecharger import necessaires, telecharger
        telecharger(Path(args.dossier), necessaires(args.moteur, args.petit or args.sans_grand_modele))
        return 0

    if args.commande == "transcrire":
        from .pipeline import transcrire_session
        p = Parametres(modele=args.modele, modele_leger=args.modele_leger, modele_spk=args.modele_spk,
                       moteur=args.moteur, petit_modele=args.petit, diarisation=args.diarisation,
                       seuil_voix=args.seuil_voix,
                       empreinte=args.empreinte, reference=args.reference,
                       poids_canaux=args.poids_canaux, nb_locuteurs=args.nb_locuteurs,
                       nb_max=args.nb_max, seuil_appariement=args.seuil_appariement,
                       pause_max_s=args.pause_max, n_extraits=args.extraits,
                       duree_extrait_s=args.duree_extrait, reprendre=not args.refaire_asr,
                       garder_pistes=args.garder_pistes,
                       duree_max_s=600.0 if args.essai else None)
        transcrire_session(Path(args.dossier), args.session, args.date, Path(args.sortie),
                           Path(args.registre), p, nouveaux_extraits=args.nouveaux_extraits)
        return 0

    if args.commande == "appliquer":
        from .pipeline import appliquer
        from .utils import lire_json
        infos = lire_json(Path(args.sortie) / args.session / "_travail" / "session.json")
        appliquer(Path(args.sortie), args.session, Path(args.registre), Parametres(**infos["parametres"]))
        return 0

    if args.commande == "evaluer":
        from .pipeline import evaluer
        res = evaluer(Path(args.sortie), args.session, collier=args.collier)
        return 0 if res else 1

    if args.commande == "calibrer":
        from .pipeline import calibrer
        res = calibrer(Path(args.sortie), Path(args.registre))
        return 0 if res.get("seuil_propose") is not None else 1

    if args.commande == "effacer-registre":
        d = Path(args.registre)
        if not args.confirmer:
            log.error("Ajoutez --confirmer pour effacer définitivement %s", d)
            return 1
        if d.exists():
            shutil.rmtree(d)
            log.info("Registre %s effacé.", d)
        else:
            log.info("Aucun registre à %s.", d)
        return 0
    return 1


def _lancer():
    try:
        return main()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(_lancer())
