"""Fenêtre de l'application (tkinter, inclus dans Python).

Toutes les étapes de la ligne de commande sont accessibles par des
boutons. Les traitements longs tournent dans un fil d'exécution séparé :
la fenêtre reste réactive et le journal s'affiche au fur et à mesure.
"""

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from datetime import date
from pathlib import Path

from . import __version__
from .config import MODELES, Parametres, dossier_donnees_utilisateur
from .utils import log

FICHIER_REGLAGES = dossier_donnees_utilisateur() / "reglages.json"
DOCUMENTS = Path.home() / "Documents"
GRAND, PETIT, SPK = "vosk-model-fr-0.22", "vosk-model-small-fr-0.22", "vosk-model-spk-0.4"


# ----------------------------------------------------------------------
def lire_reglages() -> dict:
    defaut = {"resultats": str(DOCUMENTS / "Retranscription" / "resultats"),
              "registre": str(DOCUMENTS / "Retranscription" / "registre"),
              "videos": "", "seuil": 0.7, "petit_modele": False}
    try:
        defaut.update(json.loads(FICHIER_REGLAGES.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaut


def ecrire_reglages(r: dict) -> None:
    try:
        FICHIER_REGLAGES.parent.mkdir(parents=True, exist_ok=True)
        FICHIER_REGLAGES.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def ouvrir_dossier(chemin: Path) -> None:
    chemin = Path(chemin)
    chemin.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(chemin))                      # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(chemin)])
    else:
        subprocess.Popen(["xdg-open", str(chemin)])


class AntiVeille:
    """Empêche la mise en veille pendant un traitement long."""

    def __enter__(self):
        self._proc = None
        try:
            if sys.platform == "win32":
                import ctypes
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, valable pour ce fil d'exécution
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            elif sys.platform == "darwin":
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            elif self._proc:
                self._proc.terminate()
        except Exception:
            pass
        return False


class VersFile(logging.Handler):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.file.put(("journal", self.format(record)))


# ----------------------------------------------------------------------
class Application:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.racine = tk.Tk()
        self.racine.title(f"Retranscription de réunions — version {__version__}")
        self.racine.geometry("900x740")
        self.racine.minsize(760, 600)
        self.file = queue.Queue()
        self.occupe = False
        self.reglages = lire_reglages()

        gestionnaire = VersFile(self.file)
        log.addHandler(gestionnaire)
        try:
            FICHIER_REGLAGES.parent.mkdir(parents=True, exist_ok=True)
            fichier = logging.FileHandler(FICHIER_REGLAGES.parent / "journal.txt", encoding="utf-8")
            fichier.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
            log.addHandler(fichier)
        except Exception:
            pass
        log.setLevel(logging.INFO)
        log.propagate = False

        self._construire()
        self.racine.protocol("WM_DELETE_WINDOW", self._fermer)
        self.racine.after(200, self._lire_file)
        self.racine.after(300, self._etat_modeles)

    # --- construction de la fenêtre ------------------------------------
    def _construire(self):
        tk, ttk = self.tk, self.ttk
        style = ttk.Style()
        try:
            style.theme_use("vista" if sys.platform == "win32" else ("aqua" if sys.platform == "darwin" else "clam"))
        except Exception:
            pass
        from tkinter import font as tkfont
        gras = tkfont.nametofont("TkDefaultFont").copy()
        gras.configure(weight="bold", size=max(gras.cget("size"), 10) + 1)
        self._police_titre = gras
        style.configure("Titre.TLabel", font=gras)
        petite = tkfont.nametofont("TkDefaultFont").copy()
        petite.configure(size=max(petite.cget("size") - 1, 8), slant="italic")
        self._police_aide = petite
        style.configure("Aide.TLabel", font=petite)
        style.configure("Grand.TButton", padding=(14, 8))

        principal = ttk.Frame(self.racine, padding=10)
        principal.pack(fill="both", expand=True)
        onglets = ttk.Notebook(principal)
        onglets.pack(fill="x")
        page1 = ttk.Frame(onglets, padding=10)
        page2 = ttk.Frame(onglets, padding=10)
        onglets.add(page1, text="  1. Transcrire  ")
        onglets.add(page2, text="  2. Vérifier et mesurer  ")
        lignes = {page1: 0, page2: 0}
        for page in (page1, page2):
            page.columnconfigure(1, weight=1)

        def titre(page, texte):
            ttk.Label(page, text=texte, style="Titre.TLabel").grid(row=lignes[page], column=0, columnspan=3,
                                                                   sticky="w", pady=(8, 4))
            lignes[page] += 1

        def aide(page, texte, colonne=1):
            ttk.Label(page, text=texte, style="Aide.TLabel", wraplength=600, justify="left").grid(
                row=lignes[page], column=colonne, columnspan=3 - colonne, sticky="w", pady=(0, 3))
            lignes[page] += 1

        def ligne(page, etiquette, widget, bouton=None, texte_aide=None):
            L = lignes[page]
            ttk.Label(page, text=etiquette).grid(row=L, column=0, sticky="w", padx=(0, 8), pady=3)
            widget.grid(row=L, column=1, sticky="ew", pady=3)
            if bouton:
                bouton.grid(row=L, column=2, sticky="ew", padx=(8, 0), pady=3)
            lignes[page] += 1
            if texte_aide:
                aide(page, texte_aide)

        # --- onglet 1 : transcrire -------------------------------------------
        titre(page1, "Session à transcrire")
        self.v_videos = tk.StringVar(value=self.reglages.get("videos", ""))
        ligne(page1, "Dossier des vidéos", ttk.Entry(page1, textvariable=self.v_videos),
              ttk.Button(page1, text="Choisir…", command=self._choisir_videos),
              "Un dossier par session, avec toutes ses vidéos (ou fichiers son). Rien n'est converti ni modifié.")
        self.v_reference = tk.StringVar()
        self.c_reference = ttk.Combobox(page1, textvariable=self.v_reference, state="readonly")
        ligne(page1, "Vidéo de référence", self.c_reference, None,
              "Son début = temps 0 de la transcription. Choisissez celle que vous utilisez pour l'analyse.")
        petits = ttk.Frame(page1)
        self.v_session = tk.StringVar()
        self.v_date = tk.StringVar(value=date.today().isoformat())
        self.v_nb = tk.StringVar(value="0")
        ttk.Entry(petits, textvariable=self.v_session, width=12).pack(side="left")
        ttk.Label(petits, text="   Date (AAAA-MM-JJ)").pack(side="left")
        ttk.Entry(petits, textvariable=self.v_date, width=12).pack(side="left", padx=(6, 0))
        ttk.Label(petits, text="   Personnes qui parlent").pack(side="left")
        ttk.Spinbox(petits, from_=0, to=60, textvariable=self.v_nb, width=5).pack(side="left", padx=(6, 0))
        ligne(page1, "Nom de la session", petits, None,
              "Nom court sans espace (ex. S01). Personnes : participants et animateurs ; "
              "0 = je ne sais pas (estimation automatique, moins fiable).")
        self.v_petit = tk.BooleanVar(value=bool(self.reglages.get("petit_modele")))
        ttk.Checkbutton(page1, text="Ordinateur peu puissant (moins de 8 Go de mémoire) : petit modèle, moins précis",
                        variable=self.v_petit).grid(row=lignes[page1], column=1, columnspan=2, sticky="w", pady=3)
        lignes[page1] += 1
        self.v_essai = tk.BooleanVar(value=False)
        ttk.Checkbutton(page1, text="Essai : seulement les 10 premières minutes (pour vérifier que tout fonctionne)",
                        variable=self.v_essai).grid(row=lignes[page1], column=1, columnspan=2, sticky="w", pady=3)
        lignes[page1] += 1
        titre(page1, "Dossiers de travail (mémorisés)")
        self.v_resultats = tk.StringVar(value=self.reglages["resultats"])
        ligne(page1, "Résultats", ttk.Entry(page1, textvariable=self.v_resultats),
              ttk.Button(page1, text="Choisir…", command=lambda: self._choisir(self.v_resultats)),
              "Évitez un dossier synchronisé (OneDrive, Dropbox) : le son de chaque session pèse environ 700 Mo.")
        self.v_registre = tk.StringVar(value=self.reglages["registre"])
        ligne(page1, "Empreintes vocales", ttk.Entry(page1, textvariable=self.v_registre),
              ttk.Button(page1, text="Choisir…", command=lambda: self._choisir(self.v_registre)),
              "Données biométriques : à ne pas partager. Gardez le même dossier pour toutes les sessions d'un projet.")
        self.b_transcrire = ttk.Button(page1, text="Transcrire", style="Grand.TButton", command=self._transcrire)
        self.b_transcrire.grid(row=lignes[page1], column=1, sticky="w", pady=(10, 2))
        lignes[page1] += 1

        # --- onglet 2 : vérifier et mesurer ---------------------------------
        self.v_session_suivi = tk.StringVar()
        self.c_sessions = ttk.Combobox(page2, textvariable=self.v_session_suivi, state="readonly", width=20,
                                       postcommand=self._lister_sessions)
        ligne(page2, "Session", self.c_sessions)
        etapes = [
            ("Ouvrir les résultats", self._ouvrir_resultats,
             "Lisez d'abord validation/<session>_synthese.txt : il indique ce qui est à vérifier."),
            ("Appliquer les numéros corrigés", self._appliquer,
             "Si une personne a un mauvais numéro : corrigez la colonne « speaker » de "
             "validation/<session>_correspondances.csv, enregistrez, puis cliquez ici."),
            ("Mesurer la fiabilité", self._evaluer,
             "Corrigez d'abord les extraits de validation/extraits dans ELAN (logiciel gratuit), "
             "en suivant CONSIGNES.txt. Calcule le WER et le DER."),
            ("Calibrer le seuil", self._calibrer,
             "Après 2 ou 3 sessions vérifiées : propose le seuil qui reconnaît le mieux les mêmes personnes "
             "d'une session à l'autre."),
        ]
        self._boutons_etapes = []
        for texte, commande, explication in etapes:
            b = ttk.Button(page2, text=texte, command=commande, width=30)
            b.grid(row=lignes[page2], column=0, sticky="w", pady=(8, 0))
            ttk.Label(page2, text=explication, style="Aide.TLabel", wraplength=520, justify="left").grid(
                row=lignes[page2], column=1, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 0))
            lignes[page2] += 1
            self._boutons_etapes.append(b)
        self.b_ouvrir, self.b_appliquer, self.b_evaluer, self.b_calibrer = self._boutons_etapes
        self.v_seuil = tk.StringVar(value=str(self.reglages.get("seuil", 0.7)))
        avance = ttk.Frame(page2)
        avance.grid(row=lignes[page2], column=0, columnspan=3, sticky="w", pady=(14, 0))
        lignes[page2] += 1
        ttk.Label(avance, text="Seuil d'appariement entre sessions :").pack(side="left")
        ttk.Entry(avance, textvariable=self.v_seuil, width=6).pack(side="left", padx=6)
        ttk.Label(avance, text="(0,7 par défaut ; remplacé par « Calibrer le seuil »)",
                  style="Aide.TLabel").pack(side="left")

        # --- journal, toujours visible -------------------------------------
        ttk.Label(principal, text="Journal", style="Titre.TLabel").pack(anchor="w", pady=(10, 4))
        zone = ttk.Frame(principal)
        zone.pack(fill="both", expand=True)
        zone.rowconfigure(0, weight=1)
        zone.columnconfigure(0, weight=1)
        self.journal = tk.Text(zone, height=5, wrap="word", state="disabled",
                               font=("Consolas" if sys.platform == "win32" else "Menlo", 9))
        barre = ttk.Scrollbar(zone, command=self.journal.yview)
        self.journal.configure(yscrollcommand=barre.set)
        self.journal.grid(row=0, column=0, sticky="nsew")
        barre.grid(row=0, column=1, sticky="ns")
        bas = ttk.Frame(principal)
        bas.pack(fill="x", pady=(6, 0))
        self.progression = ttk.Progressbar(bas, mode="indeterminate", length=160)
        self.progression.pack(side="left", padx=(0, 8))
        self.v_etat = tk.StringVar(value="Prêt.")
        ttk.Label(bas, textvariable=self.v_etat).pack(side="left")
        ttk.Button(bas, text="Guide d'utilisation", command=self._guide).pack(side="right")
        self._boutons = [self.b_transcrire, self.b_appliquer, self.b_evaluer, self.b_calibrer]
        if self.v_videos.get():
            self._remplir_references()

    # --- petites actions --------------------------------------------------
    def _ecrire(self, texte):
        self.journal.configure(state="normal")
        self.journal.insert("end", texte + "\n")
        self.journal.see("end")
        self.journal.configure(state="disabled")

    def _choisir(self, variable):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=variable.get() or str(Path.home()))
        if d:
            variable.set(d)

    def _choisir_videos(self):
        self._choisir(self.v_videos)
        self._remplir_references()

    def _remplir_references(self):
        from .audio import lister_medias
        d = Path(self.v_videos.get())
        if not d.is_dir():
            return
        noms = [f.name for f in lister_medias(d)]
        self.c_reference["values"] = noms
        if noms and self.v_reference.get() not in noms:
            self.v_reference.set(noms[0])
        if not noms:
            self._ecrire(f"Aucune vidéo ni fichier son reconnu dans {d}.")
        if not self.v_session.get():
            nom = re.sub(r"[^A-Za-z0-9_-]+", "_", d.name).strip("_")[:20]
            self.v_session.set(nom or "S01")

    def _lister_sessions(self):
        d = Path(self.v_resultats.get())
        sessions = sorted(p.name for p in d.glob("*") if (p / "_travail" / "session.json").exists()) if d.is_dir() else []
        self.c_sessions["values"] = sessions
        if sessions and self.v_session_suivi.get() not in sessions:
            self.v_session_suivi.set(sessions[-1])

    def _guide(self):
        import webbrowser
        from tkinter import messagebox
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        for nom in ("GUIDE_UTILISATEUR.html", "GUIDE_UTILISATEUR.md"):
            if (base / nom).exists():
                webbrowser.open((base / nom).resolve().as_uri())
                return
        messagebox.showinfo("Retranscription", "Le guide d'utilisation est fourni avec l'application "
                                               "(fichier GUIDE_UTILISATEUR).")

    def _ouvrir_resultats(self):
        s = self.v_session_suivi.get()
        d = Path(self.v_resultats.get())
        ouvrir_dossier(d / s if s and (d / s).exists() else d)

    def _sauver_reglages(self):
        self.reglages.update({"resultats": self.v_resultats.get(), "registre": self.v_registre.get(),
                              "videos": self.v_videos.get(), "seuil": self._seuil(),
                              "petit_modele": bool(self.v_petit.get())})
        ecrire_reglages(self.reglages)

    def _seuil(self) -> float:
        try:
            return float(self.v_seuil.get().replace(",", "."))
        except ValueError:
            return 0.7

    # --- modèles ----------------------------------------------------------
    def _modeles_manquants(self, petit: bool):
        requis = [PETIT, SPK] if petit else [GRAND, PETIT, SPK]
        return [m for m in requis if not (MODELES / m).exists()]

    def _etat_modeles(self):
        manq = self._modeles_manquants(bool(self.v_petit.get()))
        self._ecrire(f"Retranscription de réunions, version {__version__}.")
        if manq:
            self._ecrire("Les modèles de reconnaissance ne sont pas encore installés. Ils seront "
                         "téléchargés au premier clic sur « Transcrire » (environ 1,5 Go, une seule fois).")
        else:
            self._ecrire(f"Modèles installés dans {MODELES}.")

    # --- exécution en arrière-plan ---------------------------------------
    def _lancer(self, titre, fonction):
        if self.occupe:
            return
        self.occupe = True
        for b in self._boutons:
            b.state(["disabled"])
        self.progression.start(12)
        self.v_etat.set(titre + "…")

        def travail():
            try:
                with AntiVeille():
                    message = fonction()
                self.file.put(("fini", message or "Terminé."))
            except Exception as e:  # noqa: BLE001
                self.file.put(("journal", traceback.format_exc()))
                self.file.put(("erreur", str(e)))
        threading.Thread(target=travail, daemon=True).start()

    def _lire_file(self):
        from tkinter import messagebox
        try:
            while True:
                genre, contenu = self.file.get_nowait()
                if genre == "journal":
                    self._ecrire(contenu)
                elif genre == "seuil":
                    self.v_seuil.set(contenu)
                elif genre in ("fini", "erreur"):
                    self.occupe = False
                    for b in self._boutons:
                        b.state(["!disabled"])
                    self.progression.stop()
                    if genre == "fini":
                        self.v_etat.set("Terminé.")
                        self._lister_sessions()
                        messagebox.showinfo("Retranscription", contenu)
                    else:
                        self.v_etat.set("Erreur.")
                        messagebox.showerror("Retranscription", contenu)
        except queue.Empty:
            pass
        self.racine.after(200, self._lire_file)

    # --- actions ----------------------------------------------------------
    def _transcrire(self):
        from tkinter import messagebox
        videos = Path(self.v_videos.get())
        session = self.v_session.get().strip()
        if not videos.is_dir():
            messagebox.showwarning("Retranscription", "Choisissez d'abord le dossier des vidéos.")
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", session):
            messagebox.showwarning("Retranscription",
                                   "Nom de session : lettres, chiffres, - ou _, sans espace (ex. S01).")
            return
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.v_date.get().strip()):
            messagebox.showwarning("Retranscription", "Date au format AAAA-MM-JJ, par exemple 2026-03-12.")
            return
        try:
            nb = int(self.v_nb.get())
        except ValueError:
            nb = 0
        petit = bool(self.v_petit.get())
        sortie = Path(self.v_resultats.get())
        if (sortie / session / "_travail" / "session.json").exists():
            if not messagebox.askyesno("Retranscription",
                                       f"La session {session} existe déjà. La retranscrire à nouveau ?\n"
                                       "(Les extraits déjà corrigés et la colonne « sexe » sont conservés.)"):
                return
        manq = self._modeles_manquants(petit)
        if manq and not messagebox.askyesno(
                "Retranscription", "Les modèles de reconnaissance doivent être téléchargés une fois "
                f"({'environ 50 Mo' if petit else 'environ 1,5 Go'}). Les télécharger maintenant ?"):
            return
        self._sauver_reglages()
        p = Parametres(nb_locuteurs=nb or None, reference=self.v_reference.get() or None,
                       seuil_appariement=self._seuil(),
                       duree_max_s=600.0 if self.v_essai.get() else None)
        if petit:
            p.modele = str(MODELES / PETIT)
        date_s = self.v_date.get().strip()
        registre = Path(self.v_registre.get())

        def faire():
            from .pipeline import transcrire_session
            from .telecharger import telecharger
            if manq:
                telecharger(MODELES, manq)
            transcrire_session(videos, session, date_s, sortie, registre, p)
            return (f"Session {session} transcrite.\n\nRésultats : {sortie / session}\n"
                    f"Commencez par validation/{session}_synthese.txt.")
        self._lancer(f"Transcription de {session}", faire)

    def _session_choisie(self):
        from tkinter import messagebox
        self._lister_sessions()
        s = self.v_session_suivi.get()
        if not s:
            messagebox.showwarning("Retranscription", "Aucune session transcrite dans le dossier des résultats.")
        return s

    def _appliquer(self):
        s = self._session_choisie()
        if not s:
            return
        sortie, registre = Path(self.v_resultats.get()), Path(self.v_registre.get())

        def faire():
            from .pipeline import appliquer
            from .utils import lire_json
            infos = lire_json(sortie / s / "_travail" / "session.json")
            appliquer(sortie, s, registre, Parametres(**infos["parametres"]))
            return f"Numéros corrigés appliqués à la session {s}."
        self._lancer(f"Application des corrections ({s})", faire)

    def _evaluer(self):
        s = self._session_choisie()
        if not s:
            return
        sortie = Path(self.v_resultats.get())

        def faire():
            from .pipeline import evaluer
            r = evaluer(sortie, s)
            if not r:
                return ("Aucun extrait corrigé pour l'instant.\nCorrigez les fichiers du dossier "
                        f"validation/extraits de {s} (voir CONSIGNES.txt), puis recommencez.")
            return (f"Session {s} : {r['extraits_evalues']} extraits évalués.\n\n"
                    f"Taux d'erreur sur les mots (WER) : {100 * r['WER']:.1f} %\n"
                    f"Taux d'erreur de diarisation (DER) : {100 * r['DER']:.1f} %\n"
                    f"Erreur d'identification : {100 * r['taux_erreur_identification']:.1f} %\n\n"
                    f"Détail : validation/{s}_evaluation.csv")
        self._lancer(f"Mesure de la fiabilité ({s})", faire)

    def _calibrer(self):
        sortie, registre = Path(self.v_resultats.get()), Path(self.v_registre.get())

        def faire():
            from .pipeline import calibrer
            r = calibrer(sortie, registre)
            if r.get("seuil_propose") is None:
                return ("Pas encore assez de sessions vérifiées pour proposer un seuil.\n"
                        "Il faut au moins deux sessions, avec des locuteurs communs.")
            self.file.put(("seuil", str(r["seuil_propose"])))
            return (f"Seuil proposé : {r['seuil_propose']}\n(reporté dans la fenêtre ; il servira "
                    "aux prochaines transcriptions).\n\nDétail : calibration_seuil.csv dans les résultats.")
        self._lancer("Calibrage du seuil", faire)

    def _fermer(self):
        from tkinter import messagebox
        if self.occupe and not messagebox.askokcancel(
                "Retranscription", "Un traitement est en cours. Quitter l'arrêtera. Quitter quand même ?"):
            return
        self._sauver_reglages()
        self.racine.destroy()
        os._exit(0)

    def lancer(self):
        self.racine.mainloop()


# ----------------------------------------------------------------------
COMMANDES = {"telecharger-modeles", "transcrire", "appliquer", "evaluer", "calibrer", "effacer-registre"}


def _rediriger_sorties():
    """Application sans console : les messages vont dans un fichier journal."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        dossier = dossier_donnees_utilisateur()
        dossier.mkdir(parents=True, exist_ok=True)
        f = open(dossier / "journal.txt", "a", encoding="utf-8", buffering=1)
    except Exception:
        f = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def main(argv=None):
    import multiprocessing
    multiprocessing.freeze_support()
    _rediriger_sorties()
    argv = sys.argv[1:] if argv is None else argv
    argv = [a for a in argv if not a.startswith("-psn")]       # argument ajouté par le Finder (macOS)
    if argv and argv[0] == "--verifier":
        from .autotest import verifier
        return verifier(argv[1] if len(argv) > 1 else None)
    if argv and argv[0] in COMMANDES:                          # mêmes commandes qu'en ligne de commande
        from .__main__ import _lancer
        sys.argv = [sys.argv[0]] + list(argv)
        return _lancer()
    Application().lancer()
    return 0
