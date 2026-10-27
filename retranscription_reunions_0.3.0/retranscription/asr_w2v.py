"""Reconnaissance de la parole par wav2vec 2.0 avec décodage CTC.

Le réseau wav2vec 2.0 (Baevski et al., 2020) attribue à chaque trame de
20 ms une probabilité pour chaque caractère (lettres, apostrophe,
séparateur de mots) et pour un symbole « vide ». Le décodage CTC
(Graves et al., 2006) garde le caractère le plus probable de chaque
trame, fusionne les répétitions et retire les « vides ». Il n'y a pas de
décodeur autorégressif : le texte n'est pas généré mot après mot à partir
des mots précédents, chaque trame est classée à partir du son.

Le modèle par défaut (bofenghuang/asr-wav2vec2-ctc-french, Apache 2.0)
est pré-entraîné sur 7 000 h de français, parole spontanée comprise
(LeBenchmark ; Evain et al., 2021), puis affiné sur environ 2 200 h de
français transcrit.

Le son est traité par blocs de 30 s avec 2 s de contexte de chaque côté ;
seules les trames centrales de chaque bloc sont gardées.
"""

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from .asr import Mot
from .audio import wav_memmap
from .utils import TAUX, hms, log


class TranscripteurW2V:
    def __init__(self, dossier: str, fils: int = 0):
        import onnxruntime as ort
        dossier = Path(dossier)
        cfg = json.loads((dossier / "config_retranscription.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if fils:
            opts.intra_op_num_threads = fils
        log.info("Chargement du modèle %s…", cfg.get("modele_source", dossier.name))
        self.sess = ort.InferenceSession(str(dossier / "model.onnx"), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.vocab = {int(i): t for i, t in cfg["vocabulaire"].items()}
        self.vide = int(cfg["vide"])
        self.sep = cfg.get("separateur", "|")
        self.normaliser = bool(cfg.get("normaliser", True))
        self.ept = int(cfg.get("echantillons_par_trame", 320))
        self.speciaux = {i for i, t in self.vocab.items()
                         if t.startswith("<") and t.endswith(">") and i != self.vide}
        self.nom = Path(dossier).name

    # ------------------------------------------------------------------
    def _logits(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / 32768.0
        if self.normaliser:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        return self.sess.run(None, {"input_values": x[None, :]})[0][0]

    def trames(self, signal: np.ndarray, bloc_s: float = 30.0, contexte_s: float = 2.0):
        """Étiquette la plus probable et sa probabilité, pour chaque trame de 20 ms."""
        n = len(signal)
        pas = int(bloc_s * TAUX) // self.ept * self.ept
        ctx = int(contexte_s * TAUX) // self.ept * self.ept
        ids, probs = [], []
        t0 = time.time()
        jalon = 0.05
        for debut in range(0, n, pas):
            a, b = max(0, debut - ctx), min(n, debut + pas + ctx)
            x = np.asarray(signal[a:b])
            if len(x) < 400:
                break
            lg = self._logits(x)
            f0 = (debut - a) // self.ept
            f1 = min(len(lg), f0 + (min(debut + pas, n) - debut) // self.ept)
            bloc = lg[f0:f1].astype(np.float64)
            m = bloc.max(axis=1, keepdims=True)
            p = np.exp(bloc - m)
            p /= p.sum(axis=1, keepdims=True)
            ids.append(bloc.argmax(axis=1).astype(np.int32))
            probs.append(p.max(axis=1).astype(np.float32))
            if (debut + pas) / n >= jalon:
                log.info("  wav2vec 2.0 : %3.0f %% (%s traités, %.0f min écoulées)",
                         100 * min(1, (debut + pas) / n), hms(min(debut + pas, n) / TAUX, ms=False),
                         (time.time() - t0) / 60)
                jalon += 0.05
        if not ids:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        return np.concatenate(ids), np.concatenate(probs)

    def decoder(self, ids: np.ndarray, probs: np.ndarray, pause_enonce: float = 0.5) -> List[Mot]:
        """Décodage CTC glouton, avec horodatage et confiance de chaque mot.

        Confiance d'un mot = moyenne des probabilités des trames où l'un de
        ses caractères est émis. Un mot se termine au séparateur, ou après un
        silence de plus de pause_enonce secondes ; un tel silence ouvre aussi
        un nouvel énoncé.
        """
        duree_trame = self.ept / TAUX
        mots: List[Mot] = []
        lettres, confs = [], []
        debut = fin = None
        prec = self.vide

        def clore():
            nonlocal lettres, confs, debut, fin
            if lettres:
                mot = "".join(lettres).strip("'-")
                if mot:
                    mots.append(Mot(debut=round(debut * duree_trame, 3), fin=round((fin + 1) * duree_trame, 3),
                                    mot=mot, conf=round(float(np.mean(confs)), 4)))
            lettres, confs, debut, fin = [], [], None, None

        for g in range(len(ids)):
            i = int(ids[g])
            if i == self.vide or i in self.speciaux:
                prec = i
                continue
            tok = self.vocab.get(i, "")
            if tok in (self.sep, " "):
                clore()
            else:
                if fin is not None and (g - fin) * duree_trame > pause_enonce:
                    clore()                     # long silence sans séparateur : nouveau mot
                if i != prec:
                    if debut is None:
                        debut = g
                    lettres.append(tok)
                fin = g
                confs.append(float(probs[g]))
            prec = i
        clore()
        enonce = 0
        for k, m in enumerate(mots):
            if k and m.debut - mots[k - 1].fin > pause_enonce:
                enonce += 1
            m.enonce = enonce
        return mots

    def transcrire_wav(self, chemin_wav: Path) -> List[Mot]:
        signal = wav_memmap(chemin_wav)
        ids, probs = self.trames(signal)
        mots = self.decoder(ids, probs)
        log.info("wav2vec 2.0 : %d mots reconnus.", len(mots))
        return mots
