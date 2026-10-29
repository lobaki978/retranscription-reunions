"""Décodage CTC avec modèle de langue n-gramme et lexique du projet.

Le décodage glouton garde, à chaque trame de 20 ms, le caractère le plus
probable : un mot mal entendu devient un non-mot (« absolumet ») et deux
mots peuvent rester collés (« maisje »). Le décodage par faisceau (beam
search) garde plusieurs hypothèses à la fois et les classe par

    score = log P(son | texte) + alpha · log P(texte) + beta · nombre de mots

où P(texte) est donnée par un modèle de langue n-gramme (Jelinek, 1997) :
la probabilité d'un mot ne dépend que des n-1 mots qui le précèdent,
estimée par comptage dans un grand corpus. C'est un modèle statistique
de suites de mots, du même type que celui de Vosk/Kaldi, pas un modèle
génératif neuronal. Recherche par faisceau CTC avec modèle de langue :
Hannun et al. (2014). Implémentation : pyctcdecode ; requêtes au modèle
de langue : KenLM (Heafield, 2011).

Modèle de langue par défaut : le 5-grammes publié avec le modèle de
reconnaissance (bofenghuang/asr-wav2vec2-ctc-french, dossier
language_model : lm_5gram_big.bin, unigrams.txt, attrs.json). Les
réglages alpha et beta sont ceux de attrs.json.

Lexique du projet : fichier texte facultatif, un mot ou une expression
par ligne (noms de lieux, sigles, termes du domaine). Ces termes reçoivent
un bonus pendant le décodage (« hotwords » de pyctcdecode), même s'ils sont
absents du modèle de langue.

Le décodage se fait par morceaux séparés par des silences (au moins 0,3 s
sans caractère), de 30 s au plus : la mémoire reste faible et l'état du
modèle de langue repart à chaque pause.
"""

import json
import time
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np

from .asr import Mot
from .utils import hms, log


def disponible() -> Optional[str]:
    """None si KenLM et pyctcdecode sont présents, sinon la raison."""
    import importlib
    try:
        importlib.import_module("kenlm")
        importlib.import_module("pyctcdecode")
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__} : {e}"


def lire_lexique(chemin, alphabet: set) -> List[str]:
    """Un terme par ligne ; lignes vides et commentaires (#) ignorés.

    Les termes sont mis en minuscules et réduits aux caractères que le
    modèle sait écrire (les autres deviennent des espaces).
    """
    chemin = Path(chemin)
    try:
        texte = chemin.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        texte = chemin.read_text(encoding="latin-1")
    termes, vus = [], set()
    for ligne in texte.splitlines():
        ligne = ligne.split("#", 1)[0].strip()
        if not ligne:
            continue
        t = unicodedata.normalize("NFC", ligne.lower()).replace("’", "'").replace("œ", "oe").replace("æ", "ae")
        t = "".join(c if (c in alphabet or c == " ") else " " for c in t)
        t = " ".join(m.strip("'-") for m in t.split() if m.strip("'-"))
        if t and t not in vus:
            vus.add(t)
            termes.append(t)
    return termes


def morceaux(ids: np.ndarray, vide: int, silence_min: int = 15, longueur_max: int = 1500):
    """Découpe en morceaux [a, b) aux silences d'au moins silence_min trames.

    Un morceau plus long que longueur_max est recoupé au plus long silence
    qu'il contient (ou, à défaut, à longueur_max).
    """
    n = len(ids)
    est_vide = np.asarray(ids) == vide
    # suites de trames vides : début, fin
    bords = np.flatnonzero(np.diff(np.concatenate([[0], est_vide.astype(np.int8), [0]])))
    suites = list(zip(bords[0::2], bords[1::2]))
    coupes = [0] + [(a + b) // 2 for a, b in suites if b - a >= silence_min and 0 < a and b < n] + [n]
    resultat = []
    for a, b in zip(coupes[:-1], coupes[1:]):
        while b - a > longueur_max:
            internes = [(s, e) for s, e in suites if a < s and e < a + longueur_max and e - s >= 2]
            c = (max(internes, key=lambda se: se[1] - se[0])) if internes else None
            coupe = (c[0] + c[1]) // 2 if c else a + longueur_max
            resultat.append((a, coupe))
            a = coupe
        if b > a:
            resultat.append((a, b))
    return [(a, b) for a, b in resultat if not est_vide[a:b].all()]


class DecodeurLM:
    def __init__(self, dossier_lm, vocab: dict, vide: int, sep: str, n_sorties: int,
                 lexique: Optional[List[str]] = None, largeur: int = 64, poids_lexique: float = 10.0):
        from pyctcdecode import build_ctcdecoder
        d = Path(dossier_lm)
        fichier = next((f for f in sorted(d.glob("*.bin")) + sorted(d.glob("*.arpa"))), None)
        if fichier is None:
            raise FileNotFoundError(f"Aucun modèle de langue (.bin ou .arpa) dans {d}")
        attrs = {}
        if (d / "attrs.json").exists():
            attrs = json.loads((d / "attrs.json").read_text(encoding="utf-8"))
        unigrammes = None
        if (d / "unigrams.txt").exists():
            unigrammes = [m for m in (d / "unigrams.txt").read_text(encoding="utf-8").split("\n") if m]
        self.vide = vide
        self.speciaux = []
        etiquettes = []
        for i in range(n_sorties):
            t = vocab.get(i, "")
            if i == vide:
                etiquettes.append("")
            elif t == sep:
                etiquettes.append(" ")
            elif len(t) != 1:
                etiquettes.append(chr(0xE000 + i))     # symbole spécial (zone privée) : jamais émis
                self.speciaux.append(i)
            else:
                etiquettes.append(t)
        self.alphabet = {t for t in etiquettes if len(t) == 1 and t != " "}
        self.alpha = float(attrs.get("alpha", 0.5))
        self.beta = float(attrs.get("beta", 1.5))
        log.info("Chargement du modèle de langue %s (alpha %.2f, beta %.2f)…", fichier.name, self.alpha, self.beta)
        t0 = time.time()
        self.dec = build_ctcdecoder(etiquettes, kenlm_model_path=str(fichier), unigrams=unigrammes,
                                    alpha=self.alpha, beta=self.beta,
                                    unk_score_offset=float(attrs.get("unk_score_offset", -10.0)),
                                    lm_score_boundary=bool(attrs.get("score_boundary", True)))
        log.info("Modèle de langue chargé en %.0f s.", time.time() - t0)
        self.lexique = list(lexique or [])
        self.largeur = largeur
        self.poids_lexique = poids_lexique

    def decoder(self, lp: np.ndarray, ids: np.ndarray, probs: np.ndarray,
                duree_trame: float, pause_enonce: float = 0.5) -> List[Mot]:
        """lp : log-probabilités (trames × sorties) ; ids, probs : décodage glouton (confiances)."""
        mots: List[Mot] = []
        parts = morceaux(ids, self.vide)
        total = max(1, sum(b - a for a, b in parts))
        fait, jalon, t0 = 0, 0.1, time.time()
        for a, b in parts:
            x = lp[a:b].astype(np.float32)
            if self.speciaux:
                x[:, self.speciaux] = -1e4
            faisceaux = self.dec.decode_beams(x, beam_width=self.largeur,
                                              hotwords=self.lexique or None,
                                              hotword_weight=self.poids_lexique)
            for mot, (s, e) in faisceaux[0][2]:
                mot = mot.strip("'-")
                if not mot:
                    continue
                s, e = a + s, a + max(e, s + 1)
                pleins = probs[s:e][ids[s:e] != self.vide]
                conf = float(np.mean(pleins)) if len(pleins) else float(np.mean(probs[s:e]))
                mots.append(Mot(debut=round(s * duree_trame, 3), fin=round(e * duree_trame, 3),
                                mot=mot, conf=round(conf, 4)))
            fait += b - a
            if fait / total >= jalon:
                log.info("  modèle de langue : %3.0f %% (%s traités, %.0f min écoulées)", 100 * fait / total,
                         hms(b * duree_trame, ms=False), (time.time() - t0) / 60)
                jalon += 0.1
        enonce = 0
        for k, m in enumerate(mots):
            if k and m.debut - mots[k - 1].fin > pause_enonce:
                enonce += 1
            m.enonce = enonce
        return mots

