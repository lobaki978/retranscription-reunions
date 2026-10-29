"""Tests du décodage avec modèle de langue n-gramme et lexique du projet.

Un petit modèle de langue ARPA est écrit pour le test. Des probabilités
par trame sont construites à la main : le décodage glouton se trompe
(« maisun », mots collés), le décodage par faisceau avec modèle de langue
doit corriger ; un terme absent du modèle de langue doit être retrouvé
grâce au lexique du projet.

Lancement :  python tests/test_langue.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))

from retranscription import langue                            # noqa: E402
from retranscription.asr_w2v import TranscripteurW2V          # noqa: E402

LETTRES = list("abcdefghijklmnopqrstuvwxyz'éè")
VOCAB = {0: "<pad>", 1: "|", 2: "<unk>"}
for c in LETTRES:
    VOCAB[len(VOCAB)] = c
IDX = {t: i for i, t in VOCAB.items()}
V = len(VOCAB)

ARPA = """\\data\\
ngram 1=9
ngram 2=6

\\1-grams:
-1.0\t<s>\t-0.3
-1.0\t</s>
-3.0\t<unk>
-1.2\tla\t-0.3
-1.4\tmaison\t-0.3
-1.2\test\t-0.3
-1.3\tbelle\t-0.3
-1.2\tmais\t-0.3
-1.2\tje\t-0.3

\\2-grams:
-0.3\t<s> la
-0.2\tla maison
-0.2\tmaison est
-0.2\test belle
-0.3\tbelle mais
-0.2\tmais je

\\end\\
"""


def trames(texte, confusions=None, sans_separateur=()):
    """Probabilités par trame pour un texte : 2 trames par caractère, 1 vide entre caractères.

    confusions : {position du caractère : caractère concurrent plus probable (0,55 contre 0,40)}.
    sans_separateur : positions de « | » rendues moins probables que le vide (mots collés).
    """
    confusions = confusions or {}
    P = []

    def trame(dist):
        p = np.full(V, 1e-4)
        for i, v in dist.items():
            p[i] = v
        P.append(p / p.sum())

    for _ in range(3):
        trame({IDX["<pad>"]: 0.98})
    for pos, ch in enumerate(texte):
        car = "|" if ch == " " else ch
        if pos in confusions:
            d = {IDX[confusions[pos]]: 0.55, IDX[car]: 0.40}
        elif car == "|" and pos in sans_separateur:
            d = {IDX["<pad>"]: 0.60, IDX["|"]: 0.38}
        else:
            d = {IDX[car]: 0.95}
        trame(d)
        trame(d)
        trame({IDX["<pad>"]: 0.95})
    for _ in range(3):
        trame({IDX["<pad>"]: 0.98})
    P = np.array(P)
    lp = np.log(P).astype(np.float16)
    return lp, P.argmax(axis=1).astype(np.int32), P.max(axis=1).astype(np.float32)


def glouton(ids, probs):
    t = object.__new__(TranscripteurW2V)
    t.vocab, t.vide, t.sep, t.ept = VOCAB, 0, "|", 320
    t.speciaux = {i for i, s in VOCAB.items() if len(s) != 1 and i != 0}
    return " ".join(m.mot for m in t.decoder(ids, probs))


def main():
    raison = langue.disponible()
    if raison:
        print(f"(KenLM ou pyctcdecode absent : {raison} — tests du modèle de langue non lancés)")
        return
    d = Path(tempfile.mkdtemp(prefix="lm_test_"))
    try:
        (d / "lm").mkdir()
        (d / "lm" / "petit.arpa").write_text(ARPA, encoding="utf-8")
        (d / "lm" / "unigrams.txt").write_text("la\nmaison\nest\nbelle\nmais\nje\n", encoding="utf-8")
        (d / "lm" / "attrs.json").write_text(json.dumps({"alpha": 0.5, "beta": 1.0, "unk_score_offset": -10.0,
                                                         "score_boundary": True}), encoding="utf-8")

        # 1. lexique : normalisation
        (d / "lexique.txt").write_text("# termes du projet\nZabaobab\n\nAtténuation ; adaptation\n"
                                       "l’Œuvre\nzabaobab\n", encoding="utf-8")
        alphabet = set(LETTRES)
        termes = langue.lire_lexique(d / "lexique.txt", alphabet)
        assert termes == ["zabaobab", "atténuation adaptation", "l'oeuvre"], termes
        print("OK : lexique lu et normalisé :", termes)

        # 2. découpage aux silences
        ids = np.array([0] * 5 + [5] * 10 + [0] * 20 + [6] * 10 + [0] * 5)
        assert langue.morceaux(ids, 0) == [(0, 25), (25, 50)], langue.morceaux(ids, 0)
        long = np.array(([5] * 50 + [0] * 3) * 40)
        parts = langue.morceaux(long, 0, longueur_max=500)
        assert all(b - a <= 500 for a, b in parts) and parts[0][0] == 0 and parts[-1][1] <= len(long)
        print(f"OK : découpage aux silences ({len(parts)} morceaux de 500 trames au plus).")

        # 3. corrections par le modèle de langue
        texte = "la maison est belle mais je"
        lp, ids, probs = trames(texte, confusions={7: "u"}, sans_separateur={24})
        g = glouton(ids, probs)
        dec = langue.DecodeurLM(d / "lm", VOCAB, 0, "|", V, largeur=32)
        m = dec.decoder(lp, ids, probs, 0.02)
        avec = " ".join(x.mot for x in m)
        print(f"   glouton : {g}\n   avec modèle de langue : {avec}")
        assert g == "la maisun est belle maisje", g
        assert avec == texte, avec
        assert all(0 < x.conf <= 1 for x in m) and all(x.fin > x.debut for x in m)
        assert [x.debut for x in m] == sorted(x.debut for x in m)
        print("OK : le modèle de langue corrige « maisun » et sépare « maisje ».")

        # 4. lexique du projet : terme absent du modèle de langue
        lp, ids, probs = trames("la zabaobab", confusions={10: "p"})
        sans = " ".join(x.mot for x in dec.decoder(lp, ids, probs, 0.02))
        dec.lexique = termes
        avec = " ".join(x.mot for x in dec.decoder(lp, ids, probs, 0.02))
        print(f"   sans lexique : {sans} ; avec lexique : {avec}")
        assert avec == "la zabaobab", avec
        print("OK : le lexique du projet fait reconnaître un terme absent du modèle de langue.")

        # 5. chaîne complète avec le réseau wav2vec 2.0 miniature
        mini = ICI.parent / "retranscription" / "donnees" / "w2v_mini"
        if (mini / "model.onnx").exists():
            cfg = json.loads((mini / "config_retranscription.json").read_text(encoding="utf-8"))
            lettres = [t for t in cfg["vocabulaire"].values() if len(t) == 1 and t != "|"]
            (d / "lm2").mkdir()
            uni = ["".join(lettres[i:i + 3]) for i in range(0, len(lettres) - 3, 3)]
            arpa = "\\data\\\nngram 1=%d\nngram 2=1\n\n\\1-grams:\n" % (len(uni) + 3)
            arpa += "-1.0\t<s>\t-0.3\n-1.0\t</s>\n-3.0\t<unk>\n" + "".join(f"-1.5\t{u}\n" for u in uni)
            arpa += f"\n\\2-grams:\n-0.5\t<s> {uni[0]}\n\n\\end\\\n"
            (d / "lm2" / "mini.arpa").write_text(arpa, encoding="utf-8")
            t = TranscripteurW2V(mini, dossier_lm=d / "lm2", lexique=d / "lexique.txt")
            assert t.lm is not None
            rng = np.random.default_rng(0)
            x = (rng.standard_normal(16000 * 12) * 2000).astype("<i2")
            ids, probs, lp = t.trames(x, logits=True)
            assert lp.shape == (len(ids), t.n_sorties) and lp.dtype == np.float16
            mots = t.lm.decoder(lp, ids, probs, t.ept / 16000)
            print(f"OK : réseau miniature + modèle de langue : {len(ids)} trames, {len(mots)} mots.")
        print("TOUS LES TESTS SONT PASSÉS.")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
