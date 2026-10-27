"""Tests de la reconnaissance wav2vec 2.0 + CTC.

1. Décodage CTC glouton sur une suite d'étiquettes connue (mots, temps,
   confiances, énoncés).
2. Traitement par blocs de 30 s avec contexte : mêmes étiquettes qu'un
   traitement d'un seul tenant.
3. (si PyTorch et transformers sont installés) conversion ONNX d'un réseau
   wav2vec 2.0 miniature par outils/exporter_wav2vec2.py, puis comparaison
   des étiquettes de l'application avec celles de PyTorch. Avec l'option
   --mini, le réseau converti est copié dans retranscription/donnees/w2v_mini
   (auto-vérification de l'application).

Lancement :  python tests/test_w2v.py [--mini]
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "outils"))

from retranscription.asr_w2v import TranscripteurW2V     # noqa: E402

VOCAB = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 4: "|", 5: "b", 6: "o", 7: "n",
         8: "j", 9: "u", 10: "r", 11: "'", 12: "l", 13: "e"}


def transcripteur_nu(vocab=VOCAB, vide=0):
    """TranscripteurW2V sans réseau (pour tester le décodage seul)."""
    t = object.__new__(TranscripteurW2V)
    t.vocab, t.vide, t.sep, t.normaliser, t.ept = vocab, vide, "|", True, 320
    t.speciaux = {i for i, s in vocab.items() if s.startswith("<") and s.endswith(">") and i != vide}
    t.nom = "test"
    return t


def test_decodage():
    t = transcripteur_nu()
    #        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    ids = [0, 5, 5, 0, 6, 7, 7, 4, 4, 0, 8, 6, 0, 9, 10, 0] + [0] * 29 + \
          [12, 11, 13, 0, 13, 1, 0]           # « l'ee » : deux « e » séparés par un vide
    probs = np.full(len(ids), 0.5, np.float32)
    probs[1:3] = 1.0                          # le « b » répété : 1,0 et 1,0
    mots = t.decoder(np.array(ids), probs)
    assert [m.mot for m in mots] == ["bon", "jour", "l'ee"], [m.mot for m in mots]
    assert (mots[0].debut, mots[0].fin) == (0.02, 0.14), (mots[0].debut, mots[0].fin)
    assert (mots[1].debut, mots[1].fin) == (0.2, 0.3)
    assert mots[2].debut == 0.9 and mots[2].fin == 1.0
    assert abs(mots[0].conf - np.mean([1.0, 1.0, 0.5, 0.5, 0.5])) < 1e-4
    assert [m.enonce for m in mots] == [0, 0, 1], "pause de 0,6 s : nouvel énoncé attendu"
    print("OK : décodage CTC (fusion des répétitions, vides, séparateur, temps, confiance, énoncés).")


def test_blocs():
    """Réseau factice local : l'étiquette d'une trame dépend des 400 échantillons qu'elle couvre."""
    t = transcripteur_nu()
    rng = np.random.default_rng(0)
    n = int(16000 * 75.3)
    x = (rng.standard_normal(n) * 3000).astype("<i2")
    x[16000 * 20:16000 * 21] = 0

    def logits(seg):
        seg = seg.astype(np.float32) / 32768.0
        nf = (len(seg) - 400) // 320 + 1
        idx = np.arange(nf)[:, None] * 320 + np.arange(400)[None, :]
        e = np.abs(seg[idx]).mean(axis=1)
        q = np.digitize(e, np.quantile(np.abs(x / 32768.0), [0.3, 0.5, 0.7])) + 4
        out = np.zeros((nf, len(VOCAB)), np.float32)
        out[np.arange(nf), q] = 5.0
        return out
    t._logits = logits
    ids, probs = t.trames(x, bloc_s=30.0, contexte_s=2.0)
    ref = logits(x).argmax(axis=1)
    m = min(len(ids), len(ref))
    assert abs(len(ids) - len(ref)) <= 1, (len(ids), len(ref))
    assert (ids[:m] == ref[:m]).all(), "raccord des blocs incorrect"
    assert probs.min() > 0.9
    print(f"OK : traitement par blocs identique au traitement d'un seul tenant ({len(ids)} trames).")


def test_export(copier_mini=False):
    try:
        import torch
        from transformers import (Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                                  Wav2Vec2ForCTC, Wav2Vec2Processor)
    except ImportError:
        print("(PyTorch ou transformers absent : test de conversion ONNX non lancé)")
        return
    import exporter_wav2vec2
    d = Path(tempfile.mkdtemp(prefix="w2v_mini_"))
    try:
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
        for c in "abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿœæ'-":
            vocab[c] = len(vocab)
        (d / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        tok = Wav2Vec2CTCTokenizer(str(d / "vocab.json"), unk_token="<unk>", pad_token="<pad>",
                                   word_delimiter_token="|")
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                      do_normalize=True, return_attention_mask=True)
        source = d / "source"
        Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok).save_pretrained(source)
        torch.manual_seed(0)
        cfg = Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64,
                             conv_dim=(32,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
                             feat_extract_norm="layer", do_stable_layer_norm=True, pad_token_id=0)
        reseau = Wav2Vec2ForCTC(cfg).eval()
        with torch.no_grad():                 # sorties plus contrastées que l'initialisation
            reseau.lm_head.weight.mul_(20.0)
        reseau.save_pretrained(source)

        exporter_wav2vec2.exporter(str(source), d / "export", duree_test_s=4.0)
        cfg_app = json.loads((d / "export" / "wav2vec2-fr" / "config_retranscription.json").read_text("utf-8"))
        assert cfg_app["echantillons_par_trame"] == 320 and cfg_app["vide"] == 0
        assert cfg_app["vocabulaire"]["4"] == "|"

        rng = np.random.default_rng(1)
        x = (rng.standard_normal(16000 * 70) * 2000).astype("<i2")
        xf = x.astype(np.float32) / 32768.0
        xf = (xf - xf.mean()) / np.sqrt(xf.var() + 1e-7)
        with torch.no_grad():
            ref = reseau(torch.from_numpy(xf)[None]).logits[0].numpy().argmax(-1)
        for nom, seuil in (("wav2vec2-fr", 0.999), ("wav2vec2-fr-int8", 0.9)):
            t = TranscripteurW2V(d / "export" / nom)
            direct = t._logits(x).argmax(-1)
            accord_direct = float((direct == ref).mean())
            ids, _ = t.trames(x)
            m = min(len(ids), len(ref))
            accord_blocs = float((ids[:m] == ref[:m]).mean())
            print(f"  {nom} : accord avec PyTorch {100 * accord_direct:.2f} % (d'un seul tenant), "
                  f"{100 * accord_blocs:.2f} % (par blocs de 30 s), {len(ids)} trames / {len(ref)}")
            assert len(direct) == len(ref) and accord_direct >= seuil, nom
            assert abs(len(ids) - len(ref)) <= 1
        print("OK : conversion ONNX conforme à PyTorch ; décodage par blocs cohérent.")
        if copier_mini:
            cible = RACINE / "retranscription" / "donnees" / "w2v_mini"
            shutil.rmtree(cible, ignore_errors=True)
            cible.mkdir(parents=True)
            for f in ("model.onnx", "config_retranscription.json"):
                shutil.copy(d / "export" / "wav2vec2-fr" / f, cible / f)
            (cible / "LISEZMOI.txt").write_text(
                "Réseau wav2vec 2.0 miniature à poids aléatoires (aucune reconnaissance réelle).\n"
                "Sert uniquement à l'auto-vérification de l'application (option --verifier).\n"
                "Produit par : python tests/test_w2v.py --mini\n", encoding="utf-8")
            print(f"Réseau miniature copié dans {cible} "
                  f"({sum(f.stat().st_size for f in cible.iterdir()) / 1e3:.0f} ko).")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_decodage()
    test_blocs()
    test_export(copier_mini="--mini" in sys.argv)
    print("TOUS LES TESTS SONT PASSÉS.")
