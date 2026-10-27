"""Exporte un modèle wav2vec 2.0 + CTC (Hugging Face) au format ONNX.

Lancé par GitHub Actions (workflow « Préparer les modèles »), une seule
fois : l'application télécharge ensuite le résultat, sans PyTorch.

    python outils/exporter_wav2vec2.py --modele bofenghuang/asr-wav2vec2-ctc-french --sortie modeles_export

Produit, dans --sortie :
    wav2vec2-fr/        model.onnx (32 bits) + config_retranscription.json + LISEZMOI.txt
    wav2vec2-fr-int8/   model.onnx (quantifié 8 bits, plus rapide) + mêmes fichiers
"""

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np


def exporter(modele: str, sortie: Path, duree_test_s: float = 8.0) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    sortie = Path(sortie)
    d32, d8 = sortie / "wav2vec2-fr", sortie / "wav2vec2-fr-int8"
    for d in (d32, d8):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de {modele}…")
    processeur = Wav2Vec2Processor.from_pretrained(modele)
    reseau = Wav2Vec2ForCTC.from_pretrained(modele).eval()
    tok, fe = processeur.tokenizer, processeur.feature_extractor

    # --- export ONNX (exporteur TorchScript, axes temporels dynamiques) --------
    x = torch.randn(1, int(16000 * duree_test_s))

    class Enveloppe(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).logits

    options = dict(input_names=["input_values"], output_names=["logits"],
                   dynamic_axes={"input_values": {1: "echantillons"}, "logits": {1: "trames"}},
                   opset_version=17, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    chemin32 = d32 / "model.onnx"
    print("Export ONNX…")
    # l'enveloppe doit être en mode évaluation : l'exporteur rétablit ensuite son
    # mode, qui s'appliquerait aussi au réseau (dropout actif)
    enveloppe = Enveloppe(reseau).eval()
    torch.onnx.export(enveloppe, (x,), str(chemin32), **options)
    reseau.eval()

    # --- vérification : mêmes sorties que PyTorch -----------------------------
    import onnxruntime as ort
    with torch.no_grad():
        ref = reseau(x).logits.numpy()
    sess = ort.InferenceSession(str(chemin32), providers=["CPUExecutionProvider"])
    for duree in (duree_test_s, 3.0, 17.3):           # longueurs différentes : axes dynamiques
        xx = torch.randn(1, int(16000 * duree))
        with torch.no_grad():
            r = reseau(xx).logits.numpy()
        o = sess.run(None, {"input_values": xx.numpy()})[0]
        ecart = float(np.abs(r - o).max())
        accord = float((r.argmax(-1) == o.argmax(-1)).mean())
        print(f"  {duree:5.1f} s : écart max {ecart:.2e}, accord des étiquettes {100 * accord:.2f} %")
        assert o.shape == r.shape and accord > 0.999, "export ONNX non conforme"

    del sess, enveloppe
    import gc
    gc.collect()

    # --- quantification 8 bits ---------------------------------------------------
    # seules les multiplications de matrices (couches du transformeur, l'essentiel
    # des poids) sont quantifiées ; les convolutions restent en 32 bits
    # (ConvInteger n'est pas disponible partout dans onnxruntime)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print("Quantification 8 bits…")
    quantize_dynamic(str(chemin32), str(d8 / "model.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    s8 = ort.InferenceSession(str(d8 / "model.onnx"), providers=["CPUExecutionProvider"])
    o8 = s8.run(None, {"input_values": x.numpy()})[0]
    print(f"  accord 8 bits / 32 bits : {100 * float((o8.argmax(-1) == ref.argmax(-1)).mean()):.2f} %")

    # --- description pour l'application -----------------------------------------
    vocab = tok.get_vocab()
    config = {
        "modele_source": modele,
        "url_source": f"https://huggingface.co/{modele}",
        "taux": int(fe.sampling_rate),
        "normaliser": bool(getattr(fe, "do_normalize", True)),
        "vide": int(tok.pad_token_id),
        "separateur": tok.word_delimiter_token or "|",
        "echantillons_par_trame": int(np.prod(reseau.config.conv_stride)),
        "vocabulaire": {str(i): t for t, i in vocab.items()},
    }
    note = (f"Modèle de reconnaissance de la parole : {modele}\n"
            f"Source : https://huggingface.co/{modele} (voir sa licence, Apache 2.0 pour le modèle par défaut).\n"
            "Converti au format ONNX pour l'application Retranscription. Réseau wav2vec 2.0\n"
            "(Baevski et al., 2020) avec décodage CTC (Graves et al., 2006).\n")
    for d in (d32, d8):
        (d / "config_retranscription.json").write_text(json.dumps(config, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        (d / "LISEZMOI.txt").write_text(note, encoding="utf-8")
    for d in (d32, d8):
        taille = sum(f.stat().st_size for f in d.iterdir()) / 1e6
        print(f"{d.name} : {taille:.0f} Mo")


def archiver(sortie: Path) -> None:
    for nom in ("wav2vec2-fr", "wav2vec2-fr-int8"):
        shutil.make_archive(str(Path(sortie) / nom), "zip", root_dir=sortie, base_dir=nom)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="bofenghuang/asr-wav2vec2-ctc-french")
    ap.add_argument("--sortie", default="modeles_export")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    exporter(a.modele, Path(a.sortie))
    if a.zip:
        archiver(Path(a.sortie))
