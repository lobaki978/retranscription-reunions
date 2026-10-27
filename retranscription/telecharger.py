"""Téléchargement des modèles (une seule fois, au premier lancement).

Les modèles neuronaux (wav2vec 2.0, segmentation, empreintes vocales) sont
publiés sur la page « modeles-1 » du dépôt GitHub du projet, préparée par
le workflow « Préparer les modèles ». Les modèles Vosk viennent du site
d'Alpha Cephei.
"""

import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .utils import log

DEPOT = os.environ.get("RETRANSCRIPTION_DEPOT", "lobaki978/retranscription-reunions")
PAGE = os.environ.get("RETRANSCRIPTION_URL_MODELES",
                      f"https://github.com/{DEPOT}/releases/download/modeles-1/")
VOSK = "https://alphacephei.com/vosk/models/"
K2 = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"

# nom -> (description, [(url, type)])   type : "zip", "fichier", "tar-seg"
CATALOGUE = {
    "wav2vec2-fr": ("reconnaissance du français, wav2vec 2.0 (~1,2 Go)",
                    [(PAGE + "wav2vec2-fr.zip", "zip")]),
    "wav2vec2-fr-int8": ("reconnaissance du français, wav2vec 2.0 allégé (~400 Mo)",
                         [(PAGE + "wav2vec2-fr-int8.zip", "zip")]),
    "segmentation-3.0.onnx": ("segmentation des locuteurs, pyannote 3.0 (~6 Mo)",
                              [(PAGE + "segmentation-3.0.onnx", "fichier"),
                               (K2 + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
                                "tar-seg")]),
    "wespeaker-resnet34.onnx": ("empreintes vocales, WeSpeaker ResNet34 (~26 Mo)",
                                [(PAGE + "wespeaker-resnet34.onnx", "fichier"),
                                 (K2 + "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
                                  "fichier")]),
    "vosk-model-fr-0.22": ("Vosk, grand modèle français (~1,4 Go)", [(VOSK + "vosk-model-fr-0.22.zip", "zip")]),
    "vosk-model-small-fr-0.22": ("Vosk, petit modèle français (~40 Mo)",
                                 [(VOSK + "vosk-model-small-fr-0.22.zip", "zip")]),
    "vosk-model-spk-0.4": ("Vosk, x-vecteurs (~13 Mo)", [(VOSK + "vosk-model-spk-0.4.zip", "zip")]),
}
MODELES = {n: d for n, (d, _) in CATALOGUE.items()}   # compatibilité


def necessaires(moteur: str = "wav2vec2", petit: bool = False, diarisation: str = "neurale") -> list:
    noms = []
    if moteur == "vosk":
        noms += ["vosk-model-small-fr-0.22"] if petit else ["vosk-model-fr-0.22", "vosk-model-small-fr-0.22"]
    else:
        noms += ["wav2vec2-fr-int8" if petit else "wav2vec2-fr"]
    noms += ["segmentation-3.0.onnx", "wespeaker-resnet34.onnx"] if diarisation == "neurale" \
        else ["wespeaker-resnet34.onnx"]
    return noms


def _recuperer(url: str, cible: Path) -> None:
    requete = urllib.request.Request(url, headers={"User-Agent": "retranscription"})
    with urllib.request.urlopen(requete, timeout=60) as r, open(cible, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        lus, jalon = 0, 0.1
        while True:
            bloc = r.read(1 << 20)
            if not bloc:
                break
            f.write(bloc)
            lus += len(bloc)
            if total and lus / total >= jalon:
                log.info("  %3.0f %% (%.0f Mo)", 100 * lus / total, lus / 1e6)
                jalon += 0.1


def telecharger(dossier: Path, noms=None, url: str = None) -> None:
    """url : ancienne interface (base des modèles Vosk), gardée pour les tests."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in noms or ["vosk-model-fr-0.22", "vosk-model-small-fr-0.22", "vosk-model-spk-0.4"]:
        cible = dossier / nom
        if cible.exists():
            log.info("%s déjà présent.", nom)
            continue
        description, sources = CATALOGUE.get(nom, ("", [(VOSK + nom + ".zip", "zip")]))
        if url:
            sources = [(url + nom + ".zip", "zip")]
        log.info("Téléchargement de %s : %s", nom, description)
        erreurs = []
        for adresse, genre in sources:
            tmp = dossier / (nom + ".partiel")
            try:
                _recuperer(adresse, tmp)
                if genre == "fichier":
                    tmp.replace(cible)
                elif genre == "zip":
                    with zipfile.ZipFile(tmp) as z:
                        z.extractall(dossier)
                    tmp.unlink()
                elif genre == "tar-seg":
                    with tarfile.open(tmp, "r:bz2") as t:
                        membre = next(m for m in t.getmembers() if m.name.endswith("/model.onnx"))
                        with t.extractfile(membre) as src, open(cible, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    tmp.unlink()
                break
            except Exception as e:  # noqa: BLE001
                erreurs.append(f"{adresse} : {e}")
                tmp.unlink(missing_ok=True)
        if not cible.exists():
            raise RuntimeError(f"Impossible de télécharger {nom}.\n" + "\n".join(erreurs) +
                               "\nVérifiez la connexion Internet. Si le message parle d'une page "
                               "introuvable (404), les modèles n'ont pas encore été publiés sur la "
                               "page « modeles-1 » du dépôt GitHub.")
        log.info("%s installé dans %s", nom, cible)
