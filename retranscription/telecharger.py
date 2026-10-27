"""Téléchargement des modèles Vosk (site d'Alpha Cephei, auteurs de Vosk)."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .utils import log

URL = "https://alphacephei.com/vosk/models/"
MODELES = {
    "vosk-model-fr-0.22": "reconnaissance du français, grand modèle (~1,4 Go)",
    "vosk-model-small-fr-0.22": "reconnaissance du français, petit modèle (~40 Mo)",
    "vosk-model-spk-0.4": "x-vecteurs (modèle de locuteur, ~13 Mo)",
}


def telecharger(dossier: Path, noms=None, url: str = URL) -> None:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in noms or MODELES:
        cible = dossier / nom
        if cible.exists():
            log.info("%s déjà présent.", nom)
            continue
        archive = dossier / f"{nom}.zip"
        log.info("Téléchargement de %s : %s", nom, MODELES.get(nom, ""))
        with urllib.request.urlopen(url + archive.name) as r, open(archive, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            lus, jalon = 0, 0.1
            while True:
                bloc = r.read(1 << 20)
                if not bloc:
                    break
                f.write(bloc)
                lus += len(bloc)
                if total and lus / total >= jalon:
                    log.info("  %3.0f %%", 100 * lus / total)
                    jalon += 0.1
        with zipfile.ZipFile(archive) as z:
            z.extractall(dossier)
        archive.unlink()
        if not cible.exists():
            # archive dont le dossier racine porte un autre nom
            sous = [p for p in dossier.iterdir() if p.is_dir() and p.name.startswith(nom.split("-0")[0])]
            if len(sous) == 1:
                shutil.move(str(sous[0]), str(cible))
        log.info("%s installé dans %s", nom, cible)
