# -*- mode: python ; coding: utf-8 -*-
# Fabrication de l'application avec PyInstaller :  pyinstaller --noconfirm retranscription.spec
import sys
from pathlib import Path

import vosk
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Vosk charge sa bibliothèque (libvosk.dll / .so / .dyld) depuis son propre dossier
dossier_vosk = Path(vosk.__file__).parent
for f in dossier_vosk.iterdir():
    if f.suffix.lower() in (".dll", ".so", ".dyld", ".dylib"):
        binaries.append((str(f), "vosk"))
hiddenimports += collect_submodules("vosk") + ["_cffi_backend"]

for paquet in ("av", "jiwer", "rapidfuzz"):
    d, b, h = collect_all(paquet)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("retranscription")

# onnxruntime : moteur des réseaux wav2vec 2.0, segmentation et WeSpeaker
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += ["onnxruntime", "onnxruntime.capi._pybind_state"]

# réseau wav2vec 2.0 miniature (poids aléatoires) pour l'auto-vérification
donnees = [(str(f), str(f.parent)) for f in Path("retranscription/donnees").rglob("*") if f.is_file()]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas + donnees + [(f, ".") for f in ("README.md", "GUIDE_UTILISATEUR.md", "GUIDE_UTILISATEUR.html")
                   if Path(f).exists()],
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "pandas", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "torch", "transformers", "onnx", "pyannote", "pympi", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Retranscription",
          console=False, upx=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Retranscription", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Retranscription.app",
        bundle_identifier="fr.recherche.retranscription",
        info_plist={
            "CFBundleName": "Retranscription",
            "CFBundleDisplayName": "Retranscription",
            "CFBundleShortVersionString": "0.3.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",
        },
    )
