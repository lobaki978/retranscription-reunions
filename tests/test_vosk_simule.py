"""Tests des appels à Vosk avec une bibliothèque simulée, et du téléchargement.

Les modèles Vosk ne sont pas téléchargeables dans l'environnement de test.
On remplace le module `vosk` par un double qui respecte son interface et
le format de ses résultats JSON ; on vérifie ainsi la lecture du flux, le
découpage en énoncés, les temps des mots et le calcul des x-vecteurs.
"""

import http.server
import json
import sys
import threading
import types
import wave
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TAUX = 16000


class FauxRecognizer:
    """Imite KaldiRecognizer : un « mot » toutes les 0,5 s, un énoncé toutes les 4 s."""

    def __init__(self, modele, taux, spk=None):
        assert taux == TAUX
        self.spk = spk
        self.mots_actives = False
        self.Reset()

    def SetWords(self, v):
        self.mots_actives = v

    def Reset(self):
        self.debut_enonce = getattr(self, "position", 0)
        self.position = getattr(self, "position", 0)
        self.audio = []

    def _resultat(self):
        x = np.concatenate(self.audio) if self.audio else np.zeros(0, np.int16)
        t0, t1 = self.debut_enonce / TAUX, self.position / TAUX
        mots = []
        t = np.ceil(t0 * 2) / 2
        while t + 0.4 <= t1:
            mots.append({"conf": 0.9, "end": round(t + 0.4, 2), "start": round(t, 2), "word": f"mot{int(t * 2)}"})
            t += 0.5
        r = {"text": " ".join(m["word"] for m in mots)}
        if self.mots_actives and mots:
            r["result"] = mots
        if self.spk is not None and len(x) > 0.3 * TAUX:
            e = float(np.abs(x.astype(float)).mean())
            r["spk"] = list(np.full(128, e / 1000.0))
            r["spk_frames"] = int(len(x) / 160)
        self.audio, self.debut_enonce = [], self.position
        return json.dumps(r)

    def AcceptWaveform(self, donnees):
        x = np.frombuffer(donnees, dtype=np.int16)
        self.audio.append(x)
        avant = self.position
        self.position += len(x)
        return (avant // (4 * TAUX)) != (self.position // (4 * TAUX))

    def Result(self):
        return self._resultat()

    def FinalResult(self):
        return self._resultat()


def installer_faux_vosk():
    m = types.ModuleType("vosk")
    m.Model = lambda chemin: ("modele", chemin)
    m.SpkModel = lambda chemin: ("spk", chemin)
    m.KaldiRecognizer = FauxRecognizer
    m.SetLogLevel = lambda n: None
    sys.modules["vosk"] = m


def ecrire_wav(chemin, x):
    with wave.open(str(chemin), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(TAUX)
        w.writeframes(x.astype("<i2").tobytes())


def test_transcription_et_xvecteurs(dossier):
    installer_faux_vosk()
    from retranscription.asr import transcrire_wav, charger_modele
    from retranscription.audio import wav_memmap
    from retranscription.empreintes import ExtracteurVosk, fenetres
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(12 * TAUX) * 3000).astype(np.int16)
    chemin = dossier / "essai.wav"
    ecrire_wav(chemin, x)
    (dossier / "modele").mkdir(exist_ok=True)
    mots = transcrire_wav(chemin, charger_modele(str(dossier / "modele")))
    debuts = [m.debut for m in mots]
    assert debuts == sorted(debuts), "temps non croissants"
    assert len(mots) == 24, f"{len(mots)} mots au lieu de 24"
    assert mots[-1].fin <= 12.0 and mots[0].debut == 0.0
    assert len({m.enonce for m in mots}) == 3, "découpage en énoncés incorrect"
    print(f"OK : transcription simulée, {len(mots)} mots, {len({m.enonce for m in mots})} énoncés.")
    fen = fenetres([(0.0, 12.0)], 1.5, 0.75)
    ext = ExtracteurVosk("spk", ("modele", "petit"))
    X, valides = ext.calculer(wav_memmap(chemin), fen)
    assert X.shape == (len(fen), 128) and len(valides) == len(fen)
    print(f"OK : {len(fen)} x-vecteurs simulés de dimension {X.shape[1]}.")
    del sys.modules["vosk"]


def test_telechargement(dossier):
    from retranscription.telecharger import telecharger
    serveur_dir = dossier / "serveur"
    serveur_dir.mkdir()
    for nom in ("vosk-model-small-fr-0.22", "vosk-model-spk-0.4"):
        with zipfile.ZipFile(serveur_dir / f"{nom}.zip", "w") as z:
            z.writestr(f"{nom}/README", "faux modèle")
            z.writestr(f"{nom}/am/final.mdl", "x" * 1000)

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(serveur_dir), **k)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    telecharger(dossier / "modeles", ["vosk-model-small-fr-0.22", "vosk-model-spk-0.4"], url=url)
    srv.shutdown()
    assert (dossier / "modeles" / "vosk-model-spk-0.4" / "am" / "final.mdl").exists()
    assert not list((dossier / "modeles").glob("*.zip")), "archive non supprimée"
    print("OK : téléchargement et décompression des modèles (serveur local).")


def test_vraie_bibliotheque():
    """La vraie bibliothèque Vosk se charge ; un modèle absent donne une erreur claire."""
    import importlib
    vosk = importlib.import_module("vosk")
    from retranscription.asr import charger_modele
    try:
        charger_modele("/chemin/inexistant")
    except FileNotFoundError as e:
        assert "telecharger-modeles" in str(e)
    print(f"OK : bibliothèque Vosk chargée ({Path(vosk.__file__).parent.name}), message clair si modèle absent.")


if __name__ == "__main__":
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp())
    try:
        test_transcription_et_xvecteurs(d)
        test_telechargement(d)
        test_vraie_bibliotheque()
        print("TOUS LES TESTS SONT PASSÉS.")
    finally:
        shutil.rmtree(d, ignore_errors=True)
