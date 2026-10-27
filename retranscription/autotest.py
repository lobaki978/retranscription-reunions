"""Auto-vérification de l'application installée (option --verifier).

Utilisée après chaque fabrication automatique (GitHub Actions) : elle
vérifie que toutes les bibliothèques sont bien incluses, puis fait tourner
toute la chaîne sur un enregistrement synthétique de deux voix, avec une
transcription simulée (les modèles Vosk ne sont pas téléchargés ici).
"""

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

TAUX = 16000


def _voix(f0, echelle, duree, rng):
    """Voix synthétique source-filtre : impulsions glottiques + 3 formants."""
    from scipy.signal import lfilter
    n = int(duree * TAUX)
    t = np.arange(n) / TAUX
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 6)))
    source = (np.diff(np.floor(np.cumsum(f / TAUX)), prepend=0) > 0).astype(float)
    source = lfilter([1], [1, -0.95], source)
    y = np.zeros(n)
    for fc, bw in ((700, 80), (1200, 110), (2500, 160)):
        r = np.exp(-np.pi * bw / TAUX)
        y += lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc * echelle / TAUX), r * r], source)
    enveloppe = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2
    return y * enveloppe / (np.abs(y).max() + 1e-9)


def _enregistrement(dossier: Path, duree=90.0, graine=0):
    rng = np.random.default_rng(graine)
    voix = {"A": (115, 1.0), "B": (215, 1.2)}
    x = 0.002 * rng.standard_normal(int(duree * TAUX))
    mots, t, qui = [], 0.5, "A"
    while t < duree - 6:
        d = rng.uniform(3, 6)
        seg = _voix(*voix[qui], d, rng) * 0.3
        i = int(t * TAUX)
        x[i:i + len(seg)] += seg
        u = t
        while u + 0.35 < t + d:
            mots.append({"locuteur": qui, "debut": round(u, 3), "fin": round(u + 0.35, 3), "mot": "mot"})
            u += 0.4
        t += d + rng.uniform(0.4, 1.0)
        qui = "B" if qui == "A" else "A"
    x16 = np.clip(x * 32767 * 0.8, -32768, 32767).astype("<i2")
    chemin = dossier / "seance.m4a"
    try:                                   # AAC dans un conteneur MP4, comme une vidéo
        import av
        with av.open(str(chemin), "w") as sortie:
            flux = sortie.add_stream("aac", rate=TAUX)
            flux.layout = "mono"
            for i in range(0, len(x16), 1024):
                bloc = (x16[i:i + 1024].astype(np.float32) / 32768)[None, :]
                trame = av.AudioFrame.from_ndarray(bloc, format="fltp", layout="mono")
                trame.rate = TAUX
                for paquet in flux.encode(trame):
                    sortie.mux(paquet)
            for paquet in flux.encode(None):
                sortie.mux(paquet)
    except Exception:
        import wave
        chemin = dossier / "seance.wav"
        with wave.open(str(chemin), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(TAUX)
            w.writeframes(x16.tobytes())
    return chemin, mots


def verifier(chemin_rapport=None) -> int:
    lignes, ok = [], True

    def note(etape, reussi, detail=""):
        nonlocal ok
        ok = ok and reussi
        lignes.append(f"{'OK ' if reussi else 'ÉCHEC'}  {etape}{' : ' + detail if detail else ''}")

    try:
        import vosk
        vosk.SetLogLevel(-1)
        try:
            vosk.Model("/chemin/inexistant/modele")
            note("bibliothèque Vosk", False, "un modèle inexistant a été accepté")
        except Exception:
            note("bibliothèque Vosk", True, "chargée")
    except Exception as e:
        note("bibliothèque Vosk", False, repr(e))
    for nom in ("av", "numpy", "scipy", "sklearn", "jiwer"):
        try:
            __import__(nom)
            note(f"module {nom}", True)
        except Exception as e:
            note(f"module {nom}", False, repr(e))
    try:
        __import__("tkinter")
        note("interface tkinter", True)
    except Exception as e:
        note("interface tkinter", False, repr(e))

    tmp = Path(tempfile.mkdtemp(prefix="retranscription_verif_"))
    try:
        from .asr import Mot
        from .config import Parametres
        from .pipeline import evaluer, transcrire_session
        from .sorties import lire_csv
        medias = tmp / "medias"
        medias.mkdir()
        chemin, verite = _enregistrement(medias)
        note("enregistrement de test", True, chemin.name)
        p = Parametres(empreinte="mfcc", nb_locuteurs=2, n_extraits=1, duree_extrait_s=20)
        transcrire_session(medias, "TEST", "2026-01-01", tmp / "resultats", tmp / "registre", p,
                           transcripteur=lambda _: [Mot(m["debut"], m["fin"], m["mot"], 0.9) for m in verite])
        segs = lire_csv(tmp / "resultats" / "TEST" / "TEST_transcription.csv")
        locs = {s["speaker"] for s in segs}
        note("chaîne complète", len(locs) == 2, f"{len(segs)} segments, {len(locs)} locuteurs")
        # exactitude de l'attribution des voix
        bons = total = 0
        for m in verite:
            t = (m["debut"] + m["fin"]) / 2
            s = next((s for s in segs if float(s["debut_s"]) <= t <= float(s["fin_s"])), None)
            if s:
                total += 1
                bons += (s["speaker"] == "Speaker 01") == (m["locuteur"] == verite[0]["locuteur"])
        exact = bons / max(total, 1)
        note("attribution des voix", exact > 0.8, f"{100 * exact:.0f} % des mots")
        note("fichier ELAN", (tmp / "resultats" / "TEST" / "TEST_transcription.eaf").exists())
        note("évaluation", evaluer(tmp / "resultats", "TEST") == {}, "extraits non corrigés détectés")
    except Exception:
        note("chaîne complète", False, traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rapport = "\n".join(lignes) + f"\n\nRÉSULTAT : {'SUCCÈS' if ok else 'ÉCHEC'}\n"
    if chemin_rapport:
        Path(chemin_rapport).write_text(rapport, encoding="utf-8")
    try:
        print(rapport)
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(verifier(sys.argv[1] if len(sys.argv) > 1 else None))
