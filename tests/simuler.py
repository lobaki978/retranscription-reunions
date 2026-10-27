"""Simulation de sessions de réunion filmées par plusieurs caméras.

Sert uniquement aux tests : les modèles Vosk ne sont pas nécessaires.

* Voix synthétiques source-filtre (Fant, 1960) : train d'impulsions à la
  fréquence fondamentale du locuteur, filtré par trois formants dont les
  fréquences dépendent de la longueur du conduit vocal du locuteur.
* Salle : chaque locuteur est à une distance différente de chaque micro
  (atténuation, retard de propagation, bruit de fond propre au micro).
* Caméras : démarrages décalés, dérive d'horloge, une caméra stéréo à deux
  micros, une caméra stéréo à canaux identiques.
* Vérité terrain : mots, temps et locuteurs, dans le temps de la caméra 1.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

SR = 16000
VOYELLES = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
            "e": (530, 1840, 2480), "o": (570, 840, 2410), "y": (250, 1750, 2160)}
LEXIQUE = ("bonjour merci alors donc chaleur travail usine pause eau horaire équipe "
           "sécurité soleil température santé salarié patron proposition accord vote "
           "groupe question réponse idée solution règle été journée matin midi soir "
           "climatisation ventilation fatigue risque protection loi code mesure").split()

LOCUTEURS = {
    "A": {"f0": 110, "vtl": 1.00, "pos": (0.5, 0.5)},
    "B": {"f0": 205, "vtl": 1.18, "pos": (3.5, 0.6)},
    "C": {"f0": 135, "vtl": 0.93, "pos": (2.0, 3.0)},
    "D": {"f0": 240, "vtl": 1.10, "pos": (4.5, 3.5)},
    "E": {"f0": 160, "vtl": 1.05, "pos": (1.0, 4.0)},
}
MICROS = {"M1": (0.0, 0.0), "M2": (5.0, 0.0), "M3": (1.0, 4.5), "M4": (5.0, 4.5)}


def _resonateur(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    a = [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r]
    return lfilter([1 - r], a, x)


def synth_mot(loc, rng, duree):
    n = int(duree * SR)
    t = np.arange(n) / SR
    f0 = loc["f0"] * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t + rng.uniform(0, 6)))
    f0 *= 1 + 0.01 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1))
    phase = np.cumsum(f0 / SR)
    source = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    source = lfilter([1], [1, -0.95], source)            # inclinaison spectrale
    v1, v2 = rng.choice(list(VOYELLES), 2)
    moitie = n // 2
    y = np.zeros(n)
    for (debut, fin), v in (((0, moitie), v1), ((moitie, n), v2)):
        seg = source[debut:fin]
        out = np.zeros_like(seg)
        for f, bw in zip(VOYELLES[v], (80, 110, 160)):
            out += _resonateur(seg, f / loc["vtl"], bw)
        y[debut:fin] = out
    # consonne initiale : bruit filtré
    nc = int(0.04 * SR)
    y[:nc] += 0.3 * lfilter([1, -1], [1], rng.standard_normal(nc)) * np.std(y[nc:nc * 3] + 1e-9)
    env = np.minimum(1, np.minimum(t / 0.03, (duree - t) / 0.05))
    y *= np.clip(env, 0, 1)
    return y / (np.abs(y).max() + 1e-9)


def scenario(locuteurs, duree_s, rng):
    """Tours de parole : (locuteur, debut, fin, mot) en temps réel de la salle."""
    mots, t = [], 1.0
    precedent = None
    while t < duree_s - 5:
        loc = rng.choice([l for l in locuteurs if l != precedent])
        precedent = loc
        for _ in range(rng.integers(4, 22)):
            d = rng.uniform(0.25, 0.5)
            if t + d > duree_s - 2:
                break
            mots.append((loc, round(t, 3), round(t + d, 3), str(rng.choice(LEXIQUE))))
            t += d + rng.uniform(0.03, 0.18)
        t += rng.uniform(0.2, 1.5)
    return mots


def salle(mots, duree_s, rng):
    """Signal capté par chaque micro (temps réel de la salle)."""
    n = int(duree_s * SR)
    signaux = {m: 0.002 * rng.standard_normal(n) for m in MICROS}
    for m in MICROS:   # bruit de fond coloré propre à chaque micro
        signaux[m] = lfilter([1], [1, -0.9], signaux[m])
    for loc, a, b, _ in mots:
        y = synth_mot(LOCUTEURS[loc], rng, b - a) * 0.3
        for m, (mx, my) in MICROS.items():
            lx, ly = LOCUTEURS[loc]["pos"]
            dist = np.hypot(lx - mx, ly - my) + 0.3
            i0 = int((a + dist / 343) * SR)
            seg = y / dist
            signaux[m][i0:i0 + len(seg)] += seg[:max(0, n - i0)]
    return signaux


def camera(signal, debut_reel, derive, duree_s):
    """Ce qu'enregistre une caméra démarrée à debut_reel avec une horloge (1+derive)."""
    n_cam = int(duree_s * SR)
    t_cam = np.arange(n_cam) / SR
    t_reel = debut_reel + t_cam / (1 + derive)
    return np.interp(t_reel * SR, np.arange(len(signal)), signal, left=0, right=0)


def ecrire_video(chemin, canaux):
    wav = Path(chemin).with_suffix(".tmp.wav")
    x = np.vstack(canaux).T
    x = np.clip(x / max(1e-9, np.abs(x).max()) * 0.8 * 32767, -32768, 32767).astype("<i2")
    import wave
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=96x64:r=2",
                    "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-b:a", "128k", str(chemin)], check=True)
    wav.unlink()


def simuler_session(dossier, locuteurs, duree_s=600, graine=1):
    rng = np.random.default_rng(graine)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    mots = scenario(locuteurs, duree_s, rng)
    sig = salle(mots, duree_s + 20, rng)
    # caméra 1 (référence) : démarre à 3 s, horloge juste, stéréo à canaux identiques (micro M1)
    # caméra 2 : démarre à 0 s, +200 ppm, mono (micro M2)
    # caméra 3 : démarre à 10,5 s, -150 ppm, stéréo à deux micros (M3, M4)
    cams = {"camera1.mp4": (3.0, 0.0, ["M1", "M1"]),
            "camera2.mp4": (0.0, 200e-6, ["M2"]),
            "camera3.mp4": (10.5, -150e-6, ["M3", "M4"])}
    for nom, (debut, derive, micros) in cams.items():
        canaux = [camera(sig[m], debut, derive, duree_s) for m in micros]
        ecrire_video(dossier / nom, canaux)
    t0 = cams["camera1.mp4"][0]
    verite = [{"locuteur": l, "debut": round(a - t0, 3), "fin": round(b - t0, 3), "mot": w}
              for l, a, b, w in mots if a - t0 >= 0]
    # temps caméra k = a_k + (1 + d_k) * temps session, avec a_k = (t0 - début_k)(1 + d_k)
    attendu = {nom: {"a": round((t0 - debut) * (1 + derive), 4), "derive_ppm": derive * 1e6}
               for nom, (debut, derive, _) in cams.items()}
    (dossier.parent / f"{dossier.name}_verite.json").write_text(
        json.dumps({"mots": verite, "cameras": attendu}, ensure_ascii=False, indent=1), encoding="utf-8")
    return verite, attendu
