"""Auto-vérification de l'application installée (option --verifier).

Utilisée après chaque fabrication automatique (GitHub Actions) : elle
vérifie que toutes les bibliothèques sont bien incluses, puis fait tourner
toute la chaîne sur un enregistrement synthétique de deux voix, avec une
transcription simulée (les grands modèles ne sont pas téléchargés ici).

Le décodage wav2vec 2.0 est vérifié avec un réseau miniature (poids
aléatoires, même architecture) livré avec l'application. La diarisation
neurale est vérifiée si les modèles de segmentation et d'empreintes sont
présents dans le dossier des modèles.
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


def _signal(chemin: Path) -> np.ndarray:
    from .audio import extraire_pistes
    piste = extraire_pistes([chemin], chemin.parent / "_pistes")[0]
    return np.fromfile(piste.chemin, dtype="<i2")


def _verifier_multicanal(note, tmp: Path) -> None:
    """Fichier WAV à 10 canaux (enregistreur multipiste) : chaque canal devient une piste."""
    try:
        import wave
        from .audio import extraire_pistes
        n_c, n = 10, 48000 * 3
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((n, n_c)) * 3000).astype("<i2")
        chemin = tmp / "multipiste.WAV"
        with wave.open(str(chemin), "wb") as w:
            w.setnchannels(n_c); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes(x.tobytes())
        pistes = extraire_pistes([chemin], tmp / "_multi")
        ok = len(pistes) == n_c and all(abs(p.n - 3 * TAUX) < TAUX // 10 for p in pistes)
        note("fichier à 10 canaux", ok, f"{len(pistes)} pistes")
    except Exception:
        note("fichier à 10 canaux", False, traceback.format_exc())


def _verifier_w2v(note, chemin: Path) -> None:
    """Décodage CTC avec le réseau miniature : comparé à un décodage de référence."""
    try:
        from .asr_w2v import TranscripteurW2V
        mini = Path(__file__).resolve().parent / "donnees" / "w2v_mini"
        if not (mini / "model.onnx").exists():
            note("wav2vec 2.0 (réseau miniature)", False, f"absent : {mini}")
            return
        t = TranscripteurW2V(mini)
        x = _signal(chemin)[: 45 * TAUX]
        ids, probs = t.trames(x, bloc_s=20.0)
        attendu = len(x) // t.ept
        # référence : tout le signal d'un seul tenant, argmax direct
        ref = t._logits(x).argmax(axis=1)
        n = min(len(ids), len(ref))
        accord = float((ids[:n] == ref[:n]).mean())
        mots = t.decoder(ids, probs)
        note("wav2vec 2.0 (réseau miniature)", abs(len(ids) - attendu) <= 2 and accord > 0.9,
             f"{len(ids)} trames (attendu {attendu}), accord par blocs {100 * accord:.0f} %, {len(mots)} mots")
    except Exception:
        note("wav2vec 2.0 (réseau miniature)", False, traceback.format_exc())


ARPA_TEST = """\\data\\
ngram 1=6
ngram 2=3

\\1-grams:
-1.0\t<s>\t-0.3
-1.0\t</s>
-3.0\t<unk>
-1.2\tla\t-0.3
-1.4\tmaison\t-0.3
-1.2\tbelle\t-0.3

\\2-grams:
-0.3\t<s> la
-0.2\tla maison
-0.2\tmaison belle

\\end\\
"""


def _verifier_langue(note) -> None:
    """Décodage par faisceau avec un petit modèle de langue : « la maisun » doit devenir « la maison ».

    Non bloquant : sans KenLM, l'application fonctionne avec le décodage glouton.
    """
    from . import langue
    raison = langue.disponible()
    if raison:
        note("modèle de langue (KenLM, pyctcdecode)", False, f"absent ({raison}) : décodage glouton seulement",
             bloquant=False)
        return
    tmp = Path(tempfile.mkdtemp(prefix="retranscription_lm_"))
    try:
        (tmp / "test.arpa").write_text(ARPA_TEST, encoding="utf-8")
        vocab = {0: "<pad>", 1: "|", 2: "<unk>"}
        for c in "abcdefghijklmnopqrstuvwxyz":
            vocab[len(vocab)] = c
        idx = {t: i for i, t in vocab.items()}
        P = []
        for ch in "la maison":
            car = "|" if ch == " " else ch
            p = np.full(len(vocab), 1e-4)
            if ch == "o":
                p[idx["u"]], p[idx["o"]] = 0.55, 0.40
            else:
                p[idx[car]] = 0.95
            P += [p / p.sum(), p / p.sum()]
            b = np.full(len(vocab), 1e-4)
            b[0] = 0.95
            P.append(b / b.sum())
        P = np.array(P)
        dec = langue.DecodeurLM(tmp, vocab, 0, "|", len(vocab), largeur=16)
        mots = dec.decoder(np.log(P).astype(np.float16), P.argmax(axis=1), P.max(axis=1), 0.02)
        texte = " ".join(m.mot for m in mots)
        note("modèle de langue (KenLM, pyctcdecode)", texte == "la maison", f"« la maisun » -> « {texte} »",
             bloquant=False)
    except Exception:
        note("modèle de langue (KenLM, pyctcdecode)", False, traceback.format_exc(), bloquant=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verifier_neurale(note, chemin: Path, verite) -> None:
    from .config import Parametres
    p = Parametres()
    if not (Path(p.modele_segmentation).exists() and Path(p.modele_wespeaker).exists()):
        note("diarisation neurale", True, "non vérifiée : modèles absents (téléchargés au premier lancement)")
        return
    try:
        from .asr import Mot
        from .segmentation import ExtracteurWeSpeaker, Segmenteur, attribuer_mots_tours, diariser
        x = _signal(chemin)
        res = diariser(x, len(x) / TAUX, Segmenteur(p.modele_segmentation),
                       ExtracteurWeSpeaker(p.modele_wespeaker), k=2)
        mots = [Mot(m["debut"], m["fin"], m["mot"], 0.9) for m in verite]
        g = attribuer_mots_tours(mots, res, portee_s=float("inf"))[:, 0]
        premier = g[0]
        bons = sum((gi == premier) == (m["locuteur"] == verite[0]["locuteur"]) for gi, m in zip(g, verite))
        exact = bons / len(verite)
        note("diarisation neurale", exact > 0.8, f"{len(res.tours)} tours, {100 * exact:.0f} % des mots")
    except Exception:
        note("diarisation neurale", False, traceback.format_exc())


def verifier(chemin_rapport=None) -> int:
    lignes, ok = [], True

    def note(etape, reussi, detail="", bloquant=True):
        nonlocal ok
        ok = ok and (reussi or not bloquant)
        etat = "OK " if reussi else ("ÉCHEC" if bloquant else "ATTENTION")
        lignes.append(f"{etat}  {etape}{' : ' + detail if detail else ''}")

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
    for nom in ("av", "numpy", "scipy", "sklearn", "jiwer", "onnxruntime"):
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
        p = Parametres(diarisation="fenetres", empreinte="mfcc", nb_locuteurs=2, n_extraits=1,
                       duree_extrait_s=20)
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
        _verifier_multicanal(note, tmp)
        _verifier_w2v(note, chemin)
        _verifier_langue(note)
        _verifier_neurale(note, chemin, verite)
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
