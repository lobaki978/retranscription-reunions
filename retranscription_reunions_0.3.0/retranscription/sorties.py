"""Fichiers produits : segments de parole, CSV, TXT, ELAN (.eaf), métadonnées.

Un « segment » est une suite de mots d'un même locuteur sans pause de plus
de `pause_max` secondes. Le numéro de « tour » ne change qu'au changement
de locuteur : un tour peut donc contenir plusieurs segments.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET

import numpy as np

from .registre import numero
from .utils import hms

SEP = ";"


@dataclass
class Segment:
    speaker: str
    debut: float
    fin: float
    mots: List[str] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    sims: List[float] = field(default_factory=list)
    marges: List[float] = field(default_factory=list)
    groupe_local: int = -1
    tour: int = 0
    rang: int = 0
    motifs: List[str] = field(default_factory=list)

    @property
    def texte(self) -> str:
        return " ".join(self.mots)

    @property
    def duree(self) -> float:
        return self.fin - self.debut

    def moyenne(self, valeurs) -> float:
        v = [x for x in valeurs if x is not None and x == x]
        return float(np.mean(v)) if v else float("nan")


def segmenter(mots: List[dict], correspondance: Dict[int, str], pause_max: float = 2.0) -> List[Segment]:
    segments: List[Segment] = []
    for m in mots:
        g = m.get("groupe", -1)
        if g is None or g < 0:
            continue
        spk = correspondance.get(int(g), "Speaker ??")
        dernier = segments[-1] if segments else None
        if dernier and dernier.speaker == spk and m["debut"] - dernier.fin <= pause_max:
            s = dernier
        else:
            s = Segment(speaker=spk, debut=m["debut"], fin=m["fin"], groupe_local=int(g))
            segments.append(s)
        s.fin = max(s.fin, m["fin"])
        s.mots.append(m["mot"])
        s.confs.append(m.get("conf"))
        s.sims.append(m.get("sim"))
        s.marges.append(m.get("marge"))
    tour = 0
    for i, s in enumerate(segments):
        if i == 0 or s.speaker != segments[i - 1].speaker:
            tour += 1
        s.tour, s.rang = tour, i + 1
    return segments


def signaler(segments: List[Segment], p, groupes_a_verifier: set) -> None:
    """Motifs de vérification par segment (indicateurs automatiques)."""
    for i, s in enumerate(segments):
        s.motifs = []
        conf = s.moyenne(s.confs)
        if conf == conf and conf < p.seuil_conf_segment:
            s.motifs.append(f"confiance moyenne {conf:.2f}")
        faibles = np.mean([c < p.seuil_conf_mot for c in s.confs if c is not None]) if s.confs else 0
        if faibles > p.part_mots_faibles:
            s.motifs.append(f"{100 * faibles:.0f} % de mots peu sûrs")
        marge = s.moyenne(s.marges)
        if marge == marge and marge < p.seuil_marge:
            s.motifs.append(f"attribution incertaine (marge {marge:.2f})")
        if (0 < i < len(segments) - 1 and s.duree < 1.0
                and segments[i - 1].speaker == segments[i + 1].speaker != s.speaker):
            s.motifs.append("tour très bref entre deux tours d'un autre locuteur")
        if s.groupe_local in groupes_a_verifier:
            s.motifs.append("appariement inter-sessions à vérifier")


# ----------------------------------------------------------------------
def _ecrire_csv(chemin: Path, entetes: List[str], lignes: List[list]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=SEP)
        w.writerow(entetes)
        w.writerows(lignes)


def lire_csv(chemin: Path) -> List[dict]:
    with open(chemin, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=SEP))


def _f(x, n=3):
    return "" if x is None or x != x else round(float(x), n)


def ecrire_transcription(segments: List[Segment], session: str, date: str, dossier: Path,
                         methode: str = "wav2vec 2.0 + CTC") -> None:
    lignes = [[session, date, s.tour, s.rang, s.speaker, _f(s.debut), _f(s.fin),
               hms(s.debut), hms(s.fin), _f(s.duree), s.texte, len(s.mots), _f(s.moyenne(s.confs))]
              for s in segments]
    _ecrire_csv(dossier / f"{session}_transcription.csv",
                ["session", "date", "tour", "segment", "speaker", "debut_s", "fin_s",
                 "debut", "fin", "duree_s", "texte", "n_mots", "conf_moy"], lignes)
    with open(dossier / f"{session}_transcription.txt", "w", encoding="utf-8") as f:
        f.write(f"Session {session} — {date}\n")
        f.write(f"Transcription automatique ({methode}), locuteurs pseudonymisés.\n")
        f.write("Texte non relu : voir le dossier validation/.\n\n")
        for s in segments:
            f.write(f"[{hms(s.debut, ms=False)}] {s.speaker} : {s.texte}\n\n")


def ecrire_indicateurs(segments: List[Segment], session: str, dossier: Path) -> None:
    lignes = [[session, s.rang, s.tour, s.speaker, s.groupe_local, _f(s.debut), _f(s.fin),
               hms(s.debut), len(s.mots), _f(s.moyenne(s.confs)),
               _f(np.mean([c < 0.5 for c in s.confs]) if s.confs else None),
               _f(s.moyenne(s.sims)), _f(s.moyenne(s.marges)),
               "oui" if s.motifs else "non", " | ".join(s.motifs), s.texte]
              for s in segments]
    _ecrire_csv(dossier / f"{session}_indicateurs_segments.csv",
                ["session", "segment", "tour", "speaker", "groupe_local", "debut_s", "fin_s",
                 "debut", "n_mots", "conf_moy", "part_mots_conf_inf_0.5", "similarite_voix",
                 "marge_voix", "a_verifier", "motifs", "texte"], lignes)


# ----------------------------------------------------------------------
def ecrire_locuteurs(segments: List[Segment], session: str, date: str, dossier: Path,
                     racine: Path, decisions: List[dict]) -> None:
    """Métadonnées par locuteur. Les colonnes remplies à la main sont conservées."""
    chemin = dossier / f"{session}_locuteurs.csv"
    manuel: Dict[str, dict] = {}
    if chemin.exists():
        for ligne in lire_csv(chemin):
            manuel[ligne["speaker"]] = ligne
        extra = [c for c in lire_csv(chemin)[0].keys()] if manuel else []
    else:
        extra = []
    # sexe déjà renseigné dans une autre session pour le même locuteur
    connus: Dict[str, str] = {}
    for autre in sorted(Path(racine).glob("*/*_locuteurs.csv")):
        if autre == chemin:
            continue
        for ligne in lire_csv(autre):
            if ligne.get("sexe"):
                connus.setdefault(ligne["speaker"], ligne["sexe"])
    total = sum(s.duree for s in segments) or 1.0
    par_spk: Dict[str, List[Segment]] = {}
    for s in segments:
        par_spk.setdefault(s.speaker, []).append(s)
    info_groupes: Dict[str, List[dict]] = {}
    for d in decisions:
        info_groupes.setdefault(d["speaker"], []).append(d)
    entetes = ["session", "date", "speaker", "sexe", "temps_parole_s", "part_temps_parole",
               "nb_tours", "nb_segments", "nb_mots", "premiere_prise", "derniere_prise",
               "groupes_locaux", "statut_registre"]
    standard = set(entetes)
    supplementaires = [c for c in extra if c not in standard]
    lignes = []
    for spk in sorted(par_spk, key=numero):
        segs = par_spk[spk]
        ancien = manuel.get(spk, {})
        sexe = ancien.get("sexe") or connus.get(spk, "")
        groupes = info_groupes.get(spk, [])
        lignes.append([session, date, spk, sexe, _f(sum(s.duree for s in segs), 1),
                       _f(sum(s.duree for s in segs) / total, 4),
                       len({s.tour for s in segs}), len(segs), sum(len(s.mots) for s in segs),
                       hms(segs[0].debut, ms=False), hms(segs[-1].debut, ms=False),
                       ",".join(str(g["groupe_local"]) for g in groupes),
                       ",".join(sorted({g["statut"] for g in groupes}))]
                      + [ancien.get(c, "") for c in supplementaires])
    _ecrire_csv(chemin, entetes + supplementaires, lignes)


def ecrire_correspondances(decisions: List[dict], session: str, chemin: Path) -> None:
    lignes = [[session, d["groupe_local"], d["speaker"], d["statut"], _f(d["similarite"]),
               _f(d["second"]), _f(d["marge"]), d["duree_parole_s"], d["a_verifier"], d["motif"]]
              for d in decisions]
    _ecrire_csv(chemin, ["session", "groupe_local", "speaker", "statut", "similarite",
                         "second_meilleur", "marge", "duree_parole_s", "a_verifier", "motif"], lignes)


# ----------------------------------------------------------------------
def ecrire_eaf(chemin: Path, annotations: List[tuple], tiers: List[str],
               medias: List[tuple] = (), tier_controle: List[tuple] = ()) -> None:
    """annotations : (tier, debut_s, fin_s, texte). medias : (chemin, type MIME)."""
    racine = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "", "DATE": datetime.now().astimezone().isoformat(timespec="seconds"),
        "FORMAT": "3.0", "VERSION": "3.0",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"})
    tete = ET.SubElement(racine, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    for chemin_media, mime in medias:
        p = Path(chemin_media)
        ET.SubElement(tete, "MEDIA_DESCRIPTOR", {
            "MEDIA_URL": p.resolve().as_uri(), "MIME_TYPE": mime,
            "RELATIVE_MEDIA_URL": "./" + p.name if p.parent.resolve() == Path(chemin).parent.resolve()
            else p.resolve().as_uri()})
    toutes = [(t, a, b, x) for t, a, b, x in annotations] + \
             [("à vérifier", a, b, x) for a, b, x in tier_controle]
    # deux repères temporels propres à chaque annotation : dans ELAN, déplacer
    # une frontière ne déplace pas celle d'un autre locuteur
    noms_tiers = list(tiers) + (["à vérifier"] if tier_controle else [])
    par_tier, reperes = {nom: [] for nom in noms_tiers}, []
    for t, a, b, x in toutes:
        va, vb = int(round(a * 1000)), int(round(b * 1000))
        if vb <= va or t not in par_tier:
            continue
        i1, i2 = len(reperes), len(reperes) + 1
        reperes += [va, vb]
        par_tier[t].append((va, i1, i2, x))
    ordre_reperes = sorted(range(len(reperes)), key=lambda i: (reperes[i], i))
    ids = {i: f"ts{r + 1}" for r, i in enumerate(ordre_reperes)}
    ordre = ET.SubElement(racine, "TIME_ORDER")
    for i in ordre_reperes:
        ET.SubElement(ordre, "TIME_SLOT", {"TIME_SLOT_ID": ids[i], "TIME_VALUE": str(reperes[i])})
    n = 0
    for nom in noms_tiers:
        tier = ET.SubElement(racine, "TIER", {"LINGUISTIC_TYPE_REF": "transcription", "TIER_ID": nom})
        for va, i1, i2, x in sorted(par_tier[nom]):
            n += 1
            ann = ET.SubElement(tier, "ANNOTATION")
            al = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"a{n}", "TIME_SLOT_REF1": ids[i1], "TIME_SLOT_REF2": ids[i2]})
            ET.SubElement(al, "ANNOTATION_VALUE").text = x
    prop = ET.SubElement(tete, "PROPERTY", {"NAME": "lastUsedAnnotationId"})
    prop.text = str(n)
    ET.SubElement(racine, "LINGUISTIC_TYPE", {"GRAPHIC_REFERENCES": "false",
                                               "LINGUISTIC_TYPE_ID": "transcription",
                                               "TIME_ALIGNABLE": "true"})
    ET.indent(racine, space="    ")
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(racine).write(chemin, encoding="UTF-8", xml_declaration=True)


def lire_eaf(chemin: Path, ignorer=("à vérifier",)) -> List[tuple]:
    """Annotations alignées d'un fichier ELAN : (tier, debut_s, fin_s, texte)."""
    racine = ET.parse(chemin).getroot()
    temps = {ts.get("TIME_SLOT_ID"): ts.get("TIME_VALUE")
             for ts in racine.iter("TIME_SLOT")}
    sortie = []
    for tier in racine.iter("TIER"):
        nom = tier.get("TIER_ID")
        if nom in ignorer or nom.startswith("_"):
            continue
        for al in tier.iter("ALIGNABLE_ANNOTATION"):
            a, b = temps.get(al.get("TIME_SLOT_REF1")), temps.get(al.get("TIME_SLOT_REF2"))
            if a is None or b is None:
                continue
            v = al.findtext("ANNOTATION_VALUE") or ""
            sortie.append((nom, int(a) / 1000, int(b) / 1000, v.strip()))
    return sorted(sortie, key=lambda z: z[1])


def ecrire_eaf_session(segments: List[Segment], chemin: Path, medias: List[tuple]) -> None:
    tiers = sorted({s.speaker for s in segments}, key=numero)
    ecrire_eaf(chemin, [(s.speaker, s.debut, s.fin, s.texte) for s in segments], tiers, medias,
               [(s.debut, s.fin, " | ".join(s.motifs)) for s in segments if s.motifs])
