# Retranscription de réunions sans IA générative

Outil Python pour transcrire des réunions filmées par plusieurs caméras et
attribuer chaque prise de parole à un locuteur pseudonymisé (« Speaker 01 »,
« Speaker 02 »…), avec le même numéro d'une session à l'autre.

Aucun modèle génératif n'est utilisé (ni Whisper, ni LLM). Les réseaux
employés sont des classifieurs : ils attribuent une étiquette à chaque
trame de son, sans produire de texte à partir du texte déjà produit.

| Étape | Méthode | Référence |
|---|---|---|
| Transcription (par défaut) | wav2vec 2.0 affiné pour le français, décodage CTC glouton (un caractère par trame de 20 ms) | Baevski et al. (2020) ; Graves et al. (2006) ; Evain et al. (2021) |
| Transcription (option) | Vosk (Kaldi) : modèle acoustique TDNN, lexique fermé, modèle de langue n-gramme | Povey et al. (2011) |
| Qui parle dans chaque fenêtre de 10 s | segmentation pyannote 3.0 (« powerset », 3 voix au plus par fenêtre) | Plaquet & Bredin (2023) ; Bredin (2023) |
| Empreinte vocale | WeSpeaker ResNet34 (réseau discriminatif, ne produit ni texte ni son) | Wang et al. (2023) |
| Regroupement des voix | CAH, lien moyen, distance cosinus ; voix d'une fenêtre jamais regroupées ensemble | Sell et al. (2018) ; Bredin (2023) |
| Nombre de voix (si non fourni) | seuil de distance cosinus (0,75) | Bredin (2023) |
| Tours trop brefs | durée minimale d'un tour (1 s) | |
| Appariement entre sessions | affectation optimale (méthode hongroise) | Kuhn (1955) |
| Validation | WER (Levenshtein), DER (définition NIST) | Morris et al. (2004) ; Fiscus et al. (2006) |

Les réseaux tournent avec onnxruntime, sans PyTorch. L'ancienne méthode
(fenêtres glissantes de 1,5 s, `--diarisation fenetres`) reste disponible,
avec des empreintes WeSpeaker, des x-vecteurs Vosk (`--empreinte vosk`,
Snyder et al., 2018) ou des statistiques de MFCC sans réseau de neurones
(`--empreinte mfcc`, Davis & Mermelstein, 1980).

**Résultats sur de vraies réunions** (13 extraits de 30 s du corpus AMI,
anglais, 2 à 4 locuteurs ; `tests/test_diarisation_neurale.py`) :

| Méthode | DER, nombre de voix connu | DER, nombre estimé | Mots attribués à la bonne voix |
|---|---|---|---|
| segmentation + WeSpeaker (défaut) | 25,8 % | 21,9 % | 84 % (connu) ; 90 % (estimé) |
| fenêtres glissantes de 1,5 s + WeSpeaker | — | — | 59 % (connu) ; 46 % (estimé) |

Ces extraits sont très courts (peu de parole par voix) : sur une réunion
entière, chaque voix est mieux connue. Seule la mesure sur vos propres
extraits corrigés (section 3) vaut pour vos données.

**Trois façons de l'utiliser :**

| Pour qui | Comment | Voir |
|---|---|---|
| Personnes qui ne codent pas | Application avec fenêtre (Windows, Mac) | `GUIDE_UTILISATEUR.md` |
| Vous, dans Spyder | Script `lancer_session.py` | section 1 ci-dessous |
| Ligne de commande | `python -m retranscription …` | section 3 |

L'application est fabriquée automatiquement par GitHub pour Windows et Mac :
voir `GUIDE_GITHUB.md`.

---

## 1. Installation (Spyder ou ligne de commande)

Il faut Python 3.9 ou plus récent. Le son des vidéos est lu par PyAV, qui
contient FFmpeg : rien d'autre à installer.

```bash
cd retranscription_reunions
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -e .

python -m retranscription telecharger-modeles            # wav2vec 2.0 (~1,2 Go)
python -m retranscription telecharger-modeles --petit    # ou : version allégée (~400 Mo)
```

`pip install -e .` installe les dépendances et la commande `retranscription`
(équivalente à `python -m retranscription`), utilisable depuis n'importe quel
dossier tant que l'environnement `venv` est activé.

Le téléchargement place dans `modeles/` :

- `wav2vec2-fr` : reconnaissance du français (~1,2 Go), converti au format
  ONNX à partir de [bofenghuang/asr-wav2vec2-ctc-french](https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french)
  (Apache 2.0) ; ou `wav2vec2-fr-int8` avec `--petit` (poids en 8 bits, plus
  rapide, un peu moins précis) ;
- `segmentation-3.0.onnx` : segmentation des locuteurs (pyannote 3.0, MIT) ;
- `wespeaker-resnet34.onnx` : empreintes vocales (WeSpeaker, CC BY 4.0).

Ces fichiers viennent de la page « modeles-1 » du dépôt GitHub, préparée
une fois par le workflow « Préparer les modèles » (voir `GUIDE_GITHUB.md`).
Pour Vosk : `telecharger-modeles --moteur vosk`.

Mémoire et durée, mesurées avec un réseau de même taille (24 couches,
315 millions de paramètres) sur un processeur à 2 cœurs : environ 3 Go de
mémoire et 13 min de calcul par heure de son ; avec `--petit`, environ 2 Go
et 7 min. 8 Go de mémoire vive suffisent, 16 Go sont confortables.

### Avec Spyder (Anaconda, Windows)

Ouvrez `lancer_session.py` dans Spyder. Il est découpé en cellules
(lignes `# %%`). Cliquez dans une cellule, puis **Ctrl+Entrée** pour
l'exécuter seule. N'utilisez pas F5 (tout le fichier).

1. **Cellule RÉGLAGES** : indiquez les chemins (dossier du projet, dossier des
   vidéos de la session, dossiers de sortie et du registre), la session, la
   date, le nombre de locuteurs. Exécutez-la à chaque ouverture de Spyder.
2. **Cellule 0**, une seule fois : installe les bibliothèques et télécharge
   les modèles.
3. **Cellule 1** : transcrit la session. La console affiche l'avancement.
4. **Cellules 2 à 4** : après vos vérifications (voir section 3).

ffmpeg n'est plus nécessaire : la cellule 0 installe PyAV, qui le contient.

Conseils :

- Pour 6 h de vidéo, empêchez la mise en veille de l'ordinateur.
- Mettez le dossier `sorties` hors de OneDrive : les fichiers audio sont gros
  (~700 Mo par session) et la synchronisation ralentit tout.
- Après une erreur, redémarrez la console (Ctrl+. ou menu Consoles >
  Redémarrer le noyau) avant de relancer : Windows garde parfois des fichiers verrouillés.

## 2. Organisation des fichiers

Un dossier par session, avec toutes les vidéos (ou fichiers audio) de cette session :

```
medias/
  S01/  camera_salle.mp4  camera_table.mp4  zoom_h6.wav
  S02/  ...
```

Chaque canal de chaque fichier devient une piste. Les canaux stéréo
identiques (un seul micro enregistré deux fois) sont détectés et ignorés.
Les pistes sont alignées automatiquement (décalage de démarrage et dérive
d'horloge des caméras).

Aucune conversion préalable n'est nécessaire : l'outil lit les vidéos
(MP4, MOV, MTS, MKV, AVI…) et en extrait le son avec FFmpeg (via PyAV). Les vidéos ne
sont pas modifiées.

**Le temps 0 est le début du premier fichier** (ordre alphabétique), ou du
fichier indiqué par `--reference` (`REFERENCE` dans Spyder). Choisissez la
vidéo que vous utilisez déjà dans ELAN : les temps produits seront ceux de
cette vidéo.

**Dans ELAN**, `S01_transcription.eaf` s'ouvre avec la vidéo de référence et
le son composite déjà liés. Si vous déplacez les vidéos, ELAN demande où
elles se trouvent. Pour ajouter les autres caméras, utilisez Édition > Fichiers
liés (Edit > Linked Files) ; leur décalage par rapport à la référence est dans
`validation/S01_micros.csv` (colonne `decalage_s`).

## 3. Utilisation

### Transcrire une session

```bash
python -m retranscription transcrire medias/S01 --session S01 --date 2026-03-12 --nb-locuteurs 12
```

`--nb-locuteurs` = nombre de personnes qui parlent (participants + animateurs).
Sans elle, le nombre est estimé par un seuil de distance (`--seuil-voix`).
Comptez une à trois heures pour 6 h d'enregistrement, selon l'ordinateur (la
reconnaissance de la parole est l'étape la plus longue). Si vous relancez une session,
la transcription déjà faite est réutilisée (`--refaire-asr` pour la refaire).

**Commencez par un essai court** : l'option `--essai` (case « Essai » dans
l'application, `ESSAI = True` dans Spyder) ne traite que les 10 premières
minutes. Elle permet de vérifier l'installation et la qualité avant de lancer 6 h.

### Vérifier les numéros de locuteurs

Ouvrez `sorties/S01/validation/S01_synthese.txt`, puis
`S01_correspondances.csv`. Chaque ligne relie un groupe de voix de la session
à un numéro. La colonne `a_verifier` signale les cas douteux. Pour corriger,
modifiez la colonne `speaker`, puis :

```bash
python -m retranscription appliquer S01
```

- Deux groupes = une même personne : donnez-leur le même numéro.
- Un groupe attribué à tort à un locuteur connu : donnez-lui un nouveau numéro.
- Un groupe qui mélange deux personnes : relancez avec un `--nb-locuteurs` plus grand
  (ou un `--seuil-voix` plus petit, par exemple 0,65).

### Calibrer le seuil d'appariement entre sessions

Le seuil par défaut (0,5, cosinus entre empreintes moyennes WeSpeaker)
n'est qu'un point de départ. Sur les extraits AMI, deux personnes
différentes ne dépassent pas 0,32 ; une même personne dans deux extraits
obtient de 0,40 à 0,83 selon la quantité de parole. Après avoir vérifié
les correspondances de deux ou trois sessions :

```bash
python -m retranscription calibrer
```

La commande compare les paires « même locuteur » et « locuteurs différents »
et propose un seuil, à passer ensuite avec `--seuil-appariement`.

### Mesurer la fiabilité (WER, DER)

1. Ouvrez `sorties/S01/validation/extraits/` : 10 extraits de 2 min tirés au
   hasard dans des tranches égales de la session.
2. Lisez `CONSIGNES.txt`, puis corrigez chaque `.eaf` dans ELAN et enregistrez.
3. Lancez :

```bash
python -m retranscription evaluer S01
```

Résultat dans `S01_evaluation.csv` : WER, DER (parole manquée, fausse alarme,
confusion) et taux d'erreur d'identification, par extrait et au total.

### Fin du projet

```bash
python -m retranscription effacer-registre --confirmer
```

## 4. Fichiers produits

```
sorties/S01/
  S01_transcription.csv     une ligne par segment de parole
  S01_transcription.txt     version lisible
  S01_transcription.eaf     ELAN : une piste par locuteur + piste « à vérifier »
  S01_locuteurs.csv         métadonnées ; colonne « sexe » à remplir à la main
  S01_audio.wav             signal composite (meilleur micro à chaque instant)
  validation/
    S01_synthese.txt        indicateurs automatiques et marche à suivre
    S01_indicateurs_segments.csv
    S01_correspondances.csv groupes de voix -> numéros (modifiable)
    S01_micros.csv          alignement et part de chaque micro
    extraits/               extraits à corriger, puis S01_evaluation.csv
registre/                   empreintes moyennes par locuteur : NE PAS PARTAGER
```

**Segment et tour.** Un segment est une suite de mots d'un même locuteur
sans pause de plus de 2 s (`--pause-max`). Le numéro de tour ne change qu'au
changement de locuteur.

**Métadonnées.** La colonne `sexe` de `S01_locuteurs.csv` est conservée quand
le fichier est régénéré. Une valeur saisie pour un locuteur est reprise
automatiquement dans les autres sessions où il apparaît. Vous pouvez ajouter
vos propres colonnes : elles sont conservées aussi.

## 5. Lecture dans R

```r
library(readr)
tr  <- read_delim("sorties/S01/S01_transcription.csv", delim = ";")
loc <- read_delim("sorties/S01/S01_locuteurs.csv", delim = ";")
ind <- read_delim("sorties/S01/validation/S01_indicateurs_segments.csv", delim = ";")
# ou, en R de base :
tr <- read.csv("sorties/S01/S01_transcription.csv", sep = ";", fileEncoding = "UTF-8")
```

## 6. Principaux paramètres

| Option | Défaut | Rôle |
|---|---|---|
| `--nb-locuteurs` | estimé | nombre de personnes qui parlent |
| `--reference` | 1er fichier | fichier qui fixe le temps 0 |
| `--moteur` | wav2vec2 | `vosk` : plus rapide, moins précis |
| `--petit` | non | modèle allégé (8 bits pour wav2vec 2.0) |
| `--seuil-voix` | 0,75 | distance cosinus de coupure quand le nombre de voix est estimé ; plus petit = plus de voix |
| `--seuil-appariement` | 0,5 (WeSpeaker), 0,7 (Vosk, MFCC) | cosinus minimal pour reconnaître un locuteur d'une autre session |
| `--diarisation` | neurale | `fenetres` : ancienne méthode par fenêtres glissantes |
| `--empreinte` | wespeaker | empreintes de la méthode `fenetres` : `vosk` ou `mfcc` |
| `--poids-canaux` | 0 | méthode `fenetres` : part de la position (énergie par micro), de 0 à 1 |
| `--pause-max` | 2 s | pause qui coupe un segment |
| `--extraits`, `--duree-extrait` | 10, 120 s | échantillon de validation |

Pour comparer deux réglages (par exemple `--poids-canaux 0` et `0.3`), comparez
leur DER sur les mêmes extraits corrigés.

## 7. Limites connues

- **Parole superposée** : un seul locuteur est attribué à chaque mot.
- **Parole spontanée, micros éloignés** : la reconnaissance fait plus
  d'erreurs qu'en parole lue. Le WER mesuré sur vos extraits est la seule
  estimation fiable.
- **Orthographe** (wav2vec 2.0) : le texte est écrit lettre par lettre, sans
  lexique ni modèle de langue. Un mot mal entendu peut donc être mal
  orthographié (« prestatère ») ; les noms propres sont écrits comme ils se
  prononcent. Ces mots ont en général une confiance faible.
- **Mots hors lexique** (Vosk) : remplacés par le mot le plus proche de son lexique.
- **Ponctuation et majuscules** : ni wav2vec 2.0 ni Vosk n'en produisent.
- **Seuils** : seuils de voix et d'appariement, indicateurs de doute : à
  calibrer sur vos données.

## 8. Données personnelles (RGPD)

- Les sorties sont **pseudonymisées**, pas anonymes : la voix reste dans
  l'audio et les vidéos (considérant 26).
- Le dossier `registre/` contient des empreintes vocales moyennes, qui sont
  des **données biométriques** (art. 4(14) et 9). Il est séparé des sorties,
  sans nom, et effaçable en une commande. Conservez-le sur un support chiffré.
- Les empreintes de chaque fenêtre ne sont jamais écrites sur le disque.
- Tout tourne en local : aucun fichier n'est envoyé à un service en ligne.
- À valider avec le DPO de votre établissement.

## 9. Tests

```bash
pip install -r requirements-tests.txt
python tests/test_vosk_simule.py          # appels à Vosk (bibliothèque simulée), téléchargements
python tests/test_w2v.py                  # décodage CTC, blocs de 30 s ; conversion ONNX si PyTorch est installé
python tests/test_diarisation_neurale.py  # vraies réunions (AMI) : DER et attribution des mots
python tests/test_bout_en_bout.py         # 2 sessions simulées, 3 caméras, 4 micros
python -m retranscription.autotest        # auto-vérification (aussi : Retranscription --verifier)
```

`test_w2v.py` convertit un réseau wav2vec 2.0 miniature avec le script de
conversion du workflow et vérifie que les étiquettes calculées par
l'application sont celles de PyTorch. Le grand modèle français n'a pas pu
être téléchargé dans l'environnement où l'outil a été écrit : le premier
essai réel est à faire sur un extrait court.

Le test de bout en bout simule les voix (synthèse source-filtre) et la
transcription. Il vérifie l'alignement des caméras, le choix des micros,
le regroupement des voix, la stabilité des numéros entre sessions, les
fichiers ELAN, la conservation des métadonnées saisies et le calcul du WER
et du DER (comparés à un calcul indépendant et à pyannote.metrics).

Test de charge (session simulée de 6 h, 3 caméras, 4 micros) : toutes les
étapes hors reconnaissance de la parole prennent environ 2 minutes ; la dérive des caméras est
retrouvée à 0,1 ppm près (40,1 pour 40 ppm ; -25,1 pour -25 ppm). La durée de
la reconnaissance dépend de l'ordinateur ; elle est à mesurer sur l'essai court.

## Références

- Baevski, A., Zhou, H., Mohamed, A. & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS*.
- Bredin, H. (2017). pyannote.metrics: A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems. *Interspeech*.
- Bredin, H. (2023). pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe. *Interspeech*.
- Carletta, J. et al. (2005). The AMI meeting corpus: A pre-announcement. *MLMI*, LNCS 3869.
- Davis, S. & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Trans. ASSP*, 28(4), 357–366.
- Evain, S. et al. (2021). LeBenchmark: A reproducible framework for assessing self-supervised representation learning from speech. *Interspeech*.
- Fiscus, J. G., Ajot, J., Michel, M. & Garofolo, J. S. (2006). The Rich Transcription 2006 Spring Meeting Recognition Evaluation. *MLMI*, LNCS 4299.
- Graves, A., Fernández, S., Gomez, F. & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *ICML*.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2, 83–97.
- Morris, A. C., Maier, V. & Green, P. (2004). From WER and RIL to MER and WIL: Improved evaluation measures for connected speech recognition. *Interspeech*.
- Park, T. J., Han, K. J., Kumar, M. & Narayanan, S. (2020). Auto-tuning spectral clustering for speaker diarization using normalized maximum eigengap. *IEEE Signal Processing Letters*, 27, 381–385.
- Plaquet, A. & Bredin, H. (2023). Powerset multi-class cross entropy loss for neural speaker diarization. *Interspeech*.
- Povey, D. et al. (2011). The Kaldi speech recognition toolkit. *IEEE ASRU Workshop*.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Sell, G. et al. (2018). Diarization is hard: Some experiences and lessons learned for the JHU team in the inaugural DIHARD challenge. *Interspeech*.
- Snyder, D., Garcia-Romero, D., Sell, G., Povey, D. & Khudanpur, S. (2018). X-vectors: Robust DNN embeddings for speaker recognition. *ICASSP*.
- Wang, H. et al. (2023). WeSpeaker: A research and production oriented speaker embedding learning toolkit. *ICASSP*.
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17, 395–416.
- Règlement (UE) 2016/679 (RGPD), art. 4(5), 4(14), 9 et considérant 26.
