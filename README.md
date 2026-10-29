# Retranscription de réunions sans IA générative

Outil Python pour transcrire des réunions filmées par plusieurs caméras et
attribuer chaque prise de parole à un locuteur pseudonymisé (« Speaker 01 »,
« Speaker 02 »…), avec le même numéro d'une session à l'autre.

Aucun modèle génératif n'est utilisé (ni Whisper, ni LLM). Les réseaux
employés sont des classifieurs : ils attribuent une étiquette à chaque
trame de son, sans produire de texte à partir du texte déjà produit. Le
modèle de langue est un modèle n-gramme : des fréquences de suites de
mots comptées dans un corpus, comme celui de Vosk/Kaldi.

| Étape | Méthode | Référence |
|---|---|---|
| Transcription (par défaut) | wav2vec 2.0 affiné pour le français : une probabilité par caractère et par trame de 20 ms (CTC) | Baevski et al. (2020) ; Graves et al. (2006) ; Evain et al. (2021) |
| Décodage | recherche par faisceau CTC avec modèle de langue 5-grammes (KenLM) et lexique du projet facultatif ; à défaut, décodage glouton | Hannun et al. (2014) ; Heafield (2011) ; Jelinek (1997) |
| Transcription (option) | Vosk (Kaldi) : modèle acoustique TDNN, lexique fermé, modèle de langue n-gramme | Povey et al. (2011) |
| Qui parle dans chaque fenêtre de 10 s | segmentation pyannote 3.0 (« powerset », 3 voix au plus par fenêtre) | Plaquet & Bredin (2023) ; Bredin (2023) |
| Empreinte vocale | WeSpeaker ResNet34 (réseau discriminatif, ne produit ni texte ni son) | Wang et al. (2023) |
| Regroupement des voix | CAH, lien moyen, distance cosinus ; voix d'une fenêtre jamais regroupées ensemble ; groupes de moins de 6 empreintes rattachés à la voix la plus proche (règle de pyannote 3.1) | Sell et al. (2018) ; Bredin (2023) |
| Nombre de voix (si non fourni) | seuil de distance cosinus (0,75) | Bredin (2023) |
| Position des micros (option) | profil d'énergie des micros ajouté aux empreintes vocales | |
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

**Essais de la version 0.4** (scripts de test ; chiffres à confirmer sur vos données) :

- *Réunion à nombreux participants* : les 10 extraits AMI mis bout à bout
  (5,3 min, 24 personnes). Nombre de voix estimé : 28 sans rattachement des
  petits groupes, 20 avec ; DER 22,9 % → 22,3 %. Le regroupement de
  pyannote 3.1 tel quel (lien centroïde, seuil 0,7046, taille minimale 12)
  a été porté et essayé : il fusionne les personnes qui parlent peu
  (DER 28,0 % avec 24 voix demandées, 36,4 % en estimant le nombre) ; il
  reste disponible (`--regroupement pyannote31`).
- *Salle simulée* (réverbération 0,6 s + bruit de fond) : le DER monte à
  66 % (bruit à 10 dB) et 85 % (5 dB), surtout parce que la parole n'est
  plus détectée. Le choix des micros compte plus que les réglages.
- *Modèle de langue* : 60 phrases françaises lues par une voix de synthèse,
  en salle simulée, avec un petit modèle de langue de test (trigrammes,
  95 000 phrases de Common Voice ne contenant pas les phrases de test) :
  WER 29,3 % → 24,3 % (bruit à 10 dB) et 55,3 % → 50,7 % (5 dB) ; les
  non-mots et les mots collés disparaissent. Le modèle 5-grammes livré
  est bien plus grand ; son effet sur vos réunions reste à mesurer.

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
- `lm-fr-5gram` : modèle de langue 5-grammes (KenLM, ~1 Go) publié avec le
  modèle de reconnaissance (`lm_5gram_big.bin`, `unigrams.txt`, `attrs.json`) ;
  `--sans-modele-langue` pour ne pas le télécharger ;
- `segmentation-3.0.onnx` : segmentation des locuteurs (pyannote 3.0, MIT) ;
- `wespeaker-resnet34.onnx` : empreintes vocales (WeSpeaker, CC BY 4.0).

Ces fichiers viennent de la page « modeles-1 » du dépôt GitHub, préparée
par le workflow « Préparer les modèles » (voir `GUIDE_GITHUB.md`).
Pour Vosk : `telecharger-modeles --moteur vosk`.

**Modèle de langue : deux bibliothèques en plus.** pyctcdecode (recherche par
faisceau) et KenLM (lecture du modèle). pyctcdecode s'installe sans ses
dépendances, car il impose une ancienne version de numpy dont il n'a pas
besoin ; KenLM doit être compilé :

```bash
pip install --no-deps pyctcdecode==0.5.0 "pygtrie>=2.1,<3"
pip install kenlm        # Windows : « Outils de génération Visual Studio » nécessaires
```

Sous Anaconda, `conda install -c conda-forge kenlm` évite la compilation
si le paquet existe pour votre système. Sans ces bibliothèques, l'outil
fonctionne avec le décodage glouton et le signale dans le journal.
L'application fabriquée par GitHub les contient déjà.

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
  S01_transcription.txt     version lisible : un paragraphe par prise de parole
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
sans pause de plus de 2 s (`--pause-max`) ; au-delà de 30 s, il est coupé à
la pause suivante. Le numéro de tour ne change qu'au changement de
locuteur. Dans le `.txt`, le locuteur n'est écrit qu'au début de chaque
tour ; les segments suivants du même tour sont en retrait, avec leur heure.

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
| `--lexique` | aucun | fichier texte : termes du projet favorisés au décodage (voir ci-dessous) |
| `--sans-modele-langue` | non | décodage glouton, comme la version 0.3 |
| `--faisceau` | 64 | largeur du faisceau (plus large : un peu plus juste, plus lent) |
| `--seuil-voix` | 0,75 | distance cosinus de coupure quand le nombre de voix est estimé ; plus petit = plus de voix |
| `--seuil-appariement` | 0,5 (WeSpeaker), 0,7 (Vosk, MFCC) | cosinus minimal pour reconnaître un locuteur d'une autre session |
| `--diarisation` | neurale | `fenetres` : ancienne méthode par fenêtres glissantes |
| `--empreinte` | wespeaker | empreintes de la méthode `fenetres` : `vosk` ou `mfcc` |
| `--poids-canaux` | 0 | part de la position (énergie de chaque micro) dans le regroupement des voix, de 0 à 1 ; 0,3 si un micro est proche de chaque participant (case « Un micro près de chaque participant » de l'application) |
| `--taille-min-voix` | 6 | groupe de moins d'empreintes rattaché à la voix la plus proche |
| `--regroupement` | moyen | `pyannote31` : regroupement de pyannote 3.1 tel quel |
| `--pause-max` | 2 s | pause qui coupe un segment |
| `--extraits`, `--duree-extrait` | 10, 120 s | échantillon de validation |

Pour comparer deux réglages (par exemple `--poids-canaux 0` et `0.3`), comparez
leur DER sur les mêmes extraits corrigés.

### Lexique du projet

Un fichier texte (UTF-8), un terme par ligne : noms de lieux, sigles, mots
du domaine, expressions. Les lignes qui commencent par `#` sont ignorées.
Ces termes reçoivent un bonus pendant le décodage, même s'ils sont absents
du modèle de langue. Quelques centaines de termes suffisent ; un lexique
trop long favorise des mots qui n'ont pas été prononcés.

Avec R et UDPipe (Wijffels, 2023 ; Straka & Straková, 2017), à partir des
documents du projet (comptes rendus, supports d'atelier) :

```r
library(udpipe)
m <- udpipe_download_model(language = "french-gsd")    # une seule fois
modele <- udpipe_load_model(m$file_model)
fichiers <- list.files("documents_projet", pattern = "\\.txt$", full.names = TRUE)
textes <- vapply(fichiers, function(f) paste(readLines(f, encoding = "UTF-8"), collapse = " "), "")
ann <- as.data.frame(udpipe_annotate(modele, x = textes, doc_id = basename(fichiers)))
mots <- subset(ann, upos %in% c("NOUN", "PROPN", "ADJ") & nchar(token) > 3)
freq <- sort(table(tolower(mots$token)), decreasing = TRUE)   # formes, pas lemmes : on entend des formes
con <- file("lexique.txt", encoding = "UTF-8")
writeLines(names(freq)[seq_len(min(300, length(freq)))], con)
close(con)
```

Relisez la liste avant de l'utiliser : retirez les mots courants, ajoutez
les expressions (« filière textile ») et les noms propres oubliés.

## 7. Limites connues

- **Parole superposée** : un seul locuteur est attribué à chaque mot.
- **Micros éloignés** : c'est la première cause d'erreur. Sur le corpus AMI,
  le même système fait deux fois plus d'erreurs avec un micro de table
  qu'avec un micro-casque (WER 45 % contre 22 % ; Peddinti et al., 2016).
  Un micro près de chaque participant (enregistreur multipiste) améliore
  à la fois le texte et la reconnaissance des voix. Le WER mesuré sur vos
  extraits est la seule estimation fiable.
- **Orthographe** (wav2vec 2.0) : avec le modèle de langue, les mots écrits
  existent ; un mot mal entendu peut alors être remplacé par un autre mot,
  plausible mais faux. Sans modèle de langue, le texte est écrit lettre par
  lettre et un mot mal entendu peut être mal orthographié (« prestatère »).
  Les noms propres absents du modèle de langue sont mieux reconnus s'ils
  sont dans le lexique du projet.
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
python tests/test_langue.py               # modèle de langue et lexique (petit modèle ARPA de test)
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
- Hannun, A. Y., Maas, A. L., Jurafsky, D. & Ng, A. Y. (2014). First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs. *arXiv:1408.2873*.
- Heafield, K. (2011). KenLM: Faster and smaller language model queries. *Workshop on Statistical Machine Translation*.
- Jelinek, F. (1997). *Statistical Methods for Speech Recognition*. MIT Press.
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
- Peddinti, V., Manohar, V., Wang, Y., Povey, D. & Khudanpur, S. (2016). Far-field ASR without parallel data. *Interspeech*.
- Plaquet, A. & Bredin, H. (2023). Powerset multi-class cross entropy loss for neural speaker diarization. *Interspeech*.
- Povey, D. et al. (2011). The Kaldi speech recognition toolkit. *IEEE ASRU Workshop*.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Sell, G. et al. (2018). Diarization is hard: Some experiences and lessons learned for the JHU team in the inaugural DIHARD challenge. *Interspeech*.
- Snyder, D., Garcia-Romero, D., Sell, G., Povey, D. & Khudanpur, S. (2018). X-vectors: Robust DNN embeddings for speaker recognition. *ICASSP*.
- Straka, M. & Straková, J. (2017). Tokenizing, POS tagging, lemmatizing and parsing UD 2.0 with UDPipe. *CoNLL 2017 Shared Task*.
- Wijffels, J. (2023). *udpipe: Tokenization, Parts of Speech Tagging, Lemmatization and Dependency Parsing with the UDPipe NLP Toolkit*. Paquet R.
- Wang, H. et al. (2023). WeSpeaker: A research and production oriented speaker embedding learning toolkit. *ICASSP*.
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17, 395–416.
- Règlement (UE) 2016/679 (RGPD), art. 4(5), 4(14), 9 et considérant 26.
