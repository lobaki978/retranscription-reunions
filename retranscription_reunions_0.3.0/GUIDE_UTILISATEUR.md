# Retranscription de réunions — guide d'utilisation

L'application transcrit la parole de vidéos (ou de fichiers son) et indique
qui parle et quand. Les personnes sont numérotées (Speaker 01, Speaker 02…),
sans nom. Tout se passe sur votre ordinateur : aucun fichier n'est envoyé
sur Internet.

Aucune connaissance en programmation n'est nécessaire.

---

## 1. Installer

### Windows

1. Faites un clic droit sur `Retranscription-Windows.zip`, puis **Extraire tout**.
   Placez le dossier dans **Documents**, par exemple.
2. Ouvrez le dossier `Retranscription` et double-cliquez sur **Retranscription.exe**.
3. Si Windows affiche « Windows a protégé votre ordinateur » : cliquez sur
   **Informations complémentaires**, puis **Exécuter quand même**.
   Ce message apparaît parce que l'application n'est pas signée par un éditeur
   commercial. Il ne revient pas ensuite.

Astuce : clic droit sur `Retranscription.exe` > **Envoyer vers** > **Bureau
(créer un raccourci)**.

### Mac (puce Apple M1 à M4, macOS 14 ou plus récent)

1. Double-cliquez sur `Retranscription-Mac.zip`, puis glissez **Retranscription**
   dans le dossier **Applications**.
2. Premier lancement : double-cliquez sur l'application. macOS indique qu'il ne
   peut pas vérifier le développeur. Cliquez sur **OK**.
3. Ouvrez **Réglages Système** > **Confidentialité et sécurité**. En bas de la
   page, cliquez sur **Ouvrir quand même**, puis confirmez.
   (Sous macOS 14, un clic droit sur l'application > **Ouvrir** suffit.)
4. Si macOS demande l'accès aux dossiers Documents ou Bureau, acceptez :
   l'application doit lire vos vidéos.

Si macOS indique que l'application est « endommagée » : ouvrez le Terminal
et collez cette ligne, puis appuyez sur Entrée :

```
xattr -dr com.apple.quarantine /Applications/Retranscription.app
```

---

## 2. Avant de commencer

- **Ordinateur** : au moins 8 Go de mémoire vive. Sinon, cochez « Ordinateur
  peu puissant » (modèle allégé, un peu moins précis).
- **Internet** : nécessaire une seule fois, au premier lancement, pour
  télécharger les modèles de reconnaissance (environ 1,2 Go ; 400 Mo avec
  « Ordinateur peu puissant »).
- **Une session = un dossier** : mettez toutes les vidéos d'une même réunion
  dans un dossier. Plusieurs caméras ou micros : mettez tout ensemble,
  l'application les synchronise. Les vidéos ne sont jamais modifiées.

---

## 3. Transcrire (onglet 1)

1. **Dossier des vidéos** : cliquez sur « Dossier… » et sélectionnez le dossier de la session.
   Cette fenêtre n'affiche que les dossiers, pas les vidéos : c'est normal. Ouvrez le dossier
   qui contient les vidéos, puis cliquez sur « Sélectionner un dossier ». Vous pouvez aussi
   cliquer sur « Vidéo… » et choisir directement une des vidéos. Le journal, en bas, indique
   les fichiers trouvés.
2. **Vidéo de référence** : celle dont le début sert de temps 0. Prenez celle
   que vous utiliserez pour regarder la réunion.
3. **Nom de la session** : court, sans espace (S01, S02…).
4. **Date** de la réunion, au format AAAA-MM-JJ.
5. **Personnes qui parlent** : participants et animateurs. Indiquez le nombre
   exact si vous le connaissez : c'est le réglage qui améliore le plus le résultat.
6. **Reconnaissance** : gardez « wav2vec 2.0 » (le plus juste). « Vosk » est
   plus rapide mais fait plus d'erreurs.
7. **Dossiers de travail** : vérifiez où seront rangés les résultats. Évitez
   OneDrive ou Dropbox (gros fichiers son).
8. Cliquez sur **Transcrire**.

Durée : une à trois heures pour une réunion de 6 h, selon l'ordinateur. Laissez l'ordinateur
branché et la fenêtre ouverte ; l'application empêche la mise en veille. Le
journal, en bas de la fenêtre, montre l'avancement.

Conseil : pour la première fois, cochez **Essai** : seules les 10 premières
minutes sont traitées. Vous vérifiez ainsi que tout fonctionne et vous voyez
la qualité du résultat avant de lancer une réunion entière.

---

## 4. Les résultats

Dans le dossier des résultats, un dossier par session :

| Fichier | Contenu |
|---|---|
| `S01_transcription.txt` | Le texte, avec l'heure et le locuteur de chaque prise de parole |
| `S01_transcription.csv` | Le même contenu en tableau (Excel, R…) |
| `S01_transcription.eaf` | Pour le logiciel ELAN (facultatif) |
| `S01_locuteurs.csv` | Une ligne par personne : temps de parole, nombre de tours… La colonne « sexe » est à remplir à la main |
| `validation/` | Les contrôles de qualité (voir ci-dessous) |

Le texte est **brut** : sans ponctuation ni majuscules, et non relu. La
reconnaissance écrit les mots lettre par lettre, comme elle les entend : un
mot mal compris peut être mal orthographié, et les noms propres sont écrits
comme ils se prononcent.

---

## 5. Vérifier et mesurer la fiabilité (onglet 2)

1. **Ouvrir les résultats**, puis lire `validation/S01_synthese.txt`. Il
   indique ce qui est douteux.
2. **Une personne a un mauvais numéro ?** Ouvrez
   `validation/S01_correspondances.csv` (Excel ou LibreOffice), corrigez la
   colonne `speaker`, enregistrez au même format, puis cliquez sur
   **Appliquer les numéros corrigés**.
3. **Mesurer la fiabilité** (facultatif, recommandé pour un travail publié) :
   corrigez à la main les extraits du dossier `validation/extraits` dans ELAN
   (logiciel gratuit, en suivant `CONSIGNES.txt`), puis cliquez sur
   **Mesurer la fiabilité**. Vous obtenez le taux d'erreur sur les mots (WER)
   et sur les locuteurs (DER).
4. **Après 2 ou 3 sessions vérifiées**, cliquez sur **Calibrer le seuil** :
   l'application reconnaîtra mieux les mêmes personnes d'une session à l'autre.

---

## 6. En cas de problème

- Le détail de chaque traitement est enregistré dans un fichier journal :
  - Windows : `C:\Users\<vous>\AppData\Local\Retranscription\journal.txt`
  - Mac : `~/Library/Application Support/Retranscription/journal.txt`
  Envoyez ce fichier à la personne qui vous a transmis l'application.
- L'application se ferme pendant la transcription : la mémoire est sans doute
  insuffisante. Cochez « Ordinateur peu puissant ».
- **Deux personnes ont le même numéro** dans une session : indiquez le nombre
  exact de personnes qui parlent et relancez.
- **Mise à jour depuis une version 0.2** : les empreintes vocales ont changé.
  Les personnes reçoivent de nouveaux numéros ; pour repartir de Speaker 01,
  choisissez un nouveau dossier « Empreintes vocales » (ou videz l'ancien).
- Relancer une session déjà faite ne perd ni les extraits corrigés ni la colonne « sexe ».

---

## 7. Données personnelles

- Les résultats sont **pseudonymisés**, pas anonymes : la voix reste dans les
  vidéos et dans le fichier son produit.
- Le dossier **Empreintes vocales** contient des données biométriques :
  ne le partagez pas, rangez-le sur un support protégé, effacez-le à la fin
  du projet.
- Le traitement doit être prévu dans la déclaration RGPD de votre projet
  (consentement des participants). Voyez avec le délégué à la protection des
  données de votre établissement.
