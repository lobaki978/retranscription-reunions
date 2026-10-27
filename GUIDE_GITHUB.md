# Fabriquer l'application pour Windows et Mac avec GitHub

GitHub fabrique les deux versions de l'application à votre place, sur ses
propres ordinateurs. Vous n'avez besoin ni de Mac, ni de Python. Comptez
30 minutes la première fois.

Le dépôt ne contient que le code : aucune vidéo, aucune transcription. Il
peut donc être **public**, ce qui rend la fabrication gratuite.

Les menus de GitHub sont en anglais ; leurs noms sont indiqués en **gras**.

---

## 1. Créer le compte et le dépôt (une seule fois)

1. Allez sur github.com, cliquez sur **Sign up** et créez un compte gratuit.
2. En haut à droite, cliquez sur **+**, puis **New repository**.
3. **Repository name** : `retranscription-reunions`.
4. Cochez **Public**. Ne cochez rien d'autre.
5. Cliquez sur **Create repository**.

## 2. Déposer les fichiers

1. Décompressez `retranscription_reunions_github.zip` sur votre ordinateur.
2. Sur la page du dépôt vide, cliquez sur le lien **uploading an existing file**.
3. Ouvrez le dossier décompressé, sélectionnez **tout son contenu** (Ctrl+A
   sous Windows, Cmd+A sur Mac) et glissez-le dans la page GitHub.
   Le dossier `.github` doit en faire partie : c'est lui qui contient la
   recette de fabrication. (Sur Mac, les dossiers qui commencent par un point
   sont cachés : Cmd+Maj+point pour les afficher dans le Finder.)
4. Attendez la fin de l'envoi, puis cliquez en bas sur **Commit changes**.
5. Vérifiez que la liste des fichiers du dépôt contient bien `.github`.

Si `.github` manque : **Add file** > **Create new file**, tapez le nom
`.github/workflows/fabriquer.yml`, collez le contenu du fichier du même nom,
puis **Commit changes**.

## 3. Suivre la fabrication

1. Cliquez sur l'onglet **Actions** du dépôt. Une exécution « Fabriquer
   l'application » a démarré (rond jaune). Elle dure 15 à 25 minutes.
2. **Coche verte** : réussi. Les applications ont été fabriquées, puis
   vérifiées automatiquement sur Windows et sur Mac.
3. **Croix rouge** : échec. Cliquez sur l'exécution, puis sur la ligne en
   rouge (Windows ou Mac), ouvrez l'étape en rouge et copiez son texte.
   Envoyez-le-moi : je corrigerai.

## 3 bis. Préparer les modèles (une seule fois)

L'application télécharge ses modèles au premier lancement. Le modèle de
reconnaissance wav2vec 2.0 doit d'abord être converti et publié sur votre
dépôt. GitHub le fait pour vous :

1. Onglet **Actions**. Dans la colonne de gauche, cliquez sur
   **Préparer les modèles**.
2. À droite, cliquez sur **Run workflow**, puis sur le bouton vert
   **Run workflow** (ne changez pas le nom du modèle).
3. Attendez la coche verte (20 à 40 minutes).
4. Vérifiez : page d'accueil du dépôt > **Releases**. La page
   **Modèles (ne pas supprimer)** contient notamment `wav2vec2-fr.zip`
   (environ 1,2 Go) et `segmentation-3.0.onnx`.

Ne supprimez jamais cette page : les applications y téléchargent leurs
modèles. Il n'est pas nécessaire de recommencer à chaque nouvelle version.

## 4. Récupérer les deux applications

### Pour un essai rapide

Cliquez sur l'exécution réussie. En bas de la page, rubrique **Artifacts** :
`Retranscription-Windows` et `Retranscription-Mac`. GitHub ajoute une
enveloppe : chaque téléchargement est un .zip qui contient le .zip à partager.
Décompressez une fois.

### Pour partager (recommandé)

1. Sur la page d'accueil du dépôt, à droite, cliquez sur **Releases**, puis
   **Create a new release**.
2. **Choose a tag** : tapez `v0.3.0`, puis **Create new tag**.
3. **Release title** : `Retranscription 0.3.0`. Cliquez sur **Publish release**.
4. La fabrication se relance (15 à 25 minutes). Les deux fichiers
   `Retranscription-Windows.zip` et `Retranscription-Mac.zip` apparaissent
   ensuite sur la page de la version, rubrique **Assets**.

Vous pouvez donner directement le lien de cette page : on peut télécharger
sans compte GitHub. Vous pouvez aussi télécharger les deux fichiers et les
mettre sur Dropbox.

Joignez le mode d'emploi : `GUIDE_UTILISATEUR.html` est déjà inclus dans
chaque .zip.

## 5. Mettre à jour plus tard

1. **Add file** > **Upload files** : glissez les fichiers modifiés (ils
   remplacent les anciens), puis **Commit changes**.
2. Créez une nouvelle version comme au point 4, avec le numéro suivant
   (`v0.3.1`, `v0.4.0`…).

---

## Ce qu'il faut savoir

- **Mac** : la version fabriquée fonctionne sur les Mac à puce Apple (M1 à
  M4), sous macOS 14 ou plus récent.
- **Applications non signées** : Windows et macOS affichent un avertissement
  au premier lancement. `GUIDE_UTILISATEUR.html` explique quoi faire.
- **Taille** : environ 150 à 250 Mo par fichier. Les modèles de
  reconnaissance (1,2 Go) se téléchargent au premier lancement, sur
  l'ordinateur de chaque utilisateur.
- **Version « latest »** : la page des modèles n'est jamais marquée comme
  dernière version ; le lien « Latest release » mène bien à l'application.
- **Coût** : gratuit pour un dépôt public.
