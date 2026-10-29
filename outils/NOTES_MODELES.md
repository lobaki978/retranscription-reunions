Modèles utilisés par l'application Retranscription. Cette page est créée par le workflow « Préparer les modèles ». **Ne pas la supprimer** : l'application y télécharge ses modèles au premier lancement.

| Fichier | Rôle | Source | Licence |
|---|---|---|---|
| wav2vec2-fr.zip | reconnaissance de la parole (wav2vec 2.0 + CTC), converti en ONNX | [bofenghuang/asr-wav2vec2-ctc-french](https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french), affiné à partir de [LeBenchmark/wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) | Apache 2.0 |
| wav2vec2-fr-int8.zip | même modèle, poids des couches linéaires en 8 bits (plus léger, plus rapide) | idem | Apache 2.0 |
| lm-fr-5gram.zip | modèle de langue 5-grammes (KenLM) pour le décodage par faisceau | [language_model de bofenghuang/asr-wav2vec2-ctc-french](https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french/tree/main/language_model) | licence du dépôt source (Apache 2.0) |
| segmentation-3.0.onnx | segmentation des locuteurs (qui parle dans chaque fenêtre de 10 s) | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), conversion ONNX de [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | MIT |
| wespeaker-resnet34.onnx | empreintes vocales (256 dimensions) | [WeSpeaker](https://github.com/wenet-e2e/wespeaker), ResNet34 entraîné sur VoxCeleb, conversion ONNX de sherpa-onnx | CC BY 4.0 (licence de VoxCeleb) |

Références :
- Baevski, A., Zhou, H., Mohamed, A., Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS.
- Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. (2006). Connectionist Temporal Classification. ICML.
- Evain, S. et al. (2021). LeBenchmark: A Reproducible Framework for Assessing Self-Supervised Representation Learning from Speech. Interspeech.
- Plaquet, A., Bredin, H. (2023). Powerset multi-class cross entropy loss for neural speaker diarization. Interspeech.
- Wang, H. et al. (2023). WeSpeaker: A research and production oriented speaker embedding learning toolkit. ICASSP.
- Heafield, K. (2011). KenLM: Faster and smaller language model queries. Workshop on Statistical Machine Translation.
- Hannun, A. Y. et al. (2014). First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs. arXiv:1408.2873.
- Nagrani, A., Chung, J. S., Zisserman, A. (2017). VoxCeleb: a large-scale speaker identification dataset. Interspeech.
