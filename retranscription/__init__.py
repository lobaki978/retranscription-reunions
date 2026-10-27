"""Retranscription de réunions sans IA générative.

Chaîne de traitement :
    vidéos/audio -> pistes alignées -> meilleur micro
    -> wav2vec 2.0 + décodage CTC (ou Vosk/Kaldi)
    -> segmentation des locuteurs (pyannote 3.0) + empreintes WeSpeaker
    -> classification -> registre inter-sessions
    -> fichiers de retranscription + fichiers de validation.
"""

__version__ = "0.3.0"
