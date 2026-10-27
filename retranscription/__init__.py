"""Retranscription de réunions sans IA générative.

Chaîne de traitement :
    vidéos/audio -> pistes alignées -> meilleur micro -> Vosk (Kaldi)
    -> x-vecteurs -> classification -> registre inter-sessions
    -> fichiers de retranscription + fichiers de validation.
"""

__version__ = "0.2.0"
