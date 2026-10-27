"""Coefficients de banc de filtres (fbank) compatibles Kaldi, en numpy.

Réplique les réglages utilisés par WeSpeaker pour entraîner ses modèles
(torchaudio.compliance.kaldi.fbank) : trames de 25 ms toutes les 10 ms,
fenêtre de Hamming, préaccentuation 0,97, retrait de la composante
continue, 80 filtres sur l'échelle de Mel, logarithme de l'énergie, puis
soustraction de la moyenne sur la durée (normalisation cepstrale).
Le signal est attendu à l'échelle des entiers 16 bits.
"""

import numpy as np

TAUX = 16000
LONGUEUR, PAS, N_FFT = 400, 160, 512


def _mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def banc_mel(n_mels=80, fmin=20.0, fmax=TAUX / 2) -> np.ndarray:
    n_bins = N_FFT // 2
    largeur = TAUX / N_FFT
    mlo, mhi = _mel(fmin), _mel(fmax)
    delta = (mhi - mlo) / (n_mels + 1)
    mels = _mel(largeur * np.arange(n_bins))
    banc = np.zeros((n_mels, n_bins + 1))
    for i in range(n_mels):
        g, c, d = mlo + i * delta, mlo + (i + 1) * delta, mlo + (i + 2) * delta
        banc[i, :n_bins] = np.maximum(0.0, np.minimum((mels - g) / (c - g), (d - mels) / (d - c)))
    return banc


_BANC = banc_mel()
_FENETRE = np.hamming(LONGUEUR)


def fbank(x: np.ndarray, cmn: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = 1 + (len(x) - LONGUEUR) // PAS
    if n <= 0:
        return np.zeros((0, _BANC.shape[0]), np.float32)
    idx = np.arange(LONGUEUR)[None, :] + PAS * np.arange(n)[:, None]
    tr = x[idx]
    tr = tr - tr.mean(axis=1, keepdims=True)
    tr = np.concatenate([tr[:, :1] * 0.03, tr[:, 1:] - 0.97 * tr[:, :-1]], axis=1)
    tr *= _FENETRE
    puissance = np.abs(np.fft.rfft(tr, n=N_FFT)) ** 2
    f = np.log(np.maximum(puissance @ _BANC.T, np.finfo(np.float32).eps))
    if cmn:
        f -= f.mean(axis=0, keepdims=True)
    return f.astype(np.float32)
