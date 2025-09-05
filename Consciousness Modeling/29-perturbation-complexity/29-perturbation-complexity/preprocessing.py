import numpy as np
from typing import Optional, Tuple, Dict
from scipy.signal import butter, filtfilt, iirnotch


def zscore(X: np.ndarray, axis=-1, eps: float = 1e-8) -> np.ndarray:
    m = np.mean(X, axis=axis, keepdims=True)
    s = np.std(X, axis=axis, keepdims=True)
    return (X - m) / (s + eps)


def bandpass_filter(X: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    if l_freq is None and h_freq is None:
        return X
    nyq = 0.5 * sfreq
    if l_freq is None:
        Wn = h_freq / nyq
        b, a = butter(order, Wn, btype='low')
    elif h_freq is None:
        Wn = l_freq / nyq
        b, a = butter(order, Wn, btype='high')
    else:
        Wn = [l_freq / nyq, h_freq / nyq]
        b, a = butter(order, Wn, btype='band')
    Xf = filtfilt(b, a, X, axis=-1)
    return Xf


def notch_filter(X: np.ndarray, sfreq: float, freq: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=freq / (sfreq / 2.0), Q=q)
    return filtfilt(b, a, X, axis=-1)


def average_reference(X: np.ndarray) -> np.ndarray:
    # X shape: (..., C, T) or (C, T)
    if X.ndim == 2:
        ref = X.mean(axis=0, keepdims=True)
        return X - ref
    elif X.ndim == 3:
        ref = X.mean(axis=1, keepdims=True)
        return X - ref
    else:
        return X


BAND_DEFS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 80.0),
}


def compute_band_signals(X: np.ndarray, sfreq: float, bands: list) -> Dict[str, np.ndarray]:
    out = {}
    for b in bands:
        if b not in BAND_DEFS:
            continue
        l, h = BAND_DEFS[b]
        out[b] = bandpass_filter(X, sfreq, l, h)
    return out
