import numpy as np


def global_field_power(X: np.ndarray) -> np.ndarray:
    """Compute Global Field Power (RMS across channels) per sample.
    X shape: (C, T) or (N, C, T); returns (T,) or (N, T) respectively.
    """
    X = np.asarray(X)
    if X.ndim == 2:
        return np.sqrt(np.mean(X ** 2, axis=0))
    elif X.ndim == 3:
        return np.sqrt(np.mean(X ** 2, axis=1))
    else:
        raise ValueError('X must have shape (C,T) or (N,C,T)')


def detect_ignition(gfp: np.ndarray, z_thresh: float = 2.5, min_samples: int = 10, smooth: int = 5):
    """Detect ignition episodes using z-scored GFP threshold sustained for min_samples.
    Accepts gfp with shape (T,) or (N,T). Returns list of (start, end) for each epoch.
    """
    gfp = np.asarray(gfp)
    if gfp.ndim == 1:
        gfp = gfp[None, :]
    N, T = gfp.shape
    out = []
    for n in range(N):
        x = gfp[n]
        if smooth and smooth > 1:
            k = smooth
            x_s = np.convolve(x, np.ones(k) / k, mode='same')
        else:
            x_s = x
        mu = np.mean(x_s)
        sd = np.std(x_s) + 1e-8
        z = (x_s - mu) / sd
        above = z >= z_thresh
        episodes = []
        i = 0
        while i < T:
            if above[i]:
                j = i
                while j < T and above[j]:
                    j += 1
                if (j - i) >= min_samples:
                    episodes.append((i, j))
                i = j
            else:
                i += 1
        out.append(episodes)
    return out  # list length N, each a list of (start,end)
