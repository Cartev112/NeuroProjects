import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as e:  # pragma: no cover
    GaussianHMM = None


def detect_ignition_hmm(gfp: np.ndarray, n_components: int = 2, min_samples: int = 10, smooth: int = 5):
    """Detect ignition episodes using a Gaussian HMM on GFP.

    For each epoch, fit a 1D GaussianHMM with n_components states to the GFP
    time series (optionally smoothed). Interpret the state with the highest
    mean as the 'ignition' (high-activation) state, then extract contiguous
    segments with length >= min_samples.

    gfp: (T,) or (N,T)
    returns: list (length N) of lists of (start, end)
    """
    if GaussianHMM is None:
        raise ImportError("hmmlearn missing. Please install hmmlearn to use --ignition_method hmm")

    gfp = np.asarray(gfp)
    if gfp.ndim == 1:
        gfp = gfp[None, :]
    N, T = gfp.shape
    out = []
    for n in range(N):
        x = gfp[n]
        if smooth and smooth > 1:
            k = smooth
            xs = np.convolve(x, np.ones(k) / k, mode='same')
        else:
            xs = x
        Xobs = xs.reshape(-1, 1)
        # Fit HMM
        hmm = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=200, tol=1e-3, random_state=0)
        hmm.fit(Xobs)
        z = hmm.predict(Xobs)
        means = hmm.means_.ravel()
        high_state = int(np.argmax(means))
        mask = (z == high_state)
        # extract episodes
        episodes = []
        i = 0
        while i < T:
            if mask[i]:
                j = i
                while j < T and mask[j]:
                    j += 1
                if (j - i) >= min_samples:
                    episodes.append((i, j))
                i = j
            else:
                i += 1
        out.append(episodes)
    return out
