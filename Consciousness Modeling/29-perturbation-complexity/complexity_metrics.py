import numpy as np
from math import factorial


def _to_1d(arr):
    """Return a 1D float array view of input.

    Accepts any array-like, flattens with ravel (no copy when possible),
    and casts to float for downstream numeric stability.
    """
    x = np.asarray(arr).ravel()
    return x.astype(float)


def _binarize(ts, method="median", q: float = 0.5):
    """Binarize a time series by thresholding.

    Parameters
    - method: 'median', 'mean', or 'quantile'
    - q: quantile in (0,1) when method='quantile'
    """
    x = _to_1d(ts)
    if method == "median":
        thr = np.median(x)
    elif method == "mean":
        thr = np.mean(x)
    elif method == "quantile":
        thr = np.quantile(x, q)
    else:
        raise ValueError("binarize method must be 'median' or 'mean'")
    return (x > thr).astype(int)


def lz_complexity(ts, binarize_method="median", normalize=True, q: float = 0.5):
    """Binary Lempel–Ziv complexity with optional normalization.

    Binarizes the signal then parses it into novel substrings, counting
    the number of additions to the dictionary. Normalization uses
    (c * log2(n)) / n.
    """
    b = _binarize(ts, method=binarize_method, q=q)
    s = ''.join('1' if v == 1 else '0' for v in b)
    n = len(s)
    if n == 0:
        return np.nan
    i = 0
    c = 0
    while i < n:
        l = 1
        while i + l <= n and s[i:i + l] in _lz_dict_cache:
            l += 1
        _lz_dict_cache.add(s[i:i + l])
        c += 1
        i += l
    _lz_dict_cache.clear()
    if not normalize:
        return float(c)
    if n <= 1:
        return 0.0
    return (c * np.log2(n)) / n


_lz_dict_cache = set()


def permutation_entropy(ts, order=3, delay=1, normalize=True, tie_strategy: str = "noise", tie_eps: float = 1e-8):
    """Compute permutation entropy with simple tie handling.

    When ties are present, ordinal patterns can be ill-defined. With
    tie_strategy='noise', we add tiny iid noise to break ties in a
    stable way; with 'ordinal', we fall back to argsort's stable behavior.
    """
    x = _to_1d(ts)
    n = x.size
    if n < (order - 1) * delay + 1 or order < 2:
        return np.nan
    emb_len = n - (order - 1) * delay
    patterns = {}
    for i in range(emb_len):
        window = x[i:i + order * delay:delay]
        if tie_strategy == "noise":
            window = window + tie_eps * np.random.randn(*window.shape)
            ranks = tuple(np.argsort(window, kind='mergesort'))
        else:
            ranks = tuple(np.argsort(window, kind='mergesort'))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    pe = -np.sum(p * np.log(p + 1e-12))
    if normalize:
        pe = pe / np.log(factorial(order))
    return float(pe)


def sample_entropy(ts, m=2, r=0.2):
    """Sample entropy with Chebyshev tolerance r * std.

    Returns +inf when there are zero matches at template length m.
    """
    x = _to_1d(ts)
    n = x.size
    if n <= m + 1:
        return np.nan
    std = np.std(x)
    tol = r * std
    def _count_matches(dim):
        count = 0
        N = n - dim
        for i in range(N):
            xi = x[i:i + dim]
            for j in range(i + 1, N):
                if np.max(np.abs(xi - x[j:j + dim])) <= tol:
                    count += 1
        return count
    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return np.inf
    return -np.log((A + 1e-12) / B)


def multiscale_entropy(ts, scales=5, m=2, r=0.2):
    """Multiscale entropy via coarse-graining + sample entropy.

    For each integer scale s, average non-overlapping windows of size s
    and compute sample entropy on the coarse-grained series.
    """
    x = _to_1d(ts)
    n = x.size
    mses = []
    for s in range(1, int(scales) + 1):
        if n < s * (m + 2):
            mses.append(np.nan)
            continue
        cg = x[: (n // s) * s].reshape(-1, s).mean(axis=1)
        mses.append(sample_entropy(cg, m=m, r=r))
    return np.array(mses, dtype=float)
