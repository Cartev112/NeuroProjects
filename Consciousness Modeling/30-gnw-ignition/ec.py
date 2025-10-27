import numpy as np
from typing import Tuple


def _lag_matrix(y: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build lagged design for univariate AR.
    Returns (Y, X) where Y are responses and X has lagged terms.
    y shape: (T,)
    """
    T = y.shape[0]
    if lags < 1 or T <= lags:
        raise ValueError("Need lags>=1 and T>lags")
    Y = y[lags:]
    X = np.column_stack([y[lags - k - 1: T - k - 1] for k in range(lags)])
    return Y, X


def _lag_matrix_bivar(y: np.ndarray, x: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lagged design for bivariate VAR where y is target, x is predictor.
    Returns (Y, X) where X includes y and x lags.
    y, x shape: (T,)
    """
    T = y.shape[0]
    if lags < 1 or T <= lags:
        raise ValueError("Need lags>=1 and T>lags")
    Y = y[lags:]
    Xy = np.column_stack([y[lags - k - 1: T - k - 1] for k in range(lags)])
    Xx = np.column_stack([x[lags - k - 1: T - k - 1] for k in range(lags)])
    X = np.column_stack([Xy, Xx])
    return Y, X


def granger_pair(y: np.ndarray, x: np.ndarray, lags: int = 5) -> float:
    """Pairwise Granger strength from x->y as log variance ratio.
    Higher values => stronger directed influence.
    """
    y = np.asarray(y).ravel()
    x = np.asarray(x).ravel()
    Y_u, X_u = _lag_matrix(y, lags)
    beta_u, *_ = np.linalg.lstsq(X_u, Y_u, rcond=None)
    resid_u = Y_u - X_u @ beta_u
    var_u = float(np.var(resid_u) + 1e-12)

    Y_f, X_f = _lag_matrix_bivar(y, x, lags)
    beta_f, *_ = np.linalg.lstsq(X_f, Y_f, rcond=None)
    resid_f = Y_f - X_f @ beta_f
    var_f = float(np.var(resid_f) + 1e-12)

    return max(0.0, float(np.log(var_u / var_f)))


def granger_matrix(X: np.ndarray, lags: int = 5) -> np.ndarray:
    """Compute directed Granger strength matrix G[i,j] = i->j.
    X shape: (C, T)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be (C,T)")
    C, T = X.shape
    G = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            G[i, j] = granger_pair(X[j], X[i], lags=lags)
    return G


def broadcasting_index(G: np.ndarray) -> np.ndarray:
    """Broadcasting index per node as sum of outgoing strengths.
    G[i,j] is i->j.
    """
    G = np.asarray(G)
    return np.sum(G, axis=1)
