import numpy as np
from typing import Dict
from scipy.linalg import eigvals, solve_discrete_lyapunov


def ensure_stable(A: np.ndarray, target_radius: float = 0.95) -> np.ndarray:
    """Scale adjacency A so that spectral radius <= target_radius.
    Works for discrete-time linear system x_{t+1} = A x_t + B u_t.
    """
    A = np.asarray(A, dtype=float)
    vals = eigvals(A)
    rho = np.max(np.abs(vals)) if vals.size > 0 else 0.0
    if rho <= 0:
        return A.copy()
    scale = min(1.0, float(target_radius) / float(rho))
    return A * scale


def compute_controllability(A: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute average and modal controllability for each node.

    - Average controllability (per node i): trace(W_i), W_i solves X = A X A^T + e_i e_i^T
    - Modal controllability (per node i): sum_k (1 - |lambda_k|^2) v_{k,i}^2
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    avg = np.zeros(n, dtype=float)
    # Average controllability via Lyapunov for each canonical input
    for i in range(n):
        B = np.zeros((n, 1), dtype=float)
        B[i, 0] = 1.0
        # Solve X = A X A^T + B B^T
        X = solve_discrete_lyapunov(A, (B @ B.T))
        avg[i] = float(np.trace(X))
    # Modal controllability
    w, V = np.linalg.eig(A)
    V = np.asarray(V)
    one_minus = 1.0 - np.minimum(1.0, np.abs(w) ** 2)
    modal = (np.abs(V) ** 2 @ one_minus.real)
    return {"average": avg, "modal": modal.real}


def simulate_linear_impulses(A: np.ndarray, stim_node: int, T: int = 1000) -> np.ndarray:
    """Simulate discrete-time linear system with impulse at stim_node at t=0.

    Returns a 1D time series given by the network energy (L2 norm) over time,
    suitable for complexity metrics.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.zeros(n, dtype=float)
    y = np.zeros(T, dtype=float)
    # impulse at t=0 affects x_1
    u = np.zeros(n, dtype=float)
    u[stim_node] = 1.0
    for t in range(T):
        if t == 0:
            x = x + u
        else:
            x = A @ x
        y[t] = float(np.linalg.norm(x))
    return y
