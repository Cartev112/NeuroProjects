"""
Place–Grid Cells Demo: Grid Codes → Sparse Place Cells + Decoding

Simulates a 2D trajectory in a square arena, generates a population of grid
cell responses (multi-scale, multi-orientation interference). Then learns a
sparse set of "place cells" via dictionary learning on the grid codes and
visualizes their place fields. Also trains a linear decoder from grid codes to
position and evaluates decoding accuracy.

Run example:
  python PlaceGridDemo/placegrid_demo.py --steps 4000 --grid-cells 240 --place-cells 96 --save placegrid.png --no-show
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrajConfig:
    arena_size: float = 1.0  # units
    dt_s: float = 0.05
    steps: int = 4000
    speed_mean: float = 0.25  # units/s
    speed_std: float = 0.05
    turn_std: float = 0.6  # rad per step
    seed: int = 0


@dataclass
class GridConfig:
    num_cells: int = 240
    modules: Tuple[float, ...] = (0.30, 0.45, 0.68)  # grid scales as fraction of arena
    gain: float = 1.0
    bias: float = 0.0
    nonlinearity: str = "relu"  # or "exp"
    noise_std: float = 0.02


@dataclass
class PlaceConfig:
    num_place: int = 96
    lam: float = 0.1
    steps_per_iter: int = 25
    iters: int = 300


def simulate_trajectory(cfg: TrajConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    p = np.array([cfg.arena_size * 0.5, cfg.arena_size * 0.5], dtype=float)
    heading = rng.uniform(0, 2 * np.pi)
    pos = np.empty((cfg.steps, 2), dtype=float)
    speed = np.clip(rng.normal(cfg.speed_mean, cfg.speed_std, size=cfg.steps), 0.05, None)
    for t in range(cfg.steps):
        heading += rng.normal(0.0, cfg.turn_std)
        v = speed[t] * np.array([math.cos(heading), math.sin(heading)]) * cfg.dt_s
        p = p + v
        # Reflective boundaries
        for k in (0, 1):
            if p[k] < 0.0:
                p[k] = -p[k]
                heading = np.pi - heading if k == 0 else -heading
            if p[k] > cfg.arena_size:
                p[k] = 2 * cfg.arena_size - p[k]
                heading = np.pi - heading if k == 0 else -heading
        pos[t] = p
    t_s = np.arange(cfg.steps, dtype=float) * cfg.dt_s
    return pos, t_s


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def simulate_grid_cells(pos: np.ndarray, cfg: GridConfig, arena_size: float, seed: int = 0) -> Tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    T = pos.shape[0]
    G = cfg.num_cells
    modules = np.asarray(cfg.modules, dtype=float)
    # Pre-generate cell parameters
    cell_module = rng.integers(0, len(modules), size=G)
    base_theta = rng.uniform(0.0, np.pi, size=G)  # orientation base
    phases = rng.uniform(0.0, 2 * np.pi, size=(G, 3))

    # Frequencies from scales (cycles per unit)
    # scale s (units) ⇒ wavelength = s*arena ⇒ freq cycles/unit = 1/(s*arena)
    freqs = 1.0 / (modules * arena_size)

    R60 = _rot(np.deg2rad(60.0))
    R120 = _rot(np.deg2rad(120.0))

    X = np.empty((T, G), dtype=float)
    for g in range(G):
        scale_idx = cell_module[g]
        f = freqs[scale_idx] * 2 * np.pi  # rad/unit
        u0 = np.array([math.cos(base_theta[g]), math.sin(base_theta[g])])
        u1 = R60 @ u0
        u2 = R120 @ u0
        # Compute interference pattern along trajectory
        proj0 = pos @ u0
        proj1 = pos @ u1
        proj2 = pos @ u2
        s = (
            np.cos(f * proj0 + phases[g, 0])
            + np.cos(f * proj1 + phases[g, 1])
            + np.cos(f * proj2 + phases[g, 2])
        ) / 3.0
        if cfg.nonlinearity == "exp":
            r = np.exp(cfg.gain * s + cfg.bias)
        else:
            r = np.maximum(0.0, cfg.gain * s + cfg.bias)
        if cfg.noise_std > 0:
            r += rng.normal(0.0, cfg.noise_std, size=T)
        X[:, g] = r
    meta = {
        "cell_module": cell_module,
        "base_theta": base_theta,
        "phases": phases,
        "freqs": freqs,
    }
    return X, meta


# --------------------------- Sparse coding (ISTA) -------------------------- #

def ista(X: np.ndarray, D: np.ndarray, lam: float, steps: int, step_size: Optional[float] = None) -> np.ndarray:
    N, Din = X.shape
    K, Din2 = D.shape
    assert Din == Din2
    C = np.zeros((N, K), dtype=np.float32)
    Dt = D.T
    L = np.linalg.norm(D @ Dt, 2)
    t = (1.0 / L) if (step_size is None or step_size <= 0) else float(step_size)
    for _ in range(steps):
        grad = (C @ D - X) @ Dt
        C = C - t * grad
        C = np.sign(C) * np.maximum(0.0, np.abs(C) - t * lam)
    return C


def update_dictionary(X: np.ndarray, C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Ct = C.T
    G = Ct @ C
    P = Ct @ X
    A = G + eps * np.eye(G.shape[0], dtype=G.dtype)
    D = np.linalg.solve(A, P).astype(np.float32)
    norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-8
    D = D / norms
    return D


def train_sparse_place(X: np.ndarray, K: int, lam: float, steps_per_iter: int, iters: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    rng = np.random.default_rng(seed)
    N, Din = X.shape
    D = rng.normal(0.0, 0.01, size=(K, Din)).astype(np.float32)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    rec_trace: list[float] = []
    spars_trace: list[float] = []
    C = np.zeros((N, K), dtype=np.float32)
    for _ in range(iters):
        C = ista(X, D, lam=lam, steps=steps_per_iter)
        D = update_dictionary(X, C)
        recon = C @ D
        rec_err = float(np.mean((X - recon) ** 2))
        spars = float(np.mean(np.abs(C) > 1e-6))
        rec_trace.append(rec_err)
        spars_trace.append(spars)
    return D, C, rec_trace, spars_trace


# ---------------------------- Mapping & Decoding --------------------------- #

def occupancy_map(pos: np.ndarray, arena_size: float, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, edges_x = np.histogram(pos[:, 0], bins=bins, range=(0, arena_size))
    W, edges_y = np.histogram(pos[:, 1], bins=bins, range=(0, arena_size))
    # 2D occupancy
    occ, _, _ = np.histogram2d(pos[:, 1], pos[:, 0], bins=bins, range=[[0, arena_size], [0, arena_size]])
    return occ + 1e-6, edges_x, edges_y


def grid_field_map(meta: dict, cell_idx: int, arena_size: float, bins: int) -> np.ndarray:
    base_theta = meta["base_theta"][cell_idx]
    phases = meta["phases"][cell_idx]
    freqs = meta["freqs"][meta["cell_module"][cell_idx]]
    f = freqs * 2 * np.pi
    u0 = np.array([math.cos(base_theta), math.sin(base_theta)])
    u1 = _rot(np.deg2rad(60)) @ u0
    u2 = _rot(np.deg2rad(120)) @ u0
    xs = np.linspace(0, arena_size, bins)
    ys = np.linspace(0, arena_size, bins)
    grid = np.zeros((bins, bins), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = np.array([x, y])
            s = (
                math.cos(f * (u0 @ p) + phases[0])
                + math.cos(f * (u1 @ p) + phases[1])
                + math.cos(f * (u2 @ p) + phases[2])
            ) / 3.0
            grid[iy, ix] = max(0.0, s)
    grid /= (grid.max() + 1e-8)
    return grid


def place_field_maps(C: np.ndarray, pos: np.ndarray, arena_size: float, bins: int, k_indices: np.ndarray) -> list[np.ndarray]:
    # Accumulate activation per bin, normalized by occupancy
    occ, _, _ = np.histogram2d(pos[:, 1], pos[:, 0], bins=bins, range=[[0, arena_size], [0, arena_size]])
    occ = occ + 1e-6
    fields: list[np.ndarray] = []
    for k in k_indices:
        act = C[:, k]
        acc, _, _ = np.histogram2d(pos[:, 1], pos[:, 0], bins=bins, range=[[0, arena_size], [0, arena_size]], weights=act)
        field = acc / occ
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        fields.append(field)
    return fields


def train_linear_decoder(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    # Add bias term
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1], dtype=X.dtype)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)
    return W  # shape (G+1, 2)


def decode_positions(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    return Xb @ W


# --------------------------------- Plotting -------------------------------- #

def plot_results(
    pos: np.ndarray,
    t_s: np.ndarray,
    X: np.ndarray,
    meta: dict,
    Dp: np.ndarray,
    Cp: np.ndarray,
    rec_trace: list[float],
    spars_trace: list[float],
    pred: np.ndarray,
    arena_size: float,
    save_path: Optional[str],
    show: bool,
) -> None:
    bins = 30
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    # Path and decoded path
    ax_path = fig.add_subplot(gs[0, 0])
    ax_path.plot(pos[:, 0], pos[:, 1], color="#1f77b4", linewidth=1.0, label="true")
    ax_path.plot(pred[:, 0], pred[:, 1], color="#ff7f0e", linewidth=0.8, alpha=0.7, label="decoded")
    ax_path.set_xlim(0, arena_size)
    ax_path.set_ylim(0, arena_size)
    ax_path.set_aspect("equal")
    ax_path.set_title("Trajectory (true vs decoded)")
    ax_path.legend(fontsize=8)

    # Grid fields: sample 3
    ax_g1 = fig.add_subplot(gs[0, 1])
    ax_g2 = fig.add_subplot(gs[0, 2])
    ax_g3 = fig.add_subplot(gs[0, 3])
    for ax, idx in zip([ax_g1, ax_g2, ax_g3], np.random.choice(X.shape[1], size=3, replace=False)):
        gf = grid_field_map(meta, int(idx), arena_size, bins)
        ax.imshow(gf, origin="lower", cmap="viridis", extent=[0, arena_size, 0, arena_size])
        ax.set_title(f"Grid cell {idx}")
        ax.set_xticks([]); ax.set_yticks([])

    # Place fields: top 6 by sparsity energy
    energies = (Cp ** 2).sum(axis=0)
    top_k = np.argsort(-energies)[:6]
    pf_list = place_field_maps(Cp, pos, arena_size, bins, top_k)
    for i, pf in enumerate(pf_list):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.imshow(pf, origin="lower", cmap="magma", extent=[0, arena_size, 0, arena_size])
        ax.set_title(f"Place {top_k[i]}")
        ax.set_xticks([]); ax.set_yticks([])

    # Learning curves
    ax_l = fig.add_subplot(gs[2, 3])
    ax_l.plot(rec_trace, label="recon MSE")
    ax_l.plot(spars_trace, label="sparsity frac")
    ax_l.set_title("Sparse coding (place) learning")
    ax_l.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------ CLI --------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Place–Grid Cells: grid codes → sparse place cells + decoding")
    p.add_argument("--steps", type=int, default=4000, help="Trajectory steps")
    p.add_argument("--dt", type=float, default=0.05, help="Step size (s)")
    p.add_argument("--arena", type=float, default=1.0, help="Arena size (square)")
    p.add_argument("--grid-cells", type=int, default=240, help="# of grid cells")
    p.add_argument("--place-cells", type=int, default=96, help="# of place units")
    p.add_argument("--lam", type=float, default=0.1, help="L1 sparsity (place coding)")
    p.add_argument("--iters", type=int, default=300, help="Place coding iters")
    p.add_argument("--steps-per-iter", type=int, default=25, help="ISTA steps per iter")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save figure")
    p.add_argument("--no-show", action="store_true", help="Do not display plot window")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    traj_cfg = TrajConfig(arena_size=args.arena, dt_s=args.dt, steps=args.steps, seed=args.seed)
    grid_cfg = GridConfig(num_cells=args.grid_cells)
    place_cfg = PlaceConfig(num_place=args.place_cells, lam=args.lam, steps_per_iter=args.steps_per_iter, iters=args.iters)

    pos, t_s = simulate_trajectory(traj_cfg)
    X, meta = simulate_grid_cells(pos, grid_cfg, traj_cfg.arena_size, seed=args.seed)
    # Standardize grid codes per cell
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    # Sparse coding to get place units
    Dp, Cp, rec_trace, spars_trace = train_sparse_place(Xz, K=place_cfg.num_place, lam=place_cfg.lam, steps_per_iter=place_cfg.steps_per_iter, iters=place_cfg.iters, seed=args.seed)

    # Linear decoding train/test split
    T = X.shape[0]
    idx = np.arange(T)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    split = int(0.7 * T)
    train_idx = idx[:split]
    test_idx = idx[split:]
    W = train_linear_decoder(Xz[train_idx], pos[train_idx])
    pred = decode_positions(Xz, W)

    plot_results(pos, t_s, X, meta, Dp, Cp, rec_trace, spars_trace, pred, traj_cfg.arena_size, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




