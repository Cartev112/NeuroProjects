"""
Hopfield Network Demo: Storage, Retrieval, and Capacity vs Noise

Implements a classic Hopfield network with Hebbian weights (bipolar ±1 patterns),
supports synchronous and asynchronous updates, demonstrates retrieval from noisy
cues, and analyzes capacity (success vs alpha = P/N) and robustness to noise
(success vs flip probability).

Run:
  python HopfieldDemo/hopfield_demo.py --N 256 --P 30 --mode async --save hopfield.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def bipolar_sign(x: np.ndarray) -> np.ndarray:
    y = np.where(x >= 0, 1.0, -1.0)
    return y.astype(np.float32)


def generate_patterns(num_neurons: int, num_patterns: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=(num_patterns, num_neurons)).astype(np.float32)


def train_hebbian(patterns: np.ndarray) -> np.ndarray:
    # patterns shape: (P, N) with values in {-1, +1}
    P, N = patterns.shape
    W = patterns.T @ patterns / N
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def energy(W: np.ndarray, s: np.ndarray) -> float:
    # Hopfield energy: E = -1/2 s^T W s (bias=0)
    return float(-0.5 * s @ (W @ s))


def update_sync(W: np.ndarray, s: np.ndarray) -> np.ndarray:
    return bipolar_sign(W @ s)


def update_async(W: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = s.shape[0]
    order = rng.permutation(N)
    s_new = s.copy()
    for i in order:
        h_i = float(W[i] @ s_new)
        s_new[i] = 1.0 if h_i >= 0 else -1.0
    return s_new


def recall(
    W: np.ndarray,
    s_init: np.ndarray,
    mode: str,
    max_steps: int,
    rng: np.random.Generator,
    record_energy: bool = False,
) -> Tuple[np.ndarray, int, list[float]]:
    s = s_init.copy()
    energies: list[float] = []
    for step in range(1, max_steps + 1):
        if record_energy:
            energies.append(energy(W, s))
        s_next = update_sync(W, s) if mode == "sync" else update_async(W, s, rng)
        if np.array_equal(s_next, s):
            return s_next, step, energies
        s = s_next
    if record_energy:
        energies.append(energy(W, s))
    return s, max_steps, energies


def flip_bits(x: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape[0]) < flip_prob
    y = x.copy()
    y[mask] *= -1.0
    return y


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    # m = (1/N) a·b in [-1, 1]
    return float(np.mean(a == b) * 2.0 - 1.0)


def success_recall(retrieved: np.ndarray, target: np.ndarray, threshold: float = 0.95) -> bool:
    # Consider success if matches target or its sign-flipped version, via absolute overlap
    m = abs(overlap(retrieved, target))
    return m >= threshold


@dataclass
class HopfieldConfig:
    N: int = 256
    P: int = 30
    mode: str = "async"  # or "sync"
    max_steps: int = 50
    flip_prob: float = 0.1
    seed: Optional[int] = 0
    # Analysis sweeps
    alpha_max: float = 0.3
    cap_evals: int = 10
    noise_max: float = 0.5
    noise_evals: int = 11
    trials: int = 5


def analyze_capacity(cfg: HopfieldConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    N = cfg.N
    P_values = np.unique(np.maximum(1, (np.linspace(1, cfg.alpha_max * N, cfg.cap_evals)).astype(int)))
    success_rates = np.zeros(P_values.shape[0], dtype=float)

    for j, P in enumerate(P_values):
        succ = 0
        total = 0
        patterns = generate_patterns(N, P, rng)
        W = train_hebbian(patterns)
        for _ in range(cfg.trials):
            idx = int(rng.integers(0, P))
            target = patterns[idx]
            cue = flip_bits(target, cfg.flip_prob, rng)
            retrieved, _, _ = recall(W, cue, cfg.mode, cfg.max_steps, rng)
            if success_recall(retrieved, target):
                succ += 1
            total += 1
        success_rates[j] = succ / max(1, total)
    alpha = P_values / N
    return alpha, success_rates


def analyze_noise(cfg: HopfieldConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    N = cfg.N
    P = cfg.P
    patterns = generate_patterns(N, P, rng)
    W = train_hebbian(patterns)
    noise_levels = np.linspace(0.0, cfg.noise_max, cfg.noise_evals)
    success = np.zeros_like(noise_levels)

    for k, q in enumerate(noise_levels):
        succ = 0
        total = 0
        for _ in range(cfg.trials):
            idx = int(rng.integers(0, P))
            target = patterns[idx]
            cue = flip_bits(target, float(q), rng)
            retrieved, _, _ = recall(W, cue, cfg.mode, cfg.max_steps, rng)
            if success_recall(retrieved, target):
                succ += 1
            total += 1
        success[k] = succ / max(1, total)
    return noise_levels, success


def tile_pattern(vec: np.ndarray) -> np.ndarray:
    N = vec.shape[0]
    side = int(np.sqrt(N))
    if side * side == N:
        img = vec.reshape(side, side)
        img = (img + 1.0) / 2.0  # map -1..1 to 0..1
        return img
    # Fallback: 1 x N stripe
    return (vec[None, :] + 1.0) / 2.0


def plot_demo(
    cfg: HopfieldConfig,
    target: np.ndarray,
    cue: np.ndarray,
    retrieved: np.ndarray,
    energies: list[float],
    alpha: np.ndarray,
    cap_success: np.ndarray,
    noise_levels: np.ndarray,
    noise_success: np.ndarray,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Patterns
    ax_t = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_t.imshow(tile_pattern(target), cmap="gray", interpolation="nearest")
    ax_t.set_title("Target")
    ax_t.axis("off")
    ax_c.imshow(tile_pattern(cue), cmap="gray", interpolation="nearest")
    ax_c.set_title("Noisy cue")
    ax_c.axis("off")
    ax_r.imshow(tile_pattern(retrieved), cmap="gray", interpolation="nearest")
    ax_r.set_title("Retrieved")
    ax_r.axis("off")

    # Capacity curve
    ax_cap = fig.add_subplot(gs[1, 0])
    ax_cap.plot(alpha, cap_success, marker="o")
    ax_cap.set_xlabel("alpha = P/N")
    ax_cap.set_ylabel("Recall success")
    ax_cap.set_title("Capacity analysis")
    ax_cap.set_ylim(0, 1.05)

    # Noise robustness
    ax_noise = fig.add_subplot(gs[1, 1])
    ax_noise.plot(noise_levels, noise_success, marker="o", color="#ff7f0e")
    ax_noise.set_xlabel("Flip probability q")
    ax_noise.set_ylabel("Recall success")
    ax_noise.set_title("Noise robustness")
    ax_noise.set_ylim(0, 1.05)

    # Energy trace
    ax_E = fig.add_subplot(gs[1, 2])
    if energies:
        ax_E.plot(energies, linewidth=1.2)
    ax_E.set_title("Energy over iterations")
    ax_E.set_xlabel("Iteration")
    ax_E.set_ylabel("Energy")

    fig.suptitle(
        f"Hopfield (N={cfg.N}, P={cfg.P}, mode={cfg.mode}, steps≤{cfg.max_steps})  flip_prob={cfg.flip_prob}"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hopfield network: storage, retrieval, capacity vs noise")
    p.add_argument("--N", type=int, default=256, help="Number of neurons (use a square for nice images)")
    p.add_argument("--P", type=int, default=30, help="Number of stored patterns for demo and noise analysis")
    p.add_argument("--mode", type=str, choices=["sync", "async"], default="async", help="Update rule")
    p.add_argument("--max-steps", type=int, default=50, help="Max iterations for recall")
    p.add_argument("--flip-prob", type=float, default=0.1, help="Flip probability for noisy cue in demo/capacity")
    p.add_argument("--alpha-max", type=float, default=0.3, help="Max alpha=P/N for capacity sweep")
    p.add_argument("--cap-evals", type=int, default=10, help="Number of capacity points")
    p.add_argument("--noise-max", type=float, default=0.5, help="Max flip prob for noise sweep")
    p.add_argument("--noise-evals", type=int, default=11, help="Number of noise points")
    p.add_argument("--trials", type=int, default=5, help="Trials per sweep point")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save the figure")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = HopfieldConfig(
        N=args.N,
        P=args.P,
        mode=args.mode,
        max_steps=args.max_steps,
        flip_prob=args.flip_prob,
        seed=args.seed,
        alpha_max=args.alpha_max,
        cap_evals=args.cap_evals,
        noise_max=args.noise_max,
        noise_evals=args.noise_evals,
        trials=args.trials,
    )
    rng = np.random.default_rng(cfg.seed)

    # Example demo: store P patterns, recall one from noise
    patterns = generate_patterns(cfg.N, cfg.P, rng)
    W = train_hebbian(patterns)
    idx = int(rng.integers(0, cfg.P))
    target = patterns[idx]
    cue = flip_bits(target, cfg.flip_prob, rng)
    retrieved, iters, energies = recall(W, cue, cfg.mode, cfg.max_steps, rng, record_energy=True)
    print(f"Recall iterations: {iters}, success={success_recall(retrieved, target)}")

    # Analyze capacity and noise robustness
    alpha, cap_success = analyze_capacity(cfg, rng)
    noise_levels, noise_success = analyze_noise(cfg, rng)

    plot_demo(
        cfg,
        target,
        cue,
        retrieved,
        energies,
        alpha,
        cap_success,
        noise_levels,
        noise_success,
        save_path=args.save,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()


