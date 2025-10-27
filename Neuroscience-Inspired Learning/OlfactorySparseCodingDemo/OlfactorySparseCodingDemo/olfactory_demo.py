"""
Olfactory Bulb-Inspired Sparse Coding

Implements a simple feedforward random projection followed by lateral
inhibition / k-Winners-Take-All (kWTA) to produce sparse, decorrelated
representations of binary input patterns.

Run:
    python OlfactorySparseCodingDemo/olfactory_demo.py --patterns 200 --save olf_demo.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ModelConfig:
    input_dim: int = 256
    output_dim: int = 512
    connection_prob: float = 0.1  # sparsity of random projection
    weight_scale: float = 1.0
    inhibition_strength: float = 0.2  # subtractive lateral inhibition factor
    use_kwta: bool = True
    k_active: int = 25
    seed: Optional[int] = 0


def generate_binary_patterns(num_patterns: int, input_dim: int, activity_prob: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((num_patterns, input_dim)) < activity_prob).astype(np.float32)


def build_random_projection(input_dim: int, output_dim: int, connection_prob: float, weight_scale: float, rng: np.random.Generator) -> np.ndarray:
    W = (rng.random((output_dim, input_dim)) < connection_prob).astype(np.float32)
    # Assign +/- weights to avoid bias; normalize rows
    signs = rng.choice([-1.0, 1.0], size=W.shape).astype(np.float32)
    W = W * signs
    # Row normalization to unit norm then scale
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    W = weight_scale * (W / norms)
    return W


def lateral_inhibition(acts: np.ndarray, strength: float) -> np.ndarray:
    # Subtractive inhibition by the mean activity across units per sample
    mean_act = acts.mean(axis=1, keepdims=True)
    inhibited = acts - strength * mean_act
    return np.maximum(inhibited, 0.0)


def kwta(acts: np.ndarray, k: int) -> np.ndarray:
    if k >= acts.shape[1]:
        return np.maximum(acts, 0.0)
    thresh = np.partition(acts, -k, axis=1)[:, -k][:, None]
    y = acts - thresh
    return np.maximum(y, 0.0)


def compute_metrics(X: np.ndarray, Y: np.ndarray) -> dict:
    # Sparsity: fraction of nonzeros
    sparsity_in = float(np.count_nonzero(X) / X.size)
    sparsity_out = float(np.count_nonzero(Y) / Y.size)
    # Decorrelation: average absolute off-diagonal correlation
    def avg_offdiag_abs_corr(Z: np.ndarray) -> float:
        Zc = Z - Z.mean(axis=0, keepdims=True)
        std = Zc.std(axis=0, ddof=1, keepdims=True) + 1e-8
        Zc /= std
        C = (Zc.T @ Zc) / (Z.shape[0] - 1)
        off = C - np.eye(C.shape[0])
        return float(np.mean(np.abs(off)))
    corr_in = avg_offdiag_abs_corr(X)
    corr_out = avg_offdiag_abs_corr(Y)
    return {
        "sparsity_in": sparsity_in,
        "sparsity_out": sparsity_out,
        "avg_abs_corr_in": corr_in,
        "avg_abs_corr_out": corr_out,
    }


def run_demo(
    cfg: ModelConfig,
    num_patterns: int,
    input_activity_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = generate_binary_patterns(num_patterns, cfg.input_dim, input_activity_prob, rng)
    W = build_random_projection(cfg.input_dim, cfg.output_dim, cfg.connection_prob, cfg.weight_scale, rng)
    acts = X @ W.T  # feedforward
    if cfg.use_kwta:
        Y = kwta(acts, cfg.k_active)
    else:
        Y = lateral_inhibition(np.maximum(acts, 0.0), cfg.inhibition_strength)
    return X, Y, W


def plot_demo(X: np.ndarray, Y: np.ndarray, W: np.ndarray, metrics: dict, save_path: Optional[str], show: bool) -> None:
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Show a small sample of input and output patterns as images
    ax_in = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_W = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, :])

    n_show = min(64, X.shape[0])
    side_in = int(np.sqrt(X.shape[1])) if int(np.sqrt(X.shape[1])) ** 2 == X.shape[1] else None
    side_out = int(np.sqrt(Y.shape[1])) if int(np.sqrt(Y.shape[1])) ** 2 == Y.shape[1] else None

    def tile_patterns(Z: np.ndarray, side: Optional[int]) -> np.ndarray:
        if side is None:
            # fallback: show first vector as a horizontal stripe
            return Z[:1]
        grid_cols = int(np.ceil(np.sqrt(n_show)))
        grid_rows = int(np.ceil(n_show / grid_cols))
        pad = 1
        canvas = np.zeros((grid_rows * side + (grid_rows + 1) * pad, grid_cols * side + (grid_cols + 1) * pad), dtype=float)
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if idx >= n_show:
                    break
                patch = Z[idx].reshape(side, side)
                y = r * side + (r + 1) * pad
                x = c * side + (c + 1) * pad
                canvas[y : y + side, x : x + side] = patch
                idx += 1
        return canvas

    ax_in.imshow(tile_patterns(X, side_in), cmap="gray", interpolation="nearest")
    ax_in.set_title("Input patterns")
    ax_in.axis("off")

    # Normalize Y to [0,1] for display
    Y_disp = Y.copy()
    if Y_disp.max() > 0:
        Y_disp = (Y_disp - Y_disp.min()) / (Y_disp.max() - Y_disp.min() + 1e-8)
    ax_out.imshow(tile_patterns(Y_disp, side_out), cmap="gray", interpolation="nearest")
    ax_out.set_title("Output patterns (sparse)")
    ax_out.axis("off")

    # Show a subset of weight vectors as images if shapes are square
    if side_in is not None:
        m = min(64, W.shape[0])
        W_show = W[:m]
        # Normalize each row to [0,1]
        Wn = (W_show - W_show.min(axis=1, keepdims=True)) / (W_show.max(axis=1, keepdims=True) - W_show.min(axis=1, keepdims=True) + 1e-8)
        ax_W.imshow(tile_patterns(Wn, side_in), cmap="gray", interpolation="nearest")
        ax_W.set_title("Random projection weights")
        ax_W.axis("off")
    else:
        ax_W.axis("off")

    # Bar plot of metrics
    labels = ["sparsity_in", "sparsity_out", "avg_abs_corr_in", "avg_abs_corr_out"]
    values = [metrics[k] for k in labels]
    ax_bar.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    ax_bar.set_ylabel("Value")
    ax_bar.set_title("Sparsity and decorrelation metrics")
    ax_bar.set_ylim(0, max(1.0, max(values) * 1.1))
    for i, v in enumerate(values):
        ax_bar.text(i, v + 0.02, f"{v:.3f}", ha="center")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Olfactory bulb-inspired sparse coding via random projection + inhibition")
    p.add_argument("--input-dim", type=int, default=256, help="Input dimensionality (prefer square numbers for visualization)")
    p.add_argument("--output-dim", type=int, default=512, help="Number of output units")
    p.add_argument("--conn-prob", type=float, default=0.1, help="Connection probability for random projection")
    p.add_argument("--weight-scale", type=float, default=1.0, help="Scaling for projection weights")
    p.add_argument("--kwta", action="store_true", help="Use kWTA instead of subtractive inhibition")
    p.add_argument("--k-active", type=int, default=25, help="Active units per pattern for kWTA")
    p.add_argument("--inhib-strength", type=float, default=0.2, help="Subtractive inhibition strength (if not kWTA)")
    p.add_argument("--patterns", type=int, default=200, help="Number of random binary patterns")
    p.add_argument("--activity-prob", type=float, default=0.1, help="Probability a bit is 1 in input patterns")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save figure")
    p.add_argument("--no-show", action="store_true", help="Do not display plot")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = ModelConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        connection_prob=args.conn_prob,
        weight_scale=args.weight_scale,
        use_kwta=bool(args.kwta),
        k_active=args.k_active,
        inhibition_strength=args.inhib_strength,
        seed=args.seed,
    )
    rng = np.random.default_rng(cfg.seed)

    X, Y, W = run_demo(cfg, num_patterns=args.patterns, input_activity_prob=args.activity_prob, rng=rng)
    metrics = compute_metrics(X, Y)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_demo(X, Y, W, metrics, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()


