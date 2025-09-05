"""
Cerebellum-Inspired Motor Learning: Perceptron with Climbing-Fiber Error

Single-output perceptron learning where a climbing-fiber error signal adjusts
Purkinje cell output weights to match target outputs from random inputs.

Run:
  python CerebellumDemo/cerebellum_demo.py --inputs 256 --samples 2000 --epochs 30 --save cereb.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CerebellumConfig:
    input_dim: int = 256
    samples: int = 2000
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 0.1
    noise_std: float = 0.0
    seed: Optional[int] = 0
    target_type: str = "binary"  # or "continuous"


def generate_dataset(cfg: CerebellumConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0.0, 1.0, size=(cfg.samples, cfg.input_dim)).astype(np.float32)
    if cfg.target_type == "binary":
        # Random linear teacher with sign; adds optional noise to teacher output before sign
        teacher_w = rng.normal(0.0, 1.0, size=(cfg.input_dim, 1)).astype(np.float32)
        y_lin = X @ teacher_w
        if cfg.noise_std > 0:
            y_lin += cfg.noise_std * rng.standard_normal(size=y_lin.shape)
        y = np.where(y_lin >= 0, 1.0, -1.0).astype(np.float32)
    else:
        # Continuous target in [-1, 1]
        teacher_w = rng.normal(0.0, 1.0, size=(cfg.input_dim, 1)).astype(np.float32)
        y = np.tanh((X @ teacher_w) / np.sqrt(cfg.input_dim)).astype(np.float32)
    return X, y.squeeze(-1)


def perceptron_train(
    X: np.ndarray,
    y: np.ndarray,
    cfg: CerebellumConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, list[float]]:
    N, D = X.shape
    w = rng.normal(0.0, 0.1, size=(D,)).astype(np.float32)
    lr = cfg.learning_rate
    losses: list[float] = []

    indices = np.arange(N)
    for epoch in range(cfg.epochs):
        rng.shuffle(indices)
        for start in range(0, N, cfg.batch_size):
            batch_idx = indices[start : start + cfg.batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            o = xb @ w  # Purkinje output (before sign)
            if cfg.target_type == "binary":
                yhat = np.where(o >= 0, 1.0, -1.0)
                err = yb - yhat
                # Climbing fiber error drives synaptic update: Δw = η * err * x
                # Using perceptron-like update: move w to reduce classification error
                mis_mask = (err != 0).astype(np.float32)
                # For misclassified samples, push w towards y * x
                grad = (mis_mask[:, None] * (yb[:, None] * xb)).mean(axis=0)
                w += lr * grad
            else:
                # Continuous error: Δw = η * (y - o) * x
                err = yb - o
                grad = (err[:, None] * xb).mean(axis=0)
                w += lr * grad

        # Epoch loss
        with np.errstate(divide="ignore", invalid="ignore"):
            o_all = X @ w
            if cfg.target_type == "binary":
                yhat_all = np.where(o_all >= 0, 1.0, -1.0)
                acc = float(np.mean(yhat_all == y))
                losses.append(1.0 - acc)
            else:
                mse = float(np.mean((o_all - y) ** 2))
                losses.append(mse)

    return w, losses


def evaluate(X: np.ndarray, y: np.ndarray, w: np.ndarray, target_type: str) -> dict:
    o = X @ w
    if target_type == "binary":
        yhat = np.where(o >= 0, 1.0, -1.0)
        acc = float(np.mean(yhat == y))
        return {"accuracy": acc}
    else:
        mse = float(np.mean((o - y) ** 2))
        return {"mse": mse}


def plot_results(
    losses: list[float],
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    target_type: str,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # Learning curve
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.plot(losses, linewidth=1.5)
    ax_l.set_title("Learning curve")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("1-acc" if target_type == "binary" else "MSE")

    # Predictions vs targets (subset)
    ax_p = fig.add_subplot(gs[0, 1])
    idx = np.random.choice(X.shape[0], size=min(400, X.shape[0]), replace=False)
    o = X[idx] @ w
    if target_type == "binary":
        yhat = np.where(o >= 0, 1.0, -1.0)
        ax_p.scatter(y[idx], yhat, s=10, alpha=0.6)
        ax_p.set_xlabel("Target")
        ax_p.set_ylabel("Predicted")
        ax_p.set_title("Pred vs Target (binary)")
        ax_p.set_xlim(-1.1, 1.1)
        ax_p.set_ylim(-1.1, 1.1)
    else:
        ax_p.scatter(y[idx], o, s=10, alpha=0.6)
        ax_p.plot([-1, 1], [-1, 1], "r--", linewidth=1.0)
        ax_p.set_xlabel("Target")
        ax_p.set_ylabel("Output")
        ax_p.set_title("Pred vs Target (continuous)")

    # Weights visualization (if square)
    ax_w = fig.add_subplot(gs[0, 2])
    D = w.shape[0]
    side = int(np.sqrt(D))
    if side * side == D:
        Wimg = (w.reshape(side, side) - w.min()) / (w.max() - w.min() + 1e-8)
        ax_w.imshow(Wimg, cmap="viridis", interpolation="nearest")
        ax_w.set_title("Weights (reshaped)")
        ax_w.axis("off")
    else:
        ax_w.plot(w)
        ax_w.set_title("Weights")
        ax_w.set_xlabel("Index")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cerebellum-inspired perceptron learning with climbing-fiber error")
    p.add_argument("--inputs", type=int, default=256, help="Input dimension")
    p.add_argument("--samples", type=int, default=2000, help="Number of training samples")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--epochs", type=int, default=30, help="Epochs")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--noise-std", type=float, default=0.0, help="Noise on teacher output for binary targets")
    p.add_argument("--target-type", type=str, choices=["binary", "continuous"], default="binary", help="Type of targets")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save figure")
    p.add_argument("--no-show", action="store_true", help="Do not display plot")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = CerebellumConfig(
        input_dim=args.inputs,
        samples=args.samples,
        batch_size=args.batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        noise_std=args.noise_std,
        target_type=args.target_type,
        seed=args.seed,
    )
    rng = np.random.default_rng(cfg.seed)

    X, y = generate_dataset(cfg, rng)
    w, losses = perceptron_train(X, y, cfg, rng)
    metrics = evaluate(X, y, w, cfg.target_type)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_results(losses, X, y, w, cfg.target_type, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




