"""
Visual Cortex Receptive Field Emergence Demo

Learn Gabor-like receptive fields from naturalistic image patches using a
simple Hebbian learner (Oja's rule with competition via k-WTA) on ZCA-whitened
patches. Includes an optional 1/f synthetic image generator when no dataset is
provided.

Example:
    python ReceptiveFieldsDemo/rf_demo.py --steps 20000 --save filters.png --no-show
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ----------------------------- Data utilities ----------------------------- #

def load_grayscale_images_from_folder(folder: Path, max_images: Optional[int] = None) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    if not PIL_AVAILABLE:
        return images
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        for p in folder.glob(ext):
            try:
                with Image.open(p) as img:
                    img = img.convert("L")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    images.append(arr)
                    count += 1
                    if max_images is not None and count >= max_images:
                        return images
            except Exception:
                continue
    return images


def generate_1_over_f_image(height: int, width: int, beta: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    fy = np.fft.fftfreq(height).reshape(-1, 1)
    fx = np.fft.fftfreq(width).reshape(1, -1)
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    amplitude = 1.0 / (radius ** (beta / 2.0))
    phase = rng.uniform(0, 2 * np.pi, size=(height, width))
    spectrum = amplitude * np.exp(1j * phase)
    field = np.fft.ifft2(spectrum).real
    field = (field - field.mean()) / (field.std() + 1e-8)
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field.astype(np.float32)


def extract_random_patches(images: Iterable[np.ndarray], patch_size: int, num_patches: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    images_list = list(images)
    if not images_list:
        raise ValueError("No images provided for patch extraction.")
    patches = np.empty((num_patches, patch_size * patch_size), dtype=np.float32)
    for i in range(num_patches):
        img = images_list[rng.integers(0, len(images_list))]
        if img.ndim != 2:
            raise ValueError("Images must be grayscale 2D arrays.")
        h, w = img.shape
        if h < patch_size or w < patch_size:
            raise ValueError("Image smaller than patch size.")
        y = int(rng.integers(0, h - patch_size + 1))
        x = int(rng.integers(0, w - patch_size + 1))
        patch = img[y : y + patch_size, x : x + patch_size].copy()
        patch = patch - np.mean(patch)
        patch = patch / (np.std(patch) + 1e-6)
        patches[i] = patch.reshape(-1)
    return patches


def compute_zca_whitening_matrix(patches: np.ndarray, epsilon: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = patches.mean(axis=0, keepdims=True)
    centered = patches - feature_mean
    cov = centered.T @ centered / float(centered.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + epsilon)) @ eigvecs.T
    dewhitening = eigvecs @ np.diag(np.sqrt(eigvals + epsilon)) @ eigvecs.T
    return feature_mean, whitening.astype(np.float32), dewhitening.astype(np.float32)


def zca_whiten(patches: np.ndarray, feature_mean: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    centered = patches - feature_mean
    return centered @ whitening.T


# ----------------------------- Hebbian learner ----------------------------- #

@dataclass
class HebbianConfig:
    num_units: int = 64
    learning_rate: float = 0.05
    k_active: int = 8
    batch_size: int = 128
    steps: int = 20000
    random_seed: Optional[int] = 0


class OjaKWTA:
    """Oja's rule with competition via per-sample k-Winners-Take-All activations.

    Update: dW = eta * (Y^T X - diag(sum(Y^2, axis=0)) W) with row-wise unit-norm.
    """

    def __init__(self, input_dim: int, config: HebbianConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        w = rng.normal(0.0, 0.01, size=(config.num_units, input_dim)).astype(np.float32)
        self.weights = self._row_normalize(w)

    @staticmethod
    def _row_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        activations = x_batch @ self.weights.T  # shape: (B, U)
        if self.config.k_active >= activations.shape[1]:
            return np.maximum(activations, 0.0)
        # Per-sample threshold for kWTA
        kth = np.partition(activations, -self.config.k_active, axis=1)[:, -self.config.k_active :][:, 0]
        thresholds = kth.reshape(-1, 1)
        y = activations - thresholds
        y = np.maximum(y, 0.0)
        return y

    def update(self, x_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> float:
        # Oja's rule in batch form
        positive_term = y_batch.T @ x_batch  # (U, D)
        y_sq_sum = (y_batch * y_batch).sum(axis=0).reshape(-1, 1)  # (U, 1)
        delta_w = learning_rate * (positive_term - y_sq_sum * self.weights)
        self.weights += delta_w.astype(self.weights.dtype)
        self.weights = self._row_normalize(self.weights)
        mean_abs_update = float(np.mean(np.abs(delta_w)))
        return mean_abs_update


def train_hebbian(
    patches_whitened: np.ndarray,
    config: HebbianConfig,
    progress_every: int = 500,
) -> Tuple[OjaKWTA, list[float]]:
    rng = np.random.default_rng(config.random_seed)
    input_dim = patches_whitened.shape[1]
    model = OjaKWTA(input_dim, config)

    updates_trace: list[float] = []
    num_samples = patches_whitened.shape[0]
    for step in range(1, config.steps + 1):
        idx = rng.integers(0, num_samples, size=(config.batch_size,))
        batch = patches_whitened[idx]
        y = model.forward(batch)
        lr = config.learning_rate * (1.0 - 0.9 * (step / config.steps))
        mean_abs_update = model.update(batch, y, lr)
        updates_trace.append(mean_abs_update)
        if progress_every and step % progress_every == 0:
            print(f"Step {step}/{config.steps}  mean|dW|={mean_abs_update:.6f}")

    return model, updates_trace


# -------------------------------- Plotting -------------------------------- #

def tile_filters(weights: np.ndarray, patch_size: int, pad: int = 1) -> np.ndarray:
    num_units, dim = weights.shape
    assert dim == patch_size * patch_size
    grid_cols = int(math.ceil(math.sqrt(num_units)))
    grid_rows = int(math.ceil(num_units / grid_cols))
    tile_h = grid_rows * patch_size + (grid_rows + 1) * pad
    tile_w = grid_cols * patch_size + (grid_cols + 1) * pad
    canvas = np.ones((tile_h, tile_w), dtype=np.float32) * 0.5
    # Normalize each filter to [-1, 1] then map to [0,1]
    W = weights.copy()
    W -= W.mean(axis=1, keepdims=True)
    W /= (W.std(axis=1, keepdims=True) + 1e-8)
    W = W.reshape(num_units, patch_size, patch_size)
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= num_units:
                break
            y = r * patch_size + (r + 1) * pad
            x = c * patch_size + (c + 1) * pad
            filt = W[idx]
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            canvas[y : y + patch_size, x : x + patch_size] = filt
            idx += 1
    return canvas


def plot_results(filter_grid: np.ndarray, updates_trace: list[float], save_path: Optional[str], show: bool) -> None:
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_filters = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    ax_filters.imshow(filter_grid, cmap="gray", interpolation="nearest")
    ax_filters.set_title("Learned filters")
    ax_filters.axis("off")

    ax_curve.plot(np.arange(len(updates_trace)), updates_trace)
    ax_curve.set_title("Mean |dW| over time")
    ax_curve.set_xlabel("Step")
    ax_curve.set_ylabel("Mean |dW| (per batch)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------- CLI/Main -------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Receptive field emergence via Hebbian learning on natural patches")
    # Data
    p.add_argument("--image-folder", type=str, default=None, help="Folder of natural images to sample patches from")
    p.add_argument("--max-images", type=int, default=200, help="Max images to read from folder")
    p.add_argument("--use-1overf", action="store_true", help="Use synthetic 1/f images if no folder provided")
    p.add_argument("--dataset-size", type=int, default=50000, help="Number of random patches to sample for training")
    p.add_argument("--patch-size", type=int, default=12, help="Square patch size (pixels)")
    p.add_argument("--zca-eps", type=float, default=1e-2, help="ZCA epsilon for numerical stability")
    # Model
    p.add_argument("--units", type=int, default=64, help="Number of output units/filters")
    p.add_argument("--k-active", type=int, default=8, help="k in kWTA (winners per sample)")
    p.add_argument("--steps", type=int, default=20000, help="Training steps (batches)")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    # Output
    p.add_argument("--save", type=str, default=None, help="Path to save the figure")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    return p


def _prepare_images(args: argparse.Namespace, rng: np.random.Generator) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    if args.image_folder is not None:
        folder = Path(args.image_folder)
        if folder.exists() and folder.is_dir():
            images = load_grayscale_images_from_folder(folder, max_images=args.max_images)
    if not images and args.use_1overf:
        # Generate a handful of large 1/f images to sample patches from
        num_synth = min(32, max(4, args.max_images))
        for _ in range(num_synth):
            img = generate_1_over_f_image(256, 256, beta=1.0, rng=rng)
            images.append(img)
    if not images:
        # Fallback: simple Gaussian noise images (not ideal for Gabors, but demo runs)
        for _ in range(16):
            img = rng.normal(0.5, 0.2, size=(256, 256)).clip(0.0, 1.0).astype(np.float32)
            images.append(img)
    return images


def main() -> None:
    args = _build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    images = _prepare_images(args, rng)
    print(f"Loaded {len(images)} images for patch extraction.")

    patches = extract_random_patches(images, patch_size=args.patch_size, num_patches=args.dataset_size, rng=rng)
    print(f"Extracted patches: {patches.shape}")

    mean_vec, whitening, _ = compute_zca_whitening_matrix(patches, epsilon=args.zca_eps)
    patches_w = zca_whiten(patches, mean_vec, whitening)
    print("Applied ZCA whitening.")

    hebb_cfg = HebbianConfig(
        num_units=args.units,
        learning_rate=args.lr,
        k_active=args.k_active,
        batch_size=args.batch_size,
        steps=args.steps,
        random_seed=args.seed,
    )

    model, updates_trace = train_hebbian(patches_w, hebb_cfg, progress_every=max(1, args.steps // 20))

    grid = tile_filters(model.weights, patch_size=args.patch_size, pad=1)
    plot_results(grid, updates_trace, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()


