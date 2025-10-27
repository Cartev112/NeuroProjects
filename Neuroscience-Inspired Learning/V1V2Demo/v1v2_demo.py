"""
V1–V2 Hierarchical Unsupervised Model

Layer 1 (V1): Learns Gabor-like filters via sparse coding on ZCA-whitened
natural image patches (overcomplete dictionary).

Layer 2 (V2): Operates on 2x2 spatial neighborhoods of V1 codes and learns
composite features (e.g., corners, junctions) via sparse coding again.

Run example:
  python V1V2Demo/v1v2_demo.py --use-1overf --steps1 3000 --steps2 2000 --save v1v2.png --no-show
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


def extract_random_patches(images: Iterable[np.ndarray], patch_size: int, num_patches: int, rng: np.random.Generator) -> np.ndarray:
    images_list = list(images)
    if not images_list:
        raise ValueError("No images provided for patches.")
    patches = np.empty((num_patches, patch_size * patch_size), dtype=np.float32)
    for i in range(num_patches):
        img = images_list[int(rng.integers(0, len(images_list)))]
        h, w = img.shape
        if h < patch_size or w < patch_size:
            raise ValueError("Image smaller than patch size.")
        y = int(rng.integers(0, h - patch_size + 1))
        x = int(rng.integers(0, w - patch_size + 1))
        patch = img[y : y + patch_size, x : x + patch_size]
        patch = patch - np.mean(patch)
        patch = patch / (np.std(patch) + 1e-6)
        patches[i] = patch.reshape(-1)
    return patches


def compute_zca(patches: np.ndarray, epsilon: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = patches.mean(axis=0, keepdims=True)
    Xc = patches - mu
    cov = (Xc.T @ Xc) / float(Xc.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + epsilon)) @ eigvecs.T
    Winv = eigvecs @ np.diag(np.sqrt(eigvals + epsilon)) @ eigvecs.T
    return mu.astype(np.float32), W.astype(np.float32), Winv.astype(np.float32)


def zca_whiten(patches: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (patches - mu) @ W.T


# --------------------------- Sparse coding (ISTA) -------------------------- #

def ista(X: np.ndarray, D: np.ndarray, lam: float, steps: int, step_size: Optional[float] = None) -> np.ndarray:
    """Compute sparse codes C for X ≈ C D using ISTA (row-wise codes).

    Shapes: X (N, D_in), D (K, D_in), returns C (N, K)
    Objective: 0.5||X - C D||^2 + lam * ||C||_1
    """
    N, Din = X.shape
    K, Din2 = D.shape
    assert Din == Din2
    C = np.zeros((N, K), dtype=np.float32)
    Dt = D.T
    L = np.linalg.norm(D @ Dt, 2)  # Lipschitz bound for gradient
    t = (1.0 / L) if step_size is None else float(step_size)
    for _ in range(steps):
        grad = (C @ D - X) @ Dt  # (N,K)
        C = C - t * grad
        # Soft-threshold
        C = np.sign(C) * np.maximum(0.0, np.abs(C) - t * lam)
    return C


def update_dictionary(X: np.ndarray, C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Closed-form least-squares update for dictionary rows (atoms).

    D = (C^T X) (C^T C + eps I)^{-1}
    Rows are normalized to unit norm.
    Returns D with shape (K, D_in)
    """
    Ct = C.T
    G = Ct @ C  # (K,K)
    P = Ct @ X  # (K,D)
    A = G + eps * np.eye(G.shape[0], dtype=G.dtype)
    D = np.linalg.solve(A, P).astype(np.float32)  # (K,D)
    # Normalize atoms
    norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-8
    D = D / norms
    return D


def train_sparse_coding(
    X: np.ndarray,
    K: int,
    lam: float,
    steps_per_iter: int,
    iters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    N, Din = X.shape
    D = rng.normal(0.0, 0.01, size=(K, Din)).astype(np.float32)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    rec_trace: list[float] = []
    spars_trace: list[float] = []
    C = np.zeros((N, K), dtype=np.float32)
    for it in range(iters):
        C = ista(X, D, lam=lam, steps=steps_per_iter)
        D = update_dictionary(X, C)
        recon = C @ D
        rec_err = float(np.mean((X - recon) ** 2))
        spars = float(np.mean(np.abs(C) > 1e-6))
        rec_trace.append(rec_err)
        spars_trace.append(spars)
    return D, C, rec_trace, spars_trace


# ---------------------------- Layer 2 (V2) codes --------------------------- #

def extract_stride_patches(img: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    h, w = img.shape
    out = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            patch = patch - np.mean(patch)
            patch = patch / (np.std(patch) + 1e-6)
            out.append(patch.reshape(-1))
    return np.asarray(out, dtype=np.float32)


def v1_codes_for_image(img: np.ndarray, patch_size: int, stride: int, mu: np.ndarray, W: np.ndarray, D1: np.ndarray, lam1: float, steps: int) -> Tuple[np.ndarray, int, int]:
    patches = extract_stride_patches(img, patch_size=patch_size, stride=stride)
    Xw = zca_whiten(patches, mu, W)
    C1 = ista(Xw, D1, lam=lam1, steps=steps)
    # Put codes on a grid (Hc x Wc x K)
    h, w = img.shape
    Hc = (h - patch_size) // stride + 1
    Wc = (w - patch_size) // stride + 1
    K = D1.shape[0]
    C1_grid = C1.reshape(Hc, Wc, K)
    return C1_grid, Hc, Wc


def build_v2_training_set(C1_grid: np.ndarray) -> np.ndarray:
    """Create V2 inputs by concatenating 2x2 neighborhoods of V1 codes.

    Input grid shape: (Hc, Wc, K)
    Output samples: ((Hc-1)*(Wc-1), 4*K)
    """
    Hc, Wc, K = C1_grid.shape
    samples = []
    for y in range(Hc - 1):
        for x in range(Wc - 1):
            block = np.concatenate([
                C1_grid[y, x], C1_grid[y, x + 1], C1_grid[y + 1, x], C1_grid[y + 1, x + 1]
            ], axis=0)
            samples.append(block)
    return np.asarray(samples, dtype=np.float32)


def visualize_filters(D: np.ndarray, patch_size: int, pad: int = 1) -> np.ndarray:
    K, Din = D.shape
    assert Din == patch_size * patch_size
    grid_cols = int(math.ceil(math.sqrt(K)))
    grid_rows = int(math.ceil(K / grid_cols))
    tile_h = grid_rows * patch_size + (grid_rows + 1) * pad
    tile_w = grid_cols * patch_size + (grid_cols + 1) * pad
    canvas = np.ones((tile_h, tile_w), dtype=np.float32) * 0.5
    Dn = D.copy()
    Dn -= Dn.mean(axis=1, keepdims=True)
    Dn /= (Dn.std(axis=1, keepdims=True) + 1e-8)
    Dn = Dn.reshape(K, patch_size, patch_size)
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= K:
                break
            y = r * patch_size + (r + 1) * pad
            x = c * patch_size + (c + 1) * pad
            filt = Dn[idx]
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            canvas[y : y + patch_size, x : x + patch_size] = filt
            idx += 1
    return canvas


def visualize_v2_atoms(D2: np.ndarray, D1: np.ndarray, patch_size: int, pad: int = 1) -> np.ndarray:
    """Project each V2 atom (4*K) back to pixel space by combining V1 atoms in a 2x2 layout.

    For visualization only: not a true inverse, but illustrative.
    """
    K = D1.shape[0]
    P = patch_size
    H = 2 * P + pad
    W = 2 * P + pad
    M, dim = D2.shape
    assert dim == 4 * K
    tiles_per_row = int(math.ceil(math.sqrt(M)))
    rows = int(math.ceil(M / tiles_per_row))
    canvas = np.ones((rows * H + (rows + 1) * pad, tiles_per_row * W + (tiles_per_row + 1) * pad), dtype=np.float32) * 0.5

    # Normalize D1 filters for consistent scaling
    D1n = D1.copy()
    D1n -= D1n.mean(axis=1, keepdims=True)
    D1n /= (D1n.std(axis=1, keepdims=True) + 1e-8)
    D1n = D1n.reshape(K, P, P)

    idx = 0
    for r in range(rows):
        for c in range(tiles_per_row):
            if idx >= M:
                break
            y0 = r * H + (r + 1) * pad
            x0 = c * W + (c + 1) * pad
            # Four chunks of length K
            a = D2[idx, 0:K]
            b = D2[idx, K:2*K]
            d = D2[idx, 2*K:3*K]
            e = D2[idx, 3*K:4*K]
            tile = np.zeros((H - pad, W - pad), dtype=np.float32)
            # Top-left
            tl = (a[:, None, None] * D1n).sum(axis=0)
            tile[0:P, 0:P] = (tl - tl.min()) / (tl.max() - tl.min() + 1e-8)
            # Top-right
            tr = (b[:, None, None] * D1n).sum(axis=0)
            tile[0:P, P:2*P] = (tr - tr.min()) / (tr.max() - tr.min() + 1e-8)
            # Bottom-left
            bl = (d[:, None, None] * D1n).sum(axis=0)
            tile[P:2*P, 0:P] = (bl - bl.min()) / (bl.max() - bl.min() + 1e-8)
            # Bottom-right
            br = (e[:, None, None] * D1n).sum(axis=0)
            tile[P:2*P, P:2*P] = (br - br.min()) / (br.max() - br.min() + 1e-8)
            canvas[y0 : y0 + H - pad, x0 : x0 + W - pad] = tile
            idx += 1
    return canvas


# ---------------------------------- CLI/Main -------------------------------- #

@dataclass
class Args:
    image_folder: Optional[str]
    max_images: int
    use_1overf: bool
    dataset_size: int
    patch_size: int
    stride: int
    zca_eps: float
    K1: int
    K2: int
    lam1: float
    lam2: float
    steps1: int
    steps2: int
    iters1: int
    iters2: int
    seed: int
    save: Optional[str]
    no_show: bool


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V1–V2 hierarchical sparse coding")
    # Data
    p.add_argument("--image-folder", type=str, default=None)
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--use-1overf", action="store_true")
    p.add_argument("--dataset-size", type=int, default=60000)
    p.add_argument("--patch-size", type=int, default=12)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--zca-eps", type=float, default=1e-2)
    # Layer 1
    p.add_argument("--K1", type=int, default=128, help="# of V1 atoms (overcomplete)")
    p.add_argument("--lam1", type=float, default=0.15)
    p.add_argument("--steps1", type=int, default=50)
    p.add_argument("--iters1", type=int, default=3000)
    # Layer 2
    p.add_argument("--K2", type=int, default=128, help="# of V2 atoms")
    p.add_argument("--lam2", type=float, default=0.1)
    p.add_argument("--steps2", type=int, default=30)
    p.add_argument("--iters2", type=int, default=2000)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p


def plot_results(D1: np.ndarray, D2: np.ndarray, patch_size: int, rec1: list[float], spars1: list[float], rec2: list[float], spars2: list[float], save_path: Optional[str], show: bool) -> None:
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(visualize_filters(D1, patch_size), cmap="gray", interpolation="nearest")
    ax1.set_title("V1 filters (sparse coding)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(visualize_v2_atoms(D2, D1, patch_size), cmap="gray", interpolation="nearest")
    ax2.set_title("V2 composites (pooled 2x2 over V1)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rec1, label="V1 recon MSE")
    ax3.plot(rec2, label="V2 recon MSE")
    ax3.set_title("Reconstruction error")
    ax3.set_xlabel("Iteration")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(spars1, label="V1 sparsity frac")
    ax4.plot(spars2, label="V2 sparsity frac")
    ax4.set_title("Sparsity (fraction |c|>0)")
    ax4.set_xlabel("Iteration")
    ax4.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = _build_parser().parse_args()
    args = Args(
        image_folder=ap.image_folder,
        max_images=ap.max_images,
        use_1overf=ap.use_1overf,
        dataset_size=ap.dataset_size,
        patch_size=ap.patch_size,
        stride=ap.stride,
        zca_eps=ap.zca_eps,
        K1=ap.K1,
        K2=ap.K2,
        lam1=ap.lam1,
        lam2=ap.lam2,
        steps1=ap.steps1,
        steps2=ap.steps2,
        iters1=ap.iters1,
        iters2=ap.iters2,
        seed=ap.seed,
        save=ap.save,
        no_show=ap.no_show,
    )
    rng = np.random.default_rng(args.seed)

    # Prepare images
    images: list[np.ndarray] = []
    if args.image_folder is not None:
        folder = Path(args.image_folder)
        if folder.exists() and folder.is_dir():
            images = load_grayscale_images_from_folder(folder, max_images=args.max_images)
    if not images and args.use_1overf:
        for _ in range(min(32, max(8, args.max_images))):
            images.append(generate_1_over_f_image(256, 256, beta=1.0, rng=rng))
    if not images:
        # Fallback to Gaussian textures
        for _ in range(16):
            img = rng.normal(0.5, 0.2, size=(256, 256)).clip(0.0, 1.0).astype(np.float32)
            images.append(img)

    # V1 training data: random patches → ZCA → sparse coding
    patches = extract_random_patches(images, patch_size=args.patch_size, num_patches=args.dataset_size, rng=rng)
    mu, W, _ = compute_zca(patches, epsilon=args.zca_eps)
    Xw = zca_whiten(patches, mu, W)
    D1, C1, rec1, spars1 = train_sparse_coding(Xw, K=args.K1, lam=args.lam1, steps_per_iter=args.steps1, iters=args.iters1, rng=rng)

    # V2 training data: pick one image, tile patches with stride to form grid of V1 codes; form 2x2 neighborhoods
    img_for_v2 = images[int(rng.integers(0, len(images)))]
    C1_grid, Hc, Wc = v1_codes_for_image(img_for_v2, patch_size=args.patch_size, stride=args.stride, mu=mu, W=W, D1=D1, lam1=args.lam1, steps=max(5, args.steps1 // 5))
    V2_train = build_v2_training_set(C1_grid)
    # Normalize V2 inputs per sample
    V2_train = (V2_train - V2_train.mean(axis=0, keepdims=True)) / (V2_train.std(axis=0, keepdims=True) + 1e-6)
    D2, C2, rec2, spars2 = train_sparse_coding(V2_train, K=args.K2, lam=args.lam2, steps_per_iter=args.steps2, iters=args.iters2, rng=rng)

    plot_results(D1, D2, args.patch_size, rec1, spars1, rec2, spars2, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




