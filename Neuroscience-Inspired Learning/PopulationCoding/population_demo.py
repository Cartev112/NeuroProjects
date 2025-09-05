"""
Neural Population Coding with Poisson Spikes

Encode a circular variable (orientation) using a population of Poisson neurons
with von Mises-like tuning. Decode with Maximum Likelihood (grid search) or
Population Vector (double-angle for orientation). Includes plots and metrics.

Run:
    python PopulationCodingDemo/population_demo.py --trials 500 --decoder ml --save pop_demo.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------- Math helpers ------------------------------ #

def wrap_degrees(angle_deg: np.ndarray | float, period: float) -> np.ndarray | float:
    return (angle_deg + period) % period


def circ_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray, period: float) -> np.ndarray:
    diff = (a_deg - b_deg + period / 2.0) % period - period / 2.0
    return diff


def circ_mean_deg(angles_deg: np.ndarray, period: float) -> float:
    # Convert to radians scaled to full 2*pi for given period
    theta = angles_deg / period * 2.0 * np.pi
    mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    mean_deg = wrap_degrees(mean_angle * period / (2.0 * np.pi), period)
    return float(mean_deg)


# ------------------------------- Model config ------------------------------ #

@dataclass
class PopulationConfig:
    num_neurons: int = 32
    period_deg: float = 180.0  # orientation typically has 180° periodicity
    baseline_rate_hz: float = 5.0
    max_rate_hz: float = 50.0
    kappa: float = 2.0  # sharpness of tuning curves
    duration_ms: float = 100.0  # per trial observation window
    grid_ml_deg: float = 0.5  # resolution of ML grid search
    seed: Optional[int] = 0


def build_preferred_angles(num_neurons: int, period_deg: float) -> np.ndarray:
    return np.linspace(0.0, period_deg, num_neurons, endpoint=False)


def von_mises_tuning(theta_deg: np.ndarray, pref_deg: np.ndarray, baseline: float, max_rate: float, kappa: float, period_deg: float) -> np.ndarray:
    # Map degrees to radians on the unit circle respecting period
    theta = theta_deg / period_deg * 2.0 * np.pi
    mu = pref_deg / period_deg * 2.0 * np.pi
    # Unnormalized von Mises-like; scale to hit baseline..max_rate at peak
    responses = np.exp(kappa * np.cos(theta[:, None] - mu[None, :]))
    # Normalize per-neuron so max over theta is 1 (since cos max=1)
    peak = np.exp(kappa)
    responses = responses / peak
    rates = baseline + (max_rate - baseline) * responses
    return rates  # shape (num_thetas, num_neurons)


def simulate_poisson_counts(rates_hz: np.ndarray, duration_ms: float, rng: np.random.Generator) -> np.ndarray:
    lam = rates_hz * (duration_ms / 1000.0)
    return rng.poisson(lam)


# --------------------------------- Decoders -------------------------------- #

def decode_population_vector(counts: np.ndarray, pref_deg: np.ndarray, period_deg: float) -> float:
    # For orientation (period 180°), use double-angle trick: represent as 2*angle on circle
    angle_factor = 2.0 if abs(period_deg - 180.0) < 1e-6 else (360.0 / period_deg)
    phases = np.deg2rad(pref_deg * angle_factor)
    vector = np.sum(counts * np.exp(1j * phases))
    est_phase = np.angle(vector)
    est_deg_scaled = np.rad2deg(est_phase) / angle_factor
    est_deg = wrap_degrees(est_deg_scaled, period_deg)
    return float(est_deg)


def decode_maximum_likelihood(counts: np.ndarray, pref_deg: np.ndarray, cfg: PopulationConfig) -> float:
    grid = np.arange(0.0, cfg.period_deg, cfg.grid_ml_deg, dtype=float)
    rates = von_mises_tuning(grid, pref_deg, cfg.baseline_rate_hz, cfg.max_rate_hz, cfg.kappa, cfg.period_deg)
    lam = rates * (cfg.duration_ms / 1000.0)
    # Poisson log likelihood up to constant: sum_i [ k_i * log(lam_i) - lam_i ]
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    ll = (counts[None, :] * np.log(lam + eps) - lam).sum(axis=1)
    idx = int(np.argmax(ll))
    return float(grid[idx])


# --------------------------------- Simulation ------------------------------ #

def run_population_coding(
    cfg: PopulationConfig,
    trials: int,
    decoder: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pref = build_preferred_angles(cfg.num_neurons, cfg.period_deg)
    # Sample true orientations uniformly
    true_angles = rng.uniform(0.0, cfg.period_deg, size=(trials,)).astype(float)
    # Precompute tuning for a dense set of thetas for speed? Instead compute per trial using vectorization
    est_angles = np.empty(trials, dtype=float)
    errors = np.empty(trials, dtype=float)
    counts_all = np.empty((trials, cfg.num_neurons), dtype=int)

    for t in range(trials):
        rates = von_mises_tuning(np.array([true_angles[t]]), pref, cfg.baseline_rate_hz, cfg.max_rate_hz, cfg.kappa, cfg.period_deg)[0]
        counts = simulate_poisson_counts(rates, cfg.duration_ms, rng)
        counts_all[t] = counts
        if decoder == "ml":
            est = decode_maximum_likelihood(counts, pref, cfg)
        elif decoder == "pv":
            est = decode_population_vector(counts, pref, cfg.period_deg)
        else:
            raise ValueError("Unknown decoder: choose 'ml' or 'pv'")
        est_angles[t] = est
        errors[t] = circ_diff_deg(est, true_angles[t], cfg.period_deg)

    return pref, true_angles, est_angles, errors, counts_all


# ---------------------------------- Plotting ------------------------------- #

def plot_demo(
    pref: np.ndarray,
    cfg: PopulationConfig,
    true_angles: np.ndarray,
    est_angles: np.ndarray,
    errors: np.ndarray,
    counts_all: np.ndarray,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Tuning curves
    ax_tune = fig.add_subplot(gs[:, 0])
    thetas = np.linspace(0.0, cfg.period_deg, 361)
    rates = von_mises_tuning(thetas, pref, cfg.baseline_rate_hz, cfg.max_rate_hz, cfg.kappa, cfg.period_deg)
    for i in range(cfg.num_neurons):
        ax_tune.plot(thetas, rates[:, i], linewidth=1.0)
    ax_tune.set_title("Tuning curves")
    ax_tune.set_xlabel("Orientation (deg)")
    ax_tune.set_ylabel("Rate (Hz)")

    # One trial counts
    ax_counts = fig.add_subplot(gs[0, 1])
    trial_idx = np.random.randint(0, counts_all.shape[0])
    ax_counts.bar(np.arange(cfg.num_neurons), counts_all[trial_idx], color="#1f77b4")
    ax_counts.set_title(f"Spike counts (trial {trial_idx})")
    ax_counts.set_xlabel("Neuron index")
    ax_counts.set_ylabel("Count")

    # True vs estimated scatter (circular)
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_scatter.scatter(true_angles, est_angles, s=10, alpha=0.6)
    ax_scatter.plot([0, cfg.period_deg], [0, cfg.period_deg], "r--", linewidth=1.0)
    ax_scatter.set_xlim(0, cfg.period_deg)
    ax_scatter.set_ylim(0, cfg.period_deg)
    ax_scatter.set_xlabel("True (deg)")
    ax_scatter.set_ylabel("Estimated (deg)")
    ax_scatter.set_title("Decoding: true vs estimated")

    # Error histogram
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(errors, bins=30, color="#2ca02c")
    ax_hist.set_title("Error distribution (deg)")
    ax_hist.set_xlabel("Error (deg)")
    ax_hist.set_ylabel("Frequency")

    # Error over trials
    ax_err = fig.add_subplot(gs[1, 2])
    ax_err.plot(np.abs(errors), linewidth=0.8)
    ax_err.set_title("|Error| per trial")
    ax_err.set_xlabel("Trial")
    ax_err.set_ylabel("|Error| (deg)")

    mae = float(np.mean(np.abs(errors)))
    fig.suptitle(f"Population coding ({cfg.num_neurons} neurons, {cfg.duration_ms:.0f} ms)  MAE={mae:.2f}°")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------ CLI --------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural population coding with Poisson spikes")
    p.add_argument("--neurons", type=int, default=32, help="Number of neurons in the population")
    p.add_argument("--period", type=float, default=180.0, help="Stimulus period in degrees (180 for orientation, 360 for direction)")
    p.add_argument("--baseline", type=float, default=5.0, help="Baseline firing rate (Hz)")
    p.add_argument("--max-rate", type=float, default=50.0, help="Maximum firing rate (Hz)")
    p.add_argument("--kappa", type=float, default=2.0, help="Tuning curve sharpness (higher is narrower)")
    p.add_argument("--duration", type=float, default=100.0, help="Observation window per trial (ms)")
    p.add_argument("--trials", type=int, default=500, help="Number of trials")
    p.add_argument("--decoder", type=str, choices=["ml", "pv"], default="ml", help="Decoder: maximum likelihood (ml) or population vector (pv)")
    p.add_argument("--ml-grid", type=float, default=0.5, help="ML grid resolution in degrees")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save the figure")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = PopulationConfig(
        num_neurons=args.neurons,
        period_deg=args.period,
        baseline_rate_hz=args.baseline,
        max_rate_hz=args.max_rate,
        kappa=args.kappa,
        duration_ms=args.duration,
        grid_ml_deg=args.ml_grid,
        seed=args.seed,
    )
    rng = np.random.default_rng(cfg.seed)

    pref, true_angles, est_angles, errors, counts_all = run_population_coding(cfg, trials=args.trials, decoder=args.decoder, rng=rng)

    # Print quick metrics
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors * errors)))
    print("Decoding metrics:")
    print(f"  MAE:  {mae:.3f} deg")
    print(f"  RMSE: {rmse:.3f} deg")

    plot_demo(pref, cfg, true_angles, est_angles, errors, counts_all, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




