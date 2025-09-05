"""
Chaos in Recurrent Spiking Networks

Network of LIF neurons with random recurrent E/I connectivity. Explore how the
excitatory/inhibitory balance (g) and synaptic strength (J) shape regimes from
stable/asynchronous to irregular/chaotic.

Run:
    python ChaosDemo/chaos_demo.py --N 800 --duration 2000 --g 4.0 --J 0.15 \
        --save chaos.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LIFNetConfig:
    N: int = 1000                   # total neurons
    frac_exc: float = 0.8           # fraction excitatory
    p_connect: float = 0.1          # connection probability
    J: float = 0.15                 # excitatory weight magnitude
    g: float = 4.0                  # inhibitory/excitatory magnitude ratio
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    V_rest_mv: float = -65.0
    V_reset_mv: float = -65.0
    V_th_mv: float = -50.0
    refractory_ms: float = 2.0
    dt_ms: float = 0.1
    noise_std_mv: float = 0.5
    I_ext: float = 1.5              # constant external drive (arb)
    R_m_mohm: float = 10.0
    seed: Optional[int] = 0


class RecurrentLIFNetwork:
    def __init__(self, cfg: LIFNetConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.dt = float(cfg.dt_ms)
        self.steps_per_ms = 1.0 / self.dt
        self.N = cfg.N
        self.NE = int(round(cfg.frac_exc * cfg.N))
        self.NI = cfg.N - self.NE

        # Neuron state
        self.V = np.full(cfg.N, cfg.V_rest_mv, dtype=float)
        self.refractory_countdown = np.zeros(cfg.N, dtype=int)

        # Synaptic state (current-based, exponential)
        self.s = np.zeros(cfg.N, dtype=float)
        self.syn_decay = np.exp(-self.dt / cfg.tau_syn_ms) if cfg.tau_syn_ms > 0 else 0.0

        # Precompute LIF constants
        self.leak_factor = self.dt / cfg.tau_m_ms
        self.noise_scale = cfg.noise_std_mv * np.sqrt(self.dt)
        self.refractory_steps = int(round(cfg.refractory_ms / self.dt))

        # Build random connectivity adjacency lists: for each presynaptic neuron i, store targets and weights
        self.targets: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        cfg = self.cfg
        N = cfg.N
        NE = self.NE
        p = cfg.p_connect
        J_E = cfg.J
        J_I = -cfg.g * cfg.J

        for i in range(N):
            # Sample out-connections
            out_mask = self.rng.random(N) < p
            out_mask[i] = False  # no self-connection
            tgt_idx = np.nonzero(out_mask)[0]
            if tgt_idx.size == 0:
                self.targets.append(np.empty(0, dtype=int))
                self.weights.append(np.empty(0, dtype=float))
                continue
            if i < NE:  # excitatory presynaptic
                w = np.full(tgt_idx.size, J_E, dtype=float)
            else:       # inhibitory presynaptic
                w = np.full(tgt_idx.size, J_I, dtype=float)
            self.targets.append(tgt_idx)
            self.weights.append(w)

    def step(self, spikes_prev: np.ndarray) -> np.ndarray:
        cfg = self.cfg

        # Decay synaptic state
        self.s *= self.syn_decay

        # Add synaptic inputs from last step's spikes
        spiking_indices = np.nonzero(spikes_prev)[0]
        for i in spiking_indices:
            tgt = self.targets[i]
            w = self.weights[i]
            if tgt.size:
                self.s[tgt] += w

        # Update membrane potentials
        dV = (-(self.V - cfg.V_rest_mv) + cfg.R_m_mohm * (cfg.I_ext + self.s)) * self.leak_factor
        if cfg.noise_std_mv > 0.0:
            dV += self.noise_scale * self.rng.standard_normal(self.N)

        # Apply refractory: hold at reset and decrement countdown
        refractory_mask = self.refractory_countdown > 0
        self.V[~refractory_mask] += dV[~refractory_mask]
        self.V[refractory_mask] = cfg.V_reset_mv
        self.refractory_countdown[refractory_mask] -= 1

        # Threshold crossing
        spikes = (self.V >= cfg.V_th_mv) & (~refractory_mask)
        if np.any(spikes):
            self.V[spikes] = cfg.V_reset_mv
            self.refractory_countdown[spikes] = self.refractory_steps

        return spikes.astype(np.uint8)


def simulate(cfg: LIFNetConfig, duration_ms: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = int(round(duration_ms / cfg.dt_ms))
    time_ms = np.arange(steps) * cfg.dt_ms
    net = RecurrentLIFNetwork(cfg)

    spikes_over_time = np.zeros((steps, cfg.N), dtype=np.uint8)
    V_over_time = np.zeros((steps, cfg.N), dtype=np.float32)

    spikes_prev = np.zeros(cfg.N, dtype=np.uint8)
    for t in range(steps):
        spikes = net.step(spikes_prev)
        spikes_over_time[t] = spikes
        V_over_time[t] = net.V
        spikes_prev = spikes

    return time_ms, spikes_over_time, V_over_time, net.V


def compute_metrics(time_ms: np.ndarray, spikes: np.ndarray, cfg: LIFNetConfig) -> dict:
    steps, N = spikes.shape
    duration_s = time_ms[-1] / 1000.0 if steps > 0 else 0.0

    # Rates per neuron
    spike_counts = spikes.sum(axis=0)
    rates_hz = spike_counts / duration_s if duration_s > 0 else np.zeros(N)

    # CV of ISI per neuron
    dt = cfg.dt_ms
    cv_list = []
    for i in range(N):
        t_idx = np.nonzero(spikes[:, i])[0]
        if t_idx.size >= 3:
            isi_ms = np.diff(t_idx) * dt
            mean_isi = isi_ms.mean()
            std_isi = isi_ms.std(ddof=1) if isi_ms.size > 1 else 0.0
            cv = std_isi / mean_isi if mean_isi > 0 else np.nan
            cv_list.append(cv)
    cv_mean = float(np.nanmean(cv_list)) if cv_list else float("nan")

    # Pairwise spike count correlations (binned)
    bin_ms = 5.0
    bin_steps = max(1, int(round(bin_ms / dt)))
    num_bins = steps // bin_steps
    if num_bins > 1:
        # reshape and sum within bins
        spikes_trim = spikes[: num_bins * bin_steps]
        counts = spikes_trim.reshape(num_bins, bin_steps, N).sum(axis=1).astype(float)
        # z-score across bins for each neuron
        counts -= counts.mean(axis=0, keepdims=True)
        std = counts.std(axis=0, ddof=1, keepdims=True) + 1e-8
        z = counts / std
        C = (z.T @ z) / (num_bins - 1)
        off = C - np.eye(N)
        avg_abs_corr = float(np.mean(np.abs(off)))
    else:
        avg_abs_corr = float("nan")

    # Population stats
    mean_rate = float(rates_hz.mean())
    exc_rate = float(rates_hz[: int(cfg.frac_exc * N)].mean()) if N > 0 else 0.0
    inh_rate = float(rates_hz[int(cfg.frac_exc * N) :].mean()) if N > 0 else 0.0

    return {
        "mean_rate_hz": mean_rate,
        "exc_rate_hz": exc_rate,
        "inh_rate_hz": inh_rate,
        "cv_mean": cv_mean,
        "avg_abs_corr": avg_abs_corr,
    }


def plot_results(
    time_ms: np.ndarray,
    spikes: np.ndarray,
    cfg: LIFNetConfig,
    metrics: dict,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # Raster plot
    ax_raster = fig.add_subplot(gs[0, :])
    t_idx, n_idx = np.nonzero(spikes)
    ax_raster.scatter(time_ms[t_idx], n_idx, s=1, c="#1f77b4")
    ax_raster.set_title("Spike raster")
    ax_raster.set_ylabel("Neuron")

    # Population rate over time (binned)
    ax_rate = fig.add_subplot(gs[1, 0])
    bin_ms = 5.0
    bin_steps = max(1, int(round(bin_ms / cfg.dt_ms)))
    steps = spikes.shape[0]
    num_bins = steps // bin_steps
    if num_bins > 0:
        spikes_trim = spikes[: num_bins * bin_steps]
        binned = spikes_trim.reshape(num_bins, bin_steps, cfg.N).sum(axis=1)
        pop_rate_hz = binned.sum(axis=1) / (cfg.N * (bin_steps * cfg.dt_ms) / 1000.0)
        t_bins = time_ms[: num_bins * bin_steps : bin_steps]
        ax_rate.plot(t_bins, pop_rate_hz, linewidth=1.0)
    ax_rate.set_title("Population firing rate")
    ax_rate.set_xlabel("Time (ms)")
    ax_rate.set_ylabel("Rate (Hz)")

    # Distribution of single-neuron rates
    ax_hist = fig.add_subplot(gs[1, 1])
    duration_s = time_ms[-1] / 1000.0 if time_ms.size else 1.0
    rates = spikes.sum(axis=0) / duration_s
    ax_hist.hist(rates, bins=30, color="#2ca02c")
    ax_hist.set_title("Rate distribution")
    ax_hist.set_xlabel("Hz")

    # Text with metrics
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    msg = (
        f"mean_rate={metrics['mean_rate_hz']:.2f} Hz, exc_rate={metrics['exc_rate_hz']:.2f} Hz, "
        f"inh_rate={metrics['inh_rate_hz']:.2f} Hz, CV={metrics['cv_mean']:.2f}, "
        f"avg|corr|={metrics['avg_abs_corr']:.3f}\n"
        f"N={cfg.N}, frac_exc={cfg.frac_exc:.2f}, p={cfg.p_connect:.2f}, J={cfg.J:.3f}, g={cfg.g:.2f}, "
        f"tau_m={cfg.tau_m_ms} ms, tau_syn={cfg.tau_syn_ms} ms, dt={cfg.dt_ms} ms"
    )
    ax_txt.text(0.01, 0.7, msg, family="monospace")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chaos in recurrent LIF networks (E/I balance)")
    p.add_argument("--N", type=int, default=1000, help="Total neurons")
    p.add_argument("--frac-exc", type=float, default=0.8, help="Fraction excitatory neurons")
    p.add_argument("--p", type=float, default=0.1, help="Connection probability")
    p.add_argument("--J", type=float, default=0.15, help="Excitatory synaptic weight magnitude")
    p.add_argument("--g", type=float, default=4.0, help="Inhibitory strength multiplier (inhib weight = -g*J)")
    p.add_argument("--tau-m", type=float, default=20.0, help="Membrane time constant (ms)")
    p.add_argument("--tau-syn", type=float, default=5.0, help="Synaptic time constant (ms)")
    p.add_argument("--V-rest", type=float, default=-65.0, help="Resting potential (mV)")
    p.add_argument("--V-reset", type=float, default=-65.0, help="Reset potential (mV)")
    p.add_argument("--V-th", type=float, default=-50.0, help="Threshold (mV)")
    p.add_argument("--refractory", type=float, default=2.0, help="Refractory period (ms)")
    p.add_argument("--dt", type=float, default=0.1, help="Time step (ms)")
    p.add_argument("--noise-std", type=float, default=0.5, help="Voltage noise std (mV/√ms)")
    p.add_argument("--I-ext", type=float, default=1.5, help="External drive (arb)")
    p.add_argument("--R-m", type=float, default=10.0, help="Membrane resistance scaling")
    p.add_argument("--duration", type=float, default=2000.0, help="Simulation duration (ms)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save figure")
    p.add_argument("--no-show", action="store_true", help="Do not display plot")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = LIFNetConfig(
        N=args.N,
        frac_exc=args.frac_exc,
        p_connect=args.p,
        J=args.J,
        g=args.g,
        tau_m_ms=args.tau_m,
        tau_syn_ms=args.tau_syn,
        V_rest_mv=args.V_rest,
        V_reset_mv=args.V_reset,
        V_th_mv=args.V_th,
        refractory_ms=args.refractory,
        dt_ms=args.dt,
        noise_std_mv=args.noise_std,
        I_ext=args.I_ext,
        R_m_mohm=args.R_m,
        seed=args.seed,
    )

    time_ms, spikes, V, _ = simulate(cfg, args.duration)
    metrics = compute_metrics(time_ms, spikes, cfg)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_results(time_ms, spikes, cfg, metrics, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




