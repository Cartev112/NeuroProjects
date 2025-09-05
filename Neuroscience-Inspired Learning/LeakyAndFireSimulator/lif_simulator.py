"""
Leaky Integrate-and-Fire (LIF) neuron simulator with adjustable input current,
additive Gaussian noise, and an absolute refractory period. Includes a CLI and
matplotlib visualization.

Usage example:
    python lif_simulator.py --duration 500 --input-current 1.5 \
        --save lif_plot.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LIFConfig:
    """Configuration for a single LIF neuron.

    All time parameters are in milliseconds (ms) and voltages in millivolts (mV).
    The input current is in arbitrary units and is scaled by membrane resistance.
    """

    membrane_time_constant_ms: float = 20.0  # tau_m
    membrane_resistance_mohm: float = 10.0   # R (mV per unit input)
    resting_potential_mv: float = -65.0      # V_rest
    reset_potential_mv: float = -65.0        # V_reset
    threshold_potential_mv: float = -50.0    # V_threshold
    refractory_period_ms: float = 5.0        # absolute refractory duration
    time_step_ms: float = 0.1                # dt
    noise_std_mv: float = 0.5                # additive noise std per sqrt(ms)


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron simulator.

    Discrete-time Euler integration of:
        dV/dt = (-(V - V_rest) + R * I_in) / tau_m + sigma * xi(t)
    where xi(t) is white noise with std scaled by sqrt(dt).
    """

    def __init__(self, config: LIFConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute constants in ms domain
        self.dt = float(config.time_step_ms)
        self.tau_m = float(config.membrane_time_constant_ms)
        self.R = float(config.membrane_resistance_mohm)
        self.V_rest = float(config.resting_potential_mv)
        self.V_reset = float(config.reset_potential_mv)
        self.V_th = float(config.threshold_potential_mv)
        self.refractory_steps = int(round(config.refractory_period_ms / self.dt))
        self.noise_std = float(config.noise_std_mv)

    def simulate(
        self,
        duration_ms: float,
        input_current: float | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the simulation.

        Args:
            duration_ms: total simulation time in milliseconds.
            input_current: constant input value or a length-T array (arbitrary units).

        Returns:
            time_ms: shape (T,) time vector in ms
            membrane_potential_mv: shape (T,) membrane potential trace in mV
            spike_indices: shape (S,) indices where spikes occurred
            input_current_array: shape (T,) input current used per step
        """
        total_steps = int(round(duration_ms / self.dt))
        time_ms = np.arange(total_steps) * self.dt

        if np.isscalar(input_current):
            I = np.full(total_steps, float(input_current), dtype=float)
        else:
            I = np.asarray(input_current, dtype=float)
            if I.shape[0] != total_steps:
                raise ValueError(
                    f"input_current length {I.shape[0]} does not match steps {total_steps}"
                )

        V = np.empty(total_steps, dtype=float)
        V[0] = self.V_rest

        spike_indices_list = []
        refractory_countdown = 0

        # Precompute integration factor for efficiency
        leak_factor = self.dt / self.tau_m
        noise_scale = self.noise_std * np.sqrt(self.dt)

        for t in range(1, total_steps):
            if refractory_countdown > 0:
                V[t] = self.V_reset
                refractory_countdown -= 1
                continue

            # Euler step of LIF dynamics with noise
            dv = (
                (-(V[t - 1] - self.V_rest) + self.R * I[t - 1]) * leak_factor
            )
            if self.noise_std > 0.0:
                dv += noise_scale * self.rng.standard_normal()

            V[t] = V[t - 1] + dv

            # Spike condition
            if V[t] >= self.V_th:
                spike_indices_list.append(t)
                V[t] = self.V_reset
                refractory_countdown = self.refractory_steps

        spike_indices = np.array(spike_indices_list, dtype=int)
        return time_ms, V, spike_indices, I

    @staticmethod
    def compute_spike_metrics(time_ms: np.ndarray, spike_indices: np.ndarray) -> dict:
        """Compute simple spike train metrics: count, rate (Hz), and ISI stats."""
        duration_s = (time_ms[-1] - time_ms[0]) / 1000.0 if time_ms.size > 1 else 0.0
        spike_count = int(spike_indices.size)
        firing_rate_hz = (spike_count / duration_s) if duration_s > 0 else 0.0
        if spike_count >= 2:
            isi_ms = np.diff(time_ms[spike_indices])
            isi_stats = {
                "isi_mean_ms": float(isi_ms.mean()),
                "isi_std_ms": float(isi_ms.std(ddof=1)) if isi_ms.size > 1 else 0.0,
                "isi_min_ms": float(isi_ms.min()),
                "isi_max_ms": float(isi_ms.max()),
            }
        else:
            isi_stats = {
                "isi_mean_ms": float("nan"),
                "isi_std_ms": float("nan"),
                "isi_min_ms": float("nan"),
                "isi_max_ms": float("nan"),
            }
        return {
            "spike_count": spike_count,
            "firing_rate_hz": float(firing_rate_hz),
            **isi_stats,
        }


def plot_simulation(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    spike_indices: np.ndarray,
    input_current: np.ndarray,
    threshold_mv: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot the membrane potential, spike times, and input current."""
    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    ax_v.plot(time_ms, voltage_mv, color="#1f77b4", linewidth=1.5, label="Membrane V")
    ax_v.axhline(threshold_mv, color="#d62728", linestyle="--", linewidth=1.0, label="Threshold")
    if spike_indices.size > 0:
        ax_v.scatter(time_ms[spike_indices], voltage_mv[spike_indices], s=18, color="#d62728", zorder=3, label="Spikes")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Leaky Integrate-and-Fire Neuron")
    ax_v.legend(loc="upper right")

    ax_i.plot(time_ms, input_current, color="#2ca02c", linewidth=1.2)
    ax_i.set_xlabel("Time (ms)")
    ax_i.set_ylabel("Input (arb)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leaky Integrate-and-Fire (LIF) simulator")
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration in ms (default: 1000)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step in ms (default: 0.1)")
    # Input protocols
    parser.add_argument("--protocol", type=str, choices=["constant", "step", "sine"], default="constant", help="Input current protocol")
    parser.add_argument("--input-current", type=float, default=1.5, help="Constant current (for constant protocol)")
    parser.add_argument("--step-onset", type=float, default=200.0, help="Step onset time in ms (for step protocol)")
    parser.add_argument("--step-current", type=float, default=2.0, help="Current during step (for step protocol)")
    parser.add_argument("--sine-amplitude", type=float, default=1.0, help="Sine protocol amplitude")
    parser.add_argument("--sine-frequency", type=float, default=10.0, help="Sine protocol frequency in Hz")
    parser.add_argument("--sine-offset", type=float, default=0.0, help="Sine protocol DC offset")
    parser.add_argument("--noise-std", type=float, default=0.5, help="Noise std in mV per sqrt(ms) (default: 0.5)")
    parser.add_argument("--tau-m", type=float, default=20.0, help="Membrane time constant tau_m in ms (default: 20)")
    parser.add_argument("--resistance", type=float, default=10.0, help="Membrane resistance R scaling (default: 10)")
    parser.add_argument("--v-rest", type=float, default=-65.0, help="Resting potential V_rest in mV (default: -65)")
    parser.add_argument("--v-reset", type=float, default=-65.0, help="Reset potential V_reset in mV (default: -65)")
    parser.add_argument("--v-threshold", type=float, default=-50.0, help="Threshold potential V_th in mV (default: -50)")
    parser.add_argument("--refractory", type=float, default=5.0, help="Absolute refractory period in ms (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for noise (default: None)")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot image (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    config = LIFConfig(
        membrane_time_constant_ms=args.tau_m,
        membrane_resistance_mohm=args.resistance,
        resting_potential_mv=args.v_rest,
        reset_potential_mv=args.v_reset,
        threshold_potential_mv=args.v_threshold,
        refractory_period_ms=args.refractory,
        time_step_ms=args.dt,
        noise_std_mv=args.noise_std,
    )

    neuron = LIFNeuron(config=config, rng=rng)

    # Build input current according to protocol
    total_steps = int(round(args.duration / args.dt))
    time_ms_proto = np.arange(total_steps) * args.dt
    if args.protocol == "constant":
        input_array = float(args.input_current)
    elif args.protocol == "step":
        input_array = np.full(total_steps, float(args.input_current), dtype=float)
        onset_idx = max(0, int(round(args.step_onset / args.dt)))
        input_array[onset_idx:] = float(args.step_current)
    else:  # sine
        t_s = time_ms_proto / 1000.0
        input_array = args.sine_offset + args.sine_amplitude * np.sin(2 * np.pi * args.sine_frequency * t_s)

    time_ms, V_mv, spikes, I = neuron.simulate(duration_ms=args.duration, input_current=input_array)

    plot_simulation(
        time_ms=time_ms,
        voltage_mv=V_mv,
        spike_indices=spikes,
        input_current=I,
        threshold_mv=config.threshold_potential_mv,
        save_path=args.save,
        show=(not args.no_show),
    )

    # Print basic metrics
    metrics = LIFNeuron.compute_spike_metrics(time_ms, spikes)
    print("Spike metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


