"""
Spike-Timing Dependent Plasticity (STDP) demo with two LIF neurons connected
by a single excitatory synapse. The synapse generates a postsynaptic current
on presynaptic spikes and updates its weight via a pair-based STDP rule using
pre/post exponential traces.

Run:
    python STDPDemo/stdp_demo.py --duration 2000 --save stdp_demo.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LIFConfig:
    membrane_time_constant_ms: float = 20.0
    membrane_resistance_mohm: float = 10.0
    resting_potential_mv: float = -65.0
    reset_potential_mv: float = -65.0
    threshold_potential_mv: float = -50.0
    refractory_period_ms: float = 5.0
    noise_std_mv: float = 0.5


class LIFNeuron:
    """Stateful LIF neuron with step-wise Euler integration."""

    def __init__(self, config: LIFConfig, dt_ms: float, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.dt = float(dt_ms)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.V = float(config.resting_potential_mv)
        self.refractory_countdown = 0

        # Precompute constants
        self.leak_factor = self.dt / float(config.membrane_time_constant_ms)
        self.noise_scale = float(config.noise_std_mv) * np.sqrt(self.dt)

    def reset_state(self) -> None:
        self.V = float(self.config.resting_potential_mv)
        self.refractory_countdown = 0

    def step(self, input_current: float) -> bool:
        """Advance one time step given input current (arbitrary units).

        Returns True if a spike occurred this step.
        """
        if self.refractory_countdown > 0:
            self.V = float(self.config.reset_potential_mv)
            self.refractory_countdown -= 1
            return False

        dv = (
            (-(self.V - self.config.resting_potential_mv) + self.config.membrane_resistance_mohm * input_current)
            * self.leak_factor
        )
        if self.config.noise_std_mv > 0.0:
            dv += self.noise_scale * self.rng.standard_normal()

        self.V = self.V + dv

        if self.V >= self.config.threshold_potential_mv:
            self.V = float(self.config.reset_potential_mv)
            self.refractory_countdown = int(round(self.config.refractory_period_ms / self.dt))
            return True

        return False


@dataclass
class SynapseConfig:
    tau_syn_ms: float = 5.0
    initial_weight: float = 0.2
    min_weight: float = 0.0
    max_weight: float = 2.0


@dataclass
class STDPConfig:
    tau_plus_ms: float = 20.0   # Pre trace decay
    tau_minus_ms: float = 20.0  # Post trace decay
    a_plus: float = 0.01        # LTP magnitude
    a_minus: float = 0.012      # LTD magnitude


class PlasticSynapse:
    """Excitatory synapse with exponential PSC and pair-based STDP."""

    def __init__(self, syn_cfg: SynapseConfig, stdp_cfg: STDPConfig, dt_ms: float) -> None:
        self.dt = float(dt_ms)
        self.weight = float(syn_cfg.initial_weight)
        self.min_w = float(syn_cfg.min_weight)
        self.max_w = float(syn_cfg.max_weight)

        self.tau_syn = float(syn_cfg.tau_syn_ms)
        self.syn_trace = 0.0  # Generates postsynaptic current I_syn = weight * syn_trace

        self.tau_plus = float(stdp_cfg.tau_plus_ms)
        self.tau_minus = float(stdp_cfg.tau_minus_ms)
        self.a_plus = float(stdp_cfg.a_plus)
        self.a_minus = float(stdp_cfg.a_minus)

        # STDP traces updated on spikes
        self.pre_trace = 0.0
        self.post_trace = 0.0

        # Precompute decay factors
        self.syn_decay = np.exp(-self.dt / self.tau_syn) if self.tau_syn > 0 else 0.0
        self.pre_decay = np.exp(-self.dt / self.tau_plus) if self.tau_plus > 0 else 0.0
        self.post_decay = np.exp(-self.dt / self.tau_minus) if self.tau_minus > 0 else 0.0

    def step_begin(self) -> None:
        # Decay traces each step
        self.syn_trace *= self.syn_decay
        self.pre_trace *= self.pre_decay
        self.post_trace *= self.post_decay

    def on_pre_spike(self) -> None:
        # Generate PSC and update STDP weight (LTP on pre using post trace)
        self.syn_trace += 1.0
        self.pre_trace += 1.0
        delta_w = self.a_plus * self.post_trace
        self.weight = np.clip(self.weight + delta_w, self.min_w, self.max_w)

    def on_post_spike(self) -> None:
        # Update post trace and apply LTD using pre trace
        self.post_trace += 1.0
        delta_w = -self.a_minus * self.pre_trace
        self.weight = np.clip(self.weight + delta_w, self.min_w, self.max_w)

    def compute_syn_current(self) -> float:
        return self.weight * self.syn_trace


def simulate_stdp(
    duration_ms: float,
    dt_ms: float,
    pre_cfg: LIFConfig,
    post_cfg: LIFConfig,
    syn_cfg: SynapseConfig,
    stdp_cfg: STDPConfig,
    pre_background_current: float,
    post_background_current: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate two LIF neurons with a plastic synapse.

    Returns time (ms), V_pre, V_post, weight_trace, pre_spike_idx, post_spike_idx.
    """
    steps = int(round(duration_ms / dt_ms))
    time_ms = np.arange(steps) * dt_ms

    rng = np.random.default_rng(seed)
    pre = LIFNeuron(pre_cfg, dt_ms=dt_ms, rng=rng)
    post = LIFNeuron(post_cfg, dt_ms=dt_ms, rng=rng)
    syn = PlasticSynapse(syn_cfg, stdp_cfg, dt_ms=dt_ms)

    V_pre = np.empty(steps, dtype=float)
    V_post = np.empty(steps, dtype=float)
    W = np.empty(steps, dtype=float)

    pre_spikes = []
    post_spikes = []

    for t in range(steps):
        syn.step_begin()

        # Pre neuron: background current only
        pre_spike = pre.step(pre_background_current)
        if pre_spike:
            pre_spikes.append(t)
            syn.on_pre_spike()

        # Post neuron: background + synaptic current from pre
        I_post = post_background_current + syn.compute_syn_current()
        post_spike = post.step(I_post)
        if post_spike:
            post_spikes.append(t)
            syn.on_post_spike()

        V_pre[t] = pre.V
        V_post[t] = post.V
        W[t] = syn.weight

    return time_ms, V_pre, V_post, W, np.array(pre_spikes, dtype=int), np.array(post_spikes, dtype=int)


def plot_results(
    time_ms: np.ndarray,
    V_pre: np.ndarray,
    V_post: np.ndarray,
    W: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    v_threshold_mv: float,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    # Voltages
    ax_v.plot(time_ms, V_pre, color="#1f77b4", label="Pre V")
    ax_v.plot(time_ms, V_post, color="#ff7f0e", label="Post V")
    ax_v.axhline(v_threshold_mv, color="#d62728", linestyle="--", linewidth=1.0, label="Threshold")
    if pre_spikes.size:
        ax_v.scatter(time_ms[pre_spikes], V_pre[pre_spikes], s=16, color="#1f77b4", marker="o")
    if post_spikes.size:
        ax_v.scatter(time_ms[post_spikes], V_post[post_spikes], s=16, color="#ff7f0e", marker="o")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Two LIF Neurons with STDP")
    ax_v.legend(loc="upper right")

    # Weight
    ax_w.plot(time_ms, W, color="#2ca02c")
    ax_w.set_xlabel("Time (ms)")
    ax_w.set_ylabel("Synaptic weight")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STDP demo with two LIF neurons")
    p.add_argument("--duration", type=float, default=2000.0, help="Simulation duration (ms)")
    p.add_argument("--dt", type=float, default=0.1, help="Time step (ms)")
    # LIF
    p.add_argument("--pre-current", type=float, default=1.8, help="Pre neuron background current (arb)")
    p.add_argument("--post-current", type=float, default=0.6, help="Post neuron background current (arb)")
    p.add_argument("--noise-std", type=float, default=0.5, help="Noise std (mV/√ms) for both neurons")
    p.add_argument("--v-threshold", type=float, default=-50.0, help="Threshold potential (mV)")
    # Synapse
    p.add_argument("--w-init", type=float, default=0.2, help="Initial synaptic weight")
    p.add_argument("--w-min", type=float, default=0.0, help="Minimum synaptic weight")
    p.add_argument("--w-max", type=float, default=2.0, help="Maximum synaptic weight")
    p.add_argument("--tau-syn", type=float, default=5.0, help="Synaptic current time constant (ms)")
    # STDP
    p.add_argument("--tau-plus", type=float, default=20.0, help="Pre trace time constant (ms)")
    p.add_argument("--tau-minus", type=float, default=20.0, help="Post trace time constant (ms)")
    p.add_argument("--a-plus", type=float, default=0.01, help="LTP magnitude")
    p.add_argument("--a-minus", type=float, default=0.012, help="LTD magnitude")
    # Misc
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Save plot path")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    pre_cfg = LIFConfig(noise_std_mv=args.noise_std, threshold_potential_mv=args.v_threshold)
    post_cfg = LIFConfig(noise_std_mv=args.noise_std, threshold_potential_mv=args.v_threshold)
    syn_cfg = SynapseConfig(tau_syn_ms=args.tau_syn, initial_weight=args.w_init, min_weight=args.w_min, max_weight=args.w_max)
    stdp_cfg = STDPConfig(tau_plus_ms=args.tau_plus, tau_minus_ms=args.tau_minus, a_plus=args.a_plus, a_minus=args.a_minus)

    time_ms, V_pre, V_post, W, pre_spk, post_spk = simulate_stdp(
        duration_ms=args.duration,
        dt_ms=args.dt,
        pre_cfg=pre_cfg,
        post_cfg=post_cfg,
        syn_cfg=syn_cfg,
        stdp_cfg=stdp_cfg,
        pre_background_current=args.pre_current,
        post_background_current=args.post_current,
        seed=args.seed,
    )

    plot_results(
        time_ms=time_ms,
        V_pre=V_pre,
        V_post=V_post,
        W=W,
        pre_spikes=pre_spk,
        post_spikes=post_spk,
        v_threshold_mv=pre_cfg.threshold_potential_mv,
        save_path=args.save,
        show=(not args.no_show),
    )

    # Print quick stats
    dur_s = time_ms[-1] / 1000.0 if time_ms.size else 0.0
    pre_rate = (pre_spk.size / dur_s) if dur_s > 0 else 0.0
    post_rate = (post_spk.size / dur_s) if dur_s > 0 else 0.0
    print("Stats:")
    print(f"  Pre spikes: {pre_spk.size}, rate: {pre_rate:.2f} Hz")
    print(f"  Post spikes: {post_spk.size}, rate: {post_rate:.2f} Hz")
    print(f"  Weight: start {W[0]:.3f} → end {W[-1]:.3f}")


if __name__ == "__main__":
    main()


