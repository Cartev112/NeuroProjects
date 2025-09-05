"""
Reinforcement-Modulated STDP in a Recurrent Spiking Network

Small recurrent LIF network with eligibility-trace STDP modulated by a
global dopamine-like reward signal. The task is to regulate an output
population's firing rate toward a target that depends on context, using
only local STDP and a scalar reinforcement signal.

Run example:
  python RSTDPDemo/rstdp_demo.py --N 120 --episodes 50 --episode-ms 1500 --save rstdp.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class NetConfig:
    N: int = 120
    frac_exc: float = 0.8
    p_connect: float = 0.1
    J_init: float = 0.05
    g_inh: float = 4.0  # inhibitory stronger than excitatory
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    V_rest_mv: float = -65.0
    V_reset_mv: float = -65.0
    V_th_mv: float = -50.0
    refractory_ms: float = 2.0
    dt_ms: float = 0.5
    noise_std_mv: float = 0.3
    I_ext: float = 1.2
    R_m_mohm: float = 10.0


@dataclass
class PlasticityConfig:
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    tau_e_ms: float = 200.0  # eligibility decay
    A_plus: float = 0.01
    A_minus: float = -0.012
    eta: float = 1e-3  # learning rate scaling of dopamine-modulated eligibility
    w_exc_min: float = 0.0
    w_exc_max: float = 0.5
    w_inh_min: float = -1.0
    w_inh_max: float = 0.0
    row_norm_max: float = 2.0  # optional row-norm soft cap for stability


@dataclass
class TaskConfig:
    episode_ms: float = 1500.0
    episodes: int = 50
    context_switch_ms: float = 750.0  # two halves per episode for two contexts
    out_frac: float = 0.2  # fraction of neurons in readout population
    target_low_hz: float = 5.0
    target_high_hz: float = 25.0
    tau_readout_ms: float = 50.0  # low-pass for rate estimate
    dopamine_gain: float = 1.0
    reward_baseline_tau: float = 500.0  # baseline for advantage


class RecurrentLIFRSTDP:
    def __init__(self, net_cfg: NetConfig, pl_cfg: PlasticityConfig, task_cfg: TaskConfig, seed: Optional[int]) -> None:
        self.net_cfg = net_cfg
        self.pl_cfg = pl_cfg
        self.task_cfg = task_cfg
        self.rng = np.random.default_rng(seed)

        N = net_cfg.N
        self.NE = int(round(net_cfg.frac_exc * N))
        self.NI = N - self.NE
        self.exc_mask = np.zeros(N, dtype=bool)
        self.exc_mask[: self.NE] = True
        self.inh_mask = ~self.exc_mask

        # State variables
        self.V = np.full(N, net_cfg.V_rest_mv, dtype=float)
        self.refr = np.zeros(N, dtype=int)
        self.s = np.zeros(N, dtype=float)
        self.syn_decay = np.exp(-net_cfg.dt_ms / net_cfg.tau_syn_ms) if net_cfg.tau_syn_ms > 0 else 0.0
        self.leak = net_cfg.dt_ms / net_cfg.tau_m_ms
        self.noise_scale = net_cfg.noise_std_mv * np.sqrt(net_cfg.dt_ms)
        self.refr_steps = int(round(net_cfg.refractory_ms / net_cfg.dt_ms))

        # Connectivity
        self.W = np.zeros((N, N), dtype=float)
        self.mask = np.zeros((N, N), dtype=bool)
        self.outgoing: List[np.ndarray] = []
        self.incoming: List[np.ndarray] = []
        self._build_connectivity()

        # Plasticity traces and eligibilities
        self.pre_tr = np.zeros(N, dtype=float)
        self.post_tr = np.zeros(N, dtype=float)
        self.e = np.zeros((N, N), dtype=float)
        self.decay_pre = np.exp(-net_cfg.dt_ms / pl_cfg.tau_pre_ms) if pl_cfg.tau_pre_ms > 0 else 0.0
        self.decay_post = np.exp(-net_cfg.dt_ms / pl_cfg.tau_post_ms) if pl_cfg.tau_post_ms > 0 else 0.0
        self.decay_e = np.exp(-net_cfg.dt_ms / pl_cfg.tau_e_ms) if pl_cfg.tau_e_ms > 0 else 0.0

        # Readout subset and filters
        out_count = max(1, int(round(task_cfg.out_frac * N)))
        self.out_idx = np.arange(N - out_count, N, dtype=int)  # last fraction as output pop
        self.rate_lp = 0.0
        self.rate_decay = np.exp(-net_cfg.dt_ms / task_cfg.tau_readout_ms)
        self.reward_baseline = 0.0
        self.baseline_decay = np.exp(-net_cfg.dt_ms / task_cfg.reward_baseline_tau)

    def _build_connectivity(self) -> None:
        cfg = self.net_cfg
        N = cfg.N
        p = cfg.p_connect
        W = self.W
        mask = self.mask
        self.outgoing = []
        self.incoming = [list() for _ in range(N)]

        conn = self.rng.random((N, N)) < p
        np.fill_diagonal(conn, False)
        mask[:, :] = conn
        # Initialize E and I weights separately
        for i in range(N):
            tgt = np.nonzero(conn[i])[0]
            self.outgoing.append(tgt)
            for j in tgt:
                self.incoming[j].append(i)
            if i < self.NE:
                W[i, tgt] = self.net_cfg.J_init * self.rng.uniform(0.5, 1.0, size=tgt.size)
            else:
                W[i, tgt] = -self.net_cfg.g_inh * self.net_cfg.J_init * self.rng.uniform(0.5, 1.0, size=tgt.size)
        # Convert incoming lists to arrays
        self.incoming = [np.asarray(lst, dtype=int) if len(lst) else np.empty(0, dtype=int) for lst in self.incoming]

    def _apply_spikes(self, spikes_prev: np.ndarray) -> None:
        # Decay synapse
        self.s *= self.syn_decay
        # Add currents from spiking presynaptic neurons
        spk_idx = np.nonzero(spikes_prev)[0]
        for i in spk_idx:
            tgt = self.outgoing[i]
            if tgt.size:
                self.s[tgt] += self.W[i, tgt]

    def _lif_step(self, external_drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.net_cfg
        dV = (-(self.V - cfg.V_rest_mv) + cfg.R_m_mohm * (cfg.I_ext + external_drive + self.s)) * self.leak
        if cfg.noise_std_mv > 0.0:
            dV += self.noise_scale * self.rng.standard_normal(cfg.N)
        refr_mask = self.refr > 0
        self.V[~refr_mask] += dV[~refr_mask]
        self.V[refr_mask] = cfg.V_reset_mv
        self.refr[refr_mask] -= 1
        spikes = (self.V >= cfg.V_th_mv) & (~refr_mask)
        if np.any(spikes):
            self.V[spikes] = cfg.V_reset_mv
            self.refr[spikes] = self.refr_steps
        return spikes.astype(np.uint8), refr_mask

    def _plasticity_step(self, spikes: np.ndarray, dopamine: float) -> None:
        pc = self.pl_cfg
        # Decay traces and eligibilities
        self.pre_tr *= self.decay_pre
        self.post_tr *= self.decay_post
        self.e *= self.decay_e

        spk_idx = np.nonzero(spikes)[0]
        if spk_idx.size:
            # Update pre/post traces
            self.pre_tr[spk_idx] += 1.0
            self.post_tr[spk_idx] += 1.0
            # Post spikes: potentiation proportional to pre traces
            for j in spk_idx:
                pre_list = self.incoming[j]
                if pre_list.size:
                    self.e[pre_list, j] += pc.A_plus * self.pre_tr[pre_list]
            # Pre spikes: depression proportional to post traces
            for i in spk_idx:
                tgt = self.outgoing[i]
                if tgt.size:
                    self.e[i, tgt] += pc.A_minus * self.post_tr[tgt]

        # Dopamine-modulated weight update
        if dopamine != 0.0:
            dW = pc.eta * dopamine * self.e
            # Apply only to existing synapses
            self.W[self.mask] += dW[self.mask]
            # Row-wise constraints (E>=0, I<=0) and bounds
            exc_rows = self.exc_mask
            inh_rows = self.inh_mask
            # Excitatory rows
            W_exc = self.W[exc_rows]
            if W_exc.size:
                np.clip(W_exc, pc.w_exc_min, pc.w_exc_max, out=W_exc)
                self.W[exc_rows] = W_exc
            # Inhibitory rows
            W_inh = self.W[inh_rows]
            if W_inh.size:
                np.clip(W_inh, pc.w_inh_min, pc.w_inh_max, out=W_inh)
                self.W[inh_rows] = W_inh
            # Soft row-norm cap
            if pc.row_norm_max is not None and pc.row_norm_max > 0:
                norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8
                scale = np.minimum(1.0, pc.row_norm_max / norms)
                self.W *= scale

    def _context_drive(self, t_ms: float, ctx: int) -> np.ndarray:
        # Provide weak context-specific drive to halves of excitatory population
        drive = np.zeros(self.net_cfg.N, dtype=float)
        if ctx == 0:
            idx = np.arange(0, self.NE // 2)
        else:
            idx = np.arange(self.NE // 2, self.NE)
        drive[idx] += 0.6
        return drive

    def _update_readout_and_reward(self, spikes: np.ndarray, target_hz: float) -> Tuple[float, float, float]:
        # Low-pass filtered output rate (Hz) for output population
        dt = self.net_cfg.dt_ms
        k = 1.0 - self.rate_decay
        spk_out = float(spikes[self.out_idx].sum())
        inst_rate_hz = (spk_out / self.out_idx.size) * (1000.0 / dt)
        self.rate_lp = self.rate_decay * self.rate_lp + k * inst_rate_hz
        err = abs(self.rate_lp - target_hz)
        # Reward as error improvement (delta error); positive when error decreases
        if not hasattr(self, "prev_err"):
            self.prev_err = err
        reward = self.prev_err - err
        self.prev_err = err
        # Advantage (subtract baseline)
        self.reward_baseline = self.baseline_decay * self.reward_baseline + (1.0 - self.baseline_decay) * reward
        advantage = reward - self.reward_baseline
        dopamine = self.task_cfg.dopamine_gain * advantage
        return self.rate_lp, err, dopamine

    def train(self) -> Tuple[dict, dict, dict]:
        nc, tc = self.net_cfg, self.task_cfg
        steps_per_ep = int(round(tc.episode_ms / nc.dt_ms))
        switch_step = int(round(tc.context_switch_ms / nc.dt_ms))
        rewards_ep = []
        err_ep = []
        last_raster_spikes = None
        last_time_ms = None
        last_output_trace = []
        last_target_trace = []

        for ep in range(tc.episodes):
            # Reset some traces for plotting per episode
            output_trace = []
            target_trace = []
            ep_reward_sum = 0.0
            # Reset error history for Dopamine advantage smoothing carry across episodes
            self.prev_err = None  # initialize in first step

            spikes_prev = np.zeros(nc.N, dtype=np.uint8)
            spikes_store = np.zeros((steps_per_ep, nc.N), dtype=np.uint8)
            time_ms = np.arange(steps_per_ep) * nc.dt_ms
            for t in range(steps_per_ep):
                ctx = 0 if t < switch_step else 1
                target_hz = tc.target_low_hz if ctx == 0 else tc.target_high_hz
                drive = self._context_drive(time_ms[t], ctx)

                # Synaptic current update from previous spikes
                self._apply_spikes(spikes_prev)
                # LIF step
                spikes, _ = self._lif_step(drive)
                # Reward & dopamine
                out_rate, err, dop = self._update_readout_and_reward(spikes, target_hz)
                ep_reward_sum += dop
                # Plasticity
                self._plasticity_step(spikes, dopamine=dop)

                spikes_store[t] = spikes
                output_trace.append(out_rate)
                target_trace.append(target_hz)
                spikes_prev = spikes

            rewards_ep.append(ep_reward_sum / steps_per_ep)
            err_ep.append(np.mean(np.abs(np.asarray(output_trace) - np.asarray(target_trace))))

            # Keep last episode for plotting
            last_raster_spikes = spikes_store
            last_time_ms = time_ms
            last_output_trace = output_trace
            last_target_trace = target_trace

        logs = {
            "reward_per_ep": np.asarray(rewards_ep),
            "err_per_ep": np.asarray(err_ep),
        }
        traces = {
            "time_ms": last_time_ms,
            "spikes": last_raster_spikes,
            "output_rate": np.asarray(last_output_trace),
            "target_rate": np.asarray(last_target_trace),
        }
        stats = {
            "W_mean": float(self.W[self.mask].mean()) if np.any(self.mask) else 0.0,
            "W_std": float(self.W[self.mask].std()) if np.any(self.mask) else 0.0,
        }
        return logs, traces, stats


def plot_results(logs: dict, traces: dict, stats: dict, save_path: Optional[str], show: bool) -> None:
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # Raster of last episode
    ax_raster = fig.add_subplot(gs[0, :])
    t_idx, n_idx = np.nonzero(traces["spikes"])
    ax_raster.scatter(traces["time_ms"][t_idx], n_idx, s=1, c="#1f77b4")
    ax_raster.set_title("Spike raster (last episode)")
    ax_raster.set_ylabel("Neuron")

    # Output vs target (last episode)
    ax_out = fig.add_subplot(gs[1, 0])
    ax_out.plot(traces["time_ms"], traces["output_rate"], label="Output rate")
    ax_out.plot(traces["time_ms"], traces["target_rate"], label="Target", linestyle="--")
    ax_out.set_title("Output rate vs target")
    ax_out.set_xlabel("Time (ms)")
    ax_out.set_ylabel("Hz")
    ax_out.legend()

    # Reward and error over episodes
    ax_rew = fig.add_subplot(gs[1, 1])
    ax_rew.plot(logs["reward_per_ep"], label="Avg dopamine (adv)")
    ax_rew.plot(logs["err_per_ep"], label="Mean |error|", linestyle=":")
    ax_rew.set_title("Learning curves across episodes")
    ax_rew.set_xlabel("Episode")
    ax_rew.legend()

    # Weight stats text
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    ax_txt.text(0.01, 0.8, f"W mean={stats['W_mean']:.4f}, std={stats['W_std']:.4f}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reinforcement-modulated STDP in recurrent LIF network")
    # Net
    p.add_argument("--N", type=int, default=120, help="Total neurons")
    p.add_argument("--frac-exc", type=float, default=0.8, help="Fraction excitatory")
    p.add_argument("--p", type=float, default=0.1, help="Connection probability")
    p.add_argument("--J-init", type=float, default=0.05, help="Initial excitatory weight scale")
    p.add_argument("--g", type=float, default=4.0, help="Inhibitory strength multiplier")
    p.add_argument("--tau-m", type=float, default=20.0, help="Membrane tau (ms)")
    p.add_argument("--tau-syn", type=float, default=5.0, help="Synaptic tau (ms)")
    p.add_argument("--dt", type=float, default=0.5, help="Time step (ms)")
    p.add_argument("--noise-std", type=float, default=0.3, help="Voltage noise std (mV/√ms)")
    p.add_argument("--I-ext", type=float, default=1.2, help="External drive (arb)")
    # Plasticity
    p.add_argument("--tau-pre", type=float, default=20.0, help="Pre trace tau (ms)")
    p.add_argument("--tau-post", type=float, default=20.0, help="Post trace tau (ms)")
    p.add_argument("--tau-e", type=float, default=200.0, help="Eligibility tau (ms)")
    p.add_argument("--A-plus", type=float, default=0.01, help="STDP LTP magnitude")
    p.add_argument("--A-minus", type=float, default=-0.012, help="STDP LTD magnitude")
    p.add_argument("--eta", type=float, default=1e-3, help="Learning rate for dopamine-modulated e")
    p.add_argument("--w-exc-max", type=float, default=0.5, help="Max excitatory weight")
    p.add_argument("--w-inh-min", type=float, default=-1.0, help="Min inhibitory weight")
    p.add_argument("--row-norm-max", type=float, default=2.0, help="Row-norm soft cap")
    # Task
    p.add_argument("--episodes", type=int, default=50, help="Training episodes")
    p.add_argument("--episode-ms", type=float, default=1500.0, help="Episode duration (ms)")
    p.add_argument("--switch-ms", type=float, default=750.0, help="Context switch time (ms)")
    p.add_argument("--out-frac", type=float, default=0.2, help="Output population fraction")
    p.add_argument("--target-low", type=float, default=5.0, help="Low target rate (Hz)")
    p.add_argument("--target-high", type=float, default=25.0, help="High target rate (Hz)")
    p.add_argument("--tau-readout", type=float, default=50.0, help="Readout LPF tau (ms)")
    p.add_argument("--dopamine-gain", type=float, default=1.0, help="Dopamine gain on advantage")
    p.add_argument("--reward-baseline-tau", type=float, default=500.0, help="Baseline tau (ms)")
    # Misc
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--save", type=str, default=None, help="Path to save figure")
    p.add_argument("--no-show", action="store_true", help="Do not display plot")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    net_cfg = NetConfig(
        N=args.N,
        frac_exc=args.frac_exc,
        p_connect=args.p,
        J_init=args.J_init,
        g_inh=args.g,
        tau_m_ms=args.tau_m,
        tau_syn_ms=args.tau_syn,
        dt_ms=args.dt,
        noise_std_mv=args.noise_std,
        I_ext=args.I_ext,
    )
    pl_cfg = PlasticityConfig(
        tau_pre_ms=args.tau_pre,
        tau_post_ms=args.tau_post,
        tau_e_ms=args.tau_e,
        A_plus=args.A_plus,
        A_minus=args.A_minus,
        eta=args.eta,
        w_exc_max=args.w_exc_max,
        w_inh_min=args.w_inh_min,
        row_norm_max=args.row_norm_max,
    )
    task_cfg = TaskConfig(
        episode_ms=args.episode_ms,
        episodes=args.episodes,
        context_switch_ms=args.switch_ms,
        out_frac=args.out_frac,
        target_low_hz=args.target_low,
        target_high_hz=args.target_high,
        tau_readout_ms=args.tau_readout,
        dopamine_gain=args.dopamine_gain,
        reward_baseline_tau=args.reward_baseline_tau,
    )

    model = RecurrentLIFRSTDP(net_cfg, pl_cfg, task_cfg, seed=args.seed)
    logs, traces, stats = model.train()
    plot_results(logs, traces, stats, save_path=args.save, show=(not args.no_show))


if __name__ == "__main__":
    main()




