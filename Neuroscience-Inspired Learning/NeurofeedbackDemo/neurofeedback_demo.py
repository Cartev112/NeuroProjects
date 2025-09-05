"""
Neurofeedback Demo: Generative Visual + Audio Feedback on Signal Improvement

Concept:
- Ingest a 1D signal (simulated or from CSV) in real-time-like steps
- Smooth it, compute advantage (improvement over a running baseline)
- When advantage > 0, emit visual blooms and audio tones scaled by reward

Run examples:
  Simulated signal:
    python NeurofeedbackDemo/neurofeedback_demo.py --duration 120 --no-save
  CSV input (expects a column 'value') at ~20 Hz:
    python NeurofeedbackDemo/neurofeedback_demo.py --csv path/to/data.csv --column value --fs 20
"""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

try:
    import winsound  # Windows-only; optional
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False


@dataclass
class SignalConfig:
    source_csv: Optional[str] = None
    csv_column: str = "value"
    csv_fs: float = 20.0  # Hz
    duration_s: float = 120.0
    dt_s: float = 0.05    # step size for simulated mode (20 Hz)
    seed: int = 0


@dataclass
class FeedbackConfig:
    smooth_tau_s: float = 1.0        # EMA for signal smoothing
    baseline_tau_s: float = 5.0      # EMA for baseline for advantage
    reward_gain: float = 1.0         # scales visual/audio intensity
    reward_threshold: float = 0.0    # only > threshold triggers events
    max_blooms: int = 128
    ring_decay_s: float = 2.0        # how quickly rings fade
    palette: Tuple[str, ...] = ("#5DA5DA", "#60BD68", "#F17CB0", "#F15854", "#B2912F", "#B276B2")
    audio_min_freq: int = 300
    audio_max_freq: int = 1400
    audio_min_ms: int = 50
    audio_max_ms: int = 200
    no_audio: bool = False


class SignalSource:
    def __init__(self, cfg: SignalConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._csv_iter: Optional[Iterable[float]] = None
        self._sim_state = 0.0
        if cfg.source_csv is not None:
            self._csv_iter = self._load_csv_stream(Path(cfg.source_csv), cfg.csv_column)

    def _load_csv_stream(self, path: Path, column: str) -> Iterable[float]:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        def generator() -> Iterable[float]:
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        yield float(row[column])
                    except Exception:
                        continue
        return generator()

    def step(self) -> Tuple[float, float]:
        """Return (value, dt_s) for this step."""
        if self._csv_iter is not None:
            # Real-time-ish pacing derived from csv_fs
            dt = 1.0 / max(1e-3, self.cfg.csv_fs)
            try:
                val = next(self._csv_iter)  # type: ignore[arg-type]
            except StopIteration:
                val = np.nan
            return val, dt
        # Simulated: noisy random walk with occasional favorable drifts
        dt = self.cfg.dt_s
        drift = 0.02 * self.rng.standard_normal()
        # Occasional positive bump episodes
        if self.rng.random() < 0.02:
            drift += 0.2 * self.rng.random()
        self._sim_state = 0.98 * self._sim_state + drift
        val = self._sim_state + 0.05 * self.rng.standard_normal()
        return float(val), dt


class AudioEngine:
    def __init__(self, fb: FeedbackConfig) -> None:
        self.fb = fb
        self._lock = threading.Lock()

    def play(self, magnitude: float) -> None:
        if self.fb.no_audio:
            return
        if not WINSOUND_AVAILABLE:
            return
        # Map magnitude to frequency and duration
        mag = float(np.clip(magnitude, 0.0, 1.0))
        freq = int(self.fb.audio_min_freq + mag * (self.fb.audio_max_freq - self.fb.audio_min_freq))
        dur = int(self.fb.audio_min_ms + mag * (self.fb.audio_max_ms - self.fb.audio_min_ms))
        def run():
            try:
                winsound.Beep(freq, dur)  # blocks within thread
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()


class VisualEngine:
    def __init__(self, fb: FeedbackConfig, history_len: int = 400) -> None:
        self.fb = fb
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        self.ax.set_facecolor("#111111")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Rolling plot for signal and baseline
        self.ax2 = self.ax.inset_axes([0.08, 0.7, 0.4, 0.25])
        self.ax2.set_facecolor("#000000")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.hist_t: Deque[float] = deque(maxlen=history_len)
        self.hist_s: Deque[float] = deque(maxlen=history_len)
        self.hist_b: Deque[float] = deque(maxlen=history_len)
        (self.line_s,) = self.ax2.plot([], [], color="#5DA5DA", linewidth=1.5, label="signal")
        (self.line_b,) = self.ax2.plot([], [], color="#AAAAAA", linewidth=1.0, linestyle="--", label="baseline")
        self.ax2.legend(loc="upper right", fontsize=8, facecolor="#111111")
        # Bloom storage: list of (x, y, start_t, hue_idx)
        self._blooms: Deque[Tuple[float, float, float, int]] = deque(maxlen=fb.max_blooms)

    def emit_bloom(self, now_s: float, intensity: float) -> None:
        # Place bloom near a golden-spiral trajectory for aesthetics
        u = np.random.random()
        v = np.random.random()
        r = (0.15 + 0.35 * (u ** 0.6)) * intensity
        theta = 2 * math.pi * v * (1.0 + 0.618)
        cx = 0.5 + r * math.cos(theta)
        cy = 0.5 + r * math.sin(theta)
        hue_idx = int(np.random.randint(0, len(self.fb.palette)))
        self._blooms.append((cx, cy, now_s, hue_idx))

    def update_history(self, t: float, s_val: float, b_val: float) -> None:
        self.hist_t.append(t)
        self.hist_s.append(s_val)
        self.hist_b.append(b_val)
        self.line_s.set_data(self.hist_t, self.hist_s)
        self.line_b.set_data(self.hist_t, self.hist_b)
        if self.hist_t:
            self.ax2.set_xlim(max(0.0, self.hist_t[0]), self.hist_t[-1] + 1e-6)
            y_vals = list(self.hist_s) + list(self.hist_b)
            ymin = float(np.min(y_vals))
            ymax = float(np.max(y_vals))
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                ymin, ymax = -1.0, 1.0
            pad = 0.1 * (ymax - ymin + 1e-6)
            self.ax2.set_ylim(ymin - pad, ymax + pad)

    def draw(self, now_s: float) -> None:
        # Clear dynamic artists (keep axes and lines)
        for artist in list(self.ax.collections) + list(self.ax.patches):
            try:
                artist.remove()
            except Exception:
                pass
        # Draw blooms as expanding rings with alpha decay
        for (cx, cy, t0, hue_idx) in list(self._blooms):
            age = now_s - t0
            life = max(1e-3, self.fb.ring_decay_s)
            alpha = max(0.0, 1.0 - age / life)
            if alpha <= 0.0:
                continue
            radius = 0.03 + 0.25 * (age / life)
            color = self.fb.palette[hue_idx % len(self.fb.palette)]
            circ = plt.Circle((cx, cy), radius, fill=False, linewidth=2.0, alpha=alpha, color=color)
            self.ax.add_patch(circ)
        self.fig.canvas.draw_idle()


def run_neurofeedback(sig_cfg: SignalConfig, fb_cfg: FeedbackConfig) -> None:
    src = SignalSource(sig_cfg)
    aud = AudioEngine(fb_cfg)
    vis = VisualEngine(fb_cfg)

    # EMAs
    s_val = 0.0
    baseline = 0.0
    started = False
    t_s = 0.0
    smooth_alpha = (sig_cfg.dt_s / fb_cfg.smooth_tau_s) if sig_cfg.source_csv is None else (1.0 / (fb_cfg.smooth_tau_s * sig_cfg.csv_fs))
    base_alpha = (sig_cfg.dt_s / fb_cfg.baseline_tau_s) if sig_cfg.source_csv is None else (1.0 / (fb_cfg.baseline_tau_s * sig_cfg.csv_fs))
    smooth_alpha = float(np.clip(smooth_alpha, 1e-4, 0.5))
    base_alpha = float(np.clip(base_alpha, 1e-4, 0.5))

    t0_wall = time.time()
    while True:
        v, dt = src.step()
        if not np.isfinite(v):
            break
        if sig_cfg.source_csv is None:
            # Simulated pacing
            time.sleep(max(0.0, dt))
        else:
            # CSV pacing
            time.sleep(max(0.0, dt))
        t_s += dt
        # Smooth signal and baseline
        if not started:
            s_val = v
            baseline = v
            started = True
        else:
            s_val = (1.0 - smooth_alpha) * s_val + smooth_alpha * v
            baseline = (1.0 - base_alpha) * baseline + base_alpha * s_val
        advantage = fb_cfg.reward_gain * (s_val - baseline)
        reward_mag = float(np.clip((advantage - fb_cfg.reward_threshold), 0.0, None))
        # Normalize reward magnitude to [0,1] using a soft scale
        norm = float(1.0 - math.exp(-abs(reward_mag)))

        if reward_mag > 0.0:
            vis.emit_bloom(t_s, intensity=min(1.0, norm + 0.2))
            aud.play(norm)

        vis.update_history(t_s, s_val, baseline)
        vis.draw(t_s)

        # End condition for simulated mode
        if sig_cfg.source_csv is None and t_s >= sig_cfg.duration_s:
            break

    # Keep the last frame visible a bit
    plt.pause(0.2)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neurofeedback: creative audiovisual feedback on signal improvement")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV stream")
    p.add_argument("--column", type=str, default="value", help="CSV column for signal")
    p.add_argument("--fs", type=float, default=20.0, help="Sampling rate (Hz) for CSV pacing")
    p.add_argument("--duration", type=float, default=120.0, help="Duration (s) for simulated mode")
    p.add_argument("--dt", type=float, default=0.05, help="Step size (s) for simulated mode")
    p.add_argument("--smooth-tau", type=float, default=1.0, help="Signal EMA tau (s)")
    p.add_argument("--baseline-tau", type=float, default=5.0, help="Baseline EMA tau (s)")
    p.add_argument("--reward-gain", type=float, default=1.0, help="Reward scaling gain")
    p.add_argument("--reward-threshold", type=float, default=0.0, help="Only rewards above this emit feedback")
    p.add_argument("--no-audio", action="store_true", help="Disable audio beeps")
    p.add_argument("--seed", type=int, default=0, help="Random seed for simulation")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    sig = SignalConfig(
        source_csv=args.csv,
        csv_column=args.column,
        csv_fs=args.fs,
        duration_s=args.duration,
        dt_s=args.dt,
        seed=args.seed,
    )
    fb = FeedbackConfig(
        smooth_tau_s=args.smooth_tau,
        baseline_tau_s=args.baseline_tau,
        reward_gain=args.reward_gain,
        reward_threshold=args.reward_threshold,
        no_audio=bool(args.no_audio),
    )
    run_neurofeedback(sig, fb)


if __name__ == "__main__":
    main()




