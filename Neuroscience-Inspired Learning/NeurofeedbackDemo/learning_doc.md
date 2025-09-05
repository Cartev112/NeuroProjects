# Learning Doc: Creative Neurofeedback with Advantage-Based Reinforcement

This document explains the design and usage of the neurofeedback demo that turns improvements in a 1D signal into creative audiovisual reinforcement (expanding color blooms and short tones). It covers the high-level concept, equations, calibration, design choices, and extensions.

## 1) Neurofeedback, simply stated

- Neurofeedback is a closed loop: measure a signal (e.g., EEG band power, HRV), transform it, and feed back information to the participant in real time.
- The participant learns (implicitly or explicitly) to modulate their internal state so that feedback increases. Effective feedback is timely, informative, and pleasant.
- Here, feedback is emitted when the signal improves relative to a baseline. We visualize improvement as a “bloom” and play a short tone whose pitch/duration scales with the amount of improvement.

## 2) Signals the demo can use

- Simulated signal (default): noisy random walk with occasional positive drifts (for quick testing).
- CSV stream: any time series with a column name (default `value`). Typical sources:
  - EEG power (e.g., alpha band power), EOG artifacts removed
  - HRV metrics (RMSSD), respiration rate, GSR/EDA level, pupil diameter
  - Behavioral index (reaction time improvements)

Format for CSV mode:
- Provide `--csv path`, `--column value`, and sampling rate `--fs`. The script reads rows sequentially at approximately `1/fs` seconds per step.

## 3) Core transforms and equations

Let `x(t)` be the raw value per step and `∆t` the step size.

- Exponential moving average (EMA) smoothing of the signal:
  - `s(t) = (1 − α_s) s(t−1) + α_s x(t)`, with `α_s ≈ ∆t / τ_s` and `τ_s = --smooth-tau` (seconds)
- Baseline EMA (slower accumulator reflecting recent history):
  - `b(t) = (1 − α_b) b(t−1) + α_b s(t)`, with `α_b ≈ ∆t / τ_b` and `τ_b = --baseline-tau`
- Advantage (improvement measure) and reward magnitude:
  - `A(t) = gain · (s(t) − b(t))`, with `gain = --reward-gain`
  - Emit feedback only when `A(t) > threshold`, with `threshold = --reward-threshold`
  - Map magnitude to [0, 1] with a soft normalization (demo uses `1 − exp(−|A|)`) before rendering audio/visual strength.

Intuition:
- `s(t)` tracks the current state, `b(t)` tracks a slower “recent normal.”
- Positive `s(t) − b(t)` indicates improvement relative to recent trends; negative indicates regression.

## 4) Feedback design (what and why)

- Visual: expanding, fading rings (blooms) drawn at aesthetically spaced positions (golden-angle–inspired). The color palette cycles across pleasing hues. Multiple events can overlap, producing a generative, ambient visual that “rewards” improvement without being harsh or intrusive.
- Audio: short tones where pitch and duration scale with magnitude. This reinforces subtle improvements; tones are clipped to sensible ranges to avoid fatiguing the listener.

Design goals:
- Make success perceptually salient and pleasant (reinforcing), not jarring.
- Provide continuous cues (rolling inset plot of `s(t)` and `b(t)`) without overloading attention.

## 5) Calibration: getting good behavior quickly

- Start with defaults: `--smooth-tau 1.0`, `--baseline-tau 5.0`, `--reward-gain 1.0`, `--reward-threshold 0.0`.
- If feedback triggers too often (noisy data):
  - Increase `--baseline-tau` (e.g., 8–12 s) to demand more sustained improvements.
  - Increase `--reward-threshold` slightly (e.g., 0.02–0.05 in your signal’s units after smoothing).
- If feedback is too rare:
  - Decrease `--baseline-tau` (baseline follows faster) or increase `--reward-gain`.
  - Reduce `--reward-threshold` toward 0.
- First 10–20 seconds: allow the baseline to warm up before interpreting performance (the demo starts providing feedback immediately, but you may choose to ignore early feedback during instruction).

Tip: In CSV mode, ensure the incoming signal is already scaled reasonably (z-scored or min–max normalized). You can pre-process externally or modify the code to add live normalization.

## 6) Timing and sampling considerations

- Simulated mode steps at `--dt` seconds (default 0.05 s → 20 Hz).
- CSV mode uses `--fs` to pace reading; make sure it matches your data rate. If your source timestamps are irregular, consider resampling ahead of time.
- Visual updates run at the same loop rate; avoid very high rates to keep CPU usage modest.

## 7) Safety and comfort

- Visuals: The blooms avoid high-contrast flashing and cover a limited fraction of the screen. If working with photosensitive populations, keep ring opacity low and avoid fast pulsation.
- Audio: Tones use brief durations and moderate frequencies (default 300–1400 Hz). Use `--no-audio` when needed or route to a calm sound synthesizer.

## 8) Mapping specific biosignals

- EEG band power (e.g., posterior alpha increase during relaxation):
  - Preprocess: re-reference, band-pass, artifact rejection, compute log-power in a band, optionally z-score.
  - Feed `x(t)` = band power. Increasing target yields positive advantage.
- HRV (RMSSD):
  - Preprocess: beat detection → RMSSD per window. Smooth across windows.
  - Higher RMSSD → relaxation; map to `x(t)`.
- Breath coherence or slow respiration:
  - Track breath rate or variability and map to `x(t)` based on target (e.g., 6 breaths/min).

For adversarial/noisy channels (eye blinks, movement), gate updates: pause feedback when artifacts detected.

## 9) Customization guide (code orientation)

- `SignalSource`: implements simulated generator or CSV streaming; extend with live sources (e.g., sockets or device APIs).
- `FeedbackConfig`: all feedback parameters; tune ring decay, palette, audio ranges.
- `VisualEngine.emit_bloom`: place and color blooms; you can replace with particle systems or shader-based visuals (e.g., vispy, p5py) if desired.
- `AudioEngine.play`: current implementation uses `winsound` on Windows; replace with a cross-platform library (e.g., `sounddevice`, `pyo`, `pygame.mixer`) for richer soundscapes.
- Main loop: computes EMA smoothing, baseline, advantage, normalization; hook your own reward transforms here (e.g., percentile normalization or adaptive thresholds).

## 10) Troubleshooting

- Nothing happens (no blooms/tones):
  - Advantage may stay ≤ threshold; lower `--reward-threshold`, increase `--reward-gain`, or confirm data is varying.
  - CSV path/column wrong; check with a small print patch.
- Too many blooms (visual overload): increase `--baseline-tau`, `--reward-threshold`, or reduce `--reward-gain`.
- Audio not playing: on non-Windows systems, `winsound` is absent; use `--no-audio` or switch to a cross-platform audio backend.

## 11) Extensions and research ideas

- Adaptive thresholding: keep a rolling percentile of `s(t) − b(t)` and trigger only above a chosen percentile (e.g., 70th). Adjusts to individual variability.
- Multi-dimensional feedback: combine multiple channels (EEG + HRV) into a scalar via a weighted sum or learn a mapping with a simple linear model trained to predict session goals.
- Reward shaping with prediction: learn a small model to predict future improvement; reward prediction error yields more anticipatory feedback.
- Schedules: variable ratio reinforcement (randomize feedback opportunities) can improve engagement and reduce gaming of the signal.
- Closed-loop targeting: slowly move the target baseline (`b(t)` goal) toward a desired level when stable improvements are achieved.
- Aesthetics: replace blooms with generative art that “grows” with success (L-systems, flocking, flow fields), or music that adds harmonics/instruments when advantage rises.

## 12) Quick start recipes

- Relaxation-oriented (slow baseline):
  - `--smooth-tau 1.5 --baseline-tau 10 --reward-threshold 0.01`
- Focus bursts (fast feedback):
  - `--smooth-tau 0.5 --baseline-tau 3 --reward-threshold 0.0 --reward-gain 1.5`
- CSV example at 20 Hz:
  - `--csv data.csv --column value --fs 20 --smooth-tau 1.0 --baseline-tau 8 --reward-threshold 0.02`

## 13) Takeaways

- Advantage-based feedback (`s − b`) rewards improvements, not absolute levels, making it robust to drift and individual differences.
- Pleasant, salient feedback fosters engagement; combine visual and auditory cues thoughtfully.
- Careful calibration (taus, threshold, gain) is key to stability and usefulness; test with simulated data before connecting to a live source.


