# Neurofeedback Demo: Creative Audiovisual Reinforcement

Generates real-time visual blooms and audio tones when a 1D signal improves relative to a running baseline (advantage). Use a simulated signal or stream values from a CSV.

## Run

Simulated signal (20 Hz, 120 s):
```powershell
python NeurofeedbackDemo/neurofeedback_demo.py --duration 120
```

CSV stream (column `value`, 20 Hz pacing):
```powershell
python NeurofeedbackDemo/neurofeedback_demo.py --csv path\to\data.csv --column value --fs 20
```

## How it works

- Smooth the signal with an EMA (tau `--smooth-tau`).
- Maintain a slower EMA baseline (tau `--baseline-tau`).
- Advantage = (smoothed − baseline). When positive, issue feedback:
  - Visual: expanding, fading rings (blooms) with color cycling.
  - Audio: short tone; pitch/duration scale with magnitude (Windows `winsound`).

## Options

- Source: `--csv`, `--column`, `--fs`; or use simulated mode (`--duration`, `--dt`).
- Filtering: `--smooth-tau`, `--baseline-tau`.
- Reward: `--reward-gain`, `--reward-threshold`.
- Audio: `--no-audio` (disable tones).
- Misc: `--seed`.

## Notes

- On non-Windows systems or if `winsound` is unavailable, audio is skipped automatically.
- Tune `--reward-threshold` to reduce spurious triggers; increase `--baseline-tau` to demand more sustained improvements.
- You can feed EEG/HRV/etc. by writing a CSV with a `value` column at the desired rate.


