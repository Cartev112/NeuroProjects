# LIF Simulator

A simple Leaky Integrate-and-Fire (LIF) neuron simulator with adjustable input current, noise, and a refractory period. Includes matplotlib visualization.

## Installation

1. Create a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
```

2. Install dependencies using the venv's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

Run a simulation for 1 second (1000 ms), constant input current 1.5, save a plot, and skip showing the window:

```powershell
.\.venv\Scripts\python.exe lif_simulator.py --duration 1000 --input-current 1.5 --save lif_plot.png --no-show
```

Common options:

- `--duration` (ms): total time, default 1000
- `--dt` (ms): time step, default 0.1
- `--protocol`: input protocol: `constant`, `step`, or `sine`
- `--input-current`: constant current (arb. units), default 1.5
- `--step-onset` (ms): step onset time for `step` protocol
- `--step-current`: current during step for `step` protocol
- `--sine-amplitude`: sine protocol amplitude
- `--sine-frequency` (Hz): sine protocol frequency
- `--sine-offset`: DC offset for sine protocol
- `--noise-std` (mV/√ms): noise std, default 0.5
- `--tau-m` (ms): membrane time constant, default 20
- `--resistance`: membrane resistance scale, default 10
- `--v-rest` (mV): resting potential, default -65
- `--v-reset` (mV): reset potential, default -65
- `--v-threshold` (mV): threshold, default -50
- `--refractory` (ms): absolute refractory period, default 5
- `--seed`: RNG seed
- `--save PATH`: save plot to PATH
- `--no-show`: do not display plot window

### Examples

- Constant input:

```powershell
.\.venv\Scripts\python.exe lif_simulator.py --duration 1000 --protocol constant --input-current 1.8 --save lif_const.png --no-show
```

- Step input (step at 200 ms to current 2.5):

```powershell
.\.venv\Scripts\python.exe lif_simulator.py --duration 800 --protocol step --input-current 0.5 --step-onset 200 --step-current 2.5 --save lif_step.png --no-show
```

- Sine input (10 Hz, amplitude 1.0, offset 0.5):

```powershell
.\.venv\Scripts\python.exe lif_simulator.py --duration 1000 --protocol sine --sine-amplitude 1.0 --sine-frequency 10 --sine-offset 0.5 --save lif_sine.png --no-show
```

After each run, the script prints spike metrics (count, rate, ISI stats).

## Notes

- Units are chosen for convenience: voltages in mV and time in ms. The input current is unitless and scaled by the resistance parameter to determine its effect on membrane potential.
- Noise is additive Gaussian with standard deviation scaled by √dt to maintain proper diffusion scaling in discrete time.

