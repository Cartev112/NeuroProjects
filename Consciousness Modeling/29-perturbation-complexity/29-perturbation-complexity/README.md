## Project 29: Perturbation Complexity Mapping (CPCM)

### Overview
Compute empirical signal complexity metrics from EEG/MEG time series, per epoch/channel/window. Metrics include Lempel–Ziv Complexity (LZC), Permutation Entropy (PE), Sample Entropy (SE), and Multiscale Entropy (MSE). Phase 2 will integrate structural controllability and whole-brain perturbation simulations to relate empirical complexity to network dynamics (PCI-like indices).

### Data
- Input array in `.npy` or `.npz` format.
- Accepts shapes:
  - `(N, C, T)` = epochs × channels × time
  - `(C, T)` = channels × time (treated as one epoch)
  - `(T,)` = single time series (treated as one epoch, one channel)
- Optional channel names in a `.txt` file (one name per line). If omitted, channels are named `ch00`, `ch01`, ...

### Usage
Compute full-length metrics (no sliding windows):
```bash
python cpcm_cli.py \
  --data data.npy \
  --out_dir outputs/cpcm
```

Compute sliding-window metrics with 2 s windows and 0.5 s steps (requires `--sfreq`):
```bash
python cpcm_cli.py \
  --data data.npy \
  --out_dir outputs/cpcm_win \
  --sfreq 250 \
  --win_sec 2.0 \
  --step_sec 0.5 \
  --metrics lzc,pe,se,mse \
  --pe_order 3 \
  --mse_scales 5
```

Specify channel names:
```bash
python cpcm_cli.py --data data.npy --out_dir outputs/cpcm --channels chans.txt
```

### Outputs
- `metrics.csv`
  - Columns: `epoch, channel, t0, t1, lzc, pe, se, mse_1, mse_2, ...`
  - `t0, t1` are sample indices of the window within each epoch/channel.
- `summary.json`
  - Global summary with means/SDs across all rows per metric and dataset metadata.

### Metrics Notes
- **LZC**: binary Lempel–Ziv (median threshold). Normalized by `(log2(n) / n)`.
- **Permutation Entropy (PE)**: normalized by `log(order!)`.
- **Sample Entropy (SE)**: `m=2`, `r=0.2×std` by default; returns `inf` if no matches at `m`.
- **Multiscale Entropy (MSE)**: coarse-graining by averaging; `m=2`, `r=0.2`, configurable number of scales.

### Roadmap – Phase 2 (Planned)
- Structural connectome input (e.g., HCP template or subject DWI) and controllability indices.
- Whole-brain neural mass simulations with focal perturbations to estimate PCI-like metrics.
- Surrogate GNN to predict complexity from connectome + controllability without full simulation.
- Benchmark across vigilance/anesthesia states; correlate complexity with clinical/behavioral measures.

### Minimal Requirements
- `numpy`

### Citation
If you use this scaffold, please cite appropriate literature for LZC, PE/SE/MSE, PCI, and network control theory when applicable.
