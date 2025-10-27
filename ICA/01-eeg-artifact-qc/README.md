## Project 1: EEG Artifact Detection and Automatic Quality Reports

### Overview
This project builds an automated EEG artifact quality control pipeline that loads BIDS-formatted EEG, performs basic preprocessing (filtering, notch), estimates ICA, detects likely artifact components/segments, computes QC metrics, and generates an HTML report with plots. Outputs include a machine-readable JSON summary.

### Instructions
1) Install dependencies at `tier-1/requirements.txt`.
2) Ensure your EEG dataset is BIDS-formatted. You can test with any OpenNeuro EEG dataset.
3) Run the CLI:
```bash
python cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task rest \
  --output_dir outputs/sub-01_task-rest
```

Key arguments:
- `--subject`: Subject label (e.g., `sub-01`).
- `--task`: Task label (if multiple, choose one).
- `--run`: Optional run label.
- `--l_freq`, `--h_freq`, `--notch`: Preprocessing settings.

Artifacts produced:
- `report.html`: MNE report with raw, PSD, and ICA plots.
- `qc_summary.json`: QC metrics (duration, bad segment ratio, EOG correlation, etc.).
- `ica.fif`: Fitted ICA object for reuse.

### Concept: EEG Artifacts and ICA-based QC
EEG signals are contaminated by non-neural sources: ocular (blinks/saccades), cardiac, muscle, electrode noise, and line interference. A practical QC pipeline includes:

- Preprocessing: band-pass filtering removes slow drifts/high-frequency noise; notch suppresses line noise.
- ICA: independent component analysis attempts to decompose the mixed sensor signals into statistically independent sources. Many ocular/muscle artifacts form distinct components with characteristic topographies and spectra.
- Detection and annotation: we correlate ICA component time courses with EOG reference channels and look for stereotyped blink/saccade patterns, mark segments with high-amplitude transients, and compute summary ratios (bad-seconds/total-seconds).
- Reporting: visualization enables quick human verification, reducing the chance of over/under-correction.

In this code, ICA helps identify likely artifacts (via correlation with EOG and component spectra), while simple thresholds flag high-amplitude segments. The report stitches these pieces into a reproducible QC artifact summary.

