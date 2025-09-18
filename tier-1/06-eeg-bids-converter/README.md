## Project 6: EEG BIDS Conversion and Validation

### Overview
Convert a raw EEG dataset into BIDS format and validate it. The CLI takes paths to raw EEG files and minimal metadata, writes a BIDS directory structure, and runs the BIDS validator.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Prepare raw EEG (e.g., EDF/VHDR) and minimal metadata (subject, task, sampling rate, montage).
3) Run:
```bash
python bids_cli.py \
  --raw_file /path/to/raw.edf \
  --bids_root ./bids_out \
  --subject sub-01 \
  --task rest \
  --ses ses-01
```

### Learning: Why BIDS?
The Brain Imaging Data Structure (BIDS) standardizes folder organization, filenames, and metadata. BIDS improves reproducibility and tooling interoperability (MNE, fMRIPrep, MOABB). This code uses `mne-bids` to construct a valid BIDS dataset and optionally calls the official validator.

