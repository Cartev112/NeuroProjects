## Tier 1 Projects

This directory contains 10 foundational projects for applied ML on brain data. Each project lives in its own subfolder with code and a detailed README covering: overview, instructions, and a concept section explaining the approach used.

Projects:
- 01-eeg-artifact-qc
- 02-eeg-psd-explorer
- 03-erp-pipeline
- 04-sleep-staging-baseline
- 05-motor-imagery-csp-lda
- 06-eeg-bids-converter
- 07-fmri-confounds-qc
- 08-rsfmri-fc
- 09-meg-tfr
- 10-smri-brain-age-baseline

Environment:
```bash
conda create -n tier1 python=3.11 -y
conda activate tier1
pip install -r requirements.txt
```
