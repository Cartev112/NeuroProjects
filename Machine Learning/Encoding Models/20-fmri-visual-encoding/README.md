## Project 20: Visual Encoding with fMRI using DNN Features

### Overview
Map deep visual features to voxel responses via ridge regression; output voxelwise R² maps and ROI summaries.

### Instructions
Provide features `X.npy` (N, P) from a DNN and voxel data `Y.npy` (N, V). Run:
```bash
python visual_encoding_cli.py --X X.npy --Y Y.npy --out_dir outputs/vis_enc
```

### Learning: Encoding Models
Encoding links stimulus-derived features to brain responses. Ridge regularization handles high-dimensional features. ROI-level aggregation summarizes performance.

