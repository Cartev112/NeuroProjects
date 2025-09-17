## Project 23: ECoG High-Gamma Decoding for Motor Tasks

### Overview
Classify movement vs rest using high-gamma bandpower features extracted from ECoG channels.

### Instructions
Provide `npz` with `X` (N, C, T) and `y`. Run:
```bash
python ecog_cli.py --data_npz data.npz --out_dir outputs/ecog
```

### Learning: High-Gamma
High-gamma (70–150 Hz) correlates with local population activity in ECoG. Band-limited power over short windows yields discriminative features.

