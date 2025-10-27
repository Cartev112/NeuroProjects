## Project 18: Diffusion MRI Tractography Features for Age/Sex Prediction

### Overview
Extract tract-level metrics (FA/MD) and predict phenotype with ML baselines.

### Instructions
Provide a CSV of tract features and labels. Run:
```bash
python dmri_cli.py --features tracts.csv --label_col sex --out_dir outputs/dmri
```

### Learning: Microstructure Features
Diffusion-derived metrics summarize white-matter microstructure. Aggregating per-tract statistics forms a compact feature space for classical ML.

