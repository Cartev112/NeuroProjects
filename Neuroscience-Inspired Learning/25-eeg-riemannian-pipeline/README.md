## Project 25: EEG Riemannian Geometry Pipeline

### Overview
End-to-end covariance-based classification using tangent space mapping with PyRiemann.

### Instructions
Provide trials `X` (N, C, T) and labels `y`. Run:
```bash
python riemann_cli.py --data_npz /path/to/Xy.npz --out_dir outputs/riemann
```

### Learning: SPD Geometry
EEG covariance matrices lie on the SPD manifold. Tangent space projection allows using Euclidean classifiers while respecting geometry.

