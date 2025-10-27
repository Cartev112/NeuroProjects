# Principal Curvatures and Derived Measures on Meshes

This project estimates principal curvatures (k1, k2), mean curvature H, and Gaussian curvature K on triangular meshes using a robust normal-variation method.

## Quickstart

```bash
python curvature_cli.py --mesh path/to/mesh.obj --output-dir outputs
```

Outputs:
- `k1.npy`, `k2.npy`, `mean_curvature.npy`, `gaussian_curvature.npy`
