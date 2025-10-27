# Surface Spectral Signatures (Laplace–Beltrami + HKS)

This project computes Laplace–Beltrami eigenpairs and Heat Kernel Signatures (HKS) on cortical meshes or any watertight triangular mesh.

## Quickstart

```bash
python lb_hks_cli.py --mesh path/to/mesh.obj --num-eigs 100 --output-dir outputs
```

Outputs (by default in `outputs/`):
- `eigenvalues.npy`: Array of shape `(k,)`
- `eigenvectors.npy`: Array of shape `(num_vertices, k)`
- `hks.npy`: Array of shape `(num_vertices, num_times)`
- `times.npy`: The HKS time scales used

## CLI Usage

```bash
python lb_hks_cli.py \
  --mesh path/to/mesh.{obj|ply|stl|off} \
  --num-eigs 100 \
  --num-times 10 \
  --t-min 1e-4 \
  --t-max 1e-1 \
  --output-dir outputs
```

Optional flags:
- `--save-evecs/--no-save-evecs` (default `--save-evecs`)
- `--save-evals/--no-save-evals` (default `--save-evals`)
- `--save-hks/--no-save-hks`   (default `--save-hks`)

## Method

- Builds the cotangent Laplacian `L` and lumped mass matrix `M` for a triangular mesh
- Solves the generalized eigenproblem `L φ = λ M φ` for the smallest `k` eigenpairs
- Computes HKS at log-spaced time scales: `HKS_t(v) = Σ_i exp(-t λ_i) φ_i(v)^2`

Notes:
- The implementation uses a robust cotangent discretization with mixed Voronoi (per-vertex) areas for the mass matrix (lumped)
- Mesh should be manifold and reasonably well-conditioned; consider remeshing if numerical issues arise

## Requirements
See `requirements.txt`.

## References
- Heat Kernel Signature (HKS): Sun, Ovsjanikov, and Guibas (2010)
- Discrete cotangent Laplacian: Meyer et al. (2003)
