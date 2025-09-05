# V1–V2 Hierarchical Unsupervised Model

- Layer 1 (V1): Sparse coding (ISTA + dictionary updates) on ZCA-whitened natural image patches → Gabor-like filters.
- Layer 2 (V2): Sparse coding on 2x2 neighborhoods of V1 codes → composite features (corners, junctions).

## Run

```powershell
python V1V2Demo/v1v2_demo.py --use-1overf --steps1 3000 --steps2 2000 --save v1v2.png --no-show
```

## Options

- Data: `--image-folder`, `--max-images`, `--use-1overf`, `--dataset-size`, `--patch-size`, `--stride`, `--zca-eps`
- V1: `--K1`, `--lam1`, `--steps1`, `--iters1`
- V2: `--K2`, `--lam2`, `--steps2`, `--iters2`
- Misc: `--seed`, `--save`, `--no-show`

## Output

- Left: V1 dictionary (filters) as a grid
- Middle: V2 composites reconstructed from V1 atoms arranged in a 2x2 layout
- Right: learning curves for reconstruction error and sparsity at both layers

## Tips

- Choose `patch-size` 8–16; 12 works well. `K1` > patch_dim for overcompleteness (e.g., 128 filters for 12×12).
- `lam1` controls V1 sparsity; higher means sparser (try 0.1–0.25). `lam2` typically a bit smaller.
- If filters look noisy, increase dataset size or training iters; check whitening (`--zca-eps`).
- For real images, pass `--image-folder`. For quick tests, use `--use-1overf`.


