# Place–Grid Cells Demo

Simulate a 2D trajectory, generate grid-cell-like population codes, learn sparse “place cells,” and decode position from grid codes.

## Run

```powershell
python PlaceGridDemo/placegrid_demo.py --steps 4000 --grid-cells 240 --place-cells 96 --save placegrid.png --no-show
```

## What it shows

- Grid codes: multi-scale, multi-orientation cosine interference producing lattice-like fields.
- Place cells: sparse dictionary atoms over standardized grid codes; place fields emerge as localized activity maps.
- Decoding: linear ridge regression from grid codes to (x,y) positions.

## Plots

- True vs decoded trajectory
- Sample grid fields
- Top place fields (by activation energy)
- Sparse coding learning curves (reconstruction error, sparsity)

## Options

- Trajectory: `--steps`, `--dt`, `--arena`
- Grid: `--grid-cells`
- Place coding: `--place-cells`, `--lam`, `--iters`, `--steps-per-iter`
- Misc: `--save`, `--no-show`, `--seed`


