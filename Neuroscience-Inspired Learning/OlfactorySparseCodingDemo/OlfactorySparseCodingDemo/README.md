# Olfactory Bulb-Inspired Sparse Coding

Random projection + lateral inhibition (or kWTA) transforms binary patterns into sparse, decorrelated codes, inspired by olfactory bulb circuitry (expansion and competition).

## Run

```powershell
python OlfactorySparseCodingDemo/olfactory_demo.py --patterns 200 --kwta --k-active 25 --save olf_demo.png --no-show
```

## Options

- `--input-dim`, `--output-dim`: dimensions (use square input dims for nice grids)
- `--conn-prob`, `--weight-scale`: projection sparsity and scaling
- `--kwta` or `--inhib-strength`: use kWTA or subtractive inhibition
- `--k-active`: active outputs per pattern (kWTA)
- `--patterns`: number of input patterns
- `--activity-prob`: probability a bit is 1 in each input
- `--seed`: RNG seed

## Printed metrics

- `sparsity_in`, `sparsity_out`: fraction of nonzero entries (lower is sparser)
- `avg_abs_corr_in`, `avg_abs_corr_out`: average absolute off-diagonal correlation (lower is more decorrelated)

## Notes

- Expansion (more output than input units) + competition yields sparse, separable codes.
- Use kWTA to directly control output sparsity (≈ `k / output_dim`).
- With subtractive inhibition, tune `--inhib-strength` (typical 0.1–0.3).
