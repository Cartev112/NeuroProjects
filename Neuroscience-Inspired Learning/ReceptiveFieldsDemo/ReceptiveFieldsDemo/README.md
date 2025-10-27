# Receptive Fields Demo

Learn Gabor-like receptive fields from natural image patches using a Hebbian learner (Oja's rule with kWTA) on ZCA-whitened data.

## Quick start

Use synthetic 1/f images (no dataset needed):

```powershell
python ReceptiveFieldsDemo/rf_demo.py --use-1overf --steps 20000 --save filters.png --no-show
```

Use your own image folder (grayscale conversion is automatic):

```powershell
python ReceptiveFieldsDemo/rf_demo.py --image-folder path\to\images --max-images 200 --steps 20000 --save filters.png --no-show
```

## Tips

- Larger `--dataset-size` (e.g., 100k patches) improves filter quality.
- `--patch-size` between 8 and 16 is common; 12 is a good default.
- Whitening (`--zca-eps`) is crucial; too small epsilon may be numerically unstable.
- Competition (`--k-active`) shapes sparsity; try values around 5–10.
- Learning rate `--lr` can start at 0.05 and decay internally.

## Output

- Left: tiled learned filters (normalized per filter)
- Right: mean absolute weight update over time (convergence proxy)

