# Hopfield Network Demo

Classic Hopfield net with Hebbian weights over bipolar (±1) patterns. Supports synchronous/asynchronous updates, retrieval from noisy cues, and analyses of capacity (success vs α = P/N) and noise robustness (success vs flip probability).

## Run

```powershell
python HopfieldDemo/hopfield_demo.py --N 256 --P 30 --mode async --save hopfield.png --no-show
```

## Options

- `--N`: number of neurons (choose a square for pretty pattern images)
- `--P`: number of stored patterns (also used in noise robustness sweep)
- `--mode`: `sync` (parallel) or `async` (sequential random order)
- `--max-steps`: max recall iterations
- `--flip-prob`: cue noise for demo and capacity trials
- `--alpha-max`, `--cap-evals`: capacity sweep range and resolution
- `--noise-max`, `--noise-evals`: noise sweep range and resolution
- `--trials`: trials per sweep point
- `--seed`: RNG seed

## Notes

- Hebbian rule: W = (1/N) Σ_p x_p x_p^T with zero diagonal
- Energy decreases under asynchronous updates; synchronous may oscillate but often converges in practice
- Theoretical capacity ~ 0.138 N for random patterns (at low error); you can observe the empirical success curve approaching this scale
