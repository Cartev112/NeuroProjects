# Cerebellum-Inspired Motor Learning

Single-output perceptron trained by an error signal (climbing fiber) to adjust Purkinje weights so that outputs match targets. Supports binary (classification) and continuous targets.

## Run

```powershell
python CerebellumDemo/cerebellum_demo.py --inputs 256 --samples 2000 --epochs 30 --save cereb.png --no-show
```

## Options

- `--inputs`, `--samples`, `--batch`, `--epochs`
- `--lr`: learning rate
- `--target-type`: `binary` (perceptron-like) or `continuous` (linear with squared error)
- `--noise-std`: teacher noise for binary targets
- `--seed`: RNG seed

## Outputs

- Learning curve (1-accuracy or MSE)
- Predicted vs target scatter
- Weights visualization (reshaped if inputs form a square)

## Notes

- Binary: update is perceptron-style on misclassified samples (Δw ∝ y x)
- Continuous: update is gradient descent on squared error (Δw ∝ (y − o) x)
- Interpreting as cerebellum: parallel fibers (inputs) → Purkinje cell (output); climbing fiber conveys error to adjust synapses, reducing motor error.


