# Learning Doc: Cerebellum‑Inspired Motor Learning (Perceptron with Climbing‑Fiber Error)

This document explains the biological intuition behind cerebellar learning, the perceptron formalization used in the demo, the learning rules (binary and continuous), how to interpret the outputs, and suggested experiments and extensions.

## 1) Biological intuition (Marr–Albus–Ito)

- Parallel fibers (granule cells) provide high‑dimensional inputs to Purkinje cells.
- The Purkinje cell outputs an inhibitory signal that shapes downstream motor commands.
- A climbing fiber delivers a powerful teaching/error signal when movement outcomes deviate from desired targets.
- Synapses from parallel fibers to Purkinje cells are adjusted (LTP/LTD) when the climbing fiber signals error, gradually reducing future error.

Abstraction: A single‑output neuron (Purkinje) with many inputs (parallel fibers) learns a mapping from inputs to desired outputs using an error‑driven rule supplied by the climbing fiber.

## 2) Model overview (what the demo implements)

We use a single‑output linear unit with weights `w ∈ R^D` and input vector `x ∈ R^D`.

- Output (pre‑nonlinearity): `o = w^T x`
- Targets `y` are either binary (±1) or continuous (in [−1, 1]).
- The “climbing fiber” provides an error `e` that drives plasticity.
- Learning proceeds in mini‑batches for `epochs` over a generated dataset.

Two target regimes:
- Binary classification (perceptron‑like): teacher defines a separating hyperplane; outputs are thresholded to ±1.
- Continuous regression: teacher output is continuous (tanh of a linear teacher).

## 3) Learning rules (error‑driven plasticity)

Let `o = w^T x` and the error be `e = y − o` (continuous) or a classification‑based error signal.

- Continuous (squared error GD):
  - Update: `Δw = η · e · x = η · (y − o) · x`
  - This is standard gradient descent on `L = 1/2 (y − o)^2`.

- Binary (perceptron‑style):
  - Prediction: `ŷ = sign(o) ∈ {−1, +1}`
  - If `ŷ ≠ y`, update pushes weights toward the teacher direction:
    - `Δw = η · y · x` for misclassified samples, `0` otherwise.
  - In minibatches, we average the contributions across misclassified examples.

Interpretation: The climbing fiber conveys an error event that gates synaptic changes on parallel‑fiber inputs, adjusting `w` to reduce future error.

## 4) Data generation

- Inputs X: i.i.d. Gaussian (`N × D`).
- Teacher:
  - Binary: linear teacher `w*`; target is `y = sign(X w* + noise)`.
  - Continuous: `y = tanh( (X w*) / sqrt(D) )` producing bounded targets.

This provides random but learnable input–output mappings, analogous to arbitrary motor corrections.

## 5) What the code does

- Builds dataset `(X, y)` under chosen `--target-type`.
- Initializes weights `w ~ N(0, 0.1)`.
- Trains for `epochs` with mini‑batches:
  - Binary: perceptron‑style updates on misclassified samples.
  - Continuous: gradient descent on squared error.
- Tracks a per‑epoch loss:
  - Binary: `1 − accuracy` over the full training set.
  - Continuous: MSE over the full training set.
- Evaluates final accuracy (binary) or MSE (continuous).
- Plots:
  - Learning curve vs epoch
  - Predicted vs target scatter (subset)
  - Weight visualization (reshaped to square if possible)

CLI highlights:
- `--inputs`, `--samples`, `--batch`, `--epochs` control dataset and training loop.
- `--lr` learning rate.
- `--target-type` `binary` or `continuous`.
- `--noise-std` teacher noise for the binary case (harder classification).

## 6) Interpreting outputs

- Learning curve: should decrease and flatten as the model converges. If it stalls high, reduce noise or increase epochs/lr tuning.
- Predicted vs target scatter:
  - Binary: points at (−1, −1) and (1, 1) indicate correct predictions; off‑diagonal shows errors.
  - Continuous: points should lie near the y=x line; the red dashed line marks ideal predictions.
- Weights: inspecting `w` reveals which inputs the “Purkinje” learned to rely on; for structured inputs, hot regions indicate important features.

## 7) Suggested experiments

- Learning rate sweep: try `--lr {0.01, 0.05, 0.1, 0.2}`; too large may oscillate, too small may be slow.
- Noise robustness (binary): increase `--noise-std` to make teacher labels noisier; observe plateau accuracy.
- Sample complexity: vary `--samples` and `--inputs` (D). Fixed D, higher N improves generalization; fixed N, higher D can require more data.
- Batch size: larger batches smooth gradients but can slow escape from plateaus in perceptron learning.
- Target type: compare binary vs continuous for the same data generator.

## 8) Cerebellar links and extensions

- Eligibility traces: make plasticity depend on recent input activity convolved with a kernel; apply updates when error arrives (temporal credit assignment).
- Sign‑constrained plasticity: force only LTD (negative updates) at parallel fiber synapses, with compensatory mechanisms elsewhere.
- Multiple outputs: extend to a small vector output (e.g., simulating multiple downstream nuclei). Train with multi‑output regression/classification.
- Nonlinearities: add a static nonlinearity before the output to emulate saturation.
- Regularization: `L2` or `L1` on weights to encourage robustness or sparsity.

## 9) Quick recipe

- Binary: `--inputs 256 --samples 2000 --epochs 30 --lr 0.1 --target-type binary`
- Continuous: `--inputs 256 --samples 2000 --epochs 30 --lr 0.05 --target-type continuous`
- Expect monotonic learning curves and final metrics near 100% accuracy (low/no noise) or low MSE (continuous). Tune `--noise-std` and `--lr` to explore robustness.


