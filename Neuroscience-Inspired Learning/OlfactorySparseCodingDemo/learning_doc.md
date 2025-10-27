# Learning Doc: Olfactory Bulb–Inspired Sparse Coding

This document explains the biological intuition, mathematical model, and implementation details behind the Olfactory Bulb–inspired sparse coding demo. It also covers how to interpret outputs, tune parameters, and extend the model.

## 1) Biological intuition (why this works)

Across species, early olfactory circuits transform dense, overlapping sensory patterns into sparse, more separable representations before higher-level learning. Two key ideas recur:

- Expansion: Project inputs into a higher-dimensional space via many weak, random connections. Overcomplete, high-dimensional codes are easier to separate with simple thresholds.
- Competition: Lateral inhibition or winner-take-all suppresses most activity, leaving only a few strong responders, which yields sparse, low-overlap codes.

In mammals, olfactory receptor neurons converge to glomeruli; mitral/tufted cells relay to cortex while granule/periglomerular cells mediate lateral inhibition. In insects, projection neurons feed into a very large Kenyon-cell layer with strong competition (sparse firing). In both cases, expansion + competition improves separability and robustness.

## 2) Model overview (what we simulate)

We model the expansion and competition stages with a simple, feedforward network:

- Inputs X: binary vectors of size D_in. Each bit is 1 with probability p (simple synthetic odors).
- Random projection W: sparse matrix of size D_out × D_in with connection probability c and random ± signs. Rows of W are unit-normalized and scaled by a factor s.
- Pre-activation a = X · W^T.
- Competition stage to enforce sparsity:
  - Lateral inhibition (subtract-mean): y = ReLU(a − α · mean(a)), per sample.
  - k-Winners-Take-All (kWTA): keep the top-k responses per sample, set others to 0.

Outputs Y are sparse, decorrelated codes. No learning is performed in this minimal demo; the structure alone performs the transformation.

## 3) Mathematics (how it transforms data)

- Expansion: A random linear map a = XW^T spreads bits across many units. Signed, row-normalized W avoids bias and keeps pre-activations well-scaled.
- Competition:
  - Subtractive inhibition (global): y_i = max(0, a_i − α·ȧ) where ȧ is the mean over units for a sample. This suppresses common-mode activity and trims weaker responses.
  - kWTA: Set a hard threshold so exactly k units per sample survive (ties resolved by partition). This yields a fixed sparsity target and strong competition.

Effects:
- Sparsity: Most y_i are zero. Supports capacity, energy efficiency, and robust associative learning downstream.
- Decorrelation: Shared components are suppressed; different outputs fire together less often, simplifying classification and memory retrieval.

## 4) What the demo does (files and flow)

- `olfactory_demo.py`:
  1) Generate N binary patterns X ∈ {0,1}^{N × D_in} with bit probability p.
  2) Build W (D_out × D_in): Bernoulli(c) connectivity × random sign, row-normalize, scale.
  3) Compute pre-activations a = XW^T.
  4) Apply either subtractive inhibition or kWTA to get outputs Y.
  5) Report metrics and plot inputs/outputs/weights.

- `README.md`: usage and options.

## 5) Metrics (what to look at)

- Sparsity
  - `sparsity_in` = nonzeros(X)/|X| (≈ p).
  - `sparsity_out` = nonzeros(Y)/|Y| (target is much smaller than input; with kWTA and D_out units, ideal ≈ k/D_out).

- Decorrelation
  - `avg_abs_corr_in`: mean absolute off-diagonal correlation among input features.
  - `avg_abs_corr_out`: same among output features. Goal: lower than input.

Reduced `sparsity_out` and reduced `avg_abs_corr_out` indicate a successful expansion+competition transform toward sparse, decorrelated codes.

## 6) Interpreting the plots

- Input patterns: a tiled view of sample binary inputs (white = 1). With non-square D_in, a 1×D stripe is shown.
- Output patterns: the same samples after competition (normalized for display). You should see many fewer white pixels.
- Random projection weights: a subset of rows of W as tiles (if D_in is a perfect square). Appears as random speckle with ± structure.
- Bar chart: metrics for quick comparison.

## 7) How to use (typical settings)

- Dimensions
  - `--input-dim`: 256 is a good default; choose squares (e.g., 256, 400) for tiled visuals.
  - `--output-dim`: 2× to 8× expansion over input encourages separability (e.g., 512–2048 for D_in=256).

- Projection
  - `--conn-prob`: 0.05–0.2 keeps W sparse. Too dense reduces selectivity; too sparse can under-sample inputs.
  - `--weight-scale`: keep ~1.0; raise if pre-activations are too weak.

- Competition
  - `--kwta` + `--k-active`: controls output sparsity directly. For D_out=1024, k=20–60 yields 2–6% activity.
  - `--inhib-strength`: for subtractive inhibition, 0.1–0.3 often works; increase for more sparsity.

- Inputs
  - `--activity-prob`: 0.05–0.2 creates moderately dense inputs. Lower p can make expansion less necessary; higher p can need stronger inhibition.

## 8) Ablations and intuition checks

- Turn off competition (no kWTA, `--inhib-strength 0.0`):
  - `sparsity_out` will increase toward input sparsity.
  - `avg_abs_corr_out` often remains high; decorrelation worsens.

- Vary expansion (D_out): More output units usually reduce decoding error in downstream tasks and improve decorrelation.

- Vary connectivity (`--conn-prob`):
  - Too low: many outputs get little input → low utility.
  - Too high: outputs become similar → higher correlations.

- Signed vs. unsigned W: Removing signs biases activity and can hurt decorrelation.

## 9) Relation to biology and other models

- Olfactory bulb/cortex and insect mushroom body use expansion + competition to form sparse codes.
- Similar ideas appear in random feature methods and compressed sensing (random projections), and in neural networks via sparse coding and kWTA layers.

## 10) Extensions (where to go next)

- Divisive normalization: y_i = a_i / (ε + ||a|| or local pool), a more biological inhibition scheme.
- Structured inhibition: local/clustered pools rather than global mean.
- Learned projections: add a Hebbian/Oja stage on W to further reduce correlations while constraining W’s sparsity.
- Odor mixtures: generate inputs as mixtures of base patterns to study pattern separation.
- Noise robustness: add input noise and measure stability of sparse codes.
- Downstream tasks: train a simple linear classifier or associative memory on Y to quantify gains vs. raw X.

## 11) Quick recipe

- Start with: `--input-dim 256 --output-dim 1024 --conn-prob 0.1 --kwta --k-active 32 --activity-prob 0.1`.
- Confirm: `sparsity_out ≈ 32/1024 = 0.031` and `avg_abs_corr_out < avg_abs_corr_in`.
- Tune k (or inhibition strength) and connectivity to trade off sparsity vs. separability.
