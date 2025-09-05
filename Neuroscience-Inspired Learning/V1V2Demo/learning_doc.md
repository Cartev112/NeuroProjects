# Learning Doc: Building a V1–V2 Hierarchical Model from Scratch

This document explains the ideas behind the V1–V2 demo, how sparse coding learns Gabor-like filters from natural images, how a second layer can form composite (corner/junction) features by pooling, and how to train, tune, and interpret the model.

## 1) Why V1–V2 hierarchy with unsupervised learning?

- Natural images have strong statistical regularities (e.g., 1/f spectra, edges). Learning these regularities without labels yields useful visual features.
- V1 (primary visual cortex) neurons have localized, oriented, bandpass receptive fields (Gabor-like). These can emerge from sparsity or ICA constraints on whitened patches.
- V2 (next stage) integrates V1 outputs to build sensitivity to more complex shapes (corners, junctions, contours) via pooling/combination of V1 features.
- Unsupervised, constraint-driven learning (sparsity, independence) explains much of early visual selectivity without labels or backprop.

## 2) Data preprocessing: patches, normalization, whitening

- Patch extraction: Random small image patches (e.g., 12×12) are sampled from natural images (or 1/f synthetic textures for quick tests).
- Local contrast normalization: Each patch is mean-subtracted and scaled to unit standard deviation to remove absolute illumination.
- ZCA whitening: Decorrelates pixels and equalizes variance across directions:
  - Compute patch mean μ and covariance Σ, then eigendecompose Σ = U Λ Uᵀ
  - ZCA matrix: W = U (Λ + εI)^{-1/2} Uᵀ; whitened patches X_w = (X − μ) Wᵀ
  - Whitening avoids trivial solutions and promotes localized edge detectors during sparse coding.

## 3) Sparse coding for V1: objective and algorithm

We learn an overcomplete dictionary D (K atoms) so that whitened patches X_w are reconstructed from sparse codes C:

- Objective (row-wise codes): minimize 0.5 ||X_w − C D||²_F + λ ||C||₁
- Alternating optimization:
  1) Code step: with D fixed, estimate C via ISTA (Iterative Soft-Thresholding):
     - Gradient step on 0.5||X − C D||² then soft-threshold with λ
     - Converges to Lasso solution for each sample row
  2) Dictionary step: with C fixed, least-squares update with Tikhonov ε:
     - D = (Cᵀ C + εI)^{-1} Cᵀ X, then normalize atoms to unit norm

Why this yields Gabor-like filters:
- After whitening, edges dominate variability. Sparsity favors basis elements that compactly explain edges with few active codes, leading to localized, oriented, bandpass atoms resembling V1 simple-cell receptive fields.

## 4) From V1 to V2: pooling over neighborhoods

- Compute V1 codes over an image on a grid (patch_size, stride).
- Build V2 training samples by concatenating codes from 2×2 neighboring V1 positions → a 4K-dimensional vector per location.
- Train another sparse coding dictionary on these 4K vectors.
- Interpreting V2 atoms: Each V2 atom is a weighted combination of four V1 code vectors; projecting back by recombining the corresponding V1 filters in a 2×2 layout visualizes corner/junction-like composites.

Intuition: Pooling neighboring oriented edges naturally creates selectivity for co-occurrence patterns like corners, T-junctions, or short contours.

## 5) Training flow in the demo

1) Load images or generate 1/f textures; extract many random patches.
2) Normalize and ZCA whiten patches.
3) Train V1 by alternating ISTA (codes) and closed-form dictionary updates for several iterations; track reconstruction error and sparsity.
4) Choose one image; compute a grid of V1 codes (patch_size, stride).
5) Build V2 training set from 2×2 neighborhoods of V1 codes; normalize per feature.
6) Train V2 with the same alternating scheme; track error and sparsity.
7) Plot: V1 filters, V2 composites (projected), and learning curves (errors and sparsity fractions).

## 6) Reading the plots

- V1 filters: Grid of normalized filters. After sufficient training and data, most atoms should look like localized, oriented, bandpass patterns.
- V2 composites: Each tile is a 2×2 arrangement of V1 filters weighted by a V2 atom’s coefficients. Look for corners, junctions, or small contour fragments.
- Learning curves:
  - Reconstruction error: should decrease and stabilize.
  - Sparsity fraction (|c|>0): indicates average code density. Expect small fractions (e.g., 5–20%).

## 7) Hyperparameters and tuning

- Patch size: 8–16 (12 is a sweet spot). Smaller patches learn localized edges; larger patches can capture broader structure but need more data.
- Overcompleteness K₁: choose K₁ > patch_dim (e.g., 128 atoms for 12×12=144 dims is slightly undercomplete; 256 becomes overcomplete). Overcomplete dictionaries often yield crisper Gabors.
- Sparsity λ₁, λ₂: higher λ → sparser, potentially more Gabor-like but higher reconstruction error. Try λ₁ ∈ [0.1, 0.25], λ₂ slightly smaller.
- Iterations: thousands of outer iterations with short ISTA inner steps often suffice; increase if filters remain noisy.
- Whitening ε: too small can be unstable; 1e−2 is a robust default.
- Stride: controls overlap and the V1 code grid resolution for V2.
- Dataset size: more patches produce cleaner filters and reduce overfitting to noise.

## 8) Stability and practical tips

- Normalize dictionary atoms after each update; without normalization, atoms can explode or collapse.
- Standardize V2 inputs (mean 0, unit std per dimension) to keep scales balanced.
- If V1 filters look noisy:
  - Increase dataset size and iterations; check that whitening was applied.
  - Reduce λ₁ slightly; increase ISTA steps.
- If V2 composites look uninterpretable:
  - Ensure V1 filters are good first; increase 2×2 training samples by using more images; standardize V2 inputs.

## 9) Relationship to ICA and V1 physiology

- Sparsity on whitened data is closely related to ICA: both promote statistical independence of components. ICA on natural images also yields Gabor-like filters.
- Constraints (locality, sparsity, bandpass) align well with known V1 simple-cell receptive fields.
- Hierarchical pooling of independent components yields higher-order selectivity like corners and junctions, consistent with V2.

## 10) Extensions

- Overcomplete ICA: replace L1 sparse coding with FastICA or auxiliary-variable variants; compare filters.
- Structured sparsity: group lasso or top‑k codes to induce competition and diversity.
- Convolutional dictionary learning: learn shift-shared filters over full images instead of sampled patches.
- Multi-scale: train separate V1 dictionaries for different patch sizes; combine into V2.
- Temporal/motion stretch goal: sample spatiotemporal patches (x,y,t) and add temporal prediction (e.g., minimize future reconstruction error) to learn motion‑selective features.

## 11) Quick recipe

- Start: `--use-1overf --patch-size 12 --K1 256 --lam1 0.15 --iters1 3000`
- Then V2: `--K2 128 --lam2 0.1 --iters2 2000 --stride 6`
- Expect recognizable V1 Gabors and V2 composites showing simple corners/junctions after a few minutes.

## 12) Takeaways

- Whitening + sparsity on patches → V1‑like filters without labels.
- Pooling over local neighborhoods of V1 codes → V2‑like composite selectivity.
- Hierarchical selectivity can emerge from simple, biologically plausible constraints rather than supervision or backpropagation.


