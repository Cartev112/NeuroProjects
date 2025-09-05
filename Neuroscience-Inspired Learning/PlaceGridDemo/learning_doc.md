# Learning Doc: Place–Grid Cells – From Grid Codes to Place Fields and Decoding

This document explains the ideas behind the Place–Grid Cells demo: how multi-scale grid codes can be generated, how sparse coding over grid codes yields localized place-like fields, and how to decode position linearly from grid activity. It also includes guidance on interpretation, tuning, and extensions.

## 1) Biological context (very short)

- Grid cells (medial entorhinal cortex) fire in a periodic, hexagonal lattice over 2D space. Modules of grid cells share a scale but differ in phase and orientation.
- Place cells (hippocampus) fire in one or a few localized regions (place fields).
- Many models derive place fields by combining grid codes (e.g., linear combinations, attractors, sparse coding). Grid codes provide a basis that can represent space with high accuracy; place cells specialize as local features.

## 2) Simulating a 2D trajectory

- A random-walk-like policy with reflective boundaries explores a square arena of size `arena`.
- At each step `t`: update heading with Gaussian noise, move at a speed sampled around `speed_mean`, then reflect off walls.
- Outputs:
  - `pos[t] = (x_t, y_t)` in `[0, arena]^2`
  - `t_s[t]` timestamps in seconds (for plotting/reference)

## 3) Generating grid-cell responses

The demo builds a population of grid cells across multiple modules (scales) and random orientations/phases.

- For each cell g:
  - Pick a module scale s ∈ {s₁, s₂, …}. Wavelength λ = s · arena, spatial frequency (cycles/unit) = 1/λ, angular frequency ω = 2π/λ.
  - Choose a base orientation θ and derive three lattice directions separated by 60°: u₀ = [cos θ, sin θ], u₁ = R₆₀ u₀, u₂ = R₁₂₀ u₀.
  - Compute interference signal along the path: s(t) = (cos(ω u₀·x(t)+φ₀) + cos(ω u₁·x(t)+φ₁) + cos(ω u₂·x(t)+φ₂))/3.
  - Apply a nonlinearity (ReLU or exp) and small Gaussian noise.
- The result is an array X ∈ ℝ^{T × G} of grid responses for T steps and G cells.

Why this produces hexagonal grids: the sum of three plane waves 60° apart yields a hexagonal interference pattern in 2D.

## 4) From grid codes to place cells via sparse coding

We learn a sparse dictionary D (place units) so that standardized grid codes X_z are reconstructed from sparse place codes C:

- Standardization: X_z = (X − μ)/σ per grid cell to normalize scales.
- Objective: minimize 0.5 ||X_z − C D||²_F + λ ||C||₁ with K place atoms (K < G typically).
- Alternating optimization (as in the V1 demo):
  1) ISTA for codes C given D
  2) Closed-form least-squares update for D given C (with ε regularization) and per-row normalization

Intuition: Sparse coding finds a small number of columns in D that, linearly combined by codes C, explain the grid patterns over time. Each row of C (a place unit) tends to be active only when a specific configuration of grid phases occurs—often corresponding to a localized spatial region, producing a place field when mapped back to 2D.

## 5) Visualizing place fields

Given the trajectory `pos[t]` and the code time-series for a place unit k (the kth column of C), the place field map is formed by occupancy-normalized accumulation:

- Accumulate activity into spatial bins: A(x,y) = Σ_t 1_{bin(x_t,y_t)} · C[t,k]
- Occupancy O(x,y) = Σ_t 1_{bin(x_t,y_t)}
- Place field PF(x,y) = A(x,y) / (O(x,y)+ε), normalized to [0,1] for display.
- Localized hotspots in PF indicate strong place specificity.

## 6) Decoding position from grid codes

We fit a simple linear decoder from standardized grid activity to (x,y):

- Train/test split over time indices
- Ridge regression with bias: augment X_z with ones, solve (XᵀX + λI)W = XᵀY, where Y = positions
- Evaluate decoded path over the full sequence for visualization and error analysis (e.g., median absolute error).

Grid codes are highly informative: even linear decoders can achieve small errors if scales and coverage are diverse enough and data is sufficiently sampled.

## 7) How to read the demo plots

- Trajectory (true vs decoded): Overlap indicates good decoding; divergence indicates capacity limits or insufficient data.
- Sample grid fields: Hexagonal lattice structure for randomly chosen grid cells.
- Top place fields: Choose place units with largest activation energy; expect clear, localized hotspots.
- Learning curves: Reconstruction MSE should drop and stabilize; sparsity fraction stabilizes in a low range (e.g., 5–20%).

## 8) Hyperparameters and tuning

- Trajectory length: `--steps`; longer paths improve coverage and decoding.
- Grid cells: `--grid-cells`; more cells and more modules improve decoding and place field quality.
- Place cells: `--place-cells`; choose fewer than grid cells (e.g., 96 for 240 grid cells). Too many place atoms may reduce sparsity benefits.
- Sparse coding: `--lam` controls sparsity (higher → sparser but higher error), `--iters` and `--steps-per-iter` control convergence.
- Arena: `--arena` scales coordinates; modules are expressed as fractions of arena size.

Tips:
- Use at least 2–3 modules with distinct scales; add more grid cells to reduce aliasing.
- Standardization of X is important; without it, dictionary learning can be biased by high-variance cells.

## 9) Troubleshooting

- Place fields look diffuse:
  - Increase trajectory steps; ensure coverage across space.
  - Increase λ (more sparsity) or train longer.
- Decoding is poor:
  - Increase grid cells and module diversity; lengthen trajectory; add ridge strength in decoder if needed.
- Grid fields look odd:
  - Check module scales; too similar scales reduce diversity. Ensure nonlinearity and noise are sane (`relu` is a good default).

## 10) Extensions

- Nonlinear decoders: kernel ridge or small MLP to improve decoding.
- Boundary effects: add walls/obstacles and velocity modulation; examine grid distortions.
- Head-direction modulation: add directional tuning to grid responses.
- Place field competition: add kWTA or non-negativity constraints to D and C.
- Replay: train on a path, then drive grid activity with “imagined” sequences and reconstruct trajectories.
- 3D spaces or irregular arenas: update field computation accordingly.

## 11) Quick-start settings

- Default: `--steps 4000 --grid-cells 240 --place-cells 96 --lam 0.1 --iters 300`
- For higher fidelity decoding: `--steps 8000 --grid-cells 360 --place-cells 128`

## 12) Takeaways

- Multi-scale, multi-orientation grid codes provide a powerful, compact basis for representing space.
- Sparse coding over grid codes can naturally yield localized place fields.
- Positions can be decoded linearly from grid population activity, illustrating how downstream regions could recover spatial variables from entorhinal codes.


