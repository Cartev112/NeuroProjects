# Learning Doc: Memory with Attractor Networks (Hopfield Net)

This document introduces Hopfield networks as content-addressable memories, explains the theory (energy landscape, storage rule, capacity), and shows how the demo stores and retrieves patterns and analyzes capacity vs noise.

## 1) What is a Hopfield network?

A Hopfield network is a recurrent network of binary units that behaves like an associative memory. You store patterns by setting the synaptic weights once (Hebbian learning). Later, given a partial or noisy version of a stored pattern, the network dynamics relax into an attractor near the original, retrieving the memory.

- Units: states `s_i ∈ {−1, +1}` (bipolar convention is standard and simplifies the math)
- Weights: symmetric, zero diagonal `W_ii = 0`, no external biases here
- Dynamics: iteratively update `s` by thresholding the input `h = W s`, either synchronously (all at once) or asynchronously (one unit at a time)

Hopfield networks define an energy function that decreases under asynchronous updates, guaranteeing convergence to a stable point (local minimum). These stable points are the attractors (stored memories and some spurious mixtures).

## 2) Energy landscape and dynamics

Energy (no biases):

\[ E(s) = -\tfrac{1}{2} s^\top W s. \]

- Asynchronous update (one unit at a time) with symmetric `W` and zero diagonal monotonically decreases (or keeps) `E(s)` and therefore converges.
- Synchronous update is not guaranteed to strictly decrease `E` every step but often converges in practice; it can oscillate between states in edge cases.

Update rules (sign threshold):

- Synchronous: `s ← sign(W s)`
- Asynchronous: pick a random order of units; for each `i`, set `s_i ← sign((W s)_i)` using most recent updates (Gauss–Seidel style)

We use `sign(x) = +1` if `x ≥ 0`, else `−1`.

## 3) Storage (Hebbian learning)

For bipolar patterns `x^p ∈ {−1, +1}^N`, `p = 1..P`, the classic Hebbian weight matrix is:

\[ W = \frac{1}{N} \sum_{p=1}^{P} x^p (x^p)^\top, \quad \text{with } W_{ii} = 0. \]

This makes each stored pattern a (approximate) fixed point: if you plug `x^p` into the update rule, it tends to remain unchanged (in the low-load limit `P \ll N`). The diagonal is set to zero to avoid self-feedback.

Notes:
- Bipolar coding (±1) is preferred; if your data is 0/1, remap to ±1 (`x’ = 2x − 1`).
- Symmetry `W = W^T` and zero diagonal are important for guaranteed convergence with asynchronous updates.

## 4) Retrieval and basins of attraction

Given a cue `c` that is a noisy version of a stored pattern `x`, the network iterates updates until convergence. If the cue lies within the basin of attraction of `x`, the final state will match `x` (or its negative, which is equivalent without biases). We assess retrieval quality via overlap (magnetization):

\[ m(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} x_i \hat{x}_i \in [-1,1]. \]

We often treat retrieval as successful if `|m| ≥ 0.95` (tunable).

## 5) Capacity and noise robustness

- Capacity (random, independent patterns): The classical theoretical storage capacity of a fully connected Hopfield net with Hebbian weights is about \( P_{max} ≈ 0.138 N \) for low retrieval error rates. Beyond this load, basins shrink rapidly and spurious states proliferate.
- Noise robustness: For a fixed `P`, increasing input corruption (bit flip probability `q`) eventually pushes cues out of basins; success rate drops as `q` grows.

The demo empirically measures both:
- Capacity curve: success vs `α = P/N` for a fixed cue noise level
- Noise curve: success vs flip probability `q` for a fixed `P`

## 6) What this demo implements

- Pattern generation: `P` random bipolar patterns of length `N`
- Training: Hebbian rule with zero diagonal
- Recall: synchronous or asynchronous updates, up to `max_steps`, tracking energy
- Noise injection: independently flip each bit with probability `q`
- Metrics: overlap-based success, number of iterations to converge
- Analyses: capacity sweep over `P`, noise sweep over `q`
- Plots: target, noisy cue, retrieved; capacity curve; noise curve; energy over iterations

CLI highlights (see README for full list):
- `--N`, `--P`: size and number of stored patterns
- `--mode`: `sync` or `async`
- `--max-steps`: iterations for recall
- `--flip-prob`: cue corruption level (used in capacity trials)
- `--alpha-max`, `--cap-evals`: capacity sweep range & resolution
- `--noise-max`, `--noise-evals`: noise sweep range & resolution

## 7) Interpreting the plots

- Target / Noisy cue / Retrieved: Visual check of retrieval. If `N` is a perfect square, patterns are displayed as images; otherwise as 1×N stripes. Retrieved should match target when success is high.
- Capacity analysis: Success vs `α = P/N`. Expect a decreasing curve that drops near `α ≈ 0.1–0.2`, depending on thresholds and randomness.
- Noise robustness: Success vs flip probability `q`. Expect monotonic decrease; slope depends on `N`, `P`, and update mode.
- Energy trace: Under asynchronous updates, energy should monotonically decrease to a minimum.

## 8) Practical tips and variations

- Start with moderate `N` (e.g., 256) and `P` (e.g., 20–40). Too small `N` makes curves noisy.
- Asynchronous updates typically offer better convergence guarantees (energy decreases each microstep). Synchronous updates are faster per iteration but can occasionally oscillate.
- Overlap threshold: Tighter thresholds (e.g., 0.99) demand perfect recovery; looser thresholds (e.g., 0.9) measure partial recovery.
- Spurious states: Mixtures of stored patterns can appear as attractors. Reducing `P` or using pseudo-inverse learning can mitigate them.
- Alternative learning rules: Storkey or pseudo-inverse learning increase effective capacity at added computational cost.
- Bias terms: Adding biases (or using non-zero mean patterns) shifts attractors; keep patterns zero-mean in ±1 coding for classic behavior.

## 9) Suggested experiments

- Capacity sweep: Increase `--alpha-max` and `--cap-evals`; plot more points and observe where success drops.
- Noise sweep: Increase `--noise-max` up to 0.5; compare `async` vs `sync` performance.
- Pattern structure: Store structured patterns (e.g., letters) and observe retrieval; compare to random patterns at same `P`.
- Scaling: Double `N` keeping `α` fixed; basins and shapes should be similar while absolute `P` grows.

## 10) Key takeaways

- Hopfield nets are energy-based attractor memories with simple Hebbian storage and deterministic recall.
- Capacity scales linearly with `N` but with a small constant (~0.138 for random patterns and Hebbian rule).
- Robustness degrades as `P` increases or noise increases; asynchronous updates provide better convergence guarantees.
- This demo gives a hands-on way to explore these trade-offs and visualize retrieval and energy dynamics.


