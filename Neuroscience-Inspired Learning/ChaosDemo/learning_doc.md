# Learning Doc: Chaos in Recurrent Spiking Networks

This document explains the intuition, model, and practical usage of the recurrent LIF network in `ChaosDemo/chaos_demo.py`, and how to interpret the outputs to distinguish stable/asynchronous regimes from irregular/chaotic ones.

## 1) Why chaos in spiking networks?

Large recurrent networks of excitatory (E) and inhibitory (I) neurons can exhibit qualitatively different dynamical regimes depending on network parameters:
- Stable/asynchronous: spikes are weakly correlated, rates are steady, and variability is modest.
- Irregular/chaotic: activity is highly variable, rates fluctuate, interspike intervals are irregular, and small perturbations can significantly alter trajectories.

A central control knob is the **E/I balance**. If inhibition (relative to excitation) is too weak, activity can run away or synchronize; if it is too strong, the network is quiescent. Near the “edge of chaos,” inhibition precisely counterbalances excitation, producing asynchronous irregular (AI) dynamics reminiscent of cortical activity in vivo.

## 2) Model overview

We simulate a current-based LIF (leaky integrate-and-fire) network with exponential synapses and random E/I connectivity.

- Population: `N` neurons, with `frac_exc` fraction excitatory; the rest inhibitory.
- Connectivity: each ordered pair connects with probability `p`; no self-connections.
- Synaptic weights:
  - Excitatory: +`J`
  - Inhibitory: −`g * J` (so `g` controls relative inhibition strength)
- Neuron dynamics (per-step Euler):
  - Membrane: `dV ∝ −(V − V_rest) + R_m * (I_ext + s)` where `s` is synaptic current
  - Threshold/reset and absolute refractory
  - Additive voltage noise with std `noise_std`
- Synapse dynamics: exponential decay with time constant `tau_syn`; presynaptic spikes increment postsynaptic synaptic state by weight.

Key parameters to explore:
- `J` (excitatory scale), `g` (inhibitory multiplier), `p` (connectivity), `tau_m`, `tau_syn`, `I_ext`, `noise_std`.

## 3) How the simulation steps

At each time step `dt`:
1. Decay synaptic states: `s ← s * exp(−dt / tau_syn)`
2. Apply spikes from previous step: for every spiking neuron i, add weights to targets’ `s`
3. Update membrane potentials for non-refractory neurons via Euler integration with leak, external drive, synaptic current, and noise
4. Enforce refractory/reset and detect threshold crossings → new spikes

The code stores spike rasters and membrane traces, computes metrics, and generates plots.

## 4) Regimes and qualitative signatures

- Asynchronous/stable (balanced):
  - Mean rates moderate and relatively steady
  - ISI CV (coefficient of variation) around 0.7–1.0 (Poisson-like) without strong synchrony
  - Low pairwise correlations (near zero)
  - Population rate shows small fluctuations without large oscillations

- Irregular/chaotic / highly fluctuating:
  - Broader rate distribution across neurons
  - ISI CV ≳ 1 and visibly variable spike trains
  - Population rate exhibits large, aperiodic fluctuations
  - Slight increases in average absolute pairwise correlation (though AI chaos can still be weakly correlated)

- Over-excited/synchronous or oscillatory:
  - Clear bands in raster (synchrony), periodic oscillations in population rate
  - ISI CV can drop (< 0.5) for regular spiking or rise with bursting; correlations increase

Rule of thumb: increase `J` or `g` to push toward stronger inhibition-dominated irregular regimes; decrease to make the network quieter or more regular. The edge often lies where inhibition just reins in excitation.

## 5) Metrics reported

- Mean firing rate (overall, excitatory subset, inhibitory subset)
- ISI CV (coefficient of variation of interspike intervals, averaged over neurons with ≥3 spikes)
  - CV ≈ 1 suggests Poisson-like irregularity; CV < 0.5 suggests regularity; CV > 1 indicates burstiness/irregularity
- Average absolute pairwise correlation of spike counts in short bins (e.g., 5 ms)
  - Near zero implies asynchronous activity; larger values indicate shared fluctuations/synchrony

Use these together; no single metric uniquely defines chaos, but the constellation of signs is informative.

## 6) Plots and how to read them

- Spike raster: time on x-axis, neuron index on y-axis; look for bands (synchrony) vs cloud (asynchronous)
- Population rate (binned): fluctuations over time; steady small fluctuations for AI, large/periodic swings for oscillations
- Rate histogram: heterogeneity of single-neuron rates; broader distributions in irregular regimes

## 7) Suggested experiments

- Sweep inhibitory balance:
  - Fix `J`, vary `g` from 2 to 6. Observe transitions in CV, correlations, and raster structure.
- Sweep synaptic strength:
  - Fix `g`, increase `J` from 0.05 to 0.3. Stronger synapses amplify fluctuations; find the edge.
- Change connectivity:
  - Lower `p` reduces in-degree; may reduce correlations but also effective drive → adjust `J` accordingly.
- Synaptic time constant:
  - Increase `tau_syn` to integrate inputs longer; can promote oscillations/synchrony in some regimes.
- External drive and noise:
  - Decrease `I_ext` to reduce overall activity; increase `noise_std` to enhance irregularity without changing mean drive.

For more reproducible comparisons, keep `seed` fixed. For larger networks, increase `N` (e.g., 2k–5k) and consider increasing `duration` to gather stable statistics.

## 8) Tips and extensions

- Numerical stability: ensure `dt << tau_m, tau_syn`; defaults (0.1 ms) are conservative.
- Binning for correlations: 2–10 ms bins are typical; try different `bin_ms` in the code to see robustness.
- Extensions to try:
  - Conductance-based synapses (voltage-dependent), synaptic delays, heterogeneous thresholds/time constants
  - Structured connectivity (clusters) to induce metastable/assembly dynamics
  - External oscillatory inputs to probe resonance/phase-locking
  - Additional metrics: Fano factor of counts, power spectra of population rate, avalanche statistics

## 9) Quick start

- Start with: `--N 800 --duration 2000 --J 0.15 --g 4.0 --p 0.1`.
- If firing is too low: increase `I_ext` or decrease `g` slightly.
- If overly synchronized/oscillatory: increase `g`, reduce `J`, or shorten `tau_syn`.

These settings should expose clear differences across regimes and produce readable rasters and metrics that match expectations for balanced AI vs more chaotic/synchronous behavior.


