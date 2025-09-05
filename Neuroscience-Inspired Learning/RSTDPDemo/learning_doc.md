# Learning Doc: Reinforcement‑Modulated STDP in a Recurrent Spiking Network

This document explains the concepts behind combining biologically plausible spike‑timing dependent plasticity (STDP) with a global dopamine‑like reinforcement signal, how the provided demo implements it, how to interpret the outputs, and how to extend it to more challenging tasks.

## 1) Big picture: why R‑STDP?

- STDP is local and unsupervised: synapses strengthen or weaken based on spike timing (Hebbian/anti‑Hebbian). Powerful for learning structure, but it lacks task guidance.
- Reinforcement learning provides a global scalar signal (reward) indicating whether performance improved. Dopamine in the brain is a classic neuromodulator for this.
- R‑STDP combines the two: local synapses keep an eligibility trace of “credit” from spike‑timing, and a delayed global reward gates whether that credit turns into an actual weight change. This links local events to distant outcomes.

In this demo, a small recurrent LIF network learns to regulate the firing rate of an output population toward a context‑dependent target using only R‑STDP (no backpropagation).

## 2) Model overview

- Neurons: Leaky Integrate‑and‑Fire (LIF), absolute refractory, additive noise.
- Network: Recurrent E/I random connectivity (E: positive weights, I: negative), 80/20 split by default.
- Synapses: Current‑based with exponential decay.
- Plasticity: Pair‑based STDP (pre/post traces) → eligibility traces; dopamine‑modulated consolidation.
- Readout: A small “output population” of neurons; its low‑pass filtered rate should match a target that switches mid‑episode (two contexts).
- Reward: Decrease in absolute error between output rate and current target (advantage = reward − baseline), treated as dopamine.
- Stabilization: E/I constraints, weight bounds, row‑norm soft cap, inhibition stronger than excitation, small learning rates.

## 3) Core equations

### 3.1 Neuron and synapse dynamics (per time step ∆t)

- Synaptic current (exponential):
  \[ s(t+∆t) = s(t)·e^{−∆t/τ_{syn}} + \sum_{i∈spikes(t)} w_i \]
- Membrane potential (Euler):
  \[ V(t+∆t) = V(t) + \frac{∆t}{τ_m}\big( −(V−V_{rest}) + R_m·(I_{ext} + s + I_{ctx}) \big) + σ·\xi \]
  Threshold and reset with absolute refractory as usual.

### 3.2 STDP traces and eligibility

We track pre and post spike traces:
- \(\text{pre}_i(t) ← \text{pre}_i(t)·e^{−∆t/τ_{pre}};\, \text{post}_j(t) ← \text{post}_j(t)·e^{−∆t/τ_{post}}\)
- On spike: \(\text{pre}_i{+}{=}1\) if neuron i spikes; \(\text{post}_j{+}{=}1\) if neuron j spikes

Eligibility for synapse i→j decays with \(τ_e\) and is updated by pair‑based increments:
- \(e_{ij} ← e_{ij}·e^{−∆t/τ_e}\)
- Post spike j: \(e_{ij} {+}{=} A_+·\text{pre}_i\)
- Pre spike i: \(e_{ij} {+}{=} A_-·\text{post}_j\) (with \(A_-<0\))

Interpretation: \(e_{ij}\) accrues potentiation or depression credit based on recent spike timing, but does not immediately change \(w_{ij}\).

### 3.3 Dopamine‑modulated weight update (consolidation)

Let \(δ(t)\) be the dopamine/advantage signal (reward minus a running baseline). Then:
\[ ∆w_{ij} = η · δ(t) · e_{ij} \]
Applied only on existing synapses. After the update:
- Enforce E/I sign constraints and bounds (E in [0, w_max], I in [w_min, 0])
- Optional row‑norm soft cap: \(w_i ← w_i · \min(1, c/\lVert w_i\rVert)\)

### 3.4 Reward shaping and readout

- Output population rate is low‑pass filtered:
  \[ r(t+∆t) = r(t)·e^{−∆t/τ_{ro}} + (1−e^{−∆t/τ_{ro}})·\hat{r}(t) \]
  where \(\hat{r}(t)\) is instantaneous population rate (Hz).
- Error: \(E(t) = |r(t) − r^*(\text{context}(t))|\)
- Reward: improvement \(R(t) = E(t−∆t) − E(t)\)
- Baseline (exponential moving average) \(b(t)\); advantage \(A(t)=R(t)−b(t)\); dopamine \(δ(t) = g·A(t)\)

## 4) Training loop and task

Each episode (e.g., 1500 ms) is split into two contexts; the target rate switches halfway (e.g., 5 Hz then 25 Hz). During every step:
1) Apply synaptic decay and previous spikes → currents
2) Integrate LIF with contextual drive to a subset of excitatory neurons
3) Update readout rate, compute current error, reward, advantage/dopamine
4) Update STDP traces → eligibilities; consolidate with \(∆w ∝ δ·e\)

Across episodes, dopamine advantage should trend positive while mean absolute error decreases, indicating successful context‑dependent regulation.

## 5) Stabilization strategies (critical!)

- E/I structure: keep inhibition stronger than excitation (e.g., `g ≈ 3–5`).
- Weight bounds: clamp excitatory in [0, w_exc_max], inhibitory in [w_inh_min, 0].
- Soft row‑norm cap: limit total outgoing strength per neuron.
- Small plasticity rate: `eta ≈ 1e‑3` or smaller; increase only after stability is verified.
- Noise and refractory: mild voltage noise and refractory help desynchronize and avoid lock‑in.
- Reward shaping: advantage (reward minus baseline) reduces drift/variance and prevents weight changes when no improvement occurs.

If you see runaway excitation (rates exploding): increase `g`, reduce `w_exc_max`, lower `eta`, or shorten `tau_e`.

## 6) Interpreting the outputs

- Spike raster (last episode):
  - Asynchronous irregular activity is typical; strong bands suggest synchrony/oscillation (may need stronger inhibition or lower `J_init`).
- Output vs target (last episode):
  - Output should track each context’s target better over training. Watch transients at the switch point.
- Learning curves:
  - Dopamine advantage per episode should drift positive/settle near zero; mean |error| should decrease.
- Weight statistics:
  - Mean/std provide a coarse view of synaptic distribution; check excitatory weights aren’t saturating at `w_exc_max`.

## 7) Hyperparameter guidance

- Network size: `N=100–300` keeps it tractable; more neurons need smaller `eta`.
- Connectivity: `p=0.05–0.15`. Too dense increases correlations; too sparse lowers drive.
- Inhibition: `g=3–5` is a good starting range.
- Initial weights: small `J_init=0.03–0.08` encourage stability.
- STDP magnitudes: `A_plus≈0.01`, `A_minus≈−0.012`; shorten `tau_e` for faster credit decay.
- Learning rate: `eta=5e‑4–2e‑3`.
- Readout: `tau_readout=30–80 ms` balances smoothness vs responsiveness.
- Reward baseline: `reward_baseline_tau=300–800 ms` for stable advantage.

## 8) Troubleshooting and failure modes

- Exploding activity:
  - Increase `g`, reduce `w_exc_max`, apply stronger row‑norm cap, or lower `eta`.
- No learning (flat curves):
  - Increase `eta` slightly; ensure target difference between contexts is large enough; reduce noise; shorten `tau_e`.
- Over‑synchronization:
  - Reduce `J_init`, `tau_syn`, or increase noise; decrease connectivity `p`.
- Output stuck near one target:
  - Increase context drive separation or output population fraction; verify dopamine gain > 0.

## 9) Extending to control tasks (stretch goal)

- Replace context targets with a control error (e.g., cart‑pole angle magnitude). Define reward as reduction in error or episodic return (survival time).
- Action readout: map output rate(s) to an action probability (e.g., left/right via softmax over two output pops). Sample action each control step.
- Credit assignment: keep eligibility traces continuous over environment steps; apply dopamine based on immediate TD‑like error (or return). A simple baseline can be the running reward mean.
- Safety: begin with a strong stabilizing controller (e.g., PD) and let the SNN learn residual corrections; then anneal the baseline controller.

## 10) Connections to theory & biology

- Three‑factor learning rules: pre‑synaptic activity, post‑synaptic activity, and a modulatory factor (dopamine) are sufficient for policy gradient‑like updates.
- Dopamine as reward prediction error: advantage‑based gating mirrors temporal‑difference ideas.
- Homeostatic plasticity and inhibitory plasticity are common biological mechanisms to stabilize Hebbian learning—consider adding them for larger networks.

## 11) How to use this repo piece

- See `RSTDPDemo/README.md` for commands and options.
- Start small (`N=120`, default params). Verify output tracks context targets and curves improve.
- Then tune `eta`, `g`, `w_exc_max`, and `tau_e` in small steps.

## 12) Summary

R‑STDP couples local STDP credit assignment with a global scalar reward to achieve goal‑directed learning without backpropagation. Stabilization (E/I balance, caps, baselines) is essential. With careful tuning, spiking networks can learn simple regulation and control tasks under biological constraints.


