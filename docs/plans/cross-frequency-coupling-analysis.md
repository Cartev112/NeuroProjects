## Cross-Frequency Coupling (CFC) Analysis – Project Plan

### Goal
Quantify interactions between frequency bands (e.g., theta–gamma PAC) and test their relation to cognition and behavior.

### Data
- EEG/MEG with task markers; optional sleep datasets.
- BIDS inputs; channel locations for topographies.

### Methods
- Bandpass filter sets; Hilbert transform for phase/amplitude extraction.
- Metrics: Modulation Index (Tort), Phase–Amplitude Coupling (Canolty), Cross-Frequency Coherence, GLM-based PAC.
- Control analyses: surrogate data (time-shift, phase-shuffle), power confound regressors.

### Pipeline
1) Preprocess (filtering/notch, bad channel handling, referencing).
2) For each low/high band pair and sensor/ROI:
   - Extract phase(low) and amplitude(high) via Hilbert.
   - Compute PAC metric and null distribution via surrogates.
3) Aggregate across trials; create subject maps and group stats.
4) Optional time-resolved PAC with sliding windows.

### Outputs
- PAC comodulograms (low-phase × high-amp grid), sensor/ROI maps, CSV of PAC strength per condition.
- Statistical maps with multiple-comparison correction.

### Milestones
- M1: PAC prototype on one subject/channel.
- M2: Surrogate framework and significance testing.
- M3: Group-level comodulograms and topographies.
- M4: Condition/behavior associations.

### Risks
- Filter leakage; edge artifacts; PAC inflation by broadband power.

### Dependencies
- numpy, scipy, mne, matplotlib, scikit-learn, statsmodels.

### Project Rationale

#### What is the point of this project?
The project's purpose is to investigate a key hypothesis in neuroscience: that slow brain oscillations (e.g., theta waves) act as a temporal framework that organizes the firing of faster oscillations (e.g., gamma waves). This "cross-frequency coupling" is thought to be a fundamental mechanism for information routing, neural computation, and memory formation. This project aims to provide a robust toolkit to detect and quantify this phenomenon.

#### What can be learned from it?
A user will learn the signal processing techniques required to measure cross-frequency coupling, primarily phase-amplitude coupling (PAC). They will understand the critical importance of methodological rigor, including how to implement surrogate data testing (e.g., time-shifting) to control for false positives and how to mitigate potential confounds like non-sinusoidal waveform shape or broadband power fluctuations.

#### What does it provide?
This project provides a pipeline to:
1.  Calculate PAC using established metrics (e.g., Modulation Index).
2.  Visualize the coupling in "comodulograms" that show which pairs of frequencies are interacting.
3.  Implement a statistical framework to assess whether the observed coupling is stronger than expected by chance.
4.  Generate topographical plots to identify the brain regions where CFC is most prominent during a given task.

