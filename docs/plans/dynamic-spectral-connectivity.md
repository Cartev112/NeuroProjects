## Dynamic Spectral Connectivity – Project Plan

### Goal
Estimate time-varying spectral connectivity between regions/sensors and relate dynamics to tasks/states.

### Data
- EEG/MEG (events available) or rs-fMRI-derived band-limited power time series.
- BIDS inputs; derived epochs saved per subject.

### Methods
- Sliding-window and multitaper spectral estimates (Welch/multitaper).
- Connectivity: coherence, imaginary coherence, PLV, debiased wPLI.
- Dimensionality reduction: PCA/ICA on connectivity vectors; HMM/k-means for state discovery.
- Surrogates for significance (phase randomization) and FDR control.

### Pipeline
1) Load BIDS → preprocess (filter, montage) → epoch.
2) Per window: compute band-limited spectra and connectivity metrics.
3) Stack time-by-edges matrices; reduce dimensionality; cluster into states.
4) Extract features: dwell time, transition rates, state-specific networks.
5) Relate features to behavior/conditions via GLMs or mixed models.

### Outputs
- Time-resolved connectivity plots; state maps; subject-level CSV of features.
- Reproducible CLI and config.

### Milestones
- M1: Prototype windowed coherence on sample subject.
- M2: Add wPLI/PLV and band selection.
- M3: State modeling (HMM/k-means) and metrics.
- M4: Group analysis and reporting.

### Risks
- Window length trade-off; leakage/volume conduction; high dimensionality.

### Dependencies
- numpy, scipy, mne, scikit-learn, matplotlib, hmmlearn (optional).

### Project Rationale

#### What is the point of this project?
The core idea is to move beyond static "snapshot" views of brain connectivity. By analyzing how spectral connections between brain regions evolve over milliseconds to seconds, we can capture the fluid, dynamic nature of neural communication as the brain shifts between different tasks, processes information, or transitions between cognitive states. It aims to provide a "movie" of brain network interactions, not just a single picture.

#### What can be learned from it?
A user can learn the fundamentals of time-resolved connectivity analysis, including the trade-offs of sliding-window estimation. They will gain practical experience with different spectral connectivity metrics (e.g., coherence, wPLI) and understand why volume-conduction-resistant measures are important. Finally, it introduces an unsupervised machine learning approach (clustering) to discover and quantify recurring, large-scale brain network "states" from high-dimensional connectivity data.

#### What does it provide?
This project provides a reusable command-line tool to:
1.  Compute time-varying connectivity matrices from EEG/MEG data for various frequency bands and metrics.
2.  Optionally identify and label distinct brain network states over time.
3.  Generate quantitative outputs like state dwell times and transition rates, which can be correlated with experimental conditions or behavior.
4.  Produce visualizations of state-specific network topologies and their temporal dynamics.

