## Spectral Analysis of Music Perception – Project Plan

### Goal
Relate musical structure to brain spectral dynamics and decode musical attributes/emotion from spectral features.

### Data
- EEG/MEG during naturalistic music listening; annotations (tempo, key, rhythm onsets, sections) from MIR toolkits.
- Optional behavioral ratings (valence/arousal) per segment.

### Methods
- Spectral features: bandpower, spectral entropy, aperiodic/periodic (FOOOF) components.
- MIR features: tempo, beat strength, spectral centroid, chroma, MFCCs.
- Alignment via beat-synchronous windows; cross-correlation and encoding models (ridge) between MIR and neural spectra.
- Connectivity: coherence between auditory/frontal sensors across bands.

### Pipeline
1) Preprocess EEG/MEG; align audio and events; extract MIR features with librosa/essentia.
2) Compute per-window spectral features and coherence.
3) Encode MIR → neural (ridge/Lasso); decode emotion/section labels from neural spectra (SVM/RF).
4) Visualize time courses, importance maps, and confusion matrices.

### Outputs
- CSV of spectral features, model scores (R²/accuracy), and figures (spectral timecourses, coherence maps).

### Milestones
- M1: MIR extraction + basic bandpower alignment.
- M2: Encoding/decoding baselines.
- M3: Subject/group analysis and robustness.

### Risks
- Alignment drift; subject variability; musical diversity.

### Dependencies
- numpy, scipy, mne, librosa, scikit-learn, matplotlib, pandas.

### Project Rationale

#### What is the point of this project?
This project aims to bridge the gap between the rich, structured world of music and the language of brain signals. Its goal is to systematically explore how the brain's spectral activity tracks and represents complex musical features in real-time. It moves beyond simple tones to understand neural processing in a more naturalistic, engaging context, with the potential to decode both perceptual and emotional responses to music.

#### What can be learned from it?
A user will learn a multi-domain analysis workflow. This includes extracting meaningful acoustic and structural features from audio using Music Information Retrieval (MIR) toolkits. They will then learn how to align these features with continuous neural recordings and apply machine learning techniques, such as encoding models (predicting brain activity from music) and decoding models (predicting musical attributes from brain activity), to formally link the two domains.

#### What does it provide?
This project delivers a framework to:
1.  Process and synchronize musical audio features with corresponding EEG/MEG recordings.
2.  Implement and evaluate both encoding and decoding models to quantify the strength of the brain-music relationship.
3.  Generate visualizations that show how specific neural frequency bands track musical elements like rhythm, tempo, and harmony over time.
4.  Serve as a basis for exploring more advanced topics like neural entrainment or emotional classification from listening data.

