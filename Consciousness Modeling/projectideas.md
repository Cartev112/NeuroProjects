Global Neuronal Workspace Signatures (GNW-Ignition)
Goal: Detect “ignition” and long-range broadcasting dynamics associated with conscious access, integrating time-resolved effective connectivity and multiscale broadcasting indices.
Data:
MEG/EEG with report/no-report visual awareness or masking.
fMRI (optional) for structural/functional priors and broadcast target coverage.
Core methods:
Ignition detection: HMM/HSMM on sensor/source-level activity to detect rapid-onset, sustained high-activation episodes.
Effective connectivity: Spectral DCM or nonparametric TE/Granger to infer directed interactions.
Broadcasting index: Quantify simultaneous information flow from frontoparietal hubs to widespread targets under “seen” vs “unseen”.
Representational tests: RSA between neural states and deep vision/language models during ignition windows.
Causal structure: Constraint-based causal discovery (e.g., PCMCI+) on ignition episodes vs baseline.
Evaluation:
Seen vs unseen classification using ignition+EC features; permutation stats.
Temporal specificity of ignition relative to stimulus/report; control for motor/decision confounds.
Cross-task generalization (masking → rivalry).
Deliverables:
gnw_cli.py: fit HMM, compute broadcasting indices, run RSA and EC pipelines.
Figures: ignition rasters, hub broadcasting matrices, RSA timecourses.
Stretch:
Multimodal alignment: MEG source activity anchored by fMRI networks (e.g., frontoparietal, DMN).
Counterfactuals: remove hub connections in EC model to test broadcast loss.
3) Hierarchical Predictive Processing and Metacognition (HP3M)
Goal: Link predictive coding theory to conscious access and confidence by modeling precision-weighted prediction errors and their mapping to EEG/MEG signatures (MMN, P3b) and subjective reports.
Data:
Oddball/roving paradigms (EEG/MEG) with trial-wise confidence.
Binocular rivalry or ambiguous figures with continuous report.
Core methods:
Deep state-space modeling: Variational recurrent SSM estimating latent predictions, precision, and PE dynamics from neural time series.
Predictive coding readouts: Map latent PE/precision to canonical ERP components (MMN/P3b) and time–frequency markers (beta/gamma).
Metacognition: Jointly fit drift–diffusion or meta-d’ linking latent states to choices and confidence.
Self-supervised pretraining: Temporal contrastive/Mask modeling on raw epochs; fine-tune for latent PE decoding.
Cross-level alignment: Relate model “surprise/precision” to neural signatures and behavior on a per-trial basis.
Evaluation:
Trial-wise correlation of latent PE/precision with ERP/TF features and confidence.
Causal tests: reversing precision manipulation (attention/uncertainty) predicts P3b/behavioral changes.
Out-of-paradigm generalization (oddball → prediction under visual sequences).
Deliverables:
hp3m_cli.py: train SSM, extract PE/precision, fit DDM/meta-d’, generate trial-wise predictions.
Figures: latent trajectories vs ERP, confidence psychometrics, generalization curves.
Stretch:
Hierarchical Bayesian PCN with explicit prediction and error nodes; compare to neural network SSM.
Subject-specific vs group-level latent priors for individual differences.

1. Integrated Information Architecture Mapper (IIT-Φ Dynamics)
A multi-scale computational framework that combines:

Graph-theoretic analysis of neural connectivity patterns to compute integrated information (Φ) across different brain regions
Dynamic causal modeling to track information integration over time during different conscious states (waking, dreaming, anesthesia, psychedelics)
Topological data analysis (persistent homology) to identify high-dimensional structures in neural phase spaces that correlate with conscious vs unconscious processing
Machine learning classifiers trained on Φ-profiles to predict consciousness levels from EEG/fMRI data
Interactive 3D visualization showing how information integration networks reconfigure during state transitions

Novel aspect: Implement a "consciousness landscape" where you simulate perturbations to the system and map basins of attraction corresponding to different phenomenal states, essentially creating a phase space of consciousness.

2. Predictive Processing Hierarchy Simulator with Metacognitive Layers
A hierarchical Bayesian brain model implementing:

Predictive coding architecture with multiple levels (sensory → perceptual → conceptual → metacognitive)
Active inference framework where the system minimizes prediction error through both perception and action
Attention mechanisms modeled as precision-weighting of prediction errors
Recurrent processing loops with different temporal scales simulating feed-forward and feedback dynamics
Self-modeling module that generates predictions about the system's own internal states (metacognition)
Counterfactual reasoning engine that generates alternative scenarios and simulates "what-if" consciousness

Novel aspect: Include a "global workspace" layer where information becomes globally broadcast when prediction errors exceed thresholds, creating a computational correlate of access consciousness. Add psychophysical experiments where you manipulate attention, expectation, and compare model behavior to phenomena like change blindness, binocular rivalry, and the attentional blink.

3. Quantum-Inspired Neural Oscillator Networks for Binding Problem
A hybrid classical-quantum inspired computational model exploring:

Coupled oscillator networks modeling gamma, beta, theta rhythms with phase-locking dynamics
Quantum-inspired tensor network states representing superposition-like binding of features before "measurement" (conscious access)
Spike-timing dependent plasticity that creates temporal binding windows
Cross-frequency coupling analysis (phase-amplitude coupling) to identify hierarchical feature binding
Virtual lesion studies systematically disrupting synchronization to model disorders of consciousness
Information geometry analysis tracking how perceptual manifolds emerge from oscillatory binding
Quantum discord and entanglement measures (adapted for classical systems) as proxies for feature binding strength

Novel aspect: Implement a "conscious moment" detection algorithm that identifies when distributed features achieve sufficient coherence (measured via order parameters and mutual information) to constitute a unified conscious percept. Test whether quantum-inspired coherence measures better predict subjective reports than classical correlation measures using psychophysical data.