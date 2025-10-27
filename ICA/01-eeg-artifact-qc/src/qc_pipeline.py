# EEG Artifact Detection and Quality Control Pipeline
# This module provides tools for automated EEG quality assessment:
# 1. Load EEG data from BIDS format
# 2. Preprocess (filter, re-reference)
# 3. Run ICA to identify artifact components (eye movements, blinks)
# 4. Detect high-amplitude artifacts (muscle, movement)
# 5. Compute quality metrics and generate reports
# Used to ensure data quality before further analysis.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.report import Report
from mne_bids import BIDSPath, read_raw_bids


@dataclass
class ArtifactInfo:
    """
    Container for artifact detection results.
    
    Attributes:
        eog_indices: ICA component indices identified as eye artifacts (EOG)
        eog_scores: Correlation scores for each EOG component (higher = more likely artifact)
        bad_segment_ratio: Proportion of recording flagged as bad (0.0 to 1.0)
        high_amplitude_seconds: Total duration of high-amplitude artifacts in seconds
    """
    eog_indices: np.ndarray
    eog_scores: np.ndarray
    bad_segment_ratio: float
    high_amplitude_seconds: float


def load_bids_raw(bids_root: str, subject: str, task: str, run: Optional[str] = None) -> mne.io.BaseRaw:
    """
    Load raw EEG data from BIDS-formatted dataset.
    
    BIDS (Brain Imaging Data Structure) is a standard format for organizing
    neuroimaging data. This function handles common EEG file formats.
    
    Args:
        bids_root: Root directory of BIDS dataset
        subject: Subject ID (e.g., "sub-01" or "01")
        task: Task name (e.g., "rest", "motor")
        run: Optional run number for multiple recordings
    
    Returns:
        MNE Raw object containing the loaded EEG data
    """
    # Build BIDS path - try BrainVision format (.vhdr) first
    bids_path = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run, suffix="eeg", extension=".vhdr", datatype="eeg")
    try:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
    except Exception:
        # Fall back to EDF format if BrainVision not found
        bids_path = bids_path.update(extension=".edf")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
    
    # Load data into memory for faster processing
    raw.load_data()
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 40.0, notch: Optional[float] = 50.0) -> mne.io.BaseRaw:
    """
    Apply standard preprocessing steps to raw EEG data.
    
    Preprocessing pipeline:
    1. Notch filter to remove power line noise (50/60 Hz)
    2. Bandpass filter to focus on neural frequencies
    3. Average reference to reduce common noise
    
    Args:
        raw: Raw EEG data
        l_freq: High-pass filter cutoff in Hz (default: 1.0 Hz, removes slow drifts)
        h_freq: Low-pass filter cutoff in Hz (default: 40.0 Hz, removes high-freq noise)
        notch: Power line frequency to remove (50 Hz Europe, 60 Hz USA, None to skip)
    
    Returns:
        Preprocessed copy of the raw data
    """
    raw = raw.copy()  # Don't modify original
    
    # Remove power line noise at 50/60 Hz and its harmonic (100/120 Hz)
    if notch is not None and notch > 0:
        raw.notch_filter(freqs=[notch, notch * 2], verbose=False)
    
    # Bandpass filter: keep frequencies between l_freq and h_freq
    # Removes slow drifts (<1 Hz) and high-frequency noise (>40 Hz)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    
    # Average reference: subtract the mean of all channels from each channel
    # This is a common reference scheme that reduces common noise
    raw.set_eeg_reference("average", projection=True)
    
    return raw


def fit_ica(raw: mne.io.BaseRaw, n_components: Optional[int] = None, random_state: int = 97) -> ICA:
    """
    Fit Independent Component Analysis (ICA) to decompose EEG into independent sources.
    
    ICA separates mixed signals into independent components. For EEG:
    - Brain activity components (what we want)
    - Artifact components: eye blinks, eye movements, muscle, heartbeat
    
    We can identify and remove artifact components while preserving brain signals.
    
    Args:
        raw: Preprocessed raw EEG data
        n_components: Number of ICA components (None = use all channels)
        random_state: Random seed for reproducibility
    
    Returns:
        Fitted ICA object containing the decomposition
    """
    # Initialize ICA with FastICA algorithm (fast and robust)
    ica = ICA(method="fastica", n_components=n_components, random_state=random_state, max_iter="auto")
    
    # Fit ICA on high-pass filtered data (1 Hz) to improve convergence
    # Low frequencies can interfere with ICA decomposition
    ica.fit(raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False))
    
    return ica


def _find_eog_components(raw: mne.io.BaseRaw, ica: ICA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify ICA components that correspond to eye artifacts (EOG).
    
    Eye movements and blinks create large electrical signals that contaminate EEG.
    This function finds ICA components that correlate with eye activity.
    
    Strategy:
    1. If EOG channels exist: use them to detect eye artifacts
    2. Otherwise: use frontal EEG channels (Fp1, Fp2) as proxy for eye activity
    
    Args:
        raw: Raw EEG data
        ica: Fitted ICA decomposition
    
    Returns:
        eog_indices: Component indices identified as eye artifacts
        eog_scores: Correlation scores for each component
    """
    # Try to create epochs around detected eye blinks/movements
    eog_epochs = None
    try:
        eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
    except Exception:
        pass  # No EOG channels or detection failed
    
    if eog_epochs is None or len(eog_epochs) == 0:
        # Fall back to correlation with frontal channels if EOG not available
        # Frontal channels (Fp1, Fp2, AFz) are closest to eyes and pick up eye artifacts
        picks_frontal = mne.pick_channels_regexp(raw.ch_names, regexp=r"Fp[12]|AF[Z12]")
        
        if len(picks_frontal) == 0:
            # No frontal channels available - can't detect EOG
            return np.array([], dtype=int), np.array([])
        
        # Compute correlation between each ICA component and frontal channels
        frontal = raw.get_data(picks=picks_frontal).mean(axis=0)  # Average frontal signal
        sources = ica.get_sources(raw).get_data()  # ICA component time series
        scores = np.array([np.corrcoef(s, frontal)[0, 1] for s in sources])
        
        # Flag components with high correlation (>0.3) as likely eye artifacts
        inds = np.where(np.abs(scores) > 0.3)[0]
        return inds, scores
    else:
        # Use MNE's built-in EOG detection (more robust when EOG channels exist)
        inds, scores = ica.find_bads_eog(eog_epochs)
        return np.array(inds), np.array(scores)


def _annotate_high_amplitude_segments(raw: mne.io.BaseRaw, z_thresh: float = 5.0, min_duration: float = 0.2) -> mne.Annotations:
    """
    Detect and annotate segments with abnormally high amplitude (muscle, movement artifacts).
    
    High-amplitude artifacts are typically caused by:
    - Muscle tension (jaw clenching, frowning)
    - Head/body movements
    - Electrode issues (poor contact, cable movement)
    
    Detection method:
    1. Z-score normalize each channel
    2. Find time points where any channel exceeds threshold
    3. Group consecutive bad time points into segments
    
    Args:
        raw: Raw EEG data
        z_thresh: Z-score threshold (default: 5.0 = 5 standard deviations)
        min_duration: Minimum segment duration in seconds to flag (default: 0.2s)
    
    Returns:
        MNE Annotations object marking bad segments
    """
    # Get EEG data (channels × time points)
    data = raw.get_data(picks="eeg")
    
    # Z-score normalize each channel independently
    # This accounts for different baseline amplitudes across channels
    z = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    
    # Find peak z-score across all channels at each time point
    peak = np.max(np.abs(z), axis=0)
    
    # Flag time points exceeding threshold
    bad_mask = peak > z_thresh
    
    # Group consecutive bad time points into segments
    onsets = []  # Start times of bad segments
    durations = []  # Durations of bad segments
    srate = raw.info["sfreq"]  # Sampling rate
    
    in_seg = False  # Currently in a bad segment?
    start = 0  # Start index of current segment
    
    for i, bad in enumerate(bad_mask):
        if bad and not in_seg:
            # Start of new bad segment
            in_seg = True
            start = i
        if in_seg and (not bad or i == len(bad_mask) - 1):
            # End of bad segment
            end = i
            dur = (end - start) / srate  # Convert samples to seconds
            
            # Only keep segments longer than minimum duration
            if dur >= min_duration:
                onsets.append(start / srate)
                durations.append(dur)
            in_seg = False
    
    # Create MNE annotations
    desc = ["BAD_high_amp"] * len(onsets)
    return mne.Annotations(onset=onsets, duration=durations, description=desc)


def detect_artifacts(raw: mne.io.BaseRaw, ica: ICA) -> Tuple[mne.Annotations, ArtifactInfo]:
    """
    Run complete artifact detection pipeline.
    
    Detects two types of artifacts:
    1. Eye artifacts (blinks, movements) via ICA
    2. High-amplitude artifacts (muscle, movement) via threshold detection
    
    Args:
        raw: Preprocessed raw EEG data
        ica: Fitted ICA decomposition
    
    Returns:
        annotations: MNE Annotations marking bad time segments
        info: ArtifactInfo object with detection results and statistics
    """
    # Detect eye artifact components
    eog_inds, eog_scores = _find_eog_components(raw, ica)
    
    # Detect high-amplitude segments
    high_amp_ann = _annotate_high_amplitude_segments(raw)
    
    # Combine with any existing annotations
    annotations = raw.annotations + high_amp_ann if raw.annotations is not None else high_amp_ann

    # Calculate summary statistics
    dur_bad = sum(high_amp_ann.duration) if len(high_amp_ann) > 0 else 0.0
    bad_ratio = float(dur_bad / raw.times[-1]) if raw.times.size > 0 else 0.0

    # Package results
    info = ArtifactInfo(
        eog_indices=eog_inds,
        eog_scores=eog_scores,
        bad_segment_ratio=bad_ratio,
        high_amplitude_seconds=dur_bad,
    )
    return annotations, info


def compute_qc_metrics(raw: mne.io.BaseRaw, ica: ICA, artifact_info: ArtifactInfo) -> Dict[str, float]:
    """
    Compute quality control metrics for EEG recording.
    
    Metrics include:
    - Recording metadata (duration, channels)
    - Artifact statistics (bad segments, EOG components)
    - Spectral power in standard frequency bands:
      * Delta (1-4 Hz): slow waves, deep sleep
      * Theta (4-8 Hz): drowsiness, meditation
      * Alpha (8-12 Hz): relaxed wakefulness, eyes closed
      * Beta (12-30 Hz): active thinking, focus
    
    Args:
        raw: Preprocessed raw EEG data
        ica: Fitted ICA decomposition
        artifact_info: Results from artifact detection
    
    Returns:
        Dictionary of quality metrics
    """
    # Compute power spectral density (PSD) using Welch's method
    # This gives us the power at each frequency
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=1.0, fmax=40.0, n_fft=2048, verbose=False)
    
    # Helper function to compute mean power in a frequency band
    band = lambda fmin, fmax: float(np.mean(psd[:, (freqs >= fmin) & (freqs <= fmax)]))
    
    # Compile all metrics
    metrics = {
        # Recording info
        "duration_seconds": float(raw.times[-1]),
        "num_channels": int(len(raw.ch_names)),
        
        # Artifact statistics
        "bad_segment_ratio": float(artifact_info.bad_segment_ratio),
        "high_amplitude_seconds": float(artifact_info.high_amplitude_seconds),
        "num_eog_ica_components": int(len(artifact_info.eog_indices)),
        
        # Spectral power by frequency band
        "mean_delta_power": band(1, 4),    # 1-4 Hz
        "mean_theta_power": band(4, 8),    # 4-8 Hz
        "mean_alpha_power": band(8, 12),   # 8-12 Hz
        "mean_beta_power": band(12, 30),   # 12-30 Hz
    }
    return metrics


def generate_report(raw: mne.io.BaseRaw, ica: ICA, artifact_info: ArtifactInfo, out_html: Path) -> None:
    """
    Generate an HTML quality control report with visualizations.
    
    The report includes:
    - Raw data visualization with power spectral density
    - ICA component topographies (spatial patterns)
    - Identified artifact components with scores
    
    Args:
        raw: Preprocessed raw EEG data
        ica: Fitted ICA decomposition
        artifact_info: Results from artifact detection
        out_html: Path to save the HTML report
    """
    # Create MNE report object
    report = Report(title="EEG Artifact QC")
    
    # Add raw data visualization with PSD plot
    report.add_raw(raw, title="Raw (preprocessed)", psd=True)
    
    # Add ICA component visualizations
    try:
        report.add_ica(ica, title="ICA components")
    except Exception:
        pass  # Skip if visualization fails
    
    # Add summary of detected artifacts
    if artifact_info.eog_indices.size > 0:
        report.add_html(
            f"<h3>EOG components</h3><p>Indices: {artifact_info.eog_indices.tolist()}<br>Scores: {np.round(artifact_info.eog_scores, 3).tolist()}</p>",
            title="Artifact summary",
        )
    
    # Save report to HTML file
    report.save(out_html, overwrite=True, open_browser=False)


def save_ica(ica: ICA, path: Path) -> None:
    """
    Save fitted ICA decomposition to disk for later use.
    
    The saved ICA can be loaded and applied to remove artifact components
    from the data in downstream analysis.
    
    Args:
        ica: Fitted ICA object
        path: File path to save ICA (typically with .fif extension)
    """
    ica.save(path, overwrite=True)

