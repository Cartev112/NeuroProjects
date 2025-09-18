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
    eog_indices: np.ndarray
    eog_scores: np.ndarray
    bad_segment_ratio: float
    high_amplitude_seconds: float


def load_bids_raw(bids_root: str, subject: str, task: str, run: Optional[str] = None) -> mne.io.BaseRaw:
    bids_path = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run, suffix="eeg", extension=".vhdr", datatype="eeg")
    try:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
    except Exception:
        bids_path = bids_path.update(extension=".edf")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 40.0, notch: Optional[float] = 50.0) -> mne.io.BaseRaw:
    raw = raw.copy()
    if notch is not None and notch > 0:
        raw.notch_filter(freqs=[notch, notch * 2], verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.set_eeg_reference("average", projection=True)
    return raw


def fit_ica(raw: mne.io.BaseRaw, n_components: Optional[int] = None, random_state: int = 97) -> ICA:
    ica = ICA(method="fastica", n_components=n_components, random_state=random_state, max_iter="auto")
    ica.fit(raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False))
    return ica


def _find_eog_components(raw: mne.io.BaseRaw, ica: ICA) -> Tuple[np.ndarray, np.ndarray]:
    eog_epochs = None
    try:
        eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
    except Exception:
        pass
    if eog_epochs is None or len(eog_epochs) == 0:
        # Fall back to correlation with frontal channels if EOG not available
        picks_frontal = mne.pick_channels_regexp(raw.ch_names, regexp=r"Fp[12]|AF[Z12]")
        if len(picks_frontal) == 0:
            return np.array([], dtype=int), np.array([])
        frontal = raw.get_data(picks=picks_frontal).mean(axis=0)
        sources = ica.get_sources(raw).get_data()
        scores = np.array([np.corrcoef(s, frontal)[0, 1] for s in sources])
        inds = np.where(np.abs(scores) > 0.3)[0]
        return inds, scores
    else:
        inds, scores = ica.find_bads_eog(eog_epochs)
        return np.array(inds), np.array(scores)


def _annotate_high_amplitude_segments(raw: mne.io.BaseRaw, z_thresh: float = 5.0, min_duration: float = 0.2) -> mne.Annotations:
    data = raw.get_data(picks="eeg")
    z = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    peak = np.max(np.abs(z), axis=0)
    bad_mask = peak > z_thresh
    onsets = []
    durations = []
    srate = raw.info["sfreq"]
    in_seg = False
    start = 0
    for i, bad in enumerate(bad_mask):
        if bad and not in_seg:
            in_seg = True
            start = i
        if in_seg and (not bad or i == len(bad_mask) - 1):
            end = i
            dur = (end - start) / srate
            if dur >= min_duration:
                onsets.append(start / srate)
                durations.append(dur)
            in_seg = False
    desc = ["BAD_high_amp"] * len(onsets)
    return mne.Annotations(onset=onsets, duration=durations, description=desc)


def detect_artifacts(raw: mne.io.BaseRaw, ica: ICA) -> Tuple[mne.Annotations, ArtifactInfo]:
    eog_inds, eog_scores = _find_eog_components(raw, ica)
    high_amp_ann = _annotate_high_amplitude_segments(raw)
    annotations = raw.annotations + high_amp_ann if raw.annotations is not None else high_amp_ann

    dur_bad = sum(high_amp_ann.duration) if len(high_amp_ann) > 0 else 0.0
    bad_ratio = float(dur_bad / raw.times[-1]) if raw.times.size > 0 else 0.0

    info = ArtifactInfo(
        eog_indices=eog_inds,
        eog_scores=eog_scores,
        bad_segment_ratio=bad_ratio,
        high_amplitude_seconds=dur_bad,
    )
    return annotations, info


def compute_qc_metrics(raw: mne.io.BaseRaw, ica: ICA, artifact_info: ArtifactInfo) -> Dict[str, float]:
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=1.0, fmax=40.0, n_fft=2048, verbose=False)
    band = lambda fmin, fmax: float(np.mean(psd[:, (freqs >= fmin) & (freqs <= fmax)]))
    metrics = {
        "duration_seconds": float(raw.times[-1]),
        "num_channels": int(len(raw.ch_names)),
        "bad_segment_ratio": float(artifact_info.bad_segment_ratio),
        "high_amplitude_seconds": float(artifact_info.high_amplitude_seconds),
        "num_eog_ica_components": int(len(artifact_info.eog_indices)),
        "mean_delta_power": band(1, 4),
        "mean_theta_power": band(4, 8),
        "mean_alpha_power": band(8, 12),
        "mean_beta_power": band(12, 30),
    }
    return metrics


def generate_report(raw: mne.io.BaseRaw, ica: ICA, artifact_info: ArtifactInfo, out_html: Path) -> None:
    report = Report(title="EEG Artifact QC")
    report.add_raw(raw, title="Raw (preprocessed)", psd=True)
    try:
        report.add_ica(ica, title="ICA components")
    except Exception:
        pass
    if artifact_info.eog_indices.size > 0:
        report.add_html(
            f"<h3>EOG components</h3><p>Indices: {artifact_info.eog_indices.tolist()}<br>Scores: {np.round(artifact_info.eog_scores, 3).tolist()}</p>",
            title="Artifact summary",
        )
    report.save(out_html, overwrite=True, open_browser=False)


def save_ica(ica: ICA, path: Path) -> None:
    ica.save(path, overwrite=True)

