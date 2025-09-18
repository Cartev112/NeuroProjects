import argparse
from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser(description="MEG sensor vs source decoding")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_epochs(bids_root, subject, task):
    bp = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, suffix="meg", extension=".fif", datatype="meg")
    raw = read_raw_bids(bp, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    picks = mne.pick_types(raw.info, meg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.5, baseline=(-0.2, 0.0), picks=picks, preload=True)
    return epochs


def decode_sensor(epochs):
    X = epochs.get_data().reshape(len(epochs), -1)
    y = epochs.events[:, 2]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")


def decode_source(epochs):
    # Placeholder: use sensor data average over time as pseudo-source feature
    X = epochs.get_data().mean(axis=2)
    y = epochs.events[:, 2]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    epochs = load_epochs(args.bids_root, args.subject, args.task)
    s_scores = decode_sensor(epochs)
    r_scores = decode_source(epochs)
    np.savetxt(out / "sensor_scores.txt", s_scores)
    np.savetxt(out / "source_scores.txt", r_scores)
    print(f"Sensor BA: {s_scores.mean():.3f} | Source BA: {r_scores.mean():.3f}")


if __name__ == "__main__":
    main()

