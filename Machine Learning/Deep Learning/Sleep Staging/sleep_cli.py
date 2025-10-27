import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Sleep staging baseline")
    p.add_argument("--data_root", required=True, help="Path to Sleep-EDF or preprocessed epochs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--classifier", choices=["svm", "rf"], default="svm")
    return p.parse_args()


def extract_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    # epoch_data: shape (num_epochs, num_samples)
    import scipy.signal as sg
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]
    feats = []
    for x in epoch_data:
        f, pxx = sg.welch(x, fs=sfreq, nperseg=int(sfreq * 2))
        bp = [np.trapz(pxx[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)]) for lo, hi in bands]
        activity = np.var(x)
        mobility = np.sqrt(np.var(np.diff(x)) / (activity + 1e-12))
        complexity = np.sqrt(np.var(np.diff(np.diff(x))) / (np.var(np.diff(x)) + 1e-12)) / (mobility + 1e-12)
        feats.append(bp + [activity, mobility, complexity])
    return np.asarray(feats)


def load_dummy(data_root: str):
    # Placeholder loader: expects npz with 'epochs' (N, T), 'labels' (N,), 'sfreq'
    arr = np.load(Path(data_root) / "sleep_epochs.npz")
    return arr["epochs"], arr["labels"], float(arr["sfreq"]) 


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_epochs, y, sfreq = load_dummy(args.data_root)
    X = extract_features(X_epochs, sfreq)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = SVC(class_weight="balanced", kernel="rbf", C=2.0, gamma="scale") if args.classifier == "svm" else RandomForestClassifier(n_estimators=300, class_weight="balanced")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, Xs, y, cv=cv)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    pd.DataFrame(report).to_csv(out / "classification_report.csv")
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()

