import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne_bids import read_raw_bids, BIDSPath


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


def parse_args():
    p = argparse.ArgumentParser(description="EEG PSD Explorer")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--run")
    p.add_argument("--method", choices=["welch", "multitaper"], default="welch")
    p.add_argument("--fmin", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=40.0)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_raw(bids_root: str, subject: str, task: str, run: str | None):
    bp = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run, suffix="eeg", extension=".vhdr", datatype="eeg")
    try:
        raw = read_raw_bids(bp, verbose=False)
    except Exception:
        raw = read_raw_bids(bp.update(extension=".edf"), verbose=False)
    raw.load_data()
    raw.set_montage("standard_1020", on_missing="warn")
    return raw


def compute_psd(raw: mne.io.BaseRaw, method: str, fmin: float, fmax: float):
    if method == "welch":
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024, verbose=False)
    else:
        psd, freqs = mne.time_frequency.psd_array_multitaper(raw.get_data(), sfreq=raw.info["sfreq"], fmin=fmin, fmax=fmax, verbose=False)
    return psd, freqs


def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd[:, mask], freqs[mask], axis=1).mean())


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args.bids_root, args.subject, args.task, args.run)
    psd, freqs = compute_psd(raw, args.method, args.fmin, args.fmax)

    np.save(out_dir / "psd.npy", psd)
    np.save(out_dir / "freqs.npy", freqs)

    rows = []
    for band_name, (lo, hi) in BANDS.items():
        rows.append({"band": band_name, "bandpower": bandpower(psd, freqs, lo, hi)})
    pd.DataFrame(rows).to_csv(out_dir / "bandpower.csv", index=False)

    fig = raw.plot_psd(fmin=args.fmin, fmax=args.fmax, show=False)
    fig.savefig(out_dir / "psd_channels.png", dpi=150, bbox_inches="tight")

    # Alpha topography example
    try:
        raw_copy = raw.copy().filter(8, 12, verbose=False)
        alpha_power = raw_copy.get_data().var(axis=1)
        mne.viz.plot_topomap(alpha_power, raw.info, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(out_dir / "alpha_topomap.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print(f"Saved PSD outputs to {out_dir}")


if __name__ == "__main__":
    main()

