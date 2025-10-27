import argparse
from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids


def parse_args():
    p = argparse.ArgumentParser(description="MEG TFR pipeline")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--run")
    p.add_argument("--event_id", nargs="+", required=True, help="e.g., face:5 scrambled:6")
    p.add_argument("--tmin", type=float, default=-0.3)
    p.add_argument("--tmax", type=float, default=0.6)
    p.add_argument("--baseline", nargs=2, type=float, default=[-0.2, 0.0])
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_raw(bids_root: str, subject: str, task: str, run: str | None):
    bp = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run, suffix="meg", extension=".fif", datatype="meg")
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    return raw


def parse_event_id(pairs):
    mapping = {}
    for item in pairs:
        name, code = item.split(":")
        mapping[name] = int(code)
    return mapping


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.bids_root, args.subject, args.task, args.run)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_id = parse_event_id(args.event_id)
    events = events[np.isin(events[:, 2], list(event_id.values()))]

    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=args.tmin, tmax=args.tmax, baseline=tuple(args.baseline), picks=picks, preload=True)

    freqs = np.linspace(4, 40, 20)
    n_cycles = freqs / 2.0

    for cond in event_id:
        ep = epochs[cond]
        power = mne.time_frequency.tfr_morlet(ep, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, decim=1, n_jobs=1)
        power.apply_baseline(baseline=tuple(args.baseline), mode="logratio")
        fig = power.plot_topo(baseline=None, mode=None, title=f"{cond} TFR", show=False)
        fig.savefig(out / f"tfr_{cond}.png", dpi=150, bbox_inches="tight")

    print(f"Saved TFR figures to {out}")


if __name__ == "__main__":
    main()

