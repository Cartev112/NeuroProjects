import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids


def parse_args():
    p = argparse.ArgumentParser(description="ERP pipeline for oddball")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--run")
    p.add_argument("--event_id", nargs="+", help="e.g., standard:1 deviant:2", required=True)
    p.add_argument("--tmin", type=float, default=-0.2)
    p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument("--baseline", nargs=2, type=float, default=[-0.2, 0.0])
    p.add_argument("--l_freq", type=float, default=0.1)
    p.add_argument("--h_freq", type=float, default=30.0)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_raw(bids_root: str, subject: str, task: str, run: str | None):
    bp = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run, suffix="eeg", extension=".vhdr", datatype="eeg")
    try:
        raw = read_raw_bids(bp, verbose=False)
    except Exception:
        raw = read_raw_bids(bp.update(extension=".edf"), verbose=False)
    raw.load_data()
    return raw


def parse_event_id(pairs):
    mapping = {}
    for item in pairs:
        name, code = item.split(":")
        mapping[name] = int(code)
    return mapping


def find_peak(evoked: mne.Evoked, tmin: float, tmax: float, mode: str = "pos"):
    return evoked.copy().crop(tmin, tmax).get_peak(ch_type="eeg", tmin=tmin, tmax=tmax, mode=mode)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.bids_root, args.subject, args.task, args.run)
    raw.filter(args.l_freq, args.h_freq, verbose=False)
    raw.set_eeg_reference("average", projection=True)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Merge with provided mapping if necessary
    user_event_id = parse_event_id(args.event_id)
    # Keep only requested events
    events = events[np.isin(events[:, 2], list(user_event_id.values()))]
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")

    epochs = mne.Epochs(raw, events, event_id=user_event_id, tmin=args.tmin, tmax=args.tmax, baseline=tuple(args.baseline), picks=picks, preload=True, reject_by_annotation=True)
    # Robust average via trimmed mean (simple approximation)
    evokeds = {cond: epochs[cond].average(method="mean") for cond in user_event_id}
    # Save figures and CSV of peaks
    rows = []
    for cond, evo in evokeds.items():
        fig = evo.plot_joint(ts_args=dict(time_unit="s"), show=False)
        fig[0].savefig(out_dir / f"{cond}_evoked.png", dpi=150, bbox_inches="tight")
        t_peak_pos, ch_pos, amp_pos = find_peak(evo, 0.2, 0.5, mode="pos")
        t_peak_neg, ch_neg, amp_neg = find_peak(evo, 0.08, 0.15, mode="neg")
        rows.append({
            "condition": cond,
            "p300_latency_s": float(t_peak_pos),
            "p300_channel": ch_pos,
            "p300_amplitude_uV": float(amp_pos * 1e6),
            "n100_latency_s": float(t_peak_neg),
            "n100_channel": ch_neg,
            "n100_amplitude_uV": float(amp_neg * 1e6),
        })

    pd.DataFrame(rows).to_csv(out_dir / "component_peaks.csv", index=False)

    if len(evokeds) >= 2:
        names = list(evokeds)
        diff = mne.combine_evoked([evokeds[names[1]], evokeds[names[0]]], weights=[1, -1])
        diff_fig = diff.plot_joint(ts_args=dict(time_unit="s"), show=False)
        diff_fig[0].savefig(out_dir / "contrast_diff.png", dpi=150, bbox_inches="tight")

    print(f"Saved ERP outputs to {out_dir}")


if __name__ == "__main__":
    main()

