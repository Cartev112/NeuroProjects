import argparse
from pathlib import Path
import mne
from mne_bids import BIDSPath, write_raw_bids


def parse_args():
    p = argparse.ArgumentParser(description="EEG BIDS converter")
    p.add_argument("--raw_file", required=True)
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--ses")
    p.add_argument("--montage", default="standard_1020")
    p.add_argument("--line_freq", type=float, default=50.0)
    return p.parse_args()


def main():
    args = parse_args()
    raw = mne.io.read_raw(args.raw_file, preload=False)
    try:
        raw.set_montage(args.montage, on_missing="warn")
    except Exception:
        pass
    raw.info["line_freq"] = args.line_freq

    bp = BIDSPath(root=args.bids_root, subject=args.subject.replace("sub-", ""), task=args.task, session=args.ses, datatype="eeg")
    write_raw_bids(raw, bp, overwrite=True, format="auto")

    print(f"Wrote BIDS dataset to {args.bids_root}")


if __name__ == "__main__":
    main()

