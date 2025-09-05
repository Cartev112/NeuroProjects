import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib


def parse_args():
    p = argparse.ArgumentParser(description="fMRI confounds and motion QC")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fd_thresh", type=float, default=0.5)
    return p.parse_args()


def compute_fd(motion_params: np.ndarray) -> np.ndarray:
    # motion_params: T x 6 (3 trans, 3 rot in radians). Convert rotations to mm by 50mm brain radius.
    diff = np.vstack([np.zeros((1, motion_params.shape[1])), np.diff(motion_params, axis=0)])
    rot = diff[:, 3:] * 50.0
    trans = diff[:, :3]
    fd = np.sum(np.abs(np.hstack([trans, rot])), axis=1)
    return fd


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Placeholder: expects minimal inputs in out_dir for demo purposes
    # In real usage, you'd walk BIDS and read confounds.tsv from fMRIPrep.
    # Here we look for motion.npy (T x 6) and bold.nii.gz to compute basic metrics.
    motion_path = Path(args.bids_root) / "motion.npy"
    bold_path = Path(args.bids_root) / "bold.nii.gz"

    if motion_path.exists():
        motion = np.load(motion_path)
        fd = compute_fd(motion)
        pd.Series(fd, name="FD").to_csv(out / "framewise_displacement.csv", index=False)
        summary = {
            "fd_mean": float(fd.mean()),
            "fd_median": float(np.median(fd)),
            "fd_prop_over_thresh": float((fd > args.fd_thresh).mean()),
        }
    else:
        summary = {}

    if bold_path.exists():
        img = nib.load(str(bold_path))
        data = img.get_fdata()
        ts = data.reshape(-1, data.shape[-1]).mean(axis=0)
        dvars = np.sqrt(np.mean(np.diff(ts) ** 2))
        summary["global_signal_dvars"] = float(dvars)

    with open(out / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved QC summary to {out}")


if __name__ == "__main__":
    main()

