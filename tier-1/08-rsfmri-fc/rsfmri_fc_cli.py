import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import datasets, input_data


def parse_args():
    p = argparse.ArgumentParser(description="Resting-state FC matrices")
    p.add_argument("--func", required=True, help="Path to 4D rs-fMRI NIfTI")
    p.add_argument("--mask", required=False, help="Path to brain mask NIfTI")
    p.add_argument("--atlas", choices=["harvard_oxford", "aal", "msdl"], default="harvard_oxford")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def get_atlas(name: str):
    if name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        labels = atlas.labels
        masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
    elif name == "aal":
        atlas = datasets.fetch_atlas_aal()
        labels = atlas.labels
        masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
    else:
        atlas = datasets.fetch_atlas_msdl()
        labels = atlas.labels
        masker = input_data.NiftiMapsMasker(maps_img=atlas.maps, standardize=True)
    return masker, labels


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    masker, labels = get_atlas(args.atlas)
    if args.mask:
        masker.mask_img = args.mask

    ts = masker.fit_transform(args.func)
    corr = np.corrcoef(ts.T)
    # Regularized precision via Ledoit-Wolf covariance
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(ts)
    prec = np.linalg.pinv(lw.covariance_)

    np.save(out / "corr.npy", corr)
    np.save(out / "precision.npy", prec)
    pd.DataFrame(labels, columns=["label"]).to_csv(out / "labels.csv", index=False)
    print(f"Saved FC matrices to {out}")


if __name__ == "__main__":
    main()

