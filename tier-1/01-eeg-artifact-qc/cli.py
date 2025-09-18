import argparse
import json
from pathlib import Path

from src.qc_pipeline import (
    load_bids_raw,
    preprocess_raw,
    fit_ica,
    detect_artifacts,
    compute_qc_metrics,
    generate_report,
    save_ica,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Artifact QC and Report")
    parser.add_argument("--bids_root", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--l_freq", type=float, default=1.0)
    parser.add_argument("--h_freq", type=float, default=40.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_bids_raw(args.bids_root, args.subject, args.task, args.run)
    raw_prep = preprocess_raw(raw, l_freq=args.l_freq, h_freq=args.h_freq, notch=args.notch)

    ica = fit_ica(raw_prep)
    annotations, artifact_info = detect_artifacts(raw_prep, ica)
    raw_prep.set_annotations(annotations)

    qc = compute_qc_metrics(raw_prep, ica, artifact_info)
    with open(output_dir / "qc_summary.json", "w") as f:
        json.dump(qc, f, indent=2)

    report_path = output_dir / "report.html"
    generate_report(raw_prep, ica, artifact_info, report_path)
    save_ica(ica, output_dir / "ica.fif")

    print(f"QC complete. Report: {report_path}")


if __name__ == "__main__":
    main()

