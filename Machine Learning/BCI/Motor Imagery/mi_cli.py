# Motor Imagery Classification using CSP + LDA
# This script implements a classic pipeline for classifying motor imagery EEG data:
# 1. CSP (Common Spatial Patterns) - extracts spatial filters that maximize class separation
# 2. LDA (Linear Discriminant Analysis) - linear classifier on CSP features
# This is a standard baseline approach in Brain-Computer Interface (BCI) research.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from mne.decoding import CSP  # Common Spatial Patterns from MNE library


def parse_args():
    """
    Parse command-line arguments for the motor imagery classification pipeline.
    
    Returns:
        Parsed arguments with:
        - data_root: Directory containing the motor imagery epochs data
        - out_dir: Directory to save cross-validation results
        - n_components: Number of CSP components to extract (default: 6)
                       More components = more features but risk of overfitting
    """
    p = argparse.ArgumentParser(description="Motor imagery CSP+LDA baseline")
    p.add_argument("--data_root", required=True, help="Path to epochs npz containing X,y")
    p.add_argument("--out_dir", required=True, help="Output directory for results")
    p.add_argument("--n_components", type=int, default=6, help="Number of CSP components (spatial filters)")
    return p.parse_args()


def load_epochs_npz(path: str):
    """
    Load motor imagery epochs from a .npz file.
    
    Args:
        path: Directory containing 'mi_epochs.npz' file
    
    Returns:
        X: EEG epochs with shape (n_trials, n_channels, n_times)
           Each trial is a segment of EEG during motor imagery
        y: Labels for each trial (e.g., 0=left hand, 1=right hand)
        sfreq: Sampling frequency in Hz (optional, may be None)
    """
    arr = np.load(Path(path) / "mi_epochs.npz")
    return arr["X"], arr["y"], arr.get("sfreq", None)


def main():
    """
    Main pipeline for motor imagery classification:
    1. Load EEG epochs data
    2. Build CSP+LDA classification pipeline
    3. Evaluate using 5-fold cross-validation
    4. Save and report results
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load motor imagery data
    X, y, _ = load_epochs_npz(args.data_root)
    # X shape: (n_trials, n_channels, n_times)
    # Example: (200 trials, 22 channels, 1000 time points)
    
    # ===== Build Classification Pipeline =====
    
    # Step 1: CSP (Common Spatial Patterns)
    # 
    # ========== WHAT IS CSP? AN IN-DEPTH EXPLANATION ==========
    #
    # CSP is a spatial filtering technique that finds optimal linear combinations of EEG channels
    # to maximize the difference between two classes (e.g., left hand vs right hand imagery).
    #
    # THE PROBLEM:
    # - Raw EEG has many channels (e.g., 22 electrodes on the scalp)
    # - Each channel records a mix of signals from different brain sources
    # - We need to find which spatial patterns (combinations of channels) are most discriminative
    #
    # THE INTUITION:
    # Imagine you have 22 microphones recording a concert. Some microphones are near the drums,
    # some near the guitar. CSP finds the "best mix" - which combination of microphones lets you
    # hear the drums most clearly vs the guitar most clearly.
    #
    # For motor imagery:
    # - Left hand imagery activates RIGHT motor cortex (brain is cross-wired)
    # - Right hand imagery activates LEFT motor cortex
    # - CSP finds spatial filters that amplify these different activation patterns
    #
    # HOW IT WORKS (MATHEMATICALLY):
    # 1. Compute covariance matrices for each class:
    #    - Cov_left: how channels co-vary during left hand imagery
    #    - Cov_right: how channels co-vary during right hand imagery
    #
    # 2. Find spatial filters W that simultaneously:
    #    - Maximize variance for one class (e.g., left hand)
    #    - Minimize variance for the other class (e.g., right hand)
    #    This is done by solving a generalized eigenvalue problem:
    #    Cov_left * W = λ * Cov_right * W
    #
    # 3. The eigenvectors with largest eigenvalues (λ) are filters that:
    #    - Have HIGH variance for class 1, LOW variance for class 2
    #    The eigenvectors with smallest eigenvalues are filters that:
    #    - Have LOW variance for class 1, HIGH variance for class 2
    #
    # WHAT ARE "SPATIAL FILTERS"?
    # A spatial filter is a weight vector [w1, w2, ..., w22] that defines how to combine channels:
    #    filtered_signal = w1*Ch1 + w2*Ch2 + ... + w22*Ch22
    #
    # Example filter might be: [0.8, 0.6, -0.3, 0.1, ...] meaning:
    # - Strongly weight channels over right motor cortex (positive weights)
    # - Subtract activity from other areas (negative weights)
    # This creates a "virtual channel" that isolates left hand imagery activity
    #
    # THE OUTPUT FEATURES:
    # For each trial, CSP computes:
    # - Apply each spatial filter to get filtered signals
    # - Compute variance of each filtered signal
    # - Take log(variance) as the feature
    #
    # Why variance? During motor imagery, the relevant brain areas show:
    # - Event-Related Desynchronization (ERD): DECREASE in power/variance in mu/beta bands
    # - Event-Related Synchronization (ERS): INCREASE in power/variance
    # CSP filters capture these variance changes!
    #
    # EXAMPLE:
    # If n_components=6, CSP returns 6 features per trial:
    # - Features 1-3: variance in filters that favor class 1 (left hand)
    # - Features 4-6: variance in filters that favor class 2 (right hand)
    # These 6 numbers are much more discriminative than raw 22 channels!
    #
    # WHY IT'S POWERFUL FOR BCI:
    # - Reduces dimensionality: 22 channels × 1000 time points → 6 features
    # - Learns from data: automatically finds the best spatial patterns
    # - Robust to noise: combines multiple channels to boost signal-to-noise ratio
    # - Interpretable: can visualize the spatial filters as topographic maps
    #
    # ========== END OF CSP EXPLANATION ==========
    
    # Initialize CSP with parameters:
    # - n_components: number of spatial filters to extract (typically 4-8)
    #   More = more features but risk overfitting. We take the top and bottom eigenvectors.
    # - log=True: apply log transform to variance features (improves normality for LDA)
    # - cov_est="epoch": compute covariance for each epoch separately (more robust)
    # Output: n_components features per trial (log-variance in each spatial filter)
    csp = CSP(n_components=args.n_components, reg=None, log=True, cov_est="epoch")
    
    # Step 2: LDA (Linear Discriminant Analysis)
    # Linear classifier that finds the best linear boundary between classes
    # - solver="lsqr": least squares solver (works well for small datasets)
    # - shrinkage="auto": regularization to prevent overfitting
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    
    # Combine into a scikit-learn pipeline
    # Data flows: Raw EEG → CSP features → LDA predictions
    pipe = Pipeline([("csp", csp), ("lda", lda)])

    # ===== Cross-Validation Evaluation =====
    # Use 5-fold stratified cross-validation to get robust performance estimate
    # Stratified: ensures each fold has balanced class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate pipeline using balanced accuracy (handles class imbalance)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    
    # Save individual fold scores to CSV
    pd.DataFrame({"balanced_accuracy": scores}).to_csv(out / "cv_scores.csv", index=False)
    
    # Print summary statistics
    print(f"Mean balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Results saved to {out / 'cv_scores.csv'}")


if __name__ == "__main__":
    main()

