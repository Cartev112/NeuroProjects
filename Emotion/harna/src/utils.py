"""Utility functions for HARNA."""

import numpy as np
from typing import Dict


def generate_random_stimulus(feature_dim: int = 512) -> np.ndarray:
    """Generate random stimulus features."""
    return np.random.randn(feature_dim)


def generate_random_goals(goal_dim: int = 64) -> np.ndarray:
    """Generate random goal representation."""
    return np.random.randn(goal_dim)


def generate_random_norms(norm_dim: int = 32) -> np.ndarray:
    """Generate random norm representation."""
    return np.random.randn(norm_dim)


def emotion_label_from_valence_arousal(valence: float, arousal: float) -> str:
    """Map valence-arousal to emotion label."""
    if valence > 0.5 and arousal > 0.5:
        return "excitement"
    elif valence > 0.5 and arousal < -0.2:
        return "contentment"
    elif valence > 0.3:
        return "pleasant"
    elif valence < -0.5 and arousal > 0.5:
        return "anger"
    elif valence < -0.5 and arousal < -0.2:
        return "sadness"
    elif valence < -0.3:
        return "unpleasant"
    elif arousal > 0.5:
        return "aroused"
    elif arousal < -0.5:
        return "calm"
    else:
        return "neutral"


def create_context(
    social: bool = False,
    threat: bool = False,
    familiar: bool = False,
    positive: bool = False
) -> Dict:
    """Create context dictionary."""
    return {
        'social': social,
        'threat': threat,
        'familiar': familiar,
        'positive': positive
    }
