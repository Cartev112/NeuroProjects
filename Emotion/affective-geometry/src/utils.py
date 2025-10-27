"""
Utility functions for affective geometry analysis.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def generate_synthetic_biosignal(
    duration: float = 100.0,
    sampling_rate: float = 100.0,
    signal_type: str = 'hrv',
    noise_level: float = 0.1,
    trend: Optional[str] = None
) -> np.ndarray:
    """
    Generate synthetic biosignal data for testing.
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Samples per second
        signal_type: Type of biosignal ('hrv', 'eda', 'pupil', 'emg')
        noise_level: Noise magnitude
        trend: Optional trend ('increasing', 'decreasing', 'oscillating')
        
    Returns:
        Synthetic biosignal array
    """
    n_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Base signal characteristics
    if signal_type == 'hrv':
        # Heart rate variability: oscillatory with 1/f noise
        base_freq = 0.1  # Low frequency oscillation
        signal = np.sin(2 * np.pi * base_freq * time)
        signal += 0.5 * np.sin(2 * np.pi * 0.25 * time)  # Respiratory frequency
        
    elif signal_type == 'eda':
        # Electrodermal activity: slow changes with phasic responses
        signal = np.zeros(n_samples)
        # Tonic component
        signal += 2.0 + 0.5 * np.sin(2 * np.pi * 0.01 * time)
        # Phasic responses (random events)
        n_events = int(duration / 10)
        for _ in range(n_events):
            event_time = np.random.uniform(0, duration)
            event_idx = int(event_time * sampling_rate)
            if event_idx < n_samples:
                # Exponential rise and decay
                event_signal = np.exp(-np.abs(time - event_time) / 2.0)
                signal += 0.5 * event_signal
                
    elif signal_type == 'pupil':
        # Pupil diameter: relatively stable with task-evoked changes
        signal = 3.0 + 0.3 * np.sin(2 * np.pi * 0.05 * time)
        
    elif signal_type == 'emg':
        # Facial EMG: bursts of activity
        signal = 0.1 * np.random.randn(n_samples)
        n_bursts = int(duration / 5)
        for _ in range(n_bursts):
            burst_time = np.random.uniform(0, duration)
            burst_idx = int(burst_time * sampling_rate)
            burst_duration = int(0.5 * sampling_rate)
            if burst_idx + burst_duration < n_samples:
                signal[burst_idx:burst_idx+burst_duration] += np.random.uniform(0.5, 2.0)
    else:
        signal = np.zeros(n_samples)
    
    # Add trend if specified
    if trend == 'increasing':
        signal += np.linspace(0, 1, n_samples)
    elif trend == 'decreasing':
        signal += np.linspace(1, 0, n_samples)
    elif trend == 'oscillating':
        signal += 0.5 * np.sin(2 * np.pi * 0.02 * time)
    
    # Add noise
    signal += noise_level * np.random.randn(n_samples)
    
    return signal


def generate_emotional_trajectory(
    initial_valence: float = 0.0,
    initial_arousal: float = 0.0,
    duration: int = 1000,
    volatility: float = 0.1,
    drift_to_neutral: float = 0.01
) -> np.ndarray:
    """
    Generate synthetic emotional trajectory.
    
    Args:
        initial_valence: Starting valence
        initial_arousal: Starting arousal
        duration: Number of time steps
        volatility: Random walk volatility
        drift_to_neutral: Drift rate toward neutral state
        
    Returns:
        Trajectory array of shape (duration, 2)
    """
    trajectory = np.zeros((duration, 2))
    trajectory[0] = [initial_valence, initial_arousal]
    
    for t in range(1, duration):
        # Random walk
        change = np.random.normal(0, volatility, size=2)
        
        # Drift toward neutral
        drift = -drift_to_neutral * trajectory[t-1]
        
        # Update
        trajectory[t] = trajectory[t-1] + change + drift
        
        # Clip to bounds
        trajectory[t] = np.clip(trajectory[t], -1, 1)
    
    return trajectory


def create_standard_attractors(dimensions: int = 5):
    """
    Create standard emotional attractors based on basic emotions.
    
    Args:
        dimensions: Number of state space dimensions
        
    Returns:
        List of Attractor objects
    """
    from .attractors import Attractor, AttractorType
    
    # Basic emotions in PAD space (Pleasure, Arousal, Dominance)
    # Extended with Approach/Avoidance and Temporal dimensions
    
    attractors = []
    
    # Happiness: high valence, moderate arousal, high dominance, approach
    happiness = Attractor(
        center=[0.8, 0.5, 0.7, 0.8, 0.0],
        strength=1.2,
        name="happiness",
        attractor_type=AttractorType.POINT,
        damping=0.3
    )
    attractors.append(happiness)
    
    # Sadness: low valence, low arousal, low dominance, avoidance
    sadness = Attractor(
        center=[-0.7, -0.4, -0.5, -0.6, 0.0],
        strength=1.0,
        name="sadness",
        attractor_type=AttractorType.POINT,
        damping=0.5
    )
    attractors.append(sadness)
    
    # Anger: low valence, high arousal, high dominance, approach
    anger = Attractor(
        center=[-0.6, 0.8, 0.6, 0.7, 0.0],
        strength=1.1,
        name="anger",
        attractor_type=AttractorType.POINT,
        damping=0.2
    )
    attractors.append(anger)
    
    # Fear: low valence, high arousal, low dominance, avoidance
    fear = Attractor(
        center=[-0.7, 0.7, -0.6, -0.8, 0.0],
        strength=1.3,
        name="fear",
        attractor_type=AttractorType.POINT,
        damping=0.2
    )
    attractors.append(fear)
    
    # Calm: moderate valence, low arousal, moderate dominance
    calm = Attractor(
        center=[0.3, -0.6, 0.2, 0.1, 0.0],
        strength=0.9,
        name="calm",
        attractor_type=AttractorType.POINT,
        damping=0.6
    )
    attractors.append(calm)
    
    # Excitement: high valence, high arousal, moderate dominance, approach
    excitement = Attractor(
        center=[0.7, 0.8, 0.4, 0.9, 0.0],
        strength=1.0,
        name="excitement",
        attractor_type=AttractorType.POINT,
        damping=0.2
    )
    attractors.append(excitement)
    
    return attractors


def load_example_data(dataset: str = 'synthetic'):
    """
    Load example dataset for demonstrations.
    
    Args:
        dataset: Dataset name ('synthetic', 'depression', 'anxiety')
        
    Returns:
        Dictionary with data arrays
    """
    if dataset == 'synthetic':
        # Generate synthetic multimodal data
        duration = 100.0
        
        hrv = generate_synthetic_biosignal(duration, signal_type='hrv')
        eda = generate_synthetic_biosignal(duration, signal_type='eda')
        pupil = generate_synthetic_biosignal(duration, signal_type='pupil')
        
        # Generate corresponding emotional states
        n_samples = len(hrv)
        emotional_states = np.zeros((n_samples, 2))
        
        # Simple mapping from biosignals to emotions
        emotional_states[:, 0] = 0.5 * np.tanh(hrv / 2)  # Valence
        emotional_states[:, 1] = 0.5 * np.tanh(eda / 3)  # Arousal
        
        return {
            'hrv': hrv,
            'eda': eda,
            'pupil': pupil,
            'emotional_states': emotional_states,
            'sampling_rate': 100.0
        }
    
    elif dataset == 'depression':
        # Simulate depression: low valence, low arousal, high stability
        trajectory = generate_emotional_trajectory(
            initial_valence=-0.6,
            initial_arousal=-0.3,
            duration=1000,
            volatility=0.05,  # Low volatility
            drift_to_neutral=0.001  # Weak drift
        )
        
        return {
            'trajectory': trajectory,
            'label': 'depression',
            'characteristics': 'low_volatility_negative'
        }
    
    elif dataset == 'anxiety':
        # Simulate anxiety: variable valence, high arousal, high volatility
        trajectory = generate_emotional_trajectory(
            initial_valence=-0.3,
            initial_arousal=0.6,
            duration=1000,
            volatility=0.2,  # High volatility
            drift_to_neutral=0.005
        )
        
        return {
            'trajectory': trajectory,
            'label': 'anxiety',
            'characteristics': 'high_volatility_high_arousal'
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def plot_multimodal_signals(
    signals: dict,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot multiple biosignal modalities.
    
    Args:
        signals: Dictionary mapping signal names to arrays
        figsize: Figure size
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=True)
    
    if n_signals == 1:
        axes = [axes]
    
    for ax, (name, signal) in zip(axes, signals.items()):
        time = np.arange(len(signal))
        ax.plot(time, signal, linewidth=1.5)
        ax.set_ylabel(name.upper(), fontsize=11)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (samples)', fontsize=12)
    axes[0].set_title('Multimodal Biosignals', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


def compute_emotional_features(trajectory: np.ndarray) -> dict:
    """
    Compute summary features from emotional trajectory.
    
    Args:
        trajectory: Emotional trajectory array
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Mean and variance
    features['mean_valence'] = np.mean(trajectory[:, 0])
    features['std_valence'] = np.std(trajectory[:, 0])
    
    if trajectory.shape[1] > 1:
        features['mean_arousal'] = np.mean(trajectory[:, 1])
        features['std_arousal'] = np.std(trajectory[:, 1])
    
    # Range
    features['valence_range'] = np.ptp(trajectory[:, 0])
    
    # Transitions (zero crossings)
    valence_centered = trajectory[:, 0] - np.mean(trajectory[:, 0])
    features['n_transitions'] = np.sum(np.diff(np.sign(valence_centered)) != 0)
    
    # Time in positive vs negative
    features['time_positive'] = np.mean(trajectory[:, 0] > 0)
    features['time_negative'] = np.mean(trajectory[:, 0] < 0)
    
    return features


def save_results(results: dict, filepath: str):
    """
    Save analysis results to file.
    
    Args:
        results: Dictionary of results
        filepath: Output file path
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> dict:
    """
    Load analysis results from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary of results
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results
