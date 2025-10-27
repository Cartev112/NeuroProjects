"""Inter-brain synchrony detection."""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple


class SynchronyDetector:
    """
    Detect inter-brain synchrony from dual recordings.
    
    Implements phase locking and coherence analysis.
    """
    
    def __init__(
        self,
        sampling_rate: float = 250.0,
        frequency_bands: Dict[str, Tuple[float, float]] = None
    ):
        """
        Initialize synchrony detector.
        
        Args:
            sampling_rate: Sampling rate in Hz
            frequency_bands: Dictionary of frequency bands
        """
        self.sampling_rate = sampling_rate
        
        if frequency_bands is None:
            frequency_bands = {
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30)
            }
        self.frequency_bands = frequency_bands
    
    def compute_phase_locking(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute phase locking value for each frequency band.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Dictionary of PLV values per band
        """
        plv_values = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Bandpass filter
            filtered1 = self._bandpass_filter(signal1, low_freq, high_freq)
            filtered2 = self._bandpass_filter(signal2, low_freq, high_freq)
            
            # Compute phase locking value
            plv = self._compute_plv(filtered1, filtered2)
            plv_values[band_name] = plv
        
        return plv_values
    
    def _bandpass_filter(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float
    ) -> np.ndarray:
        """Apply bandpass filter."""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        return filtered
    
    def _compute_plv(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Compute phase locking value."""
        # Hilbert transform to get instantaneous phase
        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # PLV is magnitude of mean of complex exponential of phase difference
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def detect_synchrony_events(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        threshold: float = 0.7,
        window_size: int = 250
    ) -> List[int]:
        """
        Detect moments of high synchrony.
        
        Args:
            signal1: First signal
            signal2: Second signal
            threshold: Synchrony threshold
            window_size: Window size in samples
            
        Returns:
            List of time indices with high synchrony
        """
        high_sync_moments = []
        
        # Sliding window
        for i in range(0, len(signal1) - window_size, window_size // 2):
            window1 = signal1[i:i+window_size]
            window2 = signal2[i:i+window_size]
            
            # Compute PLV for this window
            plv_dict = self.compute_phase_locking(window1, window2)
            
            # Average across bands
            avg_plv = np.mean(list(plv_dict.values()))
            
            if avg_plv > threshold:
                high_sync_moments.append(i)
        
        return high_sync_moments
