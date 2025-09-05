"""
Recurrence Quantification Analysis (RQA) for physiological signals.

Analyzes emotional stability, transitions, and chaotic dynamics from
time series data (HRV, EDA, pupillometry, facial EMG).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RQAMetrics:
    """Container for recurrence quantification analysis metrics."""
    recurrence_rate: float
    determinism: float
    laminarity: float
    entropy_diagonal: float
    entropy_vertical: float
    average_diagonal_line: float
    max_diagonal_line: float
    average_vertical_line: float
    max_vertical_line: float
    trapping_time: float
    
    def __repr__(self) -> str:
        return (
            f"RQAMetrics(\n"
            f"  Recurrence Rate: {self.recurrence_rate:.4f}\n"
            f"  Determinism: {self.determinism:.4f}\n"
            f"  Laminarity: {self.laminarity:.4f}\n"
            f"  Diagonal Entropy: {self.entropy_diagonal:.4f}\n"
            f"  Vertical Entropy: {self.entropy_vertical:.4f}\n"
            f"  Avg Diagonal Line: {self.average_diagonal_line:.2f}\n"
            f"  Max Diagonal Line: {self.max_diagonal_line:.0f}\n"
            f"  Trapping Time: {self.trapping_time:.2f}\n"
            f")"
        )


class RecurrenceAnalysis:
    """
    Recurrence Quantification Analysis for emotional dynamics.
    
    Analyzes patterns in physiological time series to identify:
    - Emotional stability (high determinism)
    - Transitions (diagonal line breaks)
    - Chaotic dynamics (low determinism, high entropy)
    - Laminar states (vertical structures)
    """
    
    def __init__(
        self,
        embedding_dim: int = 3,
        time_delay: int = 1,
        threshold: Optional[float] = None,
        threshold_method: str = 'percentage',
        threshold_value: float = 0.1,
        min_diagonal_line: int = 2,
        min_vertical_line: int = 2
    ):
        """
        Initialize RQA analyzer.
        
        Args:
            embedding_dim: Embedding dimension for phase space reconstruction
            time_delay: Time delay for embedding
            threshold: Distance threshold for recurrence (auto-computed if None)
            threshold_method: 'percentage' or 'fixed' or 'fan'
            threshold_value: Percentage of max distance or fixed threshold
            min_diagonal_line: Minimum length for diagonal line structures
            min_vertical_line: Minimum length for vertical line structures
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_diagonal_line = min_diagonal_line
        self.min_vertical_line = min_vertical_line
        
        # Computed values
        self.recurrence_matrix = None
        self.distance_matrix = None
        self.embedded_series = None
    
    def time_delay_embedding(self, time_series: np.ndarray) -> np.ndarray:
        """
        Perform time-delay embedding for phase space reconstruction.
        
        Args:
            time_series: 1D time series
            
        Returns:
            Embedded time series of shape (n_vectors, embedding_dim)
        """
        time_series = np.asarray(time_series).flatten()
        n = len(time_series)
        m = self.embedding_dim
        tau = self.time_delay
        
        # Number of embedded vectors
        n_vectors = n - (m - 1) * tau
        
        if n_vectors <= 0:
            raise ValueError(
                f"Time series too short for embedding. Need at least "
                f"{(m - 1) * tau + 1} points, got {n}"
            )
        
        # Create embedded vectors
        embedded = np.zeros((n_vectors, m))
        for i in range(m):
            embedded[:, i] = time_series[i * tau : i * tau + n_vectors]
        
        self.embedded_series = embedded
        return embedded
    
    def compute_distance_matrix(self, embedded: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Args:
            embedded: Embedded time series
            
        Returns:
            Distance matrix
        """
        distances = squareform(pdist(embedded, metric='euclidean'))
        self.distance_matrix = distances
        return distances
    
    def compute_recurrence_matrix(
        self,
        time_series: Optional[np.ndarray] = None,
        embedded: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute recurrence matrix.
        
        Args:
            time_series: 1D time series (if embedded not provided)
            embedded: Pre-embedded time series
            
        Returns:
            Binary recurrence matrix
        """
        if embedded is None:
            if time_series is None:
                raise ValueError("Must provide either time_series or embedded")
            embedded = self.time_delay_embedding(time_series)
        
        # Compute distances
        distances = self.compute_distance_matrix(embedded)
        
        # Determine threshold
        if self.threshold is None:
            if self.threshold_method == 'percentage':
                # Percentage of maximum distance
                threshold = self.threshold_value * np.max(distances)
            elif self.threshold_method == 'fixed':
                threshold = self.threshold_value
            elif self.threshold_method == 'fan':
                # Fixed amount of nearest neighbors (FAN)
                n_neighbors = int(self.threshold_value * len(distances))
                threshold = np.percentile(distances, 
                                        100 * n_neighbors / len(distances))
            else:
                raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        else:
            threshold = self.threshold
        
        # Create recurrence matrix
        recurrence = (distances <= threshold).astype(int)
        
        self.recurrence_matrix = recurrence
        return recurrence
    
    def _find_diagonal_lines(self, recurrence: np.ndarray) -> Dict:
        """Find diagonal line structures in recurrence matrix."""
        n = len(recurrence)
        diagonal_lines = []
        
        # Check each diagonal (excluding main diagonal)
        for k in range(-(n-1), n):
            diagonal = np.diagonal(recurrence, offset=k)
            
            # Find consecutive 1s
            line_length = 0
            for val in diagonal:
                if val == 1:
                    line_length += 1
                else:
                    if line_length >= self.min_diagonal_line:
                        diagonal_lines.append(line_length)
                    line_length = 0
            
            # Check last line
            if line_length >= self.min_diagonal_line:
                diagonal_lines.append(line_length)
        
        return {
            'lengths': diagonal_lines,
            'count': len(diagonal_lines),
            'total_points': sum(diagonal_lines) if diagonal_lines else 0
        }
    
    def _find_vertical_lines(self, recurrence: np.ndarray) -> Dict:
        """Find vertical line structures in recurrence matrix."""
        n = len(recurrence)
        vertical_lines = []
        
        # Check each column
        for j in range(n):
            column = recurrence[:, j]
            
            # Find consecutive 1s
            line_length = 0
            for val in column:
                if val == 1:
                    line_length += 1
                else:
                    if line_length >= self.min_vertical_line:
                        vertical_lines.append(line_length)
                    line_length = 0
            
            # Check last line
            if line_length >= self.min_vertical_line:
                vertical_lines.append(line_length)
        
        return {
            'lengths': vertical_lines,
            'count': len(vertical_lines),
            'total_points': sum(vertical_lines) if vertical_lines else 0
        }
    
    def analyze(self, time_series: np.ndarray) -> RQAMetrics:
        """
        Perform complete RQA analysis.
        
        Args:
            time_series: 1D time series data
            
        Returns:
            RQAMetrics object with all measures
        """
        # Compute recurrence matrix
        recurrence = self.compute_recurrence_matrix(time_series)
        n = len(recurrence)
        
        # Recurrence rate (RR)
        recurrence_rate = np.sum(recurrence) / (n * n)
        
        # Find line structures
        diagonal_info = self._find_diagonal_lines(recurrence)
        vertical_info = self._find_vertical_lines(recurrence)
        
        # Determinism (DET): ratio of recurrence points in diagonal lines
        if np.sum(recurrence) > 0:
            determinism = diagonal_info['total_points'] / np.sum(recurrence)
        else:
            determinism = 0.0
        
        # Laminarity (LAM): ratio of recurrence points in vertical lines
        if np.sum(recurrence) > 0:
            laminarity = vertical_info['total_points'] / np.sum(recurrence)
        else:
            laminarity = 0.0
        
        # Average diagonal line length
        if diagonal_info['count'] > 0:
            avg_diag = np.mean(diagonal_info['lengths'])
            max_diag = np.max(diagonal_info['lengths'])
        else:
            avg_diag = 0.0
            max_diag = 0.0
        
        # Average vertical line length (trapping time)
        if vertical_info['count'] > 0:
            avg_vert = np.mean(vertical_info['lengths'])
            max_vert = np.max(vertical_info['lengths'])
            trapping_time = avg_vert
        else:
            avg_vert = 0.0
            max_vert = 0.0
            trapping_time = 0.0
        
        # Entropy of diagonal line distribution
        if diagonal_info['lengths']:
            diag_hist, _ = np.histogram(
                diagonal_info['lengths'],
                bins=range(self.min_diagonal_line, int(max_diag) + 2)
            )
            diag_prob = diag_hist / np.sum(diag_hist)
            entropy_diag = entropy(diag_prob[diag_prob > 0])
        else:
            entropy_diag = 0.0
        
        # Entropy of vertical line distribution
        if vertical_info['lengths']:
            vert_hist, _ = np.histogram(
                vertical_info['lengths'],
                bins=range(self.min_vertical_line, int(max_vert) + 2)
            )
            vert_prob = vert_hist / np.sum(vert_hist)
            entropy_vert = entropy(vert_prob[vert_prob > 0])
        else:
            entropy_vert = 0.0
        
        return RQAMetrics(
            recurrence_rate=recurrence_rate,
            determinism=determinism,
            laminarity=laminarity,
            entropy_diagonal=entropy_diag,
            entropy_vertical=entropy_vert,
            average_diagonal_line=avg_diag,
            max_diagonal_line=max_diag,
            average_vertical_line=avg_vert,
            max_vertical_line=max_vert,
            trapping_time=trapping_time
        )
    
    def plot_recurrence_plot(
        self,
        time_series: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = 'binary'
    ):
        """
        Plot recurrence plot.
        
        Args:
            time_series: Time series to analyze (uses cached if None)
            figsize: Figure size
            cmap: Colormap
        """
        if time_series is not None:
            recurrence = self.compute_recurrence_matrix(time_series)
        elif self.recurrence_matrix is not None:
            recurrence = self.recurrence_matrix
        else:
            raise ValueError("No recurrence matrix available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(recurrence, cmap=cmap, origin='lower', interpolation='nearest')
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Time Index', fontsize=12)
        ax.set_title('Recurrence Plot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_distance_matrix(
        self,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = 'viridis'
    ):
        """Plot distance matrix."""
        if self.distance_matrix is None:
            raise ValueError("No distance matrix available. Run analysis first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(
            self.distance_matrix,
            cmap=cmap,
            origin='lower',
            interpolation='nearest'
        )
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Time Index', fontsize=12)
        ax.set_title('Distance Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Distance')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_embedded_space(
        self,
        time_series: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot embedded time series in phase space (first 3 dimensions).
        
        Args:
            time_series: Time series to embed
            figsize: Figure size
        """
        if time_series is not None:
            embedded = self.time_delay_embedding(time_series)
        elif self.embedded_series is not None:
            embedded = self.embedded_series
        else:
            raise ValueError("No embedded series available")
        
        if self.embedding_dim >= 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Color by time
            colors = np.arange(len(embedded))
            scatter = ax.scatter(
                embedded[:, 0],
                embedded[:, 1],
                embedded[:, 2],
                c=colors,
                cmap='viridis',
                s=10,
                alpha=0.6
            )
            
            ax.set_xlabel('Dimension 1', fontsize=11)
            ax.set_ylabel('Dimension 2', fontsize=11)
            ax.set_zlabel('Dimension 3', fontsize=11)
            ax.set_title('Embedded Phase Space', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Time')
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if self.embedding_dim == 2:
                colors = np.arange(len(embedded))
                scatter = ax.scatter(
                    embedded[:, 0],
                    embedded[:, 1],
                    c=colors,
                    cmap='viridis',
                    s=10,
                    alpha=0.6
                )
                ax.set_xlabel('Dimension 1', fontsize=11)
                ax.set_ylabel('Dimension 2', fontsize=11)
                plt.colorbar(scatter, ax=ax, label='Time')
            else:
                ax.plot(embedded[:, 0], alpha=0.7)
                ax.set_xlabel('Time', fontsize=11)
                ax.set_ylabel('Value', fontsize=11)
            
            ax.set_title('Embedded Phase Space', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


def compare_rqa_groups(
    group1_data: list,
    group2_data: list,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    **rqa_kwargs
) -> Tuple[Dict, Dict]:
    """
    Compare RQA metrics between two groups (e.g., clinical vs. control).
    
    Args:
        group1_data: List of time series for group 1
        group2_data: List of time series for group 2
        group1_name: Name for group 1
        group2_name: Name for group 2
        **rqa_kwargs: Arguments for RecurrenceAnalysis
        
    Returns:
        Tuple of (group1_metrics_dict, group2_metrics_dict)
    """
    rqa = RecurrenceAnalysis(**rqa_kwargs)
    
    def analyze_group(data_list):
        metrics_list = []
        for data in data_list:
            metrics = rqa.analyze(data)
            metrics_list.append(metrics)
        
        # Aggregate statistics
        return {
            'recurrence_rate': [m.recurrence_rate for m in metrics_list],
            'determinism': [m.determinism for m in metrics_list],
            'laminarity': [m.laminarity for m in metrics_list],
            'entropy_diagonal': [m.entropy_diagonal for m in metrics_list],
            'trapping_time': [m.trapping_time for m in metrics_list],
        }
    
    group1_metrics = analyze_group(group1_data)
    group2_metrics = analyze_group(group2_data)
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"RQA Comparison: {group1_name} vs {group2_name}")
    print(f"{'='*60}\n")
    
    for metric_name in group1_metrics.keys():
        g1_mean = np.mean(group1_metrics[metric_name])
        g1_std = np.std(group1_metrics[metric_name])
        g2_mean = np.mean(group2_metrics[metric_name])
        g2_std = np.std(group2_metrics[metric_name])
        
        print(f"{metric_name.replace('_', ' ').title()}:")
        print(f"  {group1_name}: {g1_mean:.4f} ± {g1_std:.4f}")
        print(f"  {group2_name}: {g2_mean:.4f} ± {g2_std:.4f}")
        print()
    
    return group1_metrics, group2_metrics
