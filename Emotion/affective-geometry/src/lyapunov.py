"""
Lyapunov exponent calculation for emotional stability analysis.

Quantifies emotional volatility and sensitivity to initial conditions,
comparing clinical populations (depression, anxiety, bipolar disorder).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple, List
from scipy.spatial.distance import euclidean
import warnings


class LyapunovAnalysis:
    """
    Calculate Lyapunov exponents for emotional dynamics.
    
    Positive exponents indicate chaotic/volatile emotional dynamics.
    Negative exponents indicate stable emotional dynamics.
    Zero exponents indicate marginally stable dynamics.
    """
    
    def __init__(
        self,
        state_space=None,
        dynamics_function: Optional[Callable] = None
    ):
        """
        Initialize Lyapunov analyzer.
        
        Args:
            state_space: EmotionalStateSpace object (if available)
            dynamics_function: Custom dynamics function (state, time) -> velocity
        """
        self.state_space = state_space
        self.dynamics_function = dynamics_function
        
        if state_space is None and dynamics_function is None:
            raise ValueError("Must provide either state_space or dynamics_function")
    
    def _get_velocity(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Get velocity from state space or custom function."""
        if self.state_space is not None:
            return self.state_space.compute_velocity(state, time)
        else:
            return self.dynamics_function(state, time)
    
    def calculate_largest_exponent(
        self,
        trajectory: Optional[np.ndarray] = None,
        initial_state: Optional[np.ndarray] = None,
        duration: float = 100.0,
        dt: float = 0.01,
        perturbation_size: float = 1e-8,
        renormalization_steps: int = 10
    ) -> float:
        """
        Calculate largest Lyapunov exponent using trajectory-based method.
        
        Args:
            trajectory: Pre-computed trajectory (if None, simulate from initial_state)
            initial_state: Starting state for simulation
            duration: Simulation duration
            dt: Time step
            perturbation_size: Initial perturbation magnitude
            renormalization_steps: Steps between renormalization
            
        Returns:
            Largest Lyapunov exponent
        """
        # Get or generate trajectory
        if trajectory is None:
            if initial_state is None:
                raise ValueError("Must provide either trajectory or initial_state")
            
            if self.state_space is not None:
                trajectory = self.state_space.simulate_trajectory(
                    initial_state, duration, dt, save_history=False
                )
            else:
                # Simulate using custom dynamics
                n_steps = int(duration / dt)
                trajectory = np.zeros((n_steps, len(initial_state)))
                state = np.array(initial_state, dtype=np.float64)
                trajectory[0] = state
                
                for i in range(1, n_steps):
                    velocity = self._get_velocity(state, i * dt)
                    state = state + velocity * dt
                    trajectory[i] = state
        
        n_steps = len(trajectory)
        dimensions = trajectory.shape[1]
        
        # Initialize perturbation
        perturbation = np.random.normal(0, perturbation_size, size=dimensions)
        perturbation = perturbation / np.linalg.norm(perturbation) * perturbation_size
        
        lyapunov_sum = 0.0
        n_renormalizations = 0
        
        # Track perturbed trajectory
        perturbed_state = trajectory[0] + perturbation
        
        for i in range(1, n_steps):
            time = i * dt
            
            # Evolve perturbed trajectory
            if self.state_space is not None:
                perturbed_velocity = self.state_space.compute_velocity(perturbed_state, time)
            else:
                perturbed_velocity = self._get_velocity(perturbed_state, time)
            
            perturbed_state = perturbed_state + perturbed_velocity * dt
            
            # Renormalize periodically
            if i % renormalization_steps == 0:
                # Compute separation
                separation = perturbed_state - trajectory[i]
                distance = np.linalg.norm(separation)
                
                if distance > 1e-12:
                    # Accumulate log of growth rate
                    lyapunov_sum += np.log(distance / perturbation_size)
                    n_renormalizations += 1
                    
                    # Renormalize perturbation
                    perturbation = separation / distance * perturbation_size
                    perturbed_state = trajectory[i] + perturbation
        
        if n_renormalizations == 0:
            warnings.warn("No renormalizations occurred. Result may be unreliable.")
            return 0.0
        
        # Average over time
        largest_exponent = lyapunov_sum / (n_renormalizations * renormalization_steps * dt)
        
        return largest_exponent
    
    def calculate_spectrum(
        self,
        initial_state: np.ndarray,
        duration: float = 100.0,
        dt: float = 0.01,
        n_iterations: int = 1000
    ) -> np.ndarray:
        """
        Calculate full Lyapunov spectrum using QR decomposition method.
        
        Args:
            initial_state: Starting state
            duration: Simulation duration
            dt: Time step
            n_iterations: Number of QR decomposition iterations
            
        Returns:
            Array of Lyapunov exponents (sorted descending)
        """
        dimensions = len(initial_state)
        
        # Initialize orthonormal basis
        Q = np.eye(dimensions)
        
        # Accumulate exponents
        lyapunov_sum = np.zeros(dimensions)
        
        state = np.array(initial_state, dtype=np.float64)
        
        for iteration in range(n_iterations):
            # Evolve state and tangent vectors
            for _ in range(int(1.0 / dt)):
                # Compute Jacobian numerically
                jacobian = self._compute_jacobian(state, dt)
                
                # Evolve tangent vectors
                Q = jacobian @ Q
                
                # Evolve state
                velocity = self._get_velocity(state, iteration)
                state = state + velocity * dt
            
            # QR decomposition
            Q, R = np.linalg.qr(Q)
            
            # Accumulate log of diagonal elements
            lyapunov_sum += np.log(np.abs(np.diag(R)))
        
        # Average over time
        spectrum = lyapunov_sum / (n_iterations * 1.0)
        
        return np.sort(spectrum)[::-1]  # Sort descending
    
    def _compute_jacobian(
        self,
        state: np.ndarray,
        dt: float,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Jacobian matrix numerically using finite differences.
        
        Args:
            state: Current state
            dt: Time step
            epsilon: Perturbation size for finite differences
            
        Returns:
            Jacobian matrix
        """
        dimensions = len(state)
        jacobian = np.zeros((dimensions, dimensions))
        
        f0 = self._get_velocity(state, 0.0)
        
        for i in range(dimensions):
            # Perturb in direction i
            state_plus = state.copy()
            state_plus[i] += epsilon
            
            f_plus = self._get_velocity(state_plus, 0.0)
            
            # Finite difference
            jacobian[:, i] = (f_plus - f0) / epsilon
        
        # Convert to flow map Jacobian
        jacobian = np.eye(dimensions) + dt * jacobian
        
        return jacobian
    
    def classify_dynamics(self, largest_exponent: float) -> str:
        """
        Classify emotional dynamics based on largest Lyapunov exponent.
        
        Args:
            largest_exponent: Largest Lyapunov exponent
            
        Returns:
            Classification string
        """
        if largest_exponent > 0.01:
            return "Chaotic/Volatile (High emotional instability)"
        elif largest_exponent < -0.01:
            return "Stable (Low emotional volatility)"
        else:
            return "Marginally Stable (Moderate emotional dynamics)"
    
    def plot_exponent_evolution(
        self,
        initial_state: np.ndarray,
        duration: float = 100.0,
        dt: float = 0.01,
        window_size: int = 100,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot evolution of Lyapunov exponent over time.
        
        Args:
            initial_state: Starting state
            duration: Total duration
            dt: Time step
            window_size: Window for local exponent calculation
            figsize: Figure size
        """
        # Generate trajectory
        if self.state_space is not None:
            trajectory = self.state_space.simulate_trajectory(
                initial_state, duration, dt, save_history=False
            )
        else:
            n_steps = int(duration / dt)
            trajectory = np.zeros((n_steps, len(initial_state)))
            state = np.array(initial_state, dtype=np.float64)
            trajectory[0] = state
            
            for i in range(1, n_steps):
                velocity = self._get_velocity(state, i * dt)
                state = state + velocity * dt
                trajectory[i] = state
        
        # Calculate local exponents
        n_windows = len(trajectory) // window_size
        exponents = []
        times = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_traj = trajectory[start_idx:end_idx]
            
            exponent = self.calculate_largest_exponent(
                trajectory=window_traj,
                dt=dt,
                renormalization_steps=5
            )
            exponents.append(exponent)
            times.append((start_idx + window_size // 2) * dt)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(times, exponents, linewidth=2, color='darkblue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero line')
        ax.fill_between(times, 0, exponents, where=np.array(exponents) > 0, 
                        alpha=0.3, color='red', label='Chaotic')
        ax.fill_between(times, 0, exponents, where=np.array(exponents) < 0, 
                        alpha=0.3, color='green', label='Stable')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Largest Lyapunov Exponent', fontsize=12)
        ax.set_title('Evolution of Emotional Stability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig, ax


def compare_lyapunov_groups(
    group1_trajectories: List[np.ndarray],
    group2_trajectories: List[np.ndarray],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    state_space=None,
    dt: float = 0.01,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare Lyapunov exponents between two groups (e.g., clinical vs. control).
    
    Args:
        group1_trajectories: List of trajectories for group 1
        group2_trajectories: List of trajectories for group 2
        group1_name: Name for group 1
        group2_name: Name for group 2
        state_space: EmotionalStateSpace object
        dt: Time step
        figsize: Figure size
    """
    analyzer = LyapunovAnalysis(state_space=state_space)
    
    # Calculate exponents for both groups
    group1_exponents = []
    for traj in group1_trajectories:
        exp = analyzer.calculate_largest_exponent(trajectory=traj, dt=dt)
        group1_exponents.append(exp)
    
    group2_exponents = []
    for traj in group2_trajectories:
        exp = analyzer.calculate_largest_exponent(trajectory=traj, dt=dt)
        group2_exponents.append(exp)
    
    # Statistics
    g1_mean = np.mean(group1_exponents)
    g1_std = np.std(group1_exponents)
    g2_mean = np.mean(group2_exponents)
    g2_std = np.std(group2_exponents)
    
    print(f"\n{'='*60}")
    print(f"Lyapunov Exponent Comparison")
    print(f"{'='*60}\n")
    print(f"{group1_name}: {g1_mean:.6f} ± {g1_std:.6f}")
    print(f"{group2_name}: {g2_mean:.6f} ± {g2_std:.6f}")
    print(f"\nDifference: {g1_mean - g2_mean:.6f}")
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(group1_exponents, group2_exponents)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    ax = axes[0]
    ax.boxplot([group1_exponents, group2_exponents], 
               labels=[group1_name, group2_name])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Largest Lyapunov Exponent', fontsize=11)
    ax.set_title('Group Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Histogram
    ax = axes[1]
    ax.hist(group1_exponents, bins=15, alpha=0.6, label=group1_name, color='blue')
    ax.hist(group2_exponents, bins=15, alpha=0.6, label=group2_name, color='orange')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Largest Lyapunov Exponent', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes, {
        'group1': {'mean': g1_mean, 'std': g1_std, 'values': group1_exponents},
        'group2': {'mean': g2_mean, 'std': g2_std, 'values': group2_exponents},
        't_statistic': t_stat,
        'p_value': p_value
    }
