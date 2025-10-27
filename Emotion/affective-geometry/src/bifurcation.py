"""
Bifurcation analysis and basin of attraction mapping.

Analyzes how parameter changes (stress, medication, context) shift
the emotional landscape topology and identifies critical transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict
from scipy.integrate import odeint
from scipy.optimize import fsolve
import warnings


class BifurcationAnalysis:
    """
    Bifurcation analysis for emotional dynamics.
    
    Identifies critical transitions and tipping points where small
    parameter changes cause qualitative shifts in emotional patterns.
    """
    
    def __init__(self, state_space=None, dynamics_function: Optional[Callable] = None):
        """
        Initialize bifurcation analyzer.
        
        Args:
            state_space: EmotionalStateSpace object
            dynamics_function: Custom dynamics (state, params) -> velocity
        """
        self.state_space = state_space
        self.dynamics_function = dynamics_function
    
    def find_fixed_points(
        self,
        parameter_value: float,
        parameter_name: str,
        search_bounds: Tuple[float, float] = (-1.0, 1.0),
        n_initial_guesses: int = 20,
        dimensions: int = 5
    ) -> List[np.ndarray]:
        """
        Find fixed points for a given parameter value.
        
        Args:
            parameter_value: Value of bifurcation parameter
            parameter_name: Name of parameter being varied
            search_bounds: Bounds for initial guess search
            n_initial_guesses: Number of random initial guesses
            dimensions: State space dimensions
            
        Returns:
            List of fixed point states
        """
        fixed_points = []
        
        def dynamics_wrapper(state):
            """Wrapper for root finding."""
            if self.state_space is not None:
                # Temporarily modify parameter
                original_value = getattr(self.state_space, parameter_name, None)
                setattr(self.state_space, parameter_name, parameter_value)
                velocity = self.state_space.compute_velocity(state)
                if original_value is not None:
                    setattr(self.state_space, parameter_name, original_value)
                return velocity
            else:
                return self.dynamics_function(state, {parameter_name: parameter_value})
        
        # Try multiple initial guesses
        for _ in range(n_initial_guesses):
            initial_guess = np.random.uniform(
                search_bounds[0], search_bounds[1], size=dimensions
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solution = fsolve(dynamics_wrapper, initial_guess, full_output=True)
                    
                    if solution[2] == 1:  # Converged
                        fixed_point = solution[0]
                        
                        # Check if it's a new fixed point
                        is_new = True
                        for existing_fp in fixed_points:
                            if np.linalg.norm(fixed_point - existing_fp) < 0.01:
                                is_new = False
                                break
                        
                        if is_new:
                            fixed_points.append(fixed_point)
            except:
                continue
        
        return fixed_points
    
    def compute_bifurcation_diagram(
        self,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        n_parameter_values: int = 100,
        dimension_to_plot: int = 0,
        transient_steps: int = 1000,
        sample_steps: int = 100,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute bifurcation diagram by varying a parameter.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_range: (min, max) range for parameter
            n_parameter_values: Number of parameter values to sample
            dimension_to_plot: Which state dimension to plot
            transient_steps: Steps to discard as transient
            sample_steps: Steps to sample after transient
            dt: Time step
            
        Returns:
            Tuple of (parameter_values, state_samples_list)
        """
        parameter_values = np.linspace(
            parameter_range[0], parameter_range[1], n_parameter_values
        )
        
        state_samples = []
        
        # Initial state
        if self.state_space is not None:
            state = np.zeros(self.state_space.dimensions)
        else:
            state = np.zeros(5)  # Default
        
        for param_val in parameter_values:
            # Set parameter
            if self.state_space is not None:
                original_value = getattr(self.state_space, parameter_name, None)
                setattr(self.state_space, parameter_name, param_val)
            
            # Discard transient
            for _ in range(transient_steps):
                if self.state_space is not None:
                    state = self.state_space.step(state, dt=dt)
                else:
                    velocity = self.dynamics_function(state, {parameter_name: param_val})
                    state = state + velocity * dt
            
            # Sample steady state
            samples = []
            for _ in range(sample_steps):
                if self.state_space is not None:
                    state = self.state_space.step(state, dt=dt)
                else:
                    velocity = self.dynamics_function(state, {parameter_name: param_val})
                    state = state + velocity * dt
                
                samples.append(state[dimension_to_plot])
            
            state_samples.append(samples)
            
            # Restore parameter
            if self.state_space is not None and original_value is not None:
                setattr(self.state_space, parameter_name, original_value)
        
        return parameter_values, state_samples
    
    def plot_bifurcation_diagram(
        self,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        dimension_to_plot: int = 0,
        dimension_name: str = "Valence",
        figsize: Tuple[int, int] = (12, 8),
        **kwargs
    ):
        """
        Plot bifurcation diagram.
        
        Args:
            parameter_name: Parameter to vary
            parameter_range: Range of parameter values
            dimension_to_plot: Which dimension to show
            dimension_name: Name of dimension for label
            figsize: Figure size
            **kwargs: Additional arguments for compute_bifurcation_diagram
        """
        param_values, state_samples = self.compute_bifurcation_diagram(
            parameter_name, parameter_range, dimension_to_plot=dimension_to_plot, **kwargs
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each parameter value's samples
        for i, param_val in enumerate(param_values):
            samples = state_samples[i]
            ax.plot([param_val] * len(samples), samples, 'b.', markersize=1, alpha=0.5)
        
        ax.set_xlabel(parameter_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(dimension_name, fontsize=12)
        ax.set_title(f'Bifurcation Diagram: {dimension_name} vs {parameter_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def compute_basin_of_attraction(
        self,
        attractor_state: np.ndarray,
        grid_resolution: int = 50,
        dimensions: Tuple[int, int] = (0, 1),
        bounds: Tuple[float, float] = (-1.0, 1.0),
        max_time: float = 50.0,
        dt: float = 0.01,
        convergence_threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute basin of attraction for a specific attractor.
        
        Args:
            attractor_state: Target attractor state
            grid_resolution: Grid points per dimension
            dimensions: Which 2D slice to compute
            bounds: State space bounds
            max_time: Maximum simulation time
            dt: Time step
            convergence_threshold: Distance threshold for convergence
            
        Returns:
            Tuple of (X, Y, escape_times) where escape_times is time to reach attractor
        """
        dim1, dim2 = dimensions
        
        # Create grid
        x = np.linspace(bounds[0], bounds[1], grid_resolution)
        y = np.linspace(bounds[0], bounds[1], grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        escape_times = np.zeros_like(X)
        
        n_steps = int(max_time / dt)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Initial state
                if self.state_space is not None:
                    state = np.zeros(self.state_space.dimensions)
                else:
                    state = np.zeros(len(attractor_state))
                
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Simulate until convergence or max time
                converged = False
                for step in range(n_steps):
                    if self.state_space is not None:
                        state = self.state_space.step(state, dt=dt)
                    else:
                        velocity = self.dynamics_function(state, {})
                        state = state + velocity * dt
                    
                    # Check convergence
                    distance = np.linalg.norm(state - attractor_state)
                    if distance < convergence_threshold:
                        escape_times[i, j] = step * dt
                        converged = True
                        break
                
                if not converged:
                    escape_times[i, j] = max_time
        
        return X, Y, escape_times
    
    def plot_basin_of_attraction(
        self,
        attractor_state: np.ndarray,
        attractor_name: str = "Attractor",
        dimensions: Tuple[int, int] = (0, 1),
        dimension_names: Tuple[str, str] = ("Valence", "Arousal"),
        figsize: Tuple[int, int] = (10, 8),
        **kwargs
    ):
        """
        Plot basin of attraction with escape times.
        
        Args:
            attractor_state: Target attractor
            attractor_name: Name for title
            dimensions: Which dimensions to plot
            dimension_names: Names for axes
            figsize: Figure size
            **kwargs: Additional arguments for compute_basin_of_attraction
        """
        X, Y, escape_times = self.compute_basin_of_attraction(
            attractor_state, dimensions=dimensions, **kwargs
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot escape time heatmap
        im = ax.contourf(X, Y, escape_times, levels=20, cmap='viridis_r')
        
        # Mark attractor
        dim1, dim2 = dimensions
        ax.plot(attractor_state[dim1], attractor_state[dim2], 
               'r*', markersize=20, label=attractor_name)
        
        ax.set_xlabel(dimension_names[0], fontsize=12)
        ax.set_ylabel(dimension_names[1], fontsize=12)
        ax.set_title(f'Basin of Attraction: {attractor_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Time to Convergence', fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def identify_critical_transitions(
        self,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        n_parameter_values: int = 100,
        threshold_variance: float = 0.1
    ) -> List[float]:
        """
        Identify critical transition points (bifurcations).
        
        Detects sudden changes in variance as indicators of bifurcations.
        
        Args:
            parameter_name: Parameter to vary
            parameter_range: Range to search
            n_parameter_values: Resolution
            threshold_variance: Variance change threshold
            
        Returns:
            List of parameter values where transitions occur
        """
        param_values, state_samples = self.compute_bifurcation_diagram(
            parameter_name, parameter_range, n_parameter_values=n_parameter_values
        )
        
        # Compute variance for each parameter value
        variances = [np.var(samples) for samples in state_samples]
        
        # Find sudden changes in variance
        critical_points = []
        for i in range(1, len(variances)):
            variance_change = abs(variances[i] - variances[i-1])
            if variance_change > threshold_variance:
                critical_points.append(param_values[i])
        
        return critical_points


class BasinStabilityAnalysis:
    """
    Analyze basin stability for multiple attractors.
    
    Quantifies resilience of emotional states to perturbations.
    """
    
    def __init__(self, state_space):
        """
        Initialize basin stability analyzer.
        
        Args:
            state_space: EmotionalStateSpace object
        """
        self.state_space = state_space
    
    def compute_basin_stability(
        self,
        attractor_idx: int,
        n_samples: int = 1000,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        max_time: float = 50.0,
        dt: float = 0.01,
        convergence_threshold: float = 0.1
    ) -> float:
        """
        Compute basin stability for an attractor.
        
        Basin stability = fraction of random initial conditions that
        converge to this attractor.
        
        Args:
            attractor_idx: Index of target attractor
            n_samples: Number of random initial conditions
            bounds: Sampling bounds
            max_time: Maximum simulation time
            dt: Time step
            convergence_threshold: Distance threshold for convergence
            
        Returns:
            Basin stability (0 to 1)
        """
        if len(self.state_space.landscape) == 0:
            raise ValueError("No attractors in state space")
        
        target_attractor = self.state_space.landscape[attractor_idx]
        convergence_count = 0
        
        for _ in range(n_samples):
            # Random initial condition
            initial_state = np.random.uniform(
                bounds[0], bounds[1], size=self.state_space.dimensions
            )
            
            # Simulate
            trajectory = self.state_space.simulate_trajectory(
                initial_state, duration=max_time, dt=dt, save_history=False
            )
            
            # Check if converged to target attractor
            final_state = trajectory[-1]
            distance = target_attractor.distance_to(final_state)
            
            if distance < convergence_threshold:
                convergence_count += 1
        
        basin_stability = convergence_count / n_samples
        return basin_stability
    
    def compute_all_basin_stabilities(
        self,
        n_samples: int = 1000,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute basin stability for all attractors.
        
        Args:
            n_samples: Number of samples
            **kwargs: Additional arguments for compute_basin_stability
            
        Returns:
            Dictionary mapping attractor names to basin stabilities
        """
        stabilities = {}
        
        for i, attractor in enumerate(self.state_space.landscape.attractors):
            stability = self.compute_basin_stability(i, n_samples, **kwargs)
            stabilities[attractor.name] = stability
        
        return stabilities
    
    def plot_basin_stabilities(
        self,
        stabilities: Optional[Dict[str, float]] = None,
        n_samples: int = 1000,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot basin stabilities for all attractors.
        
        Args:
            stabilities: Pre-computed stabilities (computed if None)
            n_samples: Number of samples if computing
            figsize: Figure size
        """
        if stabilities is None:
            stabilities = self.compute_all_basin_stabilities(n_samples)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(stabilities.keys())
        values = list(stabilities.values())
        
        bars = ax.bar(names, values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Color code by stability
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.7:
                bar.set_color('green')
            elif val > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_ylabel('Basin Stability', fontsize=12)
        ax.set_title('Emotional State Resilience', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (name, val) in enumerate(zip(names, values)):
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig, ax
