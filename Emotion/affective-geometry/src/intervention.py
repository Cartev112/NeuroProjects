"""
Optimal intervention calculator for emotional regulation.

Computes minimal perturbations to shift emotional trajectories
from negative to positive attractors.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Callable
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass


@dataclass
class InterventionResult:
    """Container for intervention optimization results."""
    action: np.ndarray
    time_to_target: float
    success_prob: float
    energy_cost: float
    trajectory: np.ndarray
    intervention_type: str
    
    def __repr__(self) -> str:
        return (
            f"InterventionResult(\n"
            f"  Action: {self.action}\n"
            f"  Time to Target: {self.time_to_target:.2f}\n"
            f"  Success Probability: {self.success_prob:.2%}\n"
            f"  Energy Cost: {self.energy_cost:.4f}\n"
            f"  Type: {self.intervention_type}\n"
            f")"
        )


class InterventionOptimizer:
    """
    Optimize interventions to shift emotional trajectories.
    
    Finds minimal-energy perturbations that move the system from
    negative emotional states to positive attractors.
    """
    
    def __init__(self, state_space):
        """
        Initialize intervention optimizer.
        
        Args:
            state_space: EmotionalStateSpace object
        """
        self.state_space = state_space
    
    def find_optimal_intervention(
        self,
        current_state: np.ndarray,
        target_attractor,
        max_magnitude: float = 1.0,
        time_horizon: int = 50,
        dt: float = 0.1,
        method: str = 'impulse',
        n_trials: int = 100
    ) -> InterventionResult:
        """
        Find optimal intervention to reach target attractor.
        
        Args:
            current_state: Current emotional state
            target_attractor: Target attractor object
            max_magnitude: Maximum intervention magnitude
            time_horizon: Time horizon for simulation
            dt: Time step
            method: 'impulse' (one-time push) or 'sustained' (continuous)
            n_trials: Number of optimization trials
            
        Returns:
            InterventionResult object
        """
        current_state = np.array(current_state, dtype=np.float64)
        dimensions = len(current_state)
        
        if method == 'impulse':
            return self._optimize_impulse_intervention(
                current_state, target_attractor, max_magnitude,
                time_horizon, dt, n_trials
            )
        elif method == 'sustained':
            return self._optimize_sustained_intervention(
                current_state, target_attractor, max_magnitude,
                time_horizon, dt, n_trials
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimize_impulse_intervention(
        self,
        current_state: np.ndarray,
        target_attractor,
        max_magnitude: float,
        time_horizon: int,
        dt: float,
        n_trials: int
    ) -> InterventionResult:
        """Optimize single impulse intervention."""
        dimensions = len(current_state)
        
        def objective(intervention):
            """Objective: minimize intervention energy + time to target."""
            # Apply intervention
            perturbed_state = current_state + intervention
            
            # Simulate trajectory
            trajectory = self.state_space.simulate_trajectory(
                perturbed_state,
                duration=time_horizon * dt,
                dt=dt,
                save_history=False
            )
            
            # Compute distance to target over time
            distances = np.array([
                target_attractor.distance_to(state)
                for state in trajectory
            ])
            
            # Find time to reach target (within threshold)
            threshold = 0.2
            reached_indices = np.where(distances < threshold)[0]
            
            if len(reached_indices) > 0:
                time_to_target = reached_indices[0] * dt
                final_distance = distances[reached_indices[0]]
            else:
                time_to_target = time_horizon * dt
                final_distance = distances[-1]
            
            # Energy cost (L2 norm of intervention)
            energy = np.linalg.norm(intervention)
            
            # Combined objective: minimize energy and time, minimize final distance
            return energy + 0.5 * time_to_target + 2.0 * final_distance
        
        # Bounds for intervention
        bounds = [(-max_magnitude, max_magnitude) for _ in range(dimensions)]
        
        # Optimize using differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=n_trials,
            seed=42,
            atol=1e-4,
            tol=1e-4
        )
        
        optimal_intervention = result.x
        
        # Simulate with optimal intervention
        perturbed_state = current_state + optimal_intervention
        trajectory = self.state_space.simulate_trajectory(
            perturbed_state,
            duration=time_horizon * dt,
            dt=dt,
            save_history=False
        )
        
        # Compute metrics
        distances = np.array([
            target_attractor.distance_to(state)
            for state in trajectory
        ])
        
        threshold = 0.2
        reached_indices = np.where(distances < threshold)[0]
        
        if len(reached_indices) > 0:
            time_to_target = reached_indices[0] * dt
            success_prob = 1.0
        else:
            time_to_target = time_horizon * dt
            # Partial success based on final distance
            final_distance = distances[-1]
            success_prob = max(0, 1.0 - final_distance / 2.0)
        
        energy_cost = np.linalg.norm(optimal_intervention)
        
        return InterventionResult(
            action=optimal_intervention,
            time_to_target=time_to_target,
            success_prob=success_prob,
            energy_cost=energy_cost,
            trajectory=trajectory,
            intervention_type='impulse'
        )
    
    def _optimize_sustained_intervention(
        self,
        current_state: np.ndarray,
        target_attractor,
        max_magnitude: float,
        time_horizon: int,
        dt: float,
        n_trials: int
    ) -> InterventionResult:
        """Optimize sustained intervention (control signal over time)."""
        dimensions = len(current_state)
        
        # Simplified: constant control signal
        def objective(control_signal):
            """Objective for sustained control."""
            state = current_state.copy()
            trajectory = [state.copy()]
            
            total_energy = 0.0
            
            for t in range(time_horizon):
                # Apply control
                state = state + control_signal * dt
                
                # Evolve dynamics
                state = self.state_space.step(state, t * dt, dt)
                trajectory.append(state.copy())
                
                # Accumulate energy
                total_energy += np.linalg.norm(control_signal) * dt
            
            trajectory = np.array(trajectory)
            
            # Distance to target
            distances = np.array([
                target_attractor.distance_to(state)
                for state in trajectory
            ])
            
            final_distance = distances[-1]
            
            return total_energy + 2.0 * final_distance
        
        # Bounds
        bounds = [(-max_magnitude, max_magnitude) for _ in range(dimensions)]
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=n_trials,
            seed=42,
            atol=1e-4,
            tol=1e-4
        )
        
        optimal_control = result.x
        
        # Simulate with optimal control
        state = current_state.copy()
        trajectory = [state.copy()]
        total_energy = 0.0
        
        for t in range(time_horizon):
            state = state + optimal_control * dt
            state = self.state_space.step(state, t * dt, dt)
            trajectory.append(state.copy())
            total_energy += np.linalg.norm(optimal_control) * dt
        
        trajectory = np.array(trajectory)
        
        # Metrics
        distances = np.array([
            target_attractor.distance_to(state)
            for state in trajectory
        ])
        
        threshold = 0.2
        reached_indices = np.where(distances < threshold)[0]
        
        if len(reached_indices) > 0:
            time_to_target = reached_indices[0] * dt
            success_prob = 1.0
        else:
            time_to_target = time_horizon * dt
            final_distance = distances[-1]
            success_prob = max(0, 1.0 - final_distance / 2.0)
        
        return InterventionResult(
            action=optimal_control,
            time_to_target=time_to_target,
            success_prob=success_prob,
            energy_cost=total_energy,
            trajectory=trajectory,
            intervention_type='sustained'
        )
    
    def compare_intervention_strategies(
        self,
        current_state: np.ndarray,
        target_attractor,
        strategies: List[str] = ['impulse', 'sustained'],
        **kwargs
    ) -> Dict[str, InterventionResult]:
        """
        Compare different intervention strategies.
        
        Args:
            current_state: Current state
            target_attractor: Target attractor
            strategies: List of strategy names
            **kwargs: Additional arguments for optimization
            
        Returns:
            Dictionary mapping strategy names to results
        """
        results = {}
        
        for strategy in strategies:
            result = self.find_optimal_intervention(
                current_state,
                target_attractor,
                method=strategy,
                **kwargs
            )
            results[strategy] = result
        
        return results
    
    def plot_intervention_trajectory(
        self,
        intervention_result: InterventionResult,
        current_state: np.ndarray,
        target_attractor,
        dimensions: Tuple[int, int] = (0, 1),
        dimension_names: Tuple[str, str] = ("Valence", "Arousal"),
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize intervention trajectory.
        
        Args:
            intervention_result: Result from optimization
            current_state: Initial state
            target_attractor: Target attractor
            dimensions: Which dimensions to plot
            dimension_names: Names for axes
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        dim1, dim2 = dimensions
        trajectory = intervention_result.trajectory
        
        # Plot 1: Phase space trajectory
        ax = axes[0, 0]
        
        # Plot vector field
        X, Y, U, V = self.state_space.compute_vector_field(
            grid_resolution=15, dimensions=dimensions
        )
        ax.quiver(X, Y, U, V, alpha=0.3, scale=10, width=0.003)
        
        # Plot trajectory
        ax.plot(trajectory[:, dim1], trajectory[:, dim2], 
               'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Mark start, intervention, and target
        ax.plot(current_state[dim1], current_state[dim2], 
               'ro', markersize=12, label='Start')
        
        if intervention_result.intervention_type == 'impulse':
            perturbed = current_state + intervention_result.action
            ax.arrow(current_state[dim1], current_state[dim2],
                    intervention_result.action[dim1], 
                    intervention_result.action[dim2],
                    head_width=0.05, head_length=0.05, 
                    fc='red', ec='red', linewidth=2, alpha=0.7)
        
        ax.plot(target_attractor.center[dim1], target_attractor.center[dim2],
               'g*', markersize=20, label='Target')
        
        ax.set_xlabel(dimension_names[0], fontsize=11)
        ax.set_ylabel(dimension_names[1], fontsize=11)
        ax.set_title('Intervention Trajectory', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distance to target over time
        ax = axes[0, 1]
        
        distances = [target_attractor.distance_to(state) for state in trajectory]
        times = np.arange(len(distances)) * 0.1
        
        ax.plot(times, distances, 'b-', linewidth=2)
        ax.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='Success threshold')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Distance to Target', fontsize=11)
        ax.set_title('Convergence Progress', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: State dimensions over time
        ax = axes[1, 0]
        
        for i, name in enumerate([dimension_names[0], dimension_names[1]]):
            dim_idx = dimensions[i]
            ax.plot(times, trajectory[:, dim_idx], linewidth=2, label=name)
        
        ax.axhline(y=target_attractor.center[dim1], 
                  color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=target_attractor.center[dim2], 
                  color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('State Value', fontsize=11)
        ax.set_title('State Evolution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Intervention summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Intervention Summary
        {'='*40}
        
        Type: {intervention_result.intervention_type.title()}
        
        Action Vector:
        {intervention_result.action}
        
        Time to Target: {intervention_result.time_to_target:.2f}
        Success Probability: {intervention_result.success_prob:.1%}
        Energy Cost: {intervention_result.energy_cost:.4f}
        
        Interpretation:
        """
        
        if intervention_result.success_prob > 0.9:
            summary_text += "\n✓ Highly effective intervention"
        elif intervention_result.success_prob > 0.5:
            summary_text += "\n~ Moderately effective intervention"
        else:
            summary_text += "\n✗ Low effectiveness - consider alternative"
        
        if intervention_result.energy_cost < 0.5:
            summary_text += "\n✓ Low energy cost"
        elif intervention_result.energy_cost < 1.5:
            summary_text += "\n~ Moderate energy cost"
        else:
            summary_text += "\n! High energy cost"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        return fig, axes
    
    def sensitivity_analysis(
        self,
        current_state: np.ndarray,
        target_attractor,
        n_perturbations: int = 50,
        perturbation_scale: float = 0.1,
        **kwargs
    ) -> Dict:
        """
        Analyze sensitivity of intervention to initial state variations.
        
        Args:
            current_state: Nominal current state
            target_attractor: Target attractor
            n_perturbations: Number of perturbed initial states to test
            perturbation_scale: Scale of perturbations
            **kwargs: Arguments for optimization
            
        Returns:
            Dictionary with sensitivity statistics
        """
        success_rates = []
        energy_costs = []
        times_to_target = []
        
        for _ in range(n_perturbations):
            # Perturb initial state
            perturbation = np.random.normal(0, perturbation_scale, size=len(current_state))
            perturbed_state = current_state + perturbation
            
            # Find intervention
            result = self.find_optimal_intervention(
                perturbed_state,
                target_attractor,
                **kwargs
            )
            
            success_rates.append(result.success_prob)
            energy_costs.append(result.energy_cost)
            times_to_target.append(result.time_to_target)
        
        return {
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_energy_cost': np.mean(energy_costs),
            'std_energy_cost': np.std(energy_costs),
            'mean_time_to_target': np.mean(times_to_target),
            'std_time_to_target': np.std(times_to_target),
            'all_success_rates': success_rates,
            'all_energy_costs': energy_costs,
            'all_times_to_target': times_to_target
        }


def suggest_intervention_strategy(
    current_state: np.ndarray,
    emotional_history: Optional[np.ndarray] = None,
    personality_traits: Optional[Dict[str, float]] = None
) -> str:
    """
    Suggest personalized intervention strategy based on context.
    
    Args:
        current_state: Current emotional state
        emotional_history: Recent emotional trajectory
        personality_traits: Personality trait scores (neuroticism, etc.)
        
    Returns:
        Suggested intervention strategy
    """
    suggestions = []
    
    # Analyze current state
    valence = current_state[0]
    arousal = current_state[1] if len(current_state) > 1 else 0
    
    if valence < -0.5 and arousal < -0.3:
        suggestions.append("Low energy negative state detected.")
        suggestions.append("Recommendation: Activation-focused intervention (exercise, social engagement)")
    elif valence < -0.5 and arousal > 0.3:
        suggestions.append("High arousal negative state detected.")
        suggestions.append("Recommendation: Calming intervention (deep breathing, meditation)")
    elif valence > 0.5:
        suggestions.append("Positive state detected.")
        suggestions.append("Recommendation: Maintenance strategies (gratitude, savoring)")
    
    # Consider history if available
    if emotional_history is not None and len(emotional_history) > 5:
        volatility = np.std(emotional_history[:, 0])
        
        if volatility > 0.5:
            suggestions.append("High emotional volatility detected.")
            suggestions.append("Consider sustained intervention over impulse.")
    
    # Consider personality
    if personality_traits is not None:
        if personality_traits.get('neuroticism', 0) > 0.7:
            suggestions.append("High neuroticism: May benefit from cognitive reappraisal strategies.")
    
    return "\n".join(suggestions)
