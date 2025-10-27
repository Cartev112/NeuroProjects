"""
Emotional state space model with multidimensional phase space dynamics.

Implements the core framework for modeling emotions as trajectories through
a high-dimensional space defined by valence, arousal, dominance, approach/avoidance,
and temporal dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import seaborn as sns

from .attractors import Attractor, AttractorLandscape


@dataclass
class StateSpaceDimensions:
    """Defines the dimensions of the emotional state space."""
    VALENCE = 0  # Pleasure-displeasure (-1 to 1)
    AROUSAL = 1  # Activation-deactivation (-1 to 1)
    DOMINANCE = 2  # Control-submission (-1 to 1)
    APPROACH = 3  # Approach-avoidance motivation (-1 to 1)
    TEMPORAL = 4  # Temporal inertia/momentum (-1 to 1)


class EmotionalStateSpace:
    """
    High-dimensional emotional state space with dynamical systems modeling.
    
    Models emotions as continuous trajectories through a multidimensional phase
    space, with attractors representing stable emotional states and vector fields
    governing transitions.
    """
    
    def __init__(
        self,
        dimensions: int = 5,
        bounds: Optional[Tuple[float, float]] = None,
        viscosity: float = 0.1,
        noise_level: float = 0.05,
        dt: float = 0.01
    ):
        """
        Initialize emotional state space.
        
        Args:
            dimensions: Number of dimensions (default: 5 for valence, arousal, 
                       dominance, approach, temporal)
            bounds: (min, max) bounds for each dimension
            viscosity: Damping/friction in state space
            noise_level: Stochastic noise magnitude
            dt: Default time step for integration
        """
        self.dimensions = dimensions
        self.bounds = bounds if bounds is not None else (-1.0, 1.0)
        self.viscosity = viscosity
        self.noise_level = noise_level
        self.dt = dt
        
        # Attractor landscape
        self.landscape = AttractorLandscape()
        
        # History tracking
        self.trajectory_history = []
        self.time_history = []
        
        # Custom dynamics function
        self.custom_dynamics: Optional[Callable] = None
    
    def add_attractor(self, attractor: Attractor):
        """Add an attractor to the state space."""
        if attractor.dimensions != self.dimensions:
            raise ValueError(
                f"Attractor dimension {attractor.dimensions} does not match "
                f"state space dimension {self.dimensions}"
            )
        self.landscape.add_attractor(attractor)
    
    def add_attractors(self, attractors: List[Attractor]):
        """Add multiple attractors."""
        for attractor in attractors:
            self.add_attractor(attractor)
    
    def set_custom_dynamics(self, dynamics_fn: Callable):
        """
        Set custom dynamics function.
        
        Args:
            dynamics_fn: Function with signature (state, time) -> velocity
        """
        self.custom_dynamics = dynamics_fn
    
    def compute_velocity(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Compute velocity/derivative at a given state.
        
        Combines attractor forces, viscosity, and noise.
        
        Args:
            state: Current position in state space
            time: Current time
            
        Returns:
            Velocity vector (derivative of state)
        """
        state = np.asarray(state, dtype=np.float64)
        
        # Get force from attractors
        if len(self.landscape) > 0:
            attractor_force = self.landscape.compute_total_force(state, time)
        else:
            attractor_force = np.zeros_like(state)
        
        # Add custom dynamics if provided
        if self.custom_dynamics is not None:
            custom_force = self.custom_dynamics(state, time)
            attractor_force += custom_force
        
        # Apply viscosity (damping)
        velocity = attractor_force - self.viscosity * state
        
        # Add stochastic noise
        if self.noise_level > 0:
            velocity += np.random.normal(0, self.noise_level, size=velocity.shape)
        
        return velocity
    
    def step(self, state: np.ndarray, time: float = 0.0, dt: Optional[float] = None) -> np.ndarray:
        """
        Perform one integration step using Runge-Kutta 4th order.
        
        Args:
            state: Current state
            time: Current time
            dt: Time step (uses self.dt if None)
            
        Returns:
            New state after time step
        """
        if dt is None:
            dt = self.dt
        
        # RK4 integration
        k1 = self.compute_velocity(state, time)
        k2 = self.compute_velocity(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = self.compute_velocity(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = self.compute_velocity(state + dt * k3, time + dt)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply bounds
        new_state = np.clip(new_state, self.bounds[0], self.bounds[1])
        
        return new_state
    
    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        duration: float = 100.0,
        dt: Optional[float] = None,
        save_history: bool = True
    ) -> np.ndarray:
        """
        Simulate emotional trajectory from initial state.
        
        Args:
            initial_state: Starting position in state space
            duration: Total simulation time
            dt: Time step (uses self.dt if None)
            save_history: Whether to save trajectory in history
            
        Returns:
            Array of shape (n_steps, dimensions) containing trajectory
        """
        if dt is None:
            dt = self.dt
        
        n_steps = int(duration / dt)
        trajectory = np.zeros((n_steps, self.dimensions))
        times = np.zeros(n_steps)
        
        state = np.array(initial_state, dtype=np.float64)
        trajectory[0] = state
        times[0] = 0.0
        
        for i in range(1, n_steps):
            time = i * dt
            state = self.step(state, time, dt)
            trajectory[i] = state
            times[i] = time
        
        if save_history:
            self.trajectory_history.append(trajectory)
            self.time_history.append(times)
        
        return trajectory
    
    def simulate_multiple_trajectories(
        self,
        initial_states: np.ndarray,
        duration: float = 100.0,
        dt: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Simulate multiple trajectories from different initial conditions.
        
        Args:
            initial_states: Array of shape (n_trajectories, dimensions)
            duration: Total simulation time
            dt: Time step
            
        Returns:
            List of trajectory arrays
        """
        trajectories = []
        for initial_state in initial_states:
            traj = self.simulate_trajectory(
                initial_state, duration, dt, save_history=False
            )
            trajectories.append(traj)
        
        return trajectories
    
    def compute_vector_field(
        self,
        grid_resolution: int = 20,
        dimensions: Tuple[int, int] = (0, 1),
        time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vector field for visualization in 2D slice.
        
        Args:
            grid_resolution: Number of points per dimension
            dimensions: Which two dimensions to visualize
            time: Time point for vector field
            
        Returns:
            Tuple of (X, Y, U, V) for quiver plot
        """
        dim1, dim2 = dimensions
        
        # Create grid
        x = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Compute velocity at each grid point
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                state = np.zeros(self.dimensions)
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                velocity = self.compute_velocity(state, time)
                U[i, j] = velocity[dim1]
                V[i, j] = velocity[dim2]
        
        return X, Y, U, V
    
    def plot_phase_space(
        self,
        trajectory: Optional[np.ndarray] = None,
        dimensions: Tuple[int, int] = (0, 1),
        show_vector_field: bool = True,
        show_attractors: bool = True,
        show_basins: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize phase space with trajectory, vector field, and attractors.
        
        Args:
            trajectory: Trajectory to plot (uses last from history if None)
            dimensions: Which two dimensions to visualize
            show_vector_field: Whether to show velocity vector field
            show_attractors: Whether to mark attractor positions
            show_basins: Whether to show basin boundaries
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        dim1, dim2 = dimensions
        dim_names = ['Valence', 'Arousal', 'Dominance', 'Approach', 'Temporal']
        
        # Plot basin boundaries
        if show_basins and len(self.landscape) > 0:
            X, Y, basins = self.landscape.compute_basin_boundaries(
                grid_resolution=100, dimensions=dimensions
            )
            ax.contourf(X, Y, basins, alpha=0.2, cmap='tab10')
        
        # Plot vector field
        if show_vector_field:
            X, Y, U, V = self.compute_vector_field(
                grid_resolution=15, dimensions=dimensions
            )
            ax.quiver(X, Y, U, V, alpha=0.5, scale=10, width=0.003)
        
        # Plot attractors
        if show_attractors and len(self.landscape) > 0:
            for attractor in self.landscape.attractors:
                ax.plot(
                    attractor.center[dim1],
                    attractor.center[dim2],
                    'r*',
                    markersize=20,
                    label=attractor.name
                )
                # Draw basin circle
                circle = plt.Circle(
                    (attractor.center[dim1], attractor.center[dim2]),
                    attractor.strength,
                    fill=False,
                    linestyle='--',
                    alpha=0.5
                )
                ax.add_patch(circle)
        
        # Plot trajectory
        if trajectory is not None:
            ax.plot(
                trajectory[:, dim1],
                trajectory[:, dim2],
                'b-',
                alpha=0.7,
                linewidth=2,
                label='Trajectory'
            )
            # Mark start and end
            ax.plot(trajectory[0, dim1], trajectory[0, dim2], 'go', 
                   markersize=10, label='Start')
            ax.plot(trajectory[-1, dim1], trajectory[-1, dim2], 'ro', 
                   markersize=10, label='End')
        elif len(self.trajectory_history) > 0:
            traj = self.trajectory_history[-1]
            ax.plot(traj[:, dim1], traj[:, dim2], 'b-', alpha=0.7, linewidth=2)
            ax.plot(traj[0, dim1], traj[0, dim2], 'go', markersize=10)
            ax.plot(traj[-1, dim1], traj[-1, dim2], 'ro', markersize=10)
        
        ax.set_xlabel(dim_names[dim1] if dim1 < len(dim_names) else f'Dim {dim1}', 
                     fontsize=12)
        ax.set_ylabel(dim_names[dim2] if dim2 < len(dim_names) else f'Dim {dim2}', 
                     fontsize=12)
        ax.set_title('Emotional State Space Phase Portrait', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(self.bounds[0] - 0.1, self.bounds[1] + 0.1)
        ax.set_ylim(self.bounds[0] - 0.1, self.bounds[1] + 0.1)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_time_series(
        self,
        trajectory: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot emotional dimensions over time.
        
        Args:
            trajectory: Trajectory to plot
            times: Time points
            figsize: Figure size
        """
        if trajectory is None and len(self.trajectory_history) > 0:
            trajectory = self.trajectory_history[-1]
            times = self.time_history[-1]
        elif trajectory is None:
            raise ValueError("No trajectory to plot")
        
        if times is None:
            times = np.arange(len(trajectory)) * self.dt
        
        dim_names = ['Valence', 'Arousal', 'Dominance', 'Approach', 'Temporal']
        
        fig, axes = plt.subplots(self.dimensions, 1, figsize=figsize, sharex=True)
        
        if self.dimensions == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(times, trajectory[:, i], linewidth=2)
            ax.set_ylabel(dim_names[i] if i < len(dim_names) else f'Dim {i}', 
                         fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Mark attractor positions
            if len(self.landscape) > 0:
                for attractor in self.landscape.attractors:
                    ax.axhline(
                        y=attractor.center[i],
                        color='r',
                        linestyle=':',
                        alpha=0.5,
                        label=attractor.name if i == 0 else None
                    )
        
        axes[-1].set_xlabel('Time', fontsize=12)
        axes[0].set_title('Emotional Dynamics Over Time', fontsize=14, fontweight='bold')
        
        if len(self.landscape) > 0:
            axes[0].legend(loc='upper right')
        
        plt.tight_layout()
        return fig, axes
    
    def animate_trajectory(
        self,
        trajectory: np.ndarray,
        dimensions: Tuple[int, int] = (0, 1),
        interval: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Create animation of trajectory evolution.
        
        Args:
            trajectory: Trajectory to animate
            dimensions: Which dimensions to show
            interval: Milliseconds between frames
            save_path: Path to save animation (if provided)
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        dim1, dim2 = dimensions
        dim_names = ['Valence', 'Arousal', 'Dominance', 'Approach', 'Temporal']
        
        # Setup plot
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=10)
        
        # Plot attractors
        if len(self.landscape) > 0:
            for attractor in self.landscape.attractors:
                ax.plot(attractor.center[dim1], attractor.center[dim2], 
                       'r*', markersize=20)
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.set_xlabel(dim_names[dim1] if dim1 < len(dim_names) else f'Dim {dim1}')
        ax.set_ylabel(dim_names[dim2] if dim2 < len(dim_names) else f'Dim {dim2}')
        ax.set_title('Emotional Trajectory Animation')
        ax.grid(True, alpha=0.3)
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def animate(frame):
            line.set_data(trajectory[:frame, dim1], trajectory[:frame, dim2])
            point.set_data([trajectory[frame, dim1]], [trajectory[frame, dim2]])
            return line, point
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(trajectory), interval=interval, blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        return anim
    
    def reset_history(self):
        """Clear trajectory history."""
        self.trajectory_history = []
        self.time_history = []
