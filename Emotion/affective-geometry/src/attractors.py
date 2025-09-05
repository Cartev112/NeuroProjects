"""
Attractor dynamics for emotional state space modeling.

Implements different types of attractors (point, limit cycle, strange)
that represent stable emotional states and patterns.
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass


class AttractorType(Enum):
    """Types of attractors in emotional state space."""
    POINT = "point"  # Stable emotional state
    LIMIT_CYCLE = "limit_cycle"  # Oscillating emotional pattern
    STRANGE = "strange"  # Chaotic emotional dynamics
    SADDLE = "saddle"  # Unstable transition point


@dataclass
class Attractor:
    """
    Represents an attractor in emotional state space.
    
    Attributes:
        center: Central position in state space (valence, arousal, dominance, ...)
        strength: Strength of attraction (basin size)
        name: Human-readable name (e.g., "happiness", "sadness")
        attractor_type: Type of attractor
        noise_level: Stochastic fluctuations around attractor
        damping: Damping coefficient for approach dynamics
        frequency: For limit cycles, oscillation frequency
        phase: For limit cycles, initial phase
    """
    center: np.ndarray
    strength: float = 1.0
    name: str = "unnamed"
    attractor_type: AttractorType = AttractorType.POINT
    noise_level: float = 0.01
    damping: float = 0.5
    frequency: Optional[float] = None
    phase: float = 0.0
    
    def __post_init__(self):
        """Validate and convert center to numpy array."""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float64)
        
        if self.attractor_type == AttractorType.LIMIT_CYCLE and self.frequency is None:
            self.frequency = 0.1  # Default frequency
    
    @property
    def dimensions(self) -> int:
        """Number of dimensions in state space."""
        return len(self.center)
    
    def compute_force(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Compute the force/velocity field at a given state.
        
        Args:
            state: Current position in state space
            time: Current time (for time-dependent dynamics)
            
        Returns:
            Force vector pointing toward attractor
        """
        state = np.asarray(state, dtype=np.float64)
        
        if self.attractor_type == AttractorType.POINT:
            # Simple point attractor with damping
            displacement = self.center - state
            force = self.strength * displacement - self.damping * displacement
            
        elif self.attractor_type == AttractorType.LIMIT_CYCLE:
            # Limit cycle attractor (e.g., emotional oscillations)
            # Use Stuart-Landau oscillator dynamics
            displacement = state - self.center
            r = np.linalg.norm(displacement)
            
            if r > 1e-10:
                # Radial component: attract to radius = strength
                radial_force = self.strength * (1.0 - r) * displacement / r
                
                # Angular component: rotation
                if self.dimensions >= 2:
                    angular_force = np.zeros_like(displacement)
                    angular_force[0] = -self.frequency * displacement[1]
                    angular_force[1] = self.frequency * displacement[0]
                    force = radial_force + angular_force
                else:
                    force = radial_force
            else:
                force = np.zeros_like(state)
                
        elif self.attractor_type == AttractorType.SADDLE:
            # Saddle point: attractive in some directions, repulsive in others
            displacement = self.center - state
            # Make first dimension repulsive, others attractive
            force = displacement.copy()
            force[0] *= -self.strength  # Repulsive
            force[1:] *= self.strength  # Attractive
            
        elif self.attractor_type == AttractorType.STRANGE:
            # Strange attractor (simplified Lorenz-like dynamics)
            displacement = state - self.center
            force = self._lorenz_like_force(displacement, time)
            
        else:
            force = np.zeros_like(state)
        
        # Add noise
        if self.noise_level > 0:
            force += np.random.normal(0, self.noise_level, size=force.shape)
        
        return force
    
    def _lorenz_like_force(self, displacement: np.ndarray, time: float) -> np.ndarray:
        """
        Simplified Lorenz-like dynamics for strange attractors.
        
        Creates chaotic emotional dynamics with sensitive dependence
        on initial conditions.
        """
        if len(displacement) < 3:
            # Need at least 3D for chaos
            return self.strength * displacement
        
        x, y, z = displacement[0], displacement[1], displacement[2]
        
        # Lorenz parameters (scaled for emotional dynamics)
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        force = np.zeros_like(displacement)
        force[0] = self.strength * dx
        force[1] = self.strength * dy
        force[2] = self.strength * dz
        
        return force
    
    def distance_to(self, state: np.ndarray) -> float:
        """
        Compute distance from state to attractor center.
        
        Args:
            state: Current position in state space
            
        Returns:
            Euclidean distance to attractor center
        """
        state = np.asarray(state, dtype=np.float64)
        return np.linalg.norm(state - self.center)
    
    def is_in_basin(self, state: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Check if state is within the basin of attraction.
        
        Args:
            state: Current position in state space
            threshold: Distance threshold for basin membership
            
        Returns:
            True if state is in basin of attraction
        """
        distance = self.distance_to(state)
        return distance < (self.strength * threshold)
    
    def compute_lyapunov_exponent(
        self, 
        initial_state: np.ndarray, 
        duration: float = 100.0,
        dt: float = 0.01
    ) -> float:
        """
        Estimate largest Lyapunov exponent for this attractor.
        
        Args:
            initial_state: Starting state
            duration: Simulation duration
            dt: Time step
            
        Returns:
            Largest Lyapunov exponent (positive = chaos)
        """
        # Simulate trajectory
        state = np.array(initial_state, dtype=np.float64)
        perturbation = np.random.normal(0, 1e-8, size=state.shape)
        perturbed_state = state + perturbation
        
        lyapunov_sum = 0.0
        n_steps = int(duration / dt)
        
        for _ in range(n_steps):
            # Evolve both trajectories
            force = self.compute_force(state)
            perturbed_force = self.compute_force(perturbed_state)
            
            state += force * dt
            perturbed_state += perturbed_force * dt
            
            # Compute separation
            separation = np.linalg.norm(perturbed_state - state)
            
            if separation > 1e-10:
                # Accumulate log of separation growth
                lyapunov_sum += np.log(separation / np.linalg.norm(perturbation))
                
                # Renormalize perturbation
                perturbation = (perturbed_state - state) / separation * np.linalg.norm(perturbation)
                perturbed_state = state + perturbation
        
        return lyapunov_sum / duration
    
    def __repr__(self) -> str:
        return (f"Attractor(name='{self.name}', type={self.attractor_type.value}, "
                f"center={self.center}, strength={self.strength})")


class AttractorLandscape:
    """
    Manages multiple attractors and their interactions.
    
    Computes combined force fields and basin boundaries.
    """
    
    def __init__(self):
        self.attractors = []
    
    def add_attractor(self, attractor: Attractor):
        """Add an attractor to the landscape."""
        self.attractors.append(attractor)
    
    def add_attractors(self, attractors: list):
        """Add multiple attractors."""
        self.attractors.extend(attractors)
    
    def compute_total_force(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Compute combined force from all attractors.
        
        Uses weighted superposition based on distance.
        
        Args:
            state: Current position in state space
            time: Current time
            
        Returns:
            Total force vector
        """
        if not self.attractors:
            return np.zeros_like(state)
        
        total_force = np.zeros_like(state, dtype=np.float64)
        total_weight = 0.0
        
        for attractor in self.attractors:
            # Weight by inverse distance (closer attractors have more influence)
            distance = attractor.distance_to(state)
            weight = 1.0 / (1.0 + distance)
            
            force = attractor.compute_force(state, time)
            total_force += weight * force
            total_weight += weight
        
        if total_weight > 0:
            total_force /= total_weight
        
        return total_force
    
    def find_nearest_attractor(self, state: np.ndarray) -> Tuple[Attractor, float]:
        """
        Find the nearest attractor to a given state.
        
        Args:
            state: Current position in state space
            
        Returns:
            Tuple of (nearest attractor, distance)
        """
        if not self.attractors:
            raise ValueError("No attractors in landscape")
        
        distances = [attr.distance_to(state) for attr in self.attractors]
        min_idx = np.argmin(distances)
        
        return self.attractors[min_idx], distances[min_idx]
    
    def compute_basin_boundaries(
        self, 
        grid_resolution: int = 50,
        dimensions: Tuple[int, int] = (0, 1)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute basin of attraction boundaries in 2D slice.
        
        Args:
            grid_resolution: Number of points per dimension
            dimensions: Which two dimensions to visualize
            
        Returns:
            Tuple of (x_grid, y_grid, basin_labels)
        """
        if not self.attractors:
            raise ValueError("No attractors in landscape")
        
        # Get bounds from attractor positions
        centers = np.array([attr.center for attr in self.attractors])
        dim1, dim2 = dimensions
        
        x_min, x_max = centers[:, dim1].min() - 2, centers[:, dim1].max() + 2
        y_min, y_max = centers[:, dim2].min() - 2, centers[:, dim2].max() + 2
        
        x = np.linspace(x_min, x_max, grid_resolution)
        y = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create full-dimensional state for each grid point
        basin_labels = np.zeros_like(X, dtype=int)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Create state vector
                state = np.zeros(self.attractors[0].dimensions)
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Find nearest attractor
                nearest, _ = self.find_nearest_attractor(state)
                basin_labels[i, j] = self.attractors.index(nearest)
        
        return X, Y, basin_labels
    
    def __len__(self) -> int:
        return len(self.attractors)
    
    def __getitem__(self, idx: int) -> Attractor:
        return self.attractors[idx]
