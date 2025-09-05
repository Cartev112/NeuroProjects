"""
Basic simulation example: Emotional trajectory through state space.

Demonstrates how to create an emotional state space with attractors
and simulate trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state_space import EmotionalStateSpace
from src.attractors import Attractor, AttractorType
from src.utils import create_standard_attractors


def main():
    print("="*60)
    print("Basic Emotional Trajectory Simulation")
    print("="*60)
    
    # Create 5D emotional state space
    print("\n1. Creating emotional state space...")
    state_space = EmotionalStateSpace(
        dimensions=5,
        viscosity=0.1,
        noise_level=0.05
    )
    
    # Add standard emotional attractors
    print("2. Adding emotional attractors...")
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    for attr in attractors:
        print(f"   - {attr.name}: {attr.center[:3]}")
    
    # Simulate trajectory from neutral state
    print("\n3. Simulating trajectory from neutral state...")
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    trajectory = state_space.simulate_trajectory(
        initial_state,
        duration=100.0,
        dt=0.01
    )
    
    print(f"   Simulated {len(trajectory)} time steps")
    
    # Determine final attractor
    final_state = trajectory[-1]
    nearest_attractor, distance = state_space.landscape.find_nearest_attractor(final_state)
    print(f"   Final state converged to: {nearest_attractor.name}")
    print(f"   Distance to attractor: {distance:.4f}")
    
    # Visualize
    print("\n4. Generating visualizations...")
    
    # Phase space plot
    fig1, ax1 = state_space.plot_phase_space(
        trajectory,
        dimensions=(0, 1),  # Valence vs Arousal
        show_vector_field=True,
        show_attractors=True,
        show_basins=True
    )
    
    # Time series plot
    fig2, ax2 = state_space.plot_time_series(trajectory)
    
    # Multiple trajectories from different initial conditions
    print("\n5. Simulating multiple trajectories...")
    n_trajectories = 10
    initial_states = np.random.uniform(-0.5, 0.5, size=(n_trajectories, 5))
    
    trajectories = state_space.simulate_multiple_trajectories(
        initial_states,
        duration=50.0
    )
    
    # Plot all trajectories
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    # Plot vector field
    X, Y, U, V = state_space.compute_vector_field(dimensions=(0, 1))
    ax3.quiver(X, Y, U, V, alpha=0.3, scale=10, width=0.003)
    
    # Plot attractors
    for attr in attractors:
        ax3.plot(attr.center[0], attr.center[1], 'r*', markersize=20)
        ax3.text(attr.center[0], attr.center[1] + 0.15, attr.name, 
                ha='center', fontsize=9)
    
    # Plot all trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    for i, traj in enumerate(trajectories):
        ax3.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1.5, color=colors[i])
        ax3.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=8)
    
    ax3.set_xlabel('Valence', fontsize=12)
    ax3.set_ylabel('Arousal', fontsize=12)
    ax3.set_title('Multiple Emotional Trajectories', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    
    print("\n" + "="*60)
    print("Simulation complete! Close plots to exit.")
    print("="*60)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
