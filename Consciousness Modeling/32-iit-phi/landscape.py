"""
Consciousness landscape: basin of attraction mapping and perturbation analysis.
"""
import numpy as np


def simulate_dynamics(A, x0, T=100, noise=0.01):
    """Simulate network dynamics from initial state.
    
    A: (N, N) connectivity matrix
    x0: (N,) initial state (continuous or binary)
    T: number of time steps
    noise: Gaussian noise level
    
    Returns: (T, N) trajectory
    """
    A = np.asarray(A)
    x0 = np.asarray(x0)
    N = len(x0)
    
    trajectory = np.zeros((T, N))
    trajectory[0] = x0
    
    for t in range(1, T):
        # Linear dynamics with sigmoid nonlinearity
        activation = A @ trajectory[t - 1] + noise * np.random.randn(N)
        trajectory[t] = 1.0 / (1.0 + np.exp(-activation))
    
    return trajectory


def find_attractors(A, n_inits=100, T=200, threshold=0.01):
    """Find attractors by simulating from random initial conditions.
    
    A: (N, N) connectivity
    n_inits: number of random initializations
    T: simulation length
    threshold: distance threshold for attractor identification
    
    Returns: list of attractor states
    """
    A = np.asarray(A)
    N = A.shape[0]
    
    attractors = []
    
    for _ in range(n_inits):
        x0 = np.random.rand(N)
        traj = simulate_dynamics(A, x0, T=T, noise=0.001)
        
        # Final state as attractor candidate
        final_state = traj[-1]
        
        # Check if this is a new attractor
        is_new = True
        for att in attractors:
            if np.linalg.norm(final_state - att) < threshold:
                is_new = False
                break
        
        if is_new:
            attractors.append(final_state)
    
    return attractors


def basin_of_attraction(A, attractor, n_samples=1000, T=100):
    """Estimate basin of attraction for a given attractor.
    
    A: (N, N) connectivity
    attractor: (N,) attractor state
    n_samples: number of random initial conditions to test
    T: simulation length
    
    Returns: fraction of initial conditions that converge to attractor
    """
    A = np.asarray(A)
    attractor = np.asarray(attractor)
    N = len(attractor)
    
    converged = 0
    
    for _ in range(n_samples):
        x0 = np.random.rand(N)
        traj = simulate_dynamics(A, x0, T=T, noise=0.001)
        
        # Check if final state is near attractor
        if np.linalg.norm(traj[-1] - attractor) < 0.1:
            converged += 1
    
    return converged / n_samples


def perturbation_analysis(A, x0, perturbation_nodes, perturbation_strength=0.5, T=100):
    """Analyze effect of perturbations on dynamics.
    
    A: (N, N) connectivity
    x0: (N,) initial state
    perturbation_nodes: list of node indices to perturb
    perturbation_strength: magnitude of perturbation
    T: simulation length
    
    Returns: dict with baseline and perturbed trajectories
    """
    A = np.asarray(A)
    x0 = np.asarray(x0)
    
    # Baseline
    traj_baseline = simulate_dynamics(A, x0, T=T, noise=0.01)
    
    # Perturbed
    x0_pert = x0.copy()
    x0_pert[perturbation_nodes] += perturbation_strength
    x0_pert = np.clip(x0_pert, 0, 1)
    traj_perturbed = simulate_dynamics(A, x0_pert, T=T, noise=0.01)
    
    # Compute divergence over time
    divergence = np.linalg.norm(traj_perturbed - traj_baseline, axis=1)
    
    return {
        'baseline': traj_baseline,
        'perturbed': traj_perturbed,
        'divergence': divergence,
    }


def consciousness_landscape_grid(A, phi_func, grid_resolution=20):
    """Create 2D projection of consciousness landscape with Φ values.
    
    A: (N, N) connectivity
    phi_func: function that computes Φ from state
    grid_resolution: number of points per dimension
    
    Returns: dict with grid coordinates and Φ values
    """
    A = np.asarray(A)
    N = A.shape[0]
    
    # Use PCA to project to 2D for visualization
    # Sample random states
    n_samples = 1000
    states = np.random.rand(n_samples, N)
    
    # Compute mean and project
    mean_state = np.mean(states, axis=0)
    centered = states - mean_state
    
    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]
    
    # Create grid in PC space
    grid_range = 3.0
    x_vals = np.linspace(-grid_range, grid_range, grid_resolution)
    y_vals = np.linspace(-grid_range, grid_range, grid_resolution)
    
    phi_grid = np.zeros((grid_resolution, grid_resolution))
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            # Reconstruct state from PC coordinates
            state = mean_state + x * pc1 + y * pc2
            state = np.clip(state, 0, 1)
            
            # Compute Φ
            phi = phi_func(state)
            phi_grid[i, j] = phi
    
    return {
        'x_vals': x_vals,
        'y_vals': y_vals,
        'phi_grid': phi_grid,
        'pc1': pc1,
        'pc2': pc2,
        'mean_state': mean_state,
    }
