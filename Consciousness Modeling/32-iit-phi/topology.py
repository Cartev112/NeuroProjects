"""
Topological Data Analysis (TDA) for neural phase spaces.
Persistent homology to identify high-dimensional structures.
"""
import numpy as np


def compute_distance_matrix(X):
    """Compute pairwise Euclidean distance matrix.
    
    X: (N, D) point cloud
    
    Returns: (N, N) distance matrix
    """
    X = np.asarray(X)
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def vietoris_rips_filtration(D, max_epsilon=None, n_steps=50):
    """Compute Vietoris-Rips filtration (simplified).
    
    D: (N, N) distance matrix
    max_epsilon: maximum distance threshold
    n_steps: number of filtration steps
    
    Returns: list of (epsilon, simplicial_complex) tuples
    """
    D = np.asarray(D)
    N = D.shape[0]
    
    if max_epsilon is None:
        max_epsilon = np.max(D)
    
    epsilons = np.linspace(0, max_epsilon, n_steps)
    filtration = []
    
    for eps in epsilons:
        # 0-simplices (vertices)
        vertices = set(range(N))
        
        # 1-simplices (edges)
        edges = set()
        for i in range(N):
            for j in range(i + 1, N):
                if D[i, j] <= eps:
                    edges.add((i, j))
        
        # 2-simplices (triangles) - simplified
        triangles = set()
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    if D[i, j] <= eps and D[j, k] <= eps and D[i, k] <= eps:
                        triangles.add((i, j, k))
        
        complex_data = {
            'vertices': vertices,
            'edges': edges,
            'triangles': triangles,
        }
        filtration.append((eps, complex_data))
    
    return filtration


def compute_betti_numbers(complex_data):
    """Compute Betti numbers (simplified).
    
    complex_data: dict with vertices, edges, triangles
    
    Returns: dict with b0, b1, b2
    """
    vertices = complex_data['vertices']
    edges = complex_data['edges']
    triangles = complex_data['triangles']
    
    # b0: number of connected components (simplified)
    # For simplicity, assume all vertices in same component if any edge exists
    b0 = 1 if len(edges) > 0 else len(vertices)
    
    # b1: number of 1-cycles (loops)
    # Euler characteristic: V - E + F = 2 - 2g (for genus g)
    # b1 = E - V + 1 (simplified for connected graph)
    V = len(vertices)
    E = len(edges)
    F = len(triangles)
    b1 = max(0, E - V + 1) if E > 0 else 0
    
    # b2: number of 2-cycles (voids)
    b2 = max(0, F - E + V - 1) if F > 0 else 0
    
    return {'b0': b0, 'b1': b1, 'b2': b2}


def persistent_homology(X, max_epsilon=None, n_steps=50):
    """Compute persistent homology of point cloud.
    
    X: (N, D) point cloud (e.g., neural states in phase space)
    
    Returns: dict with persistence diagrams
    """
    D = compute_distance_matrix(X)
    filtration = vietoris_rips_filtration(D, max_epsilon, n_steps)
    
    # Track birth/death of features
    b0_persistence = []
    b1_persistence = []
    b2_persistence = []
    
    prev_betti = {'b0': 0, 'b1': 0, 'b2': 0}
    
    for eps, complex_data in filtration:
        betti = compute_betti_numbers(complex_data)
        
        # Track changes
        if betti['b0'] > prev_betti['b0']:
            b0_persistence.append((eps, None))  # Birth
        if betti['b1'] > prev_betti['b1']:
            b1_persistence.append((eps, None))
        if betti['b2'] > prev_betti['b2']:
            b2_persistence.append((eps, None))
        
        prev_betti = betti
    
    return {
        'b0_persistence': b0_persistence,
        'b1_persistence': b1_persistence,
        'b2_persistence': b2_persistence,
        'filtration': filtration,
    }


def topological_complexity(X):
    """Compute topological complexity as sum of persistent Betti numbers.
    
    X: (N, D) point cloud
    
    Returns: scalar complexity measure
    """
    ph = persistent_homology(X, n_steps=20)
    
    # Sum of persistent features
    complexity = (
        len(ph['b0_persistence']) +
        len(ph['b1_persistence']) +
        len(ph['b2_persistence'])
    )
    
    return float(complexity)
