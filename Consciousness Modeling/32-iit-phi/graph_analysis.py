"""
Graph-theoretic analysis of neural connectivity patterns.
"""
import numpy as np


def compute_graph_metrics(A):
    """Compute basic graph metrics from adjacency matrix.
    
    A: (N, N) adjacency/connectivity matrix
    
    Returns: dict with metrics
    """
    A = np.asarray(A)
    N = A.shape[0]
    
    # Degree (in/out for directed)
    out_degree = np.sum(A, axis=1)
    in_degree = np.sum(A, axis=0)
    
    # Clustering coefficient (simplified)
    clustering = np.zeros(N)
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
            continue
        # Count triangles
        triangles = 0
        for j in neighbors:
            for m in neighbors:
                if j != m and A[j, m] > 0:
                    triangles += 1
        clustering[i] = triangles / (k * (k - 1)) if k > 1 else 0.0
    
    # Global efficiency (inverse of average shortest path)
    # Simplified: use direct connectivity as proxy
    efficiency = np.mean(A[A > 0]) if np.sum(A > 0) > 0 else 0.0
    
    # Modularity (simplified via spectral partitioning)
    # Compute Laplacian
    D = np.diag(out_degree)
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)
    # Second smallest eigenvalue (Fiedler value) indicates modularity
    fiedler = eigvals[1] if len(eigvals) > 1 else 0.0
    
    return {
        'out_degree': out_degree,
        'in_degree': in_degree,
        'clustering': clustering,
        'mean_clustering': float(np.mean(clustering)),
        'efficiency': float(efficiency),
        'modularity_proxy': float(fiedler),
    }


def identify_hubs(A, threshold=0.75):
    """Identify hub nodes based on degree.
    
    A: (N, N) adjacency
    threshold: percentile for hub classification
    
    Returns: hub indices
    """
    A = np.asarray(A)
    degree = np.sum(A, axis=1) + np.sum(A, axis=0)
    thresh_val = np.percentile(degree, threshold * 100)
    hubs = np.where(degree >= thresh_val)[0]
    return hubs


def compute_rich_club(A, k_range=None):
    """Compute rich-club coefficient.
    
    A: (N, N) adjacency
    k_range: range of degrees to test
    
    Returns: dict with k -> rich-club coefficient
    """
    A = np.asarray(A)
    N = A.shape[0]
    degree = np.sum(A > 0, axis=1)
    
    if k_range is None:
        k_range = range(1, int(np.max(degree)) + 1)
    
    rc = {}
    for k in k_range:
        # Nodes with degree > k
        high_deg = np.where(degree > k)[0]
        if len(high_deg) < 2:
            rc[k] = 0.0
            continue
        
        # Edges among high-degree nodes
        sub_A = A[np.ix_(high_deg, high_deg)]
        E_high = np.sum(sub_A > 0)
        
        # Possible edges
        N_high = len(high_deg)
        E_possible = N_high * (N_high - 1)
        
        rc[k] = E_high / E_possible if E_possible > 0 else 0.0
    
    return rc
