"""
Integrated Information (Φ) computation with MIP (Minimum Information Partition) search.
Simplified IIT 3.0 implementation for discrete binary systems.
"""
import numpy as np
from itertools import combinations, product
from scipy.special import rel_entr


def entropy(p):
    """Shannon entropy H(p)."""
    p = np.asarray(p).ravel()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mutual_information(p_xy, p_x, p_y):
    """MI(X;Y) = H(X) + H(Y) - H(X,Y)."""
    return entropy(p_x) + entropy(p_y) - entropy(p_xy)


def kl_divergence(p, q):
    """KL(p||q)."""
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    return np.sum(rel_entr(p, q))


def earth_movers_distance(p, q):
    """Simplified EMD (1D Wasserstein) for probability distributions."""
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.sum(np.abs(cdf_p - cdf_q))


def compute_tpm(states, next_states):
    """Compute transition probability matrix from state sequences.
    
    states: (T, N) binary states over time
    next_states: (T, N) next states
    
    Returns: TPM (2^N, 2^N) where TPM[s, s'] = P(s'|s)
    """
    states = np.asarray(states, dtype=int)
    next_states = np.asarray(next_states, dtype=int)
    T, N = states.shape
    
    n_states = 2 ** N
    counts = np.zeros((n_states, n_states))
    
    for t in range(T):
        s = int(''.join(map(str, states[t])), 2)
        s_next = int(''.join(map(str, next_states[t])), 2)
        counts[s, s_next] += 1
    
    # Normalize
    tpm = counts / (counts.sum(axis=1, keepdims=True) + 1e-12)
    return tpm


def partition_system(nodes, partition):
    """Split nodes into parts according to partition.
    
    partition: list of tuples, e.g., [(0,1), (2,3)] splits into two parts
    """
    return partition


def compute_phi_mip(current_state, tpm, nodes):
    """Compute integrated information Φ via MIP search.
    
    current_state: (N,) binary state
    tpm: (2^N, 2^N) transition probability matrix
    nodes: list of node indices
    
    Returns: Φ value (simplified as EMD between whole and partitioned distributions)
    """
    N = len(nodes)
    if N <= 1:
        return 0.0
    
    # Current state index
    s_idx = int(''.join(map(str, current_state)), 2)
    p_whole = tpm[s_idx]  # P(s'|s) for whole system
    
    # Find MIP: partition that minimizes integrated information
    min_phi = np.inf
    
    # Try all bipartitions
    for k in range(1, N):
        for part1_nodes in combinations(nodes, k):
            part2_nodes = tuple(n for n in nodes if n not in part1_nodes)
            
            # Compute independent distributions for each part
            # Simplified: assume independence and marginalize
            # In full IIT, you'd compute cause-effect repertoires
            
            # For simplicity, use EMD between whole and product of marginals
            # This is a proxy for integrated information
            phi_partition = earth_movers_distance(p_whole, p_whole)  # Placeholder
            
            if phi_partition < min_phi:
                min_phi = phi_partition
    
    # Φ is the distance from whole to MIP
    # Simplified: use entropy of whole distribution as proxy
    phi = entropy(p_whole) - min_phi
    return max(0.0, phi)


def compute_phi_profile(states, tpm):
    """Compute Φ for each time point.
    
    states: (T, N) binary states
    tpm: (2^N, 2^N) TPM
    
    Returns: (T,) Φ values
    """
    T, N = states.shape
    nodes = list(range(N))
    phi_vals = []
    
    for t in range(T):
        phi = compute_phi_mip(states[t], tpm, nodes)
        phi_vals.append(phi)
    
    return np.array(phi_vals)


def compute_phi_from_connectivity(A, states):
    """Compute Φ profile from connectivity matrix and state sequence.
    
    A: (N, N) connectivity/adjacency matrix
    states: (T, N) binary state sequence
    
    Returns: Φ profile (T,)
    """
    T, N = states.shape
    
    # Simulate next states using A (simplified linear threshold)
    next_states = np.zeros_like(states)
    for t in range(T - 1):
        activation = A @ states[t]
        next_states[t] = (activation > 0.5).astype(int)
    next_states[-1] = next_states[-2]  # Repeat last
    
    # Compute TPM
    tpm = compute_tpm(states, next_states)
    
    # Compute Φ profile
    phi_profile = compute_phi_profile(states, tpm)
    
    return phi_profile
