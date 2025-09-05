import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phi_core import compute_phi_from_connectivity, compute_phi_profile, compute_tpm
from graph_analysis import compute_graph_metrics, identify_hubs, compute_rich_club
from topology import persistent_homology, topological_complexity
from landscape import (
    find_attractors,
    basin_of_attraction,
    perturbation_analysis,
    consciousness_landscape_grid,
    simulate_dynamics,
)


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _load_array(path: str, key='A'):
    if path.endswith('.npy'):
        arr = np.load(path)
    elif path.endswith('.npz'):
        npz = np.load(path)
        arr = npz[key] if key in npz.files else npz[npz.files[0]]
    else:
        raise ValueError('Input must be .npy or .npz')
    return np.asarray(arr)


def main():
    p = argparse.ArgumentParser(description='IIT-Φ Dynamics: Integrated Information Architecture Mapper')
    # Data
    p.add_argument('--connectivity', required=True, help='Connectivity matrix (N,N) .npy/.npz')
    p.add_argument('--states', default=None, help='Optional state sequence (T,N) for Φ profile')
    # Analysis
    p.add_argument('--compute_phi', action='store_true', help='Compute Φ profile from states')
    p.add_argument('--graph_metrics', action='store_true', help='Compute graph-theoretic metrics')
    p.add_argument('--topology', action='store_true', help='Compute topological complexity via persistent homology')
    p.add_argument('--landscape', action='store_true', help='Map consciousness landscape with attractors')
    p.add_argument('--perturbation', action='store_true', help='Run perturbation analysis')
    # Landscape params
    p.add_argument('--n_attractors', type=int, default=50, help='Number of random inits for attractor search')
    p.add_argument('--grid_res', type=int, default=20, help='Grid resolution for landscape visualization')
    p.add_argument('--perturb_nodes', default='0,1', help='Comma-separated node indices to perturb')
    p.add_argument('--perturb_strength', type=float, default=0.5)
    # Output
    p.add_argument('--out_dir', required=True)

    args = p.parse_args()
    _ensure_outdir(args.out_dir)

    # Load connectivity
    A = _load_array(args.connectivity, key='A')
    N = A.shape[0]
    print(f'Loaded connectivity matrix: {N} nodes')

    # Load states if provided
    states = None
    if args.states is not None:
        states = _load_array(args.states, key='states')
        print(f'Loaded states: {states.shape}')

    results = {}

    # Graph metrics
    if args.graph_metrics:
        print('Computing graph metrics...')
        metrics = compute_graph_metrics(A)
        hubs = identify_hubs(A, threshold=0.75)
        rich_club = compute_rich_club(A)
        
        results['graph_metrics'] = {
            'mean_clustering': metrics['mean_clustering'],
            'efficiency': metrics['efficiency'],
            'modularity_proxy': metrics['modularity_proxy'],
            'hubs': hubs.tolist(),
            'n_hubs': int(len(hubs)),
        }
        
        # Save detailed metrics
        np.savez(
            os.path.join(args.out_dir, 'graph_metrics.npz'),
            out_degree=metrics['out_degree'],
            in_degree=metrics['in_degree'],
            clustering=metrics['clustering'],
        )
        
        with open(os.path.join(args.out_dir, 'rich_club.json'), 'w') as f:
            json.dump({str(k): float(v) for k, v in rich_club.items()}, f, indent=2)

    # Φ computation
    if args.compute_phi and states is not None:
        print('Computing Φ profile...')
        phi_profile = compute_phi_from_connectivity(A, states)
        
        results['phi_stats'] = {
            'mean_phi': float(np.mean(phi_profile)),
            'std_phi': float(np.std(phi_profile)),
            'max_phi': float(np.max(phi_profile)),
            'min_phi': float(np.min(phi_profile)),
        }
        
        np.save(os.path.join(args.out_dir, 'phi_profile.npy'), phi_profile)
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(phi_profile)
        plt.xlabel('Time')
        plt.ylabel('Φ')
        plt.title('Integrated Information Over Time')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'phi_timecourse.png'), dpi=150)
        plt.close()

    # Topology
    if args.topology:
        print('Computing topological complexity...')
        if states is not None:
            topo_complexity = topological_complexity(states)
            ph = persistent_homology(states, n_steps=20)
            
            results['topology'] = {
                'complexity': float(topo_complexity),
                'n_b0_features': len(ph['b0_persistence']),
                'n_b1_features': len(ph['b1_persistence']),
                'n_b2_features': len(ph['b2_persistence']),
            }
        else:
            print('Warning: --topology requires --states')

    # Landscape
    if args.landscape:
        print('Mapping consciousness landscape...')
        
        # Find attractors
        attractors = find_attractors(A, n_inits=args.n_attractors, T=200)
        print(f'Found {len(attractors)} attractors')
        
        # Compute basin sizes
        basin_sizes = []
        for att in attractors:
            basin_size = basin_of_attraction(A, att, n_samples=500, T=100)
            basin_sizes.append(basin_size)
        
        results['landscape'] = {
            'n_attractors': len(attractors),
            'basin_sizes': [float(b) for b in basin_sizes],
        }
        
        np.savez(
            os.path.join(args.out_dir, 'attractors.npz'),
            attractors=np.array(attractors),
            basin_sizes=np.array(basin_sizes),
        )
        
        # Create 2D landscape visualization
        def phi_func(state):
            # Simplified Φ from single state
            state_binary = (state > 0.5).astype(int)
            next_state = (A @ state > 0.5).astype(int)
            tpm = compute_tpm(state_binary[None, :], next_state[None, :])
            from phi_core import compute_phi_mip
            return compute_phi_mip(state_binary, tpm, list(range(N)))
        
        landscape = consciousness_landscape_grid(A, phi_func, grid_resolution=args.grid_res)
        
        # Plot landscape
        plt.figure(figsize=(8, 6))
        plt.contourf(landscape['x_vals'], landscape['y_vals'], landscape['phi_grid'].T, levels=20, cmap='viridis')
        plt.colorbar(label='Φ')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Consciousness Landscape (Φ)')
        
        # Overlay attractors
        if len(attractors) > 0:
            att_array = np.array(attractors)
            att_proj = (att_array - landscape['mean_state']) @ np.vstack([landscape['pc1'], landscape['pc2']]).T
            plt.scatter(att_proj[:, 0], att_proj[:, 1], c='red', s=100, marker='*', edgecolors='white', label='Attractors')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'consciousness_landscape.png'), dpi=150)
        plt.close()

    # Perturbation
    if args.perturbation:
        print('Running perturbation analysis...')
        
        perturb_nodes = [int(x) for x in args.perturb_nodes.split(',')]
        x0 = np.random.rand(N)
        
        pert_result = perturbation_analysis(A, x0, perturb_nodes, args.perturb_strength, T=100)
        
        results['perturbation'] = {
            'perturbed_nodes': perturb_nodes,
            'strength': float(args.perturb_strength),
            'max_divergence': float(np.max(pert_result['divergence'])),
            'final_divergence': float(pert_result['divergence'][-1]),
        }
        
        # Plot divergence
        plt.figure(figsize=(10, 4))
        plt.plot(pert_result['divergence'])
        plt.xlabel('Time')
        plt.ylabel('Divergence')
        plt.title(f'Perturbation Divergence (nodes {perturb_nodes})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'perturbation_divergence.png'), dpi=150)
        plt.close()

    # Summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Done. Results in {args.out_dir}')


if __name__ == '__main__':
    main()
