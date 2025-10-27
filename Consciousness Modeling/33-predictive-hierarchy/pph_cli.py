import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hierarchy import PredictiveHierarchy
from active_inference import ActiveInferenceAgent
from metacognition import MetacognitiveModule, CounterfactualEngine
from global_workspace import GlobalWorkspace, AttentionMechanism
from psychophysics import (
    change_blindness_experiment,
    binocular_rivalry_experiment,
    attentional_blink_experiment,
)


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description='Predictive Processing Hierarchy Simulator with Metacognitive Layers')
    # Architecture
    p.add_argument('--n_levels', type=int, default=3, help='Number of hierarchical levels')
    p.add_argument('--input_dim', type=int, default=10, help='Sensory input dimension')
    p.add_argument('--hidden_dims', default='16,12,8', help='Comma-separated hidden dims per level')
    p.add_argument('--lr', type=float, default=0.01)
    # Simulation
    p.add_argument('--n_iterations', type=int, default=10, help='Iterations per forward pass')
    p.add_argument('--n_steps', type=int, default=100, help='Total simulation steps')
    # Modules
    p.add_argument('--active_inference', action='store_true', help='Enable active inference')
    p.add_argument('--metacognition', action='store_true', help='Enable metacognitive module')
    p.add_argument('--global_workspace', action='store_true', help='Enable global workspace')
    p.add_argument('--counterfactual', action='store_true', help='Run counterfactual simulations')
    # Psychophysics
    p.add_argument('--change_blindness', action='store_true')
    p.add_argument('--rivalry', action='store_true')
    p.add_argument('--attentional_blink', action='store_true')
    # Output
    p.add_argument('--out_dir', required=True)

    args = p.parse_args()
    _ensure_outdir(args.out_dir)

    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    if len(hidden_dims) != args.n_levels:
        raise ValueError(f'hidden_dims must have {args.n_levels} values')

    # Build level dimensions
    level_dims = []
    for i in range(args.n_levels):
        if i == 0:
            inp_dim = args.input_dim
        else:
            inp_dim = hidden_dims[i - 1]
        
        hid_dim = hidden_dims[i]
        
        if i == args.n_levels - 1:
            out_dim = hid_dim
        else:
            out_dim = hidden_dims[i]
        
        level_dims.append((inp_dim, hid_dim, out_dim))

    print(f'Building hierarchy with {args.n_levels} levels: {level_dims}')

    # Create hierarchy
    hierarchy = PredictiveHierarchy(level_dims, lr=args.lr)

    # Optional modules
    agent = None
    metacog = None
    workspace = None
    attention = None
    counterfactual = None

    if args.active_inference:
        print('Enabling active inference...')
        agent = ActiveInferenceAgent(hierarchy, action_dim=args.input_dim)

    if args.metacognition:
        print('Enabling metacognition...')
        metacog = MetacognitiveModule(hierarchy)

    if args.global_workspace:
        print('Enabling global workspace...')
        workspace = GlobalWorkspace(hierarchy)
        attention = AttentionMechanism(hierarchy)

    if args.counterfactual:
        print('Enabling counterfactual engine...')
        counterfactual = CounterfactualEngine(hierarchy)

    results = {}

    # Basic simulation
    print('Running basic simulation...')
    observations = [np.random.randn(args.input_dim) for _ in range(args.n_steps)]
    
    error_history = []
    confidence_history = []
    broadcast_history = []
    
    for step, obs in enumerate(observations):
        if agent:
            action, outputs = agent.step(obs, n_iterations=args.n_iterations)
        else:
            outputs = hierarchy.forward(obs, n_iterations=args.n_iterations)
        
        total_error = hierarchy.get_total_error()
        error_history.append(total_error)
        
        if metacog:
            conf = metacog.estimate_confidence()
            confidence_history.append(conf)
        
        if workspace:
            workspace.update()
            broadcast_history.append(workspace.get_broadcast_strength())

    results['simulation'] = {
        'mean_error': float(np.mean(error_history)),
        'std_error': float(np.std(error_history)),
    }

    # Plot error
    plt.figure(figsize=(10, 4))
    plt.plot(error_history)
    plt.xlabel('Step')
    plt.ylabel('Total Prediction Error')
    plt.title('Prediction Error Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'error_timecourse.png'), dpi=150)
    plt.close()

    if metacog:
        results['metacognition'] = {
            'mean_confidence': float(np.mean(confidence_history)),
            'std_confidence': float(np.std(confidence_history)),
        }
        
        plt.figure(figsize=(10, 4))
        plt.plot(confidence_history)
        plt.xlabel('Step')
        plt.ylabel('Confidence')
        plt.title('Metacognitive Confidence Over Time')
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'confidence_timecourse.png'), dpi=150)
        plt.close()

    if workspace:
        results['global_workspace'] = {
            'mean_broadcast': float(np.mean(broadcast_history)),
            'conscious_steps': int(sum(1 for h in workspace.broadcast_history if h['is_conscious'])),
            'consciousness_rate': float(sum(1 for h in workspace.broadcast_history if h['is_conscious']) / len(workspace.broadcast_history)),
        }
        
        plt.figure(figsize=(10, 4))
        plt.plot(broadcast_history)
        plt.xlabel('Step')
        plt.ylabel('Broadcast Strength')
        plt.title('Global Workspace Broadcasting')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'broadcast_timecourse.png'), dpi=150)
        plt.close()

    # Counterfactual
    if counterfactual:
        print('Running counterfactual simulations...')
        obs = np.random.randn(args.input_dim)
        
        # Test different interventions
        interventions = [
            {0: np.random.randn(hidden_dims[0])},
            {1: np.random.randn(hidden_dims[1]) if args.n_levels > 1 else np.random.randn(hidden_dims[0])},
        ]
        
        cf_results = counterfactual.compare_scenarios(obs, interventions)
        results['counterfactual'] = {
            'n_scenarios': len(cf_results),
            'best_error': float(cf_results[0][1]) if cf_results else None,
            'worst_error': float(cf_results[-1][1]) if cf_results else None,
        }

    # Psychophysics
    if args.change_blindness and workspace and attention:
        print('Running change blindness experiment...')
        cb_results = change_blindness_experiment(hierarchy, workspace, attention, n_trials=20)
        results['change_blindness'] = cb_results

    if args.rivalry and workspace:
        print('Running binocular rivalry experiment...')
        rivalry_results = binocular_rivalry_experiment(hierarchy, workspace, n_steps=100)
        results['rivalry'] = {
            'n_switches': rivalry_results['n_switches'],
            'switch_rate': rivalry_results['switch_rate'],
        }

    if args.attentional_blink and workspace and attention:
        print('Running attentional blink experiment...')
        ab_results = attentional_blink_experiment(hierarchy, workspace, attention, n_trials=20)
        results['attentional_blink'] = ab_results

    # Summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Done. Results in {args.out_dir}')


if __name__ == '__main__':
    main()
