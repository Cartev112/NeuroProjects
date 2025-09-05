"""
Intervention optimization example.

Demonstrates finding optimal interventions to shift emotional states
from negative to positive attractors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state_space import EmotionalStateSpace
from src.intervention import InterventionOptimizer, suggest_intervention_strategy
from src.utils import create_standard_attractors


def main():
    print("="*60)
    print("Optimal Intervention for Emotional Regulation")
    print("="*60)
    
    # Create state space
    print("\n1. Setting up emotional state space...")
    state_space = EmotionalStateSpace(dimensions=5, viscosity=0.1, noise_level=0.03)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Mild Sadness',
            'current_state': np.array([-0.4, -0.2, -0.3, -0.3, 0.0]),
            'target': 'happiness'
        },
        {
            'name': 'Severe Depression',
            'current_state': np.array([-0.8, -0.6, -0.7, -0.8, 0.0]),
            'target': 'calm'
        },
        {
            'name': 'Anxiety',
            'current_state': np.array([-0.3, 0.7, -0.4, -0.6, 0.0]),
            'target': 'calm'
        },
        {
            'name': 'Anger',
            'current_state': np.array([-0.6, 0.8, 0.5, 0.7, 0.0]),
            'target': 'calm'
        }
    ]
    
    # Create optimizer
    optimizer = InterventionOptimizer(state_space)
    
    # Analyze each scenario
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{i+2}. Analyzing scenario: {scenario['name']}")
        print(f"   Current state: {scenario['current_state'][:3]}")
        
        # Find target attractor
        target_attractor = None
        for attr in attractors:
            if attr.name == scenario['target']:
                target_attractor = attr
                break
        
        print(f"   Target: {scenario['target']}")
        
        # Get personalized suggestion
        suggestion = suggest_intervention_strategy(scenario['current_state'])
        print(f"\n   Personalized Recommendation:")
        for line in suggestion.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Optimize impulse intervention
        print(f"\n   Optimizing impulse intervention...")
        result_impulse = optimizer.find_optimal_intervention(
            scenario['current_state'],
            target_attractor,
            method='impulse',
            max_magnitude=1.0,
            time_horizon=50,
            n_trials=50
        )
        
        print(f"   ✓ Success probability: {result_impulse.success_prob:.1%}")
        print(f"   ✓ Energy cost: {result_impulse.energy_cost:.4f}")
        print(f"   ✓ Time to target: {result_impulse.time_to_target:.2f}")
        
        # Optimize sustained intervention
        print(f"\n   Optimizing sustained intervention...")
        result_sustained = optimizer.find_optimal_intervention(
            scenario['current_state'],
            target_attractor,
            method='sustained',
            max_magnitude=0.5,
            time_horizon=50,
            n_trials=50
        )
        
        print(f"   ✓ Success probability: {result_sustained.success_prob:.1%}")
        print(f"   ✓ Energy cost: {result_sustained.energy_cost:.4f}")
        print(f"   ✓ Time to target: {result_sustained.time_to_target:.2f}")
        
        # Compare strategies
        if result_impulse.energy_cost < result_sustained.energy_cost:
            print(f"\n   → Recommendation: Impulse intervention (lower energy cost)")
            best_result = result_impulse
        else:
            print(f"\n   → Recommendation: Sustained intervention (more gradual)")
            best_result = result_sustained
        
        results.append({
            'scenario': scenario,
            'impulse': result_impulse,
            'sustained': result_sustained,
            'best': best_result,
            'target_attractor': target_attractor
        })
    
    # Visualize results
    print(f"\n{len(scenarios)+2}. Generating visualizations...")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, result_data in enumerate(results):
        ax = axes[idx]
        
        scenario = result_data['scenario']
        best_result = result_data['best']
        target_attractor = result_data['target_attractor']
        current_state = scenario['current_state']
        
        # Plot vector field
        X, Y, U, V = state_space.compute_vector_field(
            grid_resolution=12, dimensions=(0, 1)
        )
        ax.quiver(X, Y, U, V, alpha=0.2, scale=10, width=0.003)
        
        # Plot trajectory
        trajectory = best_result.trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', 
               linewidth=2.5, alpha=0.8, label='Trajectory')
        
        # Mark start
        ax.plot(current_state[0], current_state[1], 'ro', 
               markersize=14, label='Start', zorder=10)
        
        # Mark intervention (if impulse)
        if best_result.intervention_type == 'impulse':
            perturbed = current_state + best_result.action
            ax.arrow(current_state[0], current_state[1],
                    best_result.action[0] * 0.8, 
                    best_result.action[1] * 0.8,
                    head_width=0.08, head_length=0.08, 
                    fc='red', ec='red', linewidth=2.5, 
                    alpha=0.7, zorder=9)
        
        # Mark target
        ax.plot(target_attractor.center[0], target_attractor.center[1],
               'g*', markersize=22, label='Target', zorder=10)
        
        # Mark other attractors
        for attr in attractors:
            if attr.name != scenario['target']:
                ax.plot(attr.center[0], attr.center[1], 
                       'k.', markersize=10, alpha=0.3)
        
        ax.set_xlabel('Valence', fontsize=11)
        ax.set_ylabel('Arousal', fontsize=11)
        ax.set_title(f"{scenario['name']} → {scenario['target'].title()}\n"
                    f"({best_result.intervention_type}, "
                    f"success: {best_result.success_prob:.0%})",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Create summary comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    
    scenario_names = [r['scenario']['name'] for r in results]
    
    # Success probabilities
    ax = axes2[0]
    impulse_success = [r['impulse'].success_prob for r in results]
    sustained_success = [r['sustained'].success_prob for r in results]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    ax.bar(x - width/2, impulse_success, width, label='Impulse', alpha=0.8)
    ax.bar(x + width/2, sustained_success, width, label='Sustained', alpha=0.8)
    ax.set_ylabel('Success Probability', fontsize=11)
    ax.set_title('Intervention Success Rates', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # Energy costs
    ax = axes2[1]
    impulse_energy = [r['impulse'].energy_cost for r in results]
    sustained_energy = [r['sustained'].energy_cost for r in results]
    
    ax.bar(x - width/2, impulse_energy, width, label='Impulse', alpha=0.8)
    ax.bar(x + width/2, sustained_energy, width, label='Sustained', alpha=0.8)
    ax.set_ylabel('Energy Cost', fontsize=11)
    ax.set_title('Intervention Energy Requirements', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Time to target
    ax = axes2[2]
    impulse_time = [r['impulse'].time_to_target for r in results]
    sustained_time = [r['sustained'].time_to_target for r in results]
    
    ax.bar(x - width/2, impulse_time, width, label='Impulse', alpha=0.8)
    ax.bar(x + width/2, sustained_time, width, label='Sustained', alpha=0.8)
    ax.set_ylabel('Time to Target', fontsize=11)
    ax.set_title('Convergence Time', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("Intervention optimization complete! Close plots to exit.")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    main()
