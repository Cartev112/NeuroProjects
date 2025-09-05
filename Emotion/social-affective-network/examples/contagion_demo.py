"""
Emotional contagion demonstration.

Shows how emotions spread through social networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import create_network
from src.propagation import EmotionPropagator


def main():
    print("="*60)
    print("Emotional Contagion Demo")
    print("="*60)
    
    # Create small-world network
    print("\n1. Creating small-world social network...")
    network = create_network('small_world', n_nodes=100, k=6, p=0.3)
    
    metrics = network.compute_network_metrics()
    print(f"   Nodes: {metrics['n_nodes']}")
    print(f"   Edges: {metrics['n_edges']}")
    print(f"   Clustering: {metrics['clustering']:.3f}")
    print(f"   Average path length: {metrics['avg_path_length']:.2f}")
    
    # Identify central agent
    print("\n2. Identifying central agent to seed emotion...")
    central_agents = network.identify_central_agents(top_k=1)
    seed_id = central_agents[0][0]
    print(f"   Most central agent: {seed_id}")
    print(f"   Betweenness centrality: {central_agents[0][1]:.3f}")
    
    # Seed positive emotion
    print("\n3. Seeding joy in central agent...")
    seed_agent = network.get_agent(seed_id)
    seed_agent.set_emotion({'valence': 0.9, 'arousal': 0.7})
    print(f"   Initial emotion: {seed_agent.emotional_state.emotion_type.value}")
    
    # Create propagator
    propagator = EmotionPropagator(
        network,
        decay_rate=0.1,
        transmission_rate=0.3
    )
    
    # Simulate
    print("\n4. Simulating emotional contagion...")
    n_steps = 50
    history = propagator.simulate(n_steps)
    
    # Analyze spread
    analysis = propagator.analyze_spread()
    
    print("\n" + "="*60)
    print("Contagion Results")
    print("="*60)
    print(f"Infection rate: {analysis['infection_rate']:.1%}")
    print(f"Infected agents: {analysis['n_infected']}/{analysis['total_agents']}")
    print(f"Peak time: {analysis['peak_time']}")
    
    # Plot results
    print("\n5. Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean valence over time
    ax = axes[0, 0]
    times = [h['time_step'] for h in history]
    valences = [h['mean_valence'] for h in history]
    
    ax.plot(times, valences, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Mean Valence', fontsize=11)
    ax.set_title('Network Emotional Valence Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Infection spread
    ax = axes[0, 1]
    infected_counts = [h.get('n_infected', 0) for h in history]
    
    ax.plot(times, infected_counts, 'r-', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Number Infected', fontsize=11)
    ax.set_title('Emotional Infection Spread', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Emotional variance
    ax = axes[1, 0]
    std_valences = [h['std_valence'] for h in history]
    
    ax.plot(times, std_valences, 'g-', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Std Valence', fontsize=11)
    ax.set_title('Emotional Diversity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Contagion Summary
    {'='*40}
    
    Network Properties:
      Nodes: {metrics['n_nodes']}
      Edges: {metrics['n_edges']}
      Clustering: {metrics['clustering']:.3f}
    
    Spread Statistics:
      Infection Rate: {analysis['infection_rate']:.1%}
      Peak Time: {analysis['peak_time']}
      Final Mean Valence: {valences[-1]:+.2f}
    
    Interpretation:
      {'High' if analysis['infection_rate'] > 0.5 else 'Moderate'} contagion
      {'Rapid' if analysis['peak_time'] < 20 else 'Gradual'} spread
      {'Successful' if valences[-1] > 0.3 else 'Limited'} emotional shift
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("Demo complete! Close plot to exit.")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    main()
