"""
Emotional epidemiology simulation.

Models emotion spread as epidemic process with interventions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import create_network
from src.epidemiology import EmotionalEpidemiology


def main():
    print("="*60)
    print("Emotional Epidemiology Simulation")
    print("="*60)
    
    # Create scale-free network (realistic social network)
    print("\n1. Creating scale-free social network...")
    network = create_network('scale_free', n_nodes=500, m=3)
    
    print(f"   Network size: {len(network)} agents")
    print(f"   Connections: {network.graph.number_of_edges()}")
    
    # Create epidemiology simulator
    print("\n2. Initializing emotional epidemic (anger)...")
    epi = EmotionalEpidemiology(
        network=network,
        emotion_type='anger',
        transmission_rate=0.3,
        recovery_rate=0.1,
        threshold=0.5
    )
    
    r0 = epi.compute_r0()
    print(f"   R₀ = {r0:.2f}")
    
    if r0 > 1:
        print("   ⚠ Epidemic will spread (R₀ > 1)")
    else:
        print("   ✓ Epidemic will die out (R₀ < 1)")
    
    # Seed initial infections
    print("\n3. Seeding patient zeros...")
    patient_zeros = [str(i) for i in range(5)]
    epi.seed_emotion(patient_zeros, intensity=0.9)
    print(f"   Seeded {len(patient_zeros)} initial infections")
    
    # Simulate without intervention
    print("\n4. Simulating epidemic (no intervention)...")
    history_baseline = epi.simulate(steps=100)
    
    peak_info = epi.get_peak_info()
    print(f"   Peak infected: {peak_info['peak_count']} ({peak_info['peak_proportion']:.1%})")
    print(f"   Peak time: {peak_info['peak_time']}")
    
    # Identify influencers
    print("\n5. Identifying emotional influencers...")
    influencers = epi.identify_influencers(top_k=20)
    print(f"   Identified {len(influencers)} high-influence agents")
    
    # Reset and simulate with intervention
    print("\n6. Simulating with targeted intervention...")
    
    # Create new epidemic
    network2 = create_network('scale_free', n_nodes=500, m=3)
    epi2 = EmotionalEpidemiology(
        network=network2,
        emotion_type='anger',
        transmission_rate=0.3,
        recovery_rate=0.1
    )
    
    epi2.seed_emotion(patient_zeros, intensity=0.9)
    
    # Simulate first phase
    history_intervention = epi2.simulate(steps=30)
    
    # Apply intervention
    influencers2 = epi2.identify_influencers(top_k=20)
    epi2.apply_intervention(
        influencers2,
        intervention_type='regulation',
        effectiveness=0.7
    )
    print("   Applied regulation intervention to influencers")
    
    # Continue simulation
    history_post = epi2.simulate(steps=70)
    
    peak_info2 = epi2.get_peak_info()
    print(f"   Post-intervention peak: {peak_info2['peak_count']} ({peak_info2['peak_proportion']:.1%})")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison: Baseline vs Intervention")
    print("="*60)
    print(f"Baseline peak: {peak_info['peak_count']} agents")
    print(f"Intervention peak: {peak_info2['peak_count']} agents")
    print(f"Reduction: {peak_info['peak_count'] - peak_info2['peak_count']} agents "
          f"({(1 - peak_info2['peak_count']/peak_info['peak_count']):.1%})")
    
    # Visualize
    print("\n7. Generating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline epidemic curve
    ax = axes[0]
    epi.plot_epidemic_curve()
    ax = plt.gca()
    ax.set_title('Baseline: No Intervention', fontsize=14, fontweight='bold')
    
    # Intervention epidemic curve
    plt.figure()
    epi2.plot_epidemic_curve()
    ax = plt.gca()
    ax.axvline(x=30, color='purple', linestyle='--', linewidth=2, 
              label='Intervention Applied')
    ax.set_title('With Targeted Intervention', fontsize=14, fontweight='bold')
    ax.legend()
    
    print("\n" + "="*60)
    print("Simulation complete! Close plots to exit.")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    main()
