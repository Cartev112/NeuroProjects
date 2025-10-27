"""
Emotion regulation demonstration.

Shows how different regulation strategies affect emotional responses.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import HARNAModel, Stimulus
from src.regulation import RegulationStrategy
from src.utils import generate_random_stimulus, create_context


def main():
    print("="*60)
    print("Emotion Regulation Strategies Demo")
    print("="*60)
    
    # Create model
    print("\n1. Creating HARNA model...")
    model = HARNAModel(
        theory='scherer',
        individual_traits={
            'neuroticism': 0.6,
            'emotional_intelligence': 0.7
        }
    )
    
    # Create negative emotional stimulus
    print("\n2. Creating negative emotional stimulus...")
    negative_features = generate_random_stimulus() * -2.0
    negative_stimulus = Stimulus(
        features=negative_features,
        context=create_context(threat=True)
    )
    
    # Generate baseline emotional response
    print("\n3. Baseline emotional response (no regulation)...")
    baseline_response = model.process(negative_stimulus)
    
    print(f"\n  Emotion: {baseline_response.emotion_label}")
    print(f"  Valence: {baseline_response.valence:+.2f}")
    print(f"  Arousal: {baseline_response.arousal:+.2f}")
    print(f"  Dominance: {baseline_response.dominance:+.2f}")
    
    # Test each regulation strategy
    strategies = [
        (RegulationStrategy.REAPPRAISAL, "Cognitive Reappraisal"),
        (RegulationStrategy.SUPPRESSION, "Response Suppression"),
        (RegulationStrategy.DISTRACTION, "Attentional Distraction"),
        (RegulationStrategy.ACCEPTANCE, "Acceptance")
    ]
    
    print("\n4. Testing regulation strategies...\n")
    
    results = []
    
    for strategy, name in strategies:
        print(f"{name}:")
        print("-" * 60)
        
        # Apply regulation
        regulated_response = model.process(
            negative_stimulus,
            regulate=True,
            regulation_strategy=strategy
        )
        
        # Compute changes
        valence_change = regulated_response.valence - baseline_response.valence
        arousal_change = regulated_response.arousal - baseline_response.arousal
        
        print(f"  Regulated Emotion: {regulated_response.emotion_label}")
        print(f"  Valence: {baseline_response.valence:+.2f} → {regulated_response.valence:+.2f} "
              f"(Δ {valence_change:+.2f})")
        print(f"  Arousal: {baseline_response.arousal:+.2f} → {regulated_response.arousal:+.2f} "
              f"(Δ {arousal_change:+.2f})")
        
        # Effectiveness
        effectiveness = abs(valence_change) + abs(arousal_change)
        print(f"  Effectiveness: {effectiveness:.2f}")
        print()
        
        results.append({
            'strategy': name,
            'valence_change': valence_change,
            'arousal_change': arousal_change,
            'effectiveness': effectiveness
        })
    
    # Summary comparison
    print("\n5. Strategy Comparison Summary:")
    print("="*60)
    print(f"{'Strategy':<25} {'Valence Δ':<12} {'Arousal Δ':<12} {'Effectiveness':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['strategy']:<25} "
              f"{result['valence_change']:+.2f}        "
              f"{result['arousal_change']:+.2f}        "
              f"{result['effectiveness']:.2f}")
    
    # Find most effective
    most_effective = max(results, key=lambda x: x['effectiveness'])
    print(f"\nMost Effective Strategy: {most_effective['strategy']}")
    
    # Context-dependent regulation
    print("\n6. Context-dependent regulation selection...\n")
    
    contexts = [
        ('High-stress work situation', create_context(threat=True), False),
        ('Social embarrassment', create_context(social=True), True),
        ('Mild disappointment', create_context(), True)
    ]
    
    for scenario_name, context, time_available in contexts:
        stimulus = Stimulus(
            features=generate_random_stimulus() * -1.0,
            context=context
        )
        
        # Let model select optimal strategy
        emotional_state = {
            'valence': -0.5,
            'arousal': 0.6,
            'dominance': -0.3
        }
        
        context['time_available'] = time_available
        
        optimal_strategy = model.regulation.select_optimal_strategy(
            emotional_state,
            context
        )
        
        print(f"{scenario_name}:")
        print(f"  Recommended: {optimal_strategy.value}")
        print()
    
    print("="*60)
    print("Regulation demo complete!")
    print("="*60)


if __name__ == '__main__':
    main()
