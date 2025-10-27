"""
Basic emotion generation example.

Demonstrates how to use HARNA to generate emotional responses
to different types of stimuli.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import HARNAModel, Stimulus
from src.utils import generate_random_stimulus, create_context


def main():
    print("="*60)
    print("HARNA Basic Emotion Generation Demo")
    print("="*60)
    
    # Create HARNA model
    print("\n1. Creating HARNA model...")
    model = HARNAModel(
        theory='scherer',
        individual_traits={
            'neuroticism': 0.6,
            'resilience': 0.7,
            'emotional_intelligence': 0.65
        }
    )
    print("   ✓ Model created with Scherer's appraisal theory")
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'Neutral Everyday Event',
            'features': generate_random_stimulus() * 0.3,
            'context': create_context()
        },
        {
            'name': 'Threatening Situation',
            'features': generate_random_stimulus() * -2.0,
            'context': create_context(threat=True)
        },
        {
            'name': 'Positive Social Interaction',
            'features': generate_random_stimulus() * 1.5,
            'context': create_context(social=True, positive=True)
        },
        {
            'name': 'Familiar Positive Event',
            'features': generate_random_stimulus() * 1.0,
            'context': create_context(familiar=True, positive=True)
        },
        {
            'name': 'Novel Uncertain Situation',
            'features': generate_random_stimulus() * 0.8,
            'context': create_context(familiar=False)
        }
    ]
    
    print("\n2. Processing different scenarios...\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print("-" * 60)
        
        # Create stimulus
        stimulus = Stimulus(
            features=scenario['features'],
            context=scenario['context']
        )
        
        # Process through HARNA
        response = model.process(stimulus)
        
        # Display results
        print(f"  Emotion: {response.emotion_label}")
        print(f"  Valence: {response.valence:+.2f}")
        print(f"  Arousal: {response.arousal:+.2f}")
        print(f"  Dominance: {response.dominance:+.2f}")
        print(f"\n  Subcortical Response:")
        print(f"    Threat Level: {response.subcortical_threat:.2f}")
        print(f"    Arousal: {response.subcortical_arousal:+.2f}")
        print(f"\n  Appraisal Dimensions:")
        print(f"    Novelty: {response.appraisal_result['novelty']:+.2f}")
        print(f"    Pleasantness: {response.appraisal_result['intrinsic_pleasantness']:+.2f}")
        print(f"    Goal Relevance: {response.appraisal_result['goal_relevance']:.2f}")
        print(f"    Coping Potential: {response.appraisal_result['coping_potential']:.2f}")
        print(f"\n  Processing Info:")
        print(f"    Time: {response.processing_time_ms:.1f}ms")
        print(f"    Bottom-up weight: {response.pathway_weights['bottom_up']:.2f}")
        print(f"    Top-down weight: {response.pathway_weights['top_down']:.2f}")
        print()
    
    # Compare with different personality
    print("\n3. Comparing with different personality traits...\n")
    
    # High neuroticism model
    high_neuroticism_model = HARNAModel(
        theory='scherer',
        individual_traits={'neuroticism': 0.9, 'resilience': 0.3}
    )
    
    # Low neuroticism model
    low_neuroticism_model = HARNAModel(
        theory='scherer',
        individual_traits={'neuroticism': 0.2, 'resilience': 0.8}
    )
    
    # Test on same threatening stimulus
    threat_stimulus = Stimulus(
        features=generate_random_stimulus() * -1.5,
        context=create_context(threat=True)
    )
    
    print("Same threatening stimulus processed by different personalities:")
    print("-" * 60)
    
    response_high = high_neuroticism_model.process(threat_stimulus)
    response_low = low_neuroticism_model.process(threat_stimulus)
    
    print(f"High Neuroticism (0.9):")
    print(f"  Emotion: {response_high.emotion_label}")
    print(f"  Valence: {response_high.valence:+.2f}")
    print(f"  Threat: {response_high.subcortical_threat:.2f}")
    
    print(f"\nLow Neuroticism (0.2):")
    print(f"  Emotion: {response_low.emotion_label}")
    print(f"  Valence: {response_low.valence:+.2f}")
    print(f"  Threat: {response_low.subcortical_threat:.2f}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == '__main__':
    main()
