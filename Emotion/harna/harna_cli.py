#!/usr/bin/env python3
"""
Command-line interface for HARNA model.

Provides tools for emotion generation, regulation, RL training,
and theory comparison.
"""

import click
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model import HARNAModel, Stimulus
from src.regulation import RegulationStrategy
from src.individual_differences import PersonalityTraits
from src.utils import generate_random_stimulus, create_context


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    HARNA: Hierarchical Appraisal-Regulation Neural Architecture
    
    A multi-layered computational model integrating cognitive appraisal
    theory with affective neuroscience.
    """
    pass


@cli.command()
@click.option('--theory', default='scherer', help='Emotion theory to use')
@click.option('--neuroticism', default=0.5, type=float, help='Neuroticism level (0-1)')
@click.option('--context', default='neutral', help='Context type (threat/social/positive/neutral)')
def generate(theory, neuroticism, context):
    """Generate emotional response to stimulus."""
    click.echo(f"Generating emotion using {theory} theory...")
    
    # Create model with personality traits
    model = HARNAModel(
        theory=theory,
        individual_traits={'neuroticism': neuroticism}
    )
    
    # Create stimulus
    features = generate_random_stimulus()
    
    # Create context
    if context == 'threat':
        ctx = create_context(threat=True)
    elif context == 'social':
        ctx = create_context(social=True)
    elif context == 'positive':
        ctx = create_context(positive=True)
    else:
        ctx = create_context()
    
    stimulus = Stimulus(features=features, context=ctx)
    
    # Process
    response = model.process(stimulus)
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("Emotional Response")
    click.echo("="*60)
    click.echo(str(response))
    click.echo("\nAppraisal Details:")
    for key, value in response.appraisal_result.items():
        click.echo(f"  {key}: {value:.3f}")
    click.echo("\nPathway Weights:")
    for key, value in response.pathway_weights.items():
        click.echo(f"  {key}: {value:.3f}")


@cli.command()
@click.option('--emotion', default='anger', help='Initial emotion')
@click.option('--strategies', default='reappraisal,suppression', help='Strategies to test')
def regulate(emotion, strategies):
    """Test emotion regulation strategies."""
    click.echo(f"Testing regulation strategies for {emotion}...")
    
    # Create model
    model = HARNAModel()
    
    # Create negative stimulus
    features = generate_random_stimulus()
    features *= -2  # Make it negative
    
    stimulus = Stimulus(features=features)
    
    # Generate initial emotion
    initial_response = model.process(stimulus)
    
    click.echo("\n" + "="*60)
    click.echo("Initial Emotional State")
    click.echo("="*60)
    click.echo(f"Emotion: {initial_response.emotion_label}")
    click.echo(f"Valence: {initial_response.valence:+.2f}")
    click.echo(f"Arousal: {initial_response.arousal:+.2f}")
    
    # Test each strategy
    strategy_list = strategies.split(',')
    
    for strategy_name in strategy_list:
        click.echo(f"\n{'='*60}")
        click.echo(f"Testing {strategy_name.upper()}")
        click.echo("="*60)
        
        # Map name to enum
        strategy_map = {
            'reappraisal': RegulationStrategy.REAPPRAISAL,
            'suppression': RegulationStrategy.SUPPRESSION,
            'distraction': RegulationStrategy.DISTRACTION,
            'acceptance': RegulationStrategy.ACCEPTANCE
        }
        
        strategy = strategy_map.get(strategy_name.strip(), RegulationStrategy.REAPPRAISAL)
        
        # Apply regulation
        regulated_response = model.process(stimulus, regulate=True, regulation_strategy=strategy)
        
        click.echo(f"Regulated Emotion: {regulated_response.emotion_label}")
        click.echo(f"Valence: {initial_response.valence:+.2f} → {regulated_response.valence:+.2f}")
        click.echo(f"Arousal: {initial_response.arousal:+.2f} → {regulated_response.arousal:+.2f}")
        
        valence_change = regulated_response.valence - initial_response.valence
        click.echo(f"Valence Change: {valence_change:+.2f}")


@cli.command()
@click.option('--trait', default='neuroticism', help='Trait to vary')
@click.option('--range', 'trait_range', default='0,1', help='Range (min,max)')
@click.option('--steps', default=5, type=int, help='Number of steps')
def individual_diff(trait, trait_range, steps):
    """Analyze individual differences effects."""
    click.echo(f"Analyzing effect of {trait}...")
    
    # Parse range
    min_val, max_val = [float(x) for x in trait_range.split(',')]
    trait_values = np.linspace(min_val, max_val, steps)
    
    # Fixed stimulus
    features = generate_random_stimulus()
    stimulus = Stimulus(features=features)
    
    click.echo("\n" + "="*60)
    click.echo(f"Effect of {trait.upper()} on Emotional Response")
    click.echo("="*60)
    click.echo(f"\n{'Trait Value':<12} {'Emotion':<15} {'Valence':<10} {'Arousal':<10}")
    click.echo("-" * 60)
    
    for value in trait_values:
        # Create model with trait
        model = HARNAModel(individual_traits={trait: value})
        
        # Process
        response = model.process(stimulus)
        
        click.echo(f"{value:<12.2f} {response.emotion_label:<15} "
                  f"{response.valence:+.2f}      {response.arousal:+.2f}")


@cli.command()
def demo():
    """Run interactive demonstration."""
    click.echo("="*60)
    click.echo("HARNA Interactive Demo")
    click.echo("="*60)
    
    # Create model
    click.echo("\n1. Creating HARNA model with default personality...")
    model = HARNAModel(
        theory='scherer',
        individual_traits={
            'neuroticism': 0.5,
            'resilience': 0.7,
            'emotional_intelligence': 0.6
        }
    )
    click.echo("   ✓ Model initialized")
    
    # Test different stimuli
    scenarios = [
        ('Neutral stimulus', generate_random_stimulus() * 0.5, {}),
        ('Threatening stimulus', generate_random_stimulus() * -2, {'threat': True}),
        ('Positive social event', generate_random_stimulus() * 1.5, {'social': True, 'positive': True}),
    ]
    
    for i, (name, features, context_dict) in enumerate(scenarios, 2):
        click.echo(f"\n{i}. Processing: {name}")
        
        stimulus = Stimulus(features=features, context=create_context(**context_dict))
        response = model.process(stimulus)
        
        click.echo(f"   Emotion: {response.emotion_label}")
        click.echo(f"   Valence: {response.valence:+.2f}, Arousal: {response.arousal:+.2f}")
        click.echo(f"   Threat Detection: {response.subcortical_threat:.2f}")
        click.echo(f"   Processing Time: {response.processing_time_ms:.1f}ms")
    
    # Test regulation
    click.echo(f"\n{len(scenarios)+2}. Testing emotion regulation...")
    
    # Create negative stimulus
    negative_features = generate_random_stimulus() * -2
    negative_stimulus = Stimulus(features=negative_features)
    
    # Without regulation
    unregulated = model.process(negative_stimulus)
    click.echo(f"   Without regulation: {unregulated.emotion_label} "
              f"(valence: {unregulated.valence:+.2f})")
    
    # With reappraisal
    regulated = model.process(negative_stimulus, regulate=True, 
                             regulation_strategy=RegulationStrategy.REAPPRAISAL)
    click.echo(f"   With reappraisal: {regulated.emotion_label} "
              f"(valence: {regulated.valence:+.2f})")
    
    click.echo("\n" + "="*60)
    click.echo("Demo complete!")
    click.echo("="*60)


@cli.command()
@click.option('--theories', default='all', help='Theories to compare (comma-separated or "all")')
def compare(theories):
    """Compare different emotion theories."""
    click.echo("Comparing emotion theories...")
    
    if theories == 'all':
        theory_list = ['james_lange', 'cannon_bard', 'schachter_singer', 'constructionist', 'scherer']
    else:
        theory_list = [t.strip() for t in theories.split(',')]
    
    # Fixed stimulus
    features = generate_random_stimulus()
    stimulus = Stimulus(features=features, context={'social': True})
    
    click.echo("\n" + "="*60)
    click.echo("Theory Comparison")
    click.echo("="*60)
    click.echo(f"\n{'Theory':<20} {'Emotion':<15} {'Valence':<10} {'Arousal':<10}")
    click.echo("-" * 60)
    
    for theory in theory_list:
        try:
            model = HARNAModel(theory=theory)
            response = model.process(stimulus)
            
            click.echo(f"{theory:<20} {response.emotion_label:<15} "
                      f"{response.valence:+.2f}      {response.arousal:+.2f}")
        except Exception as e:
            click.echo(f"{theory:<20} Error: {str(e)}")
    
    click.echo("\nNote: Different theories may produce different emotional responses")
    click.echo("to the same stimulus due to different causal architectures.")


if __name__ == '__main__':
    cli()
