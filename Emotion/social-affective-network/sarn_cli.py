#!/usr/bin/env python3
"""
Command-line interface for SARN (Social Affective Resonance Network).

Provides tools for simulating emotional contagion, epidemiology,
Theory of Mind reasoning, and network analysis.
"""

import click
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.network import SocialNetwork, create_network
from src.agent import Agent
from src.propagation import EmotionPropagator
from src.epidemiology import EmotionalEpidemiology
from src.theory_of_mind import TheoryOfMindModule
from src.cultural_norms import CulturalNorms


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    SARN: Social Affective Resonance Network
    
    Multi-agent system for modeling emotional contagion, empathy,
    and collective emotional dynamics.
    """
    pass


@cli.command()
@click.option('--network-type', default='small_world', help='Network topology')
@click.option('--n-nodes', default=100, type=int, help='Number of nodes')
@click.option('--seed-emotion', default='joy', help='Emotion to seed')
@click.option('--steps', default=50, type=int, help='Simulation steps')
def simulate(network_type, n_nodes, seed_emotion, steps):
    """Simulate emotional contagion in network."""
    click.echo(f"Simulating {seed_emotion} contagion in {network_type} network...")
    
    # Create network
    network = create_network(network_type, n_nodes)
    
    # Seed emotion in central node
    central_agents = network.identify_central_agents(top_k=1)
    seed_agent_id = central_agents[0][0]
    
    seed_agent = network.get_agent(seed_agent_id)
    if seed_emotion == 'joy':
        seed_agent.set_emotion({'valence': 0.8, 'arousal': 0.6})
    elif seed_emotion == 'anger':
        seed_agent.set_emotion({'valence': -0.7, 'arousal': 0.8})
    elif seed_emotion == 'sadness':
        seed_agent.set_emotion({'valence': -0.7, 'arousal': -0.3})
    
    # Create propagator
    propagator = EmotionPropagator(network, decay_rate=0.1, transmission_rate=0.3)
    
    # Simulate
    click.echo(f"\nSimulating {steps} steps...")
    history = propagator.simulate(steps)
    
    # Analyze
    analysis = propagator.analyze_spread()
    
    click.echo("\n" + "="*60)
    click.echo("Simulation Results")
    click.echo("="*60)
    click.echo(f"Infection rate: {analysis['infection_rate']:.2%}")
    click.echo(f"Infected agents: {analysis['n_infected']}/{analysis['total_agents']}")
    click.echo(f"Peak time: {analysis['peak_time']}")
    
    # Final network state
    final_state = network.get_network_emotion_state()
    click.echo(f"\nFinal network state:")
    click.echo(f"  Mean valence: {final_state['mean_valence']:+.2f}")
    click.echo(f"  Mean arousal: {final_state['mean_arousal']:+.2f}")
    click.echo(f"  Std valence: {final_state['std_valence']:.2f}")


@cli.command()
@click.option('--network-size', default=1000, type=int, help='Network size')
@click.option('--emotion', default='anger', help='Emotion type')
@click.option('--intervention', default='none', help='Intervention type (none/targeted/random)')
def epidemic(network_size, emotion, intervention):
    """Run emotional epidemiology simulation."""
    click.echo(f"Running {emotion} epidemic on network of {network_size} nodes...")
    
    # Create network
    network = create_network('scale_free', network_size, m=3)
    
    # Create epidemiology simulator
    epi = EmotionalEpidemiology(
        network=network,
        emotion_type=emotion,
        transmission_rate=0.3,
        recovery_rate=0.1
    )
    
    # Seed emotion
    patient_zeros = [str(i) for i in range(5)]
    epi.seed_emotion(patient_zeros, intensity=0.9)
    
    click.echo(f"R₀ = {epi.compute_r0():.2f}")
    
    # Simulate
    click.echo("\nSimulating epidemic...")
    history = epi.simulate(steps=100)
    
    # Apply intervention if requested
    if intervention == 'targeted':
        click.echo("\nApplying targeted intervention...")
        influencers = epi.identify_influencers(top_k=20)
        epi.apply_intervention(influencers, intervention_type='regulation', effectiveness=0.7)
        
        # Continue simulation
        history_post = epi.simulate(steps=50)
    
    # Get peak info
    peak_info = epi.get_peak_info()
    
    click.echo("\n" + "="*60)
    click.echo("Epidemic Results")
    click.echo("="*60)
    click.echo(f"Peak infected: {peak_info['peak_count']} ({peak_info['peak_proportion']:.1%})")
    click.echo(f"Peak time: {peak_info['peak_time']}")
    
    if intervention != 'none':
        click.echo(f"\nIntervention: {intervention}")
        click.echo("Post-intervention simulation completed")


@cli.command()
@click.option('--scenario', default='basic', help='ToM scenario (basic/false-belief)')
@click.option('--recursion-depth', default=2, type=int, help='Recursion depth')
def tom(scenario, recursion_depth):
    """Test Theory of Mind reasoning."""
    click.echo(f"Testing Theory of Mind: {scenario} scenario...")
    
    # Create agents
    alice = Agent('alice', initial_emotion={'valence': 0.3, 'arousal': 0.2})
    bob = Agent('bob', initial_emotion={'valence': -0.4, 'arousal': 0.5})
    
    # Create ToM module
    tom_module = TheoryOfMindModule(max_recursion=recursion_depth)
    
    if scenario == 'basic':
        # First-order ToM
        click.echo("\nFirst-order ToM: Alice infers Bob's emotion")
        
        alice_belief = tom_module.infer_other_emotion(
            alice, bob,
            observed_behavior={'facial_expression': 'frown', 'voice_tone': 'tense'}
        )
        
        click.echo(f"Bob's actual emotion: {bob.emotional_state.emotion_type.value}")
        click.echo(f"  Valence: {bob.emotional_state.valence:+.2f}")
        click.echo(f"  Arousal: {bob.emotional_state.arousal:+.2f}")
        
        click.echo(f"\nAlice's belief about Bob:")
        click.echo(f"  Valence: {alice_belief.valence:+.2f}")
        click.echo(f"  Arousal: {alice_belief.arousal:+.2f}")
        
    elif scenario == 'false-belief':
        # False belief reasoning
        click.echo("\nFalse belief scenario:")
        
        true_state = {'valence': 0.5}
        bob_belief = {'valence': -0.3}  # Bob has false belief
        
        understands = tom_module.false_belief_reasoning(
            alice, bob, true_state, bob_belief
        )
        
        click.echo(f"True state: valence = {true_state['valence']:+.2f}")
        click.echo(f"Bob's belief: valence = {bob_belief['valence']:+.2f}")
        click.echo(f"Alice understands Bob has false belief: {understands}")
    
    # Recursive mentalizing
    if recursion_depth > 1:
        click.echo(f"\nRecursive mentalizing (depth={recursion_depth}):")
        mental_state = tom_module.recursive_mentalizing(alice, bob, depth=recursion_depth)
        
        click.echo(f"Alice's nested belief structure:")
        click.echo(f"  Depth 0: {mental_state.emotion.emotion_type.value}")
        if mental_state.belief_about:
            click.echo(f"  Has nested beliefs about {len(mental_state.belief_about)} agents")


@cli.command()
@click.option('--cultures', default='western,eastern', help='Cultures to compare')
@click.option('--emotion', default='anger', help='Emotion to test')
def culture(cultures, emotion):
    """Compare cultural emotion norms."""
    click.echo(f"Comparing cultural norms for {emotion}...")
    
    culture_list = cultures.split(',')
    
    # Define cultural contexts
    cultural_norms = {}
    for culture_name in culture_list:
        if culture_name == 'western':
            cultural_norms[culture_name] = CulturalNorms(
                culture='western',
                individualism=0.8,
                power_distance=0.3
            )
        elif culture_name == 'eastern':
            cultural_norms[culture_name] = CulturalNorms(
                culture='eastern',
                individualism=0.3,
                power_distance=0.7
            )
        elif culture_name == 'african':
            cultural_norms[culture_name] = CulturalNorms(
                culture='african',
                individualism=0.4,
                power_distance=0.6
            )
    
    # Test emotion expression
    raw_emotion = {'valence': -0.6, 'arousal': 0.8, 'intensity': 0.9}
    
    click.echo("\n" + "="*60)
    click.echo(f"Raw emotion: valence={raw_emotion['valence']:+.2f}, "
              f"arousal={raw_emotion['arousal']:+.2f}")
    click.echo("="*60)
    
    for culture_name, norms in cultural_norms.items():
        modulated = norms.modulate_expression(raw_emotion)
        
        click.echo(f"\n{culture_name.title()} culture:")
        click.echo(f"  Individualism: {norms.individualism:.2f}")
        click.echo(f"  Expression intensity: {modulated['intensity']:.2f}")
        click.echo(f"  Change: {(modulated['intensity'] - raw_emotion['intensity']):.2f}")


@cli.command()
def demo():
    """Run interactive demonstration."""
    click.echo("="*60)
    click.echo("SARN Interactive Demo")
    click.echo("="*60)
    
    # 1. Create network
    click.echo("\n1. Creating social network...")
    network = create_network('small_world', n_nodes=50, k=6, p=0.3)
    metrics = network.compute_network_metrics()
    
    click.echo(f"   Network: {metrics['n_nodes']} nodes, {metrics['n_edges']} edges")
    click.echo(f"   Clustering: {metrics['clustering']:.3f}")
    click.echo(f"   Density: {metrics['density']:.3f}")
    
    # 2. Emotional contagion
    click.echo("\n2. Simulating emotional contagion...")
    
    # Seed joy in one agent
    network.get_agent('0').set_emotion({'valence': 0.8, 'arousal': 0.6})
    
    propagator = EmotionPropagator(network, decay_rate=0.1, transmission_rate=0.3)
    history = propagator.simulate(30)
    
    analysis = propagator.analyze_spread()
    click.echo(f"   Infection rate: {analysis['infection_rate']:.1%}")
    
    # 3. Theory of Mind
    click.echo("\n3. Testing Theory of Mind...")
    
    alice = network.get_agent('0')
    bob = network.get_agent('1')
    
    tom = TheoryOfMindModule()
    belief = tom.infer_other_emotion(alice, bob)
    
    click.echo(f"   Alice's belief about Bob: {belief.emotion_type.value}")
    
    # 4. Cultural comparison
    click.echo("\n4. Cultural norm comparison...")
    
    western = CulturalNorms(culture='western', individualism=0.8)
    eastern = CulturalNorms(culture='eastern', individualism=0.3)
    
    emotion = {'valence': -0.5, 'arousal': 0.7, 'intensity': 0.8}
    
    western_expr = western.modulate_expression(emotion)
    eastern_expr = eastern.modulate_expression(emotion)
    
    click.echo(f"   Western expression: {western_expr['intensity']:.2f}")
    click.echo(f"   Eastern expression: {eastern_expr['intensity']:.2f}")
    
    click.echo("\n" + "="*60)
    click.echo("Demo complete!")
    click.echo("="*60)


if __name__ == '__main__':
    cli()
