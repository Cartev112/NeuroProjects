#!/usr/bin/env python3
"""
Command-line interface for Affective Geometry analysis.

Provides tools for analyzing emotional dynamics, computing Lyapunov exponents,
training predictors, and optimizing interventions.
"""

import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.state_space import EmotionalStateSpace
from src.attractors import Attractor, AttractorType
from src.recurrence import RecurrenceAnalysis, compare_rqa_groups
from src.lyapunov import LyapunovAnalysis, compare_lyapunov_groups
from src.bifurcation import BifurcationAnalysis, BasinStabilityAnalysis
from src.topology import TopologicalAnalysis
from src.reservoir import EmotionalPredictor, MultimodalEmotionalPredictor
from src.intervention import InterventionOptimizer, suggest_intervention_strategy
from src.utils import (
    create_standard_attractors,
    load_example_data,
    generate_emotional_trajectory,
    save_results,
    load_results
)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    Affective Geometry: Dynamical Systems Mapping of Emotional Landscapes
    
    A comprehensive framework for modeling emotions as trajectories through
    high-dimensional state spaces.
    """
    pass


@cli.command()
@click.option('--duration', default=100.0, help='Simulation duration')
@click.option('--initial-state', default='0,0,0,0,0', help='Initial state (comma-separated)')
@click.option('--output', '-o', help='Output directory for plots')
@click.option('--show/--no-show', default=True, help='Show plots')
def simulate(duration, initial_state, output, show):
    """Simulate emotional trajectory in state space."""
    click.echo("Simulating emotional trajectory...")
    
    # Parse initial state
    initial = np.array([float(x) for x in initial_state.split(',')])
    
    # Create state space with standard attractors
    state_space = EmotionalStateSpace(dimensions=5)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    # Simulate
    trajectory = state_space.simulate_trajectory(initial, duration=duration)
    
    # Plot
    fig1, _ = state_space.plot_phase_space(trajectory)
    fig2, _ = state_space.plot_time_series(trajectory)
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / 'phase_space.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
        click.echo(f"Plots saved to {output}")
    
    if show:
        plt.show()
    
    click.echo("✓ Simulation complete")


@cli.command()
@click.option('--input', '-i', required=True, help='Input data file (CSV)')
@click.option('--embedding-dim', default=3, help='Embedding dimension')
@click.option('--time-delay', default=10, help='Time delay')
@click.option('--output', '-o', help='Output directory')
def rqa(input, embedding_dim, time_delay, output):
    """Perform Recurrence Quantification Analysis."""
    click.echo(f"Performing RQA on {input}...")
    
    # Load data
    data = np.loadtxt(input, delimiter=',')
    if data.ndim > 1:
        data = data[:, 0]  # Use first column
    
    # Perform RQA
    rqa_analyzer = RecurrenceAnalysis(
        embedding_dim=embedding_dim,
        time_delay=time_delay
    )
    
    metrics = rqa_analyzer.analyze(data)
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo("RQA Results")
    click.echo("="*60)
    click.echo(str(metrics))
    
    # Plot
    fig1, _ = rqa_analyzer.plot_recurrence_plot()
    fig2, _ = rqa_analyzer.plot_embedded_space()
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / 'recurrence_plot.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'embedded_space.png', dpi=300, bbox_inches='tight')
        
        # Save metrics
        results = {
            'recurrence_rate': metrics.recurrence_rate,
            'determinism': metrics.determinism,
            'laminarity': metrics.laminarity,
            'entropy_diagonal': metrics.entropy_diagonal,
            'trapping_time': metrics.trapping_time
        }
        save_results(results, output_dir / 'rqa_metrics.json')
        click.echo(f"Results saved to {output}")
    
    plt.show()
    click.echo("✓ RQA analysis complete")


@cli.command()
@click.option('--initial-state', default='0.1,0.1,0,0,0', help='Initial state')
@click.option('--duration', default=100.0, help='Simulation duration')
@click.option('--output', '-o', help='Output directory')
def lyapunov(initial_state, duration, output):
    """Calculate Lyapunov exponent for emotional dynamics."""
    click.echo("Calculating Lyapunov exponent...")
    
    # Parse initial state
    initial = np.array([float(x) for x in initial_state.split(',')])
    
    # Create state space
    state_space = EmotionalStateSpace(dimensions=5)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    # Calculate Lyapunov exponent
    analyzer = LyapunovAnalysis(state_space=state_space)
    exponent = analyzer.calculate_largest_exponent(
        initial_state=initial,
        duration=duration
    )
    
    # Classify dynamics
    classification = analyzer.classify_dynamics(exponent)
    
    click.echo("\n" + "="*60)
    click.echo("Lyapunov Exponent Analysis")
    click.echo("="*60)
    click.echo(f"Largest Lyapunov Exponent: {exponent:.6f}")
    click.echo(f"Classification: {classification}")
    click.echo("="*60)
    
    # Plot evolution
    fig, _ = analyzer.plot_exponent_evolution(initial, duration=duration)
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'lyapunov_evolution.png', dpi=300, bbox_inches='tight')
        
        results = {
            'largest_exponent': exponent,
            'classification': classification
        }
        save_results(results, output_dir / 'lyapunov_results.json')
        click.echo(f"Results saved to {output}")
    
    plt.show()
    click.echo("✓ Lyapunov analysis complete")


@cli.command()
@click.option('--parameter', default='noise_level', help='Parameter to vary')
@click.option('--range', 'param_range', default='0,0.5', help='Parameter range (min,max)')
@click.option('--output', '-o', help='Output directory')
def bifurcation(parameter, param_range, output):
    """Perform bifurcation analysis."""
    click.echo(f"Performing bifurcation analysis on {parameter}...")
    
    # Parse range
    min_val, max_val = [float(x) for x in param_range.split(',')]
    
    # Create state space
    state_space = EmotionalStateSpace(dimensions=5)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    # Bifurcation analysis
    analyzer = BifurcationAnalysis(state_space=state_space)
    
    fig, _ = analyzer.plot_bifurcation_diagram(
        parameter,
        (min_val, max_val),
        dimension_to_plot=0,
        dimension_name='Valence'
    )
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'bifurcation_diagram.png', dpi=300, bbox_inches='tight')
        click.echo(f"Plot saved to {output}")
    
    plt.show()
    click.echo("✓ Bifurcation analysis complete")


@cli.command()
@click.option('--dataset', default='synthetic', help='Dataset to use')
@click.option('--horizon', default=100, help='Prediction horizon')
@click.option('--output', '-o', help='Output directory')
def train_predictor(dataset, horizon, output):
    """Train emotional trajectory predictor."""
    click.echo(f"Training predictor on {dataset} dataset...")
    
    # Load data
    data = load_example_data(dataset)
    
    if 'emotional_states' in data:
        # Multimodal data
        biosignals = {
            'hrv': data['hrv'],
            'eda': data['eda'],
            'pupil': data['pupil']
        }
        
        predictor = MultimodalEmotionalPredictor(
            reservoir_size=300,
            spectral_radius=0.95
        )
        
        # Train
        predictor.train(biosignals, data['emotional_states'], washout=100)
        
        # Test prediction
        test_biosignals = {k: v[:1000] for k, v in biosignals.items()}
        predictions = predictor.predict(test_biosignals, horizon=horizon)
        
        # Plot
        fig, _ = predictor.predictor.plot_forecast(
            predictions,
            ground_truth=data['emotional_states'][1000:1000+horizon]
        )
        
    else:
        # Simple trajectory data
        trajectory = data['trajectory']
        
        predictor = EmotionalPredictor(reservoir_size=300)
        
        # Use trajectory as both input and target (autoregressive)
        predictor.train(trajectory[:-1], trajectory[1:], washout=100)
        
        # Predict
        predictions = predictor.predict(trajectory[0], horizon=horizon, autonomous=True)
        
        fig, _ = predictor.plot_forecast(predictions, ground_truth=trajectory[1:horizon+1])
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'prediction.png', dpi=300, bbox_inches='tight')
        click.echo(f"Results saved to {output}")
    
    plt.show()
    click.echo("✓ Predictor training complete")


@cli.command()
@click.option('--current-state', required=True, help='Current state (comma-separated)')
@click.option('--target', default='happiness', help='Target attractor name')
@click.option('--method', default='impulse', help='Intervention method (impulse/sustained)')
@click.option('--output', '-o', help='Output directory')
def optimize_intervention(current_state, target, method, output):
    """Find optimal intervention to reach target emotional state."""
    click.echo(f"Optimizing {method} intervention to reach {target}...")
    
    # Parse current state
    current = np.array([float(x) for x in current_state.split(',')])
    
    # Create state space
    state_space = EmotionalStateSpace(dimensions=5)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    
    # Find target attractor
    target_attractor = None
    for attr in attractors:
        if attr.name == target:
            target_attractor = attr
            break
    
    if target_attractor is None:
        click.echo(f"Error: Target attractor '{target}' not found")
        return
    
    # Optimize intervention
    optimizer = InterventionOptimizer(state_space)
    
    with click.progressbar(length=100, label='Optimizing') as bar:
        result = optimizer.find_optimal_intervention(
            current,
            target_attractor,
            method=method,
            n_trials=100
        )
        bar.update(100)
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo("Intervention Optimization Results")
    click.echo("="*60)
    click.echo(str(result))
    click.echo("="*60)
    
    # Suggest strategy
    suggestion = suggest_intervention_strategy(current)
    click.echo("\nPersonalized Recommendations:")
    click.echo(suggestion)
    
    # Plot
    fig, _ = optimizer.plot_intervention_trajectory(
        result, current, target_attractor
    )
    
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'intervention.png', dpi=300, bbox_inches='tight')
        
        results = {
            'action': result.action.tolist(),
            'time_to_target': result.time_to_target,
            'success_prob': result.success_prob,
            'energy_cost': result.energy_cost,
            'intervention_type': result.intervention_type
        }
        save_results(results, output_dir / 'intervention_results.json')
        click.echo(f"Results saved to {output}")
    
    plt.show()
    click.echo("✓ Intervention optimization complete")


@cli.command()
def demo():
    """Run interactive demonstration of all features."""
    click.echo("="*60)
    click.echo("Affective Geometry - Interactive Demo")
    click.echo("="*60)
    
    # Create state space
    click.echo("\n1. Creating emotional state space with attractors...")
    state_space = EmotionalStateSpace(dimensions=5)
    attractors = create_standard_attractors()
    state_space.add_attractors(attractors)
    click.echo(f"   ✓ Created {len(attractors)} emotional attractors")
    
    # Simulate trajectory
    click.echo("\n2. Simulating emotional trajectory...")
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    trajectory = state_space.simulate_trajectory(initial_state, duration=50.0)
    click.echo(f"   ✓ Simulated {len(trajectory)} time steps")
    
    # Lyapunov analysis
    click.echo("\n3. Computing Lyapunov exponent...")
    lyap_analyzer = LyapunovAnalysis(state_space=state_space)
    exponent = lyap_analyzer.calculate_largest_exponent(trajectory=trajectory)
    click.echo(f"   ✓ Largest Lyapunov exponent: {exponent:.6f}")
    click.echo(f"   ✓ {lyap_analyzer.classify_dynamics(exponent)}")
    
    # Basin stability
    click.echo("\n4. Computing basin stability...")
    basin_analyzer = BasinStabilityAnalysis(state_space)
    stabilities = basin_analyzer.compute_all_basin_stabilities(n_samples=100)
    click.echo("   ✓ Basin stabilities:")
    for name, stability in stabilities.items():
        click.echo(f"      {name}: {stability:.2%}")
    
    # Intervention
    click.echo("\n5. Optimizing intervention...")
    optimizer = InterventionOptimizer(state_space)
    target = attractors[0]  # Happiness
    current = np.array([-0.5, -0.3, -0.2, -0.4, 0.0])
    
    result = optimizer.find_optimal_intervention(
        current, target, method='impulse', n_trials=50
    )
    click.echo(f"   ✓ Intervention optimized")
    click.echo(f"      Success probability: {result.success_prob:.1%}")
    click.echo(f"      Energy cost: {result.energy_cost:.4f}")
    
    # Visualize
    click.echo("\n6. Generating visualizations...")
    fig1, _ = state_space.plot_phase_space(trajectory)
    fig2, _ = basin_analyzer.plot_basin_stabilities(stabilities)
    fig3, _ = optimizer.plot_intervention_trajectory(result, current, target)
    
    click.echo("\n" + "="*60)
    click.echo("Demo complete! Close plots to exit.")
    click.echo("="*60)
    
    plt.show()


if __name__ == '__main__':
    cli()
