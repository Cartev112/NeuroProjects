"""
Emotional trajectory prediction demo using Echo State Networks.

Demonstrates training a reservoir computing model to forecast
emotional states from multimodal biosignals.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reservoir import EmotionalPredictor, MultimodalEmotionalPredictor
from src.utils import load_example_data, generate_emotional_trajectory


def main():
    print("="*60)
    print("Emotional Trajectory Prediction Demo")
    print("="*60)
    
    # Generate synthetic emotional trajectory
    print("\n1. Generating synthetic emotional data...")
    trajectory = generate_emotional_trajectory(
        initial_valence=-0.3,
        initial_arousal=0.2,
        duration=2000,
        volatility=0.1,
        drift_to_neutral=0.01
    )
    
    print(f"   Generated {len(trajectory)} time steps")
    print(f"   Mean valence: {np.mean(trajectory[:, 0]):.3f}")
    print(f"   Mean arousal: {np.mean(trajectory[:, 1]):.3f}")
    
    # Split into train and test
    train_size = 1500
    train_data = trajectory[:train_size]
    test_data = trajectory[train_size:]
    
    # Create and train predictor
    print("\n2. Training Echo State Network predictor...")
    predictor = EmotionalPredictor(
        reservoir_size=500,
        spectral_radius=0.95,
        input_scaling=0.5,
        leak_rate=0.3,
        random_seed=42
    )
    
    # Use trajectory as both input and target (autoregressive)
    predictor.train(
        train_data[:-1],  # Input: t
        train_data[1:],   # Target: t+1
        washout=100,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n3. Evaluating on test set...")
    metrics = predictor.evaluate(
        test_data[:-1],
        test_data[1:],
        washout=50
    )
    
    print(f"   Test RMSE: {metrics['rmse']:.4f}")
    print(f"   Test MAE: {metrics['mae']:.4f}")
    print(f"   Test R²: {metrics['r2']:.4f}")
    
    # Make predictions
    print("\n4. Generating predictions...")
    
    # Short-term prediction (next 100 steps)
    horizon_short = 100
    predictions_short = predictor.predict(
        test_data[0],
        horizon=horizon_short,
        autonomous=True
    )
    
    # Long-term prediction (next 300 steps)
    horizon_long = 300
    predictions_long = predictor.predict(
        test_data[0],
        horizon=horizon_long,
        autonomous=True
    )
    
    # Visualize results
    print("\n5. Generating visualizations...")
    
    # Plot 1: Training data
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax = axes[0, 0]
    time_train = np.arange(len(train_data))
    ax.plot(time_train, train_data[:, 0], 'b-', alpha=0.7, label='Valence')
    ax.plot(time_train, train_data[:, 1], 'r-', alpha=0.7, label='Arousal')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Training Data', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Short-term prediction
    ax = axes[0, 1]
    time_pred = np.arange(horizon_short)
    ax.plot(time_pred, predictions_short[:, 0], 'b-', linewidth=2, 
           label='Predicted Valence', alpha=0.8)
    ax.plot(time_pred, test_data[1:horizon_short+1, 0], 'b--', linewidth=2,
           label='True Valence', alpha=0.6)
    ax.plot(time_pred, predictions_short[:, 1], 'r-', linewidth=2,
           label='Predicted Arousal', alpha=0.8)
    ax.plot(time_pred, test_data[1:horizon_short+1, 1], 'r--', linewidth=2,
           label='True Arousal', alpha=0.6)
    ax.set_xlabel('Steps Ahead', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Short-term Prediction ({horizon_short} steps)', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Long-term prediction
    ax = axes[1, 0]
    time_pred_long = np.arange(horizon_long)
    ax.plot(time_pred_long, predictions_long[:, 0], 'b-', linewidth=2,
           label='Predicted Valence')
    ax.plot(time_pred_long, predictions_long[:, 1], 'r-', linewidth=2,
           label='Predicted Arousal')
    ax.set_xlabel('Steps Ahead', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Long-term Prediction ({horizon_long} steps)', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Phase space comparison
    ax = axes[1, 1]
    ax.plot(test_data[:horizon_short, 0], test_data[:horizon_short, 1],
           'k--', linewidth=2, alpha=0.5, label='True Trajectory')
    ax.plot(predictions_short[:, 0], predictions_short[:, 1],
           'b-', linewidth=2, alpha=0.7, label='Predicted Trajectory')
    ax.plot(test_data[0, 0], test_data[0, 1], 'go', markersize=12, label='Start')
    ax.set_xlabel('Valence', fontsize=11)
    ax.set_ylabel('Arousal', fontsize=11)
    ax.set_title('Phase Space Prediction', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Multimodal prediction demo
    print("\n6. Demonstrating multimodal prediction...")
    
    # Load multimodal data
    multimodal_data = load_example_data('synthetic')
    
    # Create multimodal predictor
    multimodal_predictor = MultimodalEmotionalPredictor(
        reservoir_size=400,
        spectral_radius=0.95
    )
    
    # Prepare biosignals
    biosignals = {
        'hrv': multimodal_data['hrv'][:1500],
        'eda': multimodal_data['eda'][:1500],
        'pupil': multimodal_data['pupil'][:1500]
    }
    
    emotional_states = multimodal_data['emotional_states'][:1500]
    
    # Train
    print("   Training multimodal predictor...")
    multimodal_predictor.train(biosignals, emotional_states, washout=100, verbose=False)
    
    # Predict
    test_biosignals = {
        'hrv': multimodal_data['hrv'][1500:1600],
        'eda': multimodal_data['eda'][1500:1600],
        'pupil': multimodal_data['pupil'][1500:1600]
    }
    
    multimodal_predictions = multimodal_predictor.predict(
        test_biosignals,
        horizon=100,
        autonomous=False
    )
    
    # Plot modality importance
    fig2, _ = multimodal_predictor.plot_modality_importance()
    
    print("\n" + "="*60)
    print("Prediction demo complete! Close plots to exit.")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    main()
