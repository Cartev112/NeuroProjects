# Quick Start Guide

## Installation

```bash
cd affective-geometry
pip install -r requirements.txt
```

## Running Examples

### 1. Basic Simulation
Simulate emotional trajectories through state space:

```bash
python examples/basic_simulation.py
```

This demonstrates:
- Creating emotional state space with attractors
- Simulating trajectories from different initial conditions
- Visualizing phase space and basins of attraction

### 2. Prediction Demo
Train echo state network to forecast emotional trajectories:

```bash
python examples/prediction_demo.py
```

This demonstrates:
- Training reservoir computing predictor
- Short-term and long-term forecasting
- Multimodal biosignal integration

### 3. Intervention Optimization
Find optimal interventions to shift emotional states:

```bash
python examples/intervention_optimization.py
```

This demonstrates:
- Optimizing impulse vs sustained interventions
- Comparing strategies across different scenarios
- Personalized intervention recommendations

## CLI Usage

### Interactive Demo
Run the full interactive demo:

```bash
python affective_cli.py demo
```

### Simulate Trajectory
```bash
python affective_cli.py simulate --duration 100 --initial-state "0,0,0,0,0" --output results/
```

### Recurrence Analysis
```bash
python affective_cli.py rqa --input data/signal.csv --embedding-dim 3 --output results/
```

### Lyapunov Exponent
```bash
python affective_cli.py lyapunov --initial-state "0.1,0.1,0,0,0" --duration 100 --output results/
```

### Bifurcation Analysis
```bash
python affective_cli.py bifurcation --parameter noise_level --range "0,0.5" --output results/
```

### Train Predictor
```bash
python affective_cli.py train-predictor --dataset synthetic --horizon 100 --output results/
```

### Optimize Intervention
```bash
python affective_cli.py optimize-intervention \
    --current-state "-0.6,-0.3,-0.4,-0.5,0.0" \
    --target happiness \
    --method impulse \
    --output results/
```

## Python API Examples

### Create State Space and Simulate

```python
from src.state_space import EmotionalStateSpace
from src.utils import create_standard_attractors
import numpy as np

# Create state space
state_space = EmotionalStateSpace(dimensions=5)
attractors = create_standard_attractors()
state_space.add_attractors(attractors)

# Simulate trajectory
initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
trajectory = state_space.simulate_trajectory(initial_state, duration=100.0)

# Visualize
state_space.plot_phase_space(trajectory)
```

### Recurrence Analysis

```python
from src.recurrence import RecurrenceAnalysis
import numpy as np

# Load or generate signal
signal = np.random.randn(1000)

# Perform RQA
rqa = RecurrenceAnalysis(embedding_dim=3, time_delay=10)
metrics = rqa.analyze(signal)

print(f"Determinism: {metrics.determinism:.3f}")
print(f"Laminarity: {metrics.laminarity:.3f}")

# Plot
rqa.plot_recurrence_plot()
```

### Train Predictor

```python
from src.reservoir import EmotionalPredictor
import numpy as np

# Generate data
trajectory = np.random.randn(1000, 2)

# Create and train predictor
predictor = EmotionalPredictor(reservoir_size=500)
predictor.train(trajectory[:-1], trajectory[1:], washout=100)

# Predict
predictions = predictor.predict(trajectory[0], horizon=100, autonomous=True)
```

### Optimize Intervention

```python
from src.state_space import EmotionalStateSpace
from src.intervention import InterventionOptimizer
from src.utils import create_standard_attractors
import numpy as np

# Setup
state_space = EmotionalStateSpace(dimensions=5)
attractors = create_standard_attractors()
state_space.add_attractors(attractors)

# Find target
happiness = [a for a in attractors if a.name == 'happiness'][0]

# Optimize
optimizer = InterventionOptimizer(state_space)
current_state = np.array([-0.6, -0.3, -0.4, -0.5, 0.0])

result = optimizer.find_optimal_intervention(
    current_state,
    happiness,
    method='impulse'
)

print(f"Success probability: {result.success_prob:.1%}")
print(f"Energy cost: {result.energy_cost:.4f}")
```

## Key Concepts

### Emotional State Space
- **Dimensions**: Valence, Arousal, Dominance, Approach/Avoidance, Temporal
- **Attractors**: Stable emotional states (happiness, sadness, anger, etc.)
- **Trajectories**: Paths through state space representing emotional dynamics

### Analysis Methods

1. **Recurrence Quantification Analysis (RQA)**
   - Identifies patterns in emotional time series
   - Measures stability, transitions, and chaos

2. **Lyapunov Exponents**
   - Quantifies emotional volatility
   - Positive = chaotic, Negative = stable

3. **Bifurcation Analysis**
   - Maps how parameters affect emotional landscape
   - Identifies critical transitions

4. **Topological Data Analysis**
   - Finds persistent emotional structures
   - Compares patterns across individuals

5. **Reservoir Computing**
   - Forecasts future emotional states
   - Integrates multimodal biosignals

6. **Intervention Optimization**
   - Finds minimal-energy interventions
   - Shifts from negative to positive states

## Next Steps

- Explore the full API documentation in README.md
- Modify examples for your own data
- Experiment with different attractor configurations
- Try different intervention strategies
- Integrate with real biosignal data

## Troubleshooting

### Import Errors
Make sure you're running from the project directory and have installed all dependencies:
```bash
pip install -r requirements.txt
```

### Slow Optimization
Reduce `n_trials` parameter for faster (but less optimal) results:
```python
result = optimizer.find_optimal_intervention(..., n_trials=20)
```

### Memory Issues
Reduce `reservoir_size` or trajectory length:
```python
predictor = EmotionalPredictor(reservoir_size=200)
```

## Support

For issues or questions, please refer to the main README.md or open an issue on GitHub.
