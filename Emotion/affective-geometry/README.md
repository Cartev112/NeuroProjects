# Affective Geometry: Dynamical Systems Mapping of Emotional Landscapes

A comprehensive computational framework that models emotions as trajectories through high-dimensional state spaces using nonlinear dynamical systems theory, recurrence analysis, and reservoir computing.

## Theoretical Background

This project implements a novel approach to emotion science by treating emotions not as discrete categories but as continuous trajectories through a multidimensional phase space. Drawing from affective neuroscience, dynamical systems theory, and computational psychiatry, we model:

- **Emotional attractors**: Stable emotional states (happiness, sadness, anger) as attractors in phase space
- **Transitions**: Emotional shifts as trajectories between attractors
- **Stability**: Individual differences in emotional volatility via Lyapunov exponents
- **Regulation**: Interventions as perturbations that shift trajectories toward positive attractors

## Key Features

### 1. Nonlinear Dynamical Systems Modeling
- Multi-dimensional phase space (valence, arousal, dominance, approach/avoidance)
- Attractor identification and characterization
- Saddle points and limit cycles for emotional transitions
- Vector field reconstruction from longitudinal data

### 2. Recurrence Quantification Analysis (RQA)
- Analysis of physiological signals: HRV, EDA, pupillometry, facial EMG
- Identification of emotional stability, transitions, and chaotic dynamics
- Recurrence plots and quantification measures (determinism, laminarity, entropy)

### 3. Lyapunov Exponent Calculation
- Quantifies emotional stability vs. volatility
- Compares clinical populations (depression, anxiety, bipolar disorder)
- Identifies sensitive dependence on initial conditions

### 4. Bifurcation Analysis
- Maps how parameters (stress, medication, context) shift emotional landscape topology
- Identifies critical transitions and tipping points
- Basin of attraction mapping with personalized escape times

### 5. Topological Data Analysis
- Persistent homology for identifying emotional structures
- Cross-individual comparison of emotional topology
- Mapper algorithm for visualization

### 6. Reservoir Computing for Prediction
- Echo state networks for forecasting emotional trajectories
- Real-time multimodal biosignal integration
- Hours-ahead prediction of emotional states

### 7. Optimal Intervention Calculator
- Perturbation experiments to "nudge" emotional trajectories
- Calculates minimal interventions to move from negative to positive attractors
- Personalized emotion regulation strategies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Emotional State Space Modeling

```python
from affective_geometry import EmotionalStateSpace, Attractor

# Create 5D emotional state space (valence, arousal, dominance, approach, temporal)
state_space = EmotionalStateSpace(dimensions=5)

# Define attractors for basic emotions
happiness = Attractor(center=[0.8, 0.6, 0.7, 0.8, 0.0], strength=1.5, name="happiness")
sadness = Attractor(center=[-0.7, -0.4, -0.5, -0.6, 0.0], strength=1.2, name="sadness")
anger = Attractor(center=[-0.6, 0.8, 0.6, 0.7, 0.0], strength=1.3, name="anger")

state_space.add_attractors([happiness, sadness, anger])

# Simulate emotional trajectory
initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]
trajectory = state_space.simulate_trajectory(initial_state, duration=100, dt=0.1)

# Visualize
state_space.plot_phase_space(trajectory)
```

### Recurrence Quantification Analysis

```python
from affective_geometry import RecurrenceAnalysis

# Load physiological data
hrv_data = load_hrv_signal("subject_01.csv")

# Perform RQA
rqa = RecurrenceAnalysis(embedding_dim=3, time_delay=10, threshold=0.2)
metrics = rqa.analyze(hrv_data)

print(f"Recurrence Rate: {metrics['recurrence_rate']:.3f}")
print(f"Determinism: {metrics['determinism']:.3f}")
print(f"Laminarity: {metrics['laminarity']:.3f}")
print(f"Entropy: {metrics['entropy']:.3f}")

# Plot recurrence plot
rqa.plot_recurrence_plot()
```

### Lyapunov Exponent Calculation

```python
from affective_geometry import LyapunovAnalysis

# Calculate largest Lyapunov exponent
lyap = LyapunovAnalysis(state_space)
exponent = lyap.calculate_largest_exponent(trajectory)

if exponent > 0:
    print("Chaotic emotional dynamics detected")
elif exponent < 0:
    print("Stable emotional dynamics")
else:
    print("Marginally stable dynamics")
```

### Emotional Weather Prediction

```python
from affective_geometry import EmotionalPredictor

# Create reservoir computing predictor
predictor = EmotionalPredictor(
    reservoir_size=500,
    spectral_radius=0.95,
    input_scaling=0.5,
    leak_rate=0.3
)

# Train on historical multimodal data
predictor.train(biosignal_data, emotional_labels, washout=100)

# Predict future emotional trajectory
current_state = get_current_biosignals()
future_trajectory = predictor.predict(current_state, horizon=360)  # 6 hours ahead

# Visualize prediction
predictor.plot_forecast(future_trajectory)
```

### Optimal Intervention

```python
from affective_geometry import InterventionOptimizer

# Find minimal intervention to shift from negative to positive attractor
optimizer = InterventionOptimizer(state_space)

current_state = [-0.6, -0.3, -0.4, -0.5, 0.0]  # Sad state
target_attractor = happiness

intervention = optimizer.find_optimal_intervention(
    current_state,
    target_attractor,
    max_magnitude=1.0,
    time_horizon=50
)

print(f"Optimal intervention: {intervention['action']}")
print(f"Expected time to target: {intervention['time_to_target']:.1f} steps")
print(f"Success probability: {intervention['success_prob']:.2f}")
```

## CLI Interface

```bash
# Analyze emotional dynamics from physiological data
python affective_cli.py analyze --input data/subject_01.csv --output results/

# Calculate Lyapunov exponents for a cohort
python affective_cli.py lyapunov --cohort data/depression_group/ --control data/healthy_group/

# Train emotional predictor
python affective_cli.py train-predictor --data data/training_set/ --model models/predictor.pkl

# Run intervention optimization
python affective_cli.py optimize-intervention --current-state "[-0.6,-0.3,-0.4,-0.5,0.0]" --target happiness
```

## Project Structure

```
affective-geometry/
├── README.md
├── requirements.txt
├── affective_cli.py
├── src/
│   ├── __init__.py
│   ├── state_space.py          # Core emotional state space model
│   ├── attractors.py            # Attractor dynamics
│   ├── recurrence.py            # RQA implementation
│   ├── lyapunov.py              # Lyapunov exponent calculation
│   ├── bifurcation.py           # Bifurcation analysis
│   ├── topology.py              # Topological data analysis
│   ├── reservoir.py             # Echo state network predictor
│   ├── intervention.py          # Optimal intervention calculator
│   └── utils.py                 # Utility functions
├── examples/
│   ├── basic_simulation.py
│   ├── rqa_analysis.py
│   ├── lyapunov_comparison.py
│   └── prediction_demo.py
└── tests/
    ├── test_state_space.py
    ├── test_recurrence.py
    └── test_predictor.py
```

## Scientific Background

### Emotional State Space Theory
- Russell's circumplex model (valence-arousal)
- PAD model (Pleasure-Arousal-Dominance)
- Approach-avoidance motivation systems
- Temporal dynamics and inertia

### Dynamical Systems in Affective Science
- Lewis, M. D. (2005). Bridging emotion theory and neurobiology through dynamic systems modeling
- Hollenstein, T. (2015). This time, it's real: Affective flexibility, time scales, feedback loops, and the regulation of emotion
- Kuppens, P., & Verduyn, P. (2017). Emotion dynamics

### Clinical Applications
- Depression as stuck in negative attractors
- Bipolar disorder as unstable dynamics with rapid transitions
- Anxiety as high Lyapunov exponents (volatility)
- Emotion regulation as attractor modification

## Future Extensions

- Integration with real-time fMRI neurofeedback
- Multi-person emotional synchrony analysis
- Cultural differences in emotional landscape topology
- VR/AR environments for controlled perturbation experiments
- Pharmacological intervention modeling

## Citation

If you use this code in your research, please cite:

```bibtex
@software{affective_geometry,
  title={Affective Geometry: Dynamical Systems Mapping of Emotional Landscapes},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/affective-geometry}
}
```

## License

MIT License
