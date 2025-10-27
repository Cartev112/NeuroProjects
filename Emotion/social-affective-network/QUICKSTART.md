# SARN Quick Start Guide

## Installation

```bash
cd social-affective-network
pip install -r requirements.txt
```

## Running Examples

### 1. Emotional Contagion Demo
```bash
python examples/contagion_demo.py
```

Demonstrates:
- Small-world network creation
- Emotion seeding in central agent
- Contagion spread dynamics
- Visualization of spread patterns

### 2. Epidemic Simulation
```bash
python examples/epidemic_simulation.py
```

Demonstrates:
- SIR-like epidemic modeling
- R₀ calculation
- Influencer identification
- Targeted intervention effects

## CLI Usage

### Interactive Demo
```bash
python sarn_cli.py demo
```

### Simulate Emotional Contagion
```bash
python sarn_cli.py simulate --network-type small_world --n-nodes 100 --seed-emotion joy --steps 50
```

Options:
- `--network-type`: small_world, scale_free, random, complete
- `--seed-emotion`: joy, anger, sadness, fear

### Run Epidemic Simulation
```bash
python sarn_cli.py epidemic --network-size 1000 --emotion anger --intervention targeted
```

Options:
- `--intervention`: none, targeted, random

### Test Theory of Mind
```bash
python sarn_cli.py tom --scenario false-belief --recursion-depth 3
```

### Compare Cultural Norms
```bash
python sarn_cli.py culture --cultures western,eastern,african --emotion anger
```

## Python API Examples

### Basic Emotional Contagion

```python
from src.network import create_network
from src.propagation import EmotionPropagator

# Create network
network = create_network('small_world', n_nodes=100, k=6, p=0.3)

# Seed emotion
network.get_agent('0').set_emotion({'valence': 0.8, 'arousal': 0.6})

# Create propagator
propagator = EmotionPropagator(network, decay_rate=0.1, transmission_rate=0.3)

# Simulate
history = propagator.simulate(n_steps=50)

# Analyze
analysis = propagator.analyze_spread()
print(f"Infection rate: {analysis['infection_rate']:.1%}")
```

### Emotional Epidemiology

```python
from src.epidemiology import EmotionalEpidemiology

# Create epidemic simulator
epi = EmotionalEpidemiology(
    network=network,
    emotion_type='anger',
    transmission_rate=0.3,
    recovery_rate=0.1
)

# Seed patient zeros
epi.seed_emotion(['0', '1', '2'], intensity=0.9)

# Simulate
history = epi.simulate(steps=100)

# Get peak info
peak_info = epi.get_peak_info()
print(f"Peak: {peak_info['peak_count']} at time {peak_info['peak_time']}")

# Plot
epi.plot_epidemic_curve()
```

### Theory of Mind Reasoning

```python
from src.agent import Agent
from src.theory_of_mind import TheoryOfMindModule

# Create agents
alice = Agent('alice', initial_emotion={'valence': 0.5, 'arousal': 0.3})
bob = Agent('bob', initial_emotion={'valence': -0.3, 'arousal': 0.6})

# Create ToM module
tom = TheoryOfMindModule(max_recursion=3)

# First-order: Alice infers Bob's emotion
belief = tom.infer_other_emotion(alice, bob)
print(f"Alice thinks Bob feels: {belief.emotion_type.value}")

# Recursive mentalizing
nested_belief = tom.recursive_mentalizing(alice, bob, depth=2)
print(f"Nested belief depth: {nested_belief.depth}")
```

### GNN Emotion Propagation

```python
from src.propagation import GNNPropagator

# Create GNN propagator
gnn = GNNPropagator(emotion_dim=3, hidden_dim=64, n_layers=2)

# Propagate emotions
history = gnn.propagate_emotions(network, n_steps=10)
```

### Cultural Norms

```python
from src.cultural_norms import CulturalNorms

# Western culture
western = CulturalNorms(culture='western', individualism=0.8)

# Eastern culture
eastern = CulturalNorms(culture='eastern', individualism=0.3)

# Modulate expression
emotion = {'valence': -0.6, 'arousal': 0.8, 'intensity': 0.9}

western_expr = western.modulate_expression(emotion)
eastern_expr = eastern.modulate_expression(emotion)

print(f"Western: {western_expr['intensity']:.2f}")
print(f"Eastern: {eastern_expr['intensity']:.2f}")
```

### Inter-Brain Synchrony

```python
from src.synchrony import SynchronyDetector
import numpy as np

# Create detector
detector = SynchronyDetector(sampling_rate=250)

# Generate sample EEG data
eeg1 = np.random.randn(10000)
eeg2 = np.random.randn(10000)

# Compute phase locking
plv = detector.compute_phase_locking(eeg1, eeg2)

print(f"Theta PLV: {plv['theta']:.3f}")
print(f"Alpha PLV: {plv['alpha']:.3f}")
print(f"Beta PLV: {plv['beta']:.3f}")
```

### Identify Influencers

```python
# Identify central agents
influencers = network.identify_central_agents(top_k=10)

for agent_id, centrality in influencers:
    print(f"Agent {agent_id}: centrality = {centrality:.3f}")
```

## Key Concepts

### Network Topologies
- **Small-world**: High clustering, short paths (realistic social networks)
- **Scale-free**: Power-law degree distribution (hubs exist)
- **Random**: Erdős-Rényi random graph
- **Complete**: All agents connected

### Emotional Contagion
- **Transmission rate**: Probability of emotion transfer
- **Decay rate**: Rate of return to neutral
- **Susceptibility**: Individual vulnerability to contagion

### Epidemiology
- **R₀**: Basic reproduction number (average infections per case)
- **SIR model**: Susceptible → Infected → Recovered
- **Intervention**: Targeted regulation of high-influence nodes

### Theory of Mind
- **First-order**: "I think she is happy"
- **Second-order**: "I think she thinks I am sad"
- **Recursive**: Nested mental state attributions

### Cultural Dimensions
- **Individualism**: Individual vs collective focus (0 to 1)
- **Power distance**: Acceptance of hierarchy (0 to 1)
- **Display rules**: Culture-specific expression norms

## Troubleshooting

### Import Errors
```bash
cd social-affective-network
python examples/contagion_demo.py
```

### NetworkX Issues
```bash
pip install networkx>=3.1
```

### PyTorch Geometric
For GNN features:
```bash
pip install torch-geometric
```

### Memory Issues
Reduce network size:
```python
network = create_network('small_world', n_nodes=50)  # Smaller network
```

## Next Steps

1. **Integrate Real Data**: Use actual social network data
2. **Train GNN**: Train on empirical emotion spread data
3. **Validate Models**: Compare with real contagion studies
4. **Add Interventions**: Design and test regulation strategies
5. **Multi-Modal**: Integrate facial, vocal, textual cues

## Citation

```bibtex
@software{sarn,
  title={SARN: Social Affective Resonance Network},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sarn}
}
```
