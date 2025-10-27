# HARNA Quick Start Guide

## Installation

```bash
cd harna
pip install -r requirements.txt
```

## Running Examples

### 1. Basic Emotion Generation
```bash
python examples/basic_emotion_generation.py
```

Demonstrates:
- Creating HARNA model with personality traits
- Processing different emotional scenarios
- Comparing responses across personalities

### 2. Emotion Regulation
```bash
python examples/regulation_demo.py
```

Demonstrates:
- Different regulation strategies (reappraisal, suppression, distraction, acceptance)
- Effectiveness comparison
- Context-dependent strategy selection

## CLI Usage

### Interactive Demo
```bash
python harna_cli.py demo
```

### Generate Emotional Response
```bash
python harna_cli.py generate --theory scherer --neuroticism 0.7 --context threat
```

### Test Regulation Strategies
```bash
python harna_cli.py regulate --emotion anger --strategies reappraisal,suppression,distraction
```

### Analyze Individual Differences
```bash
python harna_cli.py individual-diff --trait neuroticism --range 0,1 --steps 10
```

### Compare Emotion Theories
```bash
python harna_cli.py compare --theories all
```

Or specific theories:
```bash
python harna_cli.py compare --theories james_lange,scherer,constructionist
```

## Python API Examples

### Basic Emotion Generation

```python
from src.model import HARNAModel, Stimulus
from src.utils import generate_random_stimulus, create_context

# Create model
model = HARNAModel(
    theory='scherer',
    individual_traits={
        'neuroticism': 0.6,
        'resilience': 0.7
    }
)

# Create stimulus
features = generate_random_stimulus()
context = create_context(threat=True)
stimulus = Stimulus(features=features, context=context)

# Process
response = model.process(stimulus)

print(f"Emotion: {response.emotion_label}")
print(f"Valence: {response.valence:+.2f}")
print(f"Arousal: {response.arousal:+.2f}")
```

### Emotion Regulation

```python
from src.regulation import RegulationStrategy

# Generate emotional response
response = model.process(stimulus)

# Apply regulation
regulated = model.process(
    stimulus,
    regulate=True,
    regulation_strategy=RegulationStrategy.REAPPRAISAL
)

print(f"Original: {response.emotion_label} (valence: {response.valence:+.2f})")
print(f"Regulated: {regulated.emotion_label} (valence: {regulated.valence:+.2f})")
```

### Train Threat Detector

```python
from src.subcortical import generate_synthetic_threat_data

# Generate training data
features, labels = generate_synthetic_threat_data(n_samples=1000)

# Train
model.subcortical.train_threat_detector(
    features, labels,
    epochs=50,
    batch_size=32
)

# Evaluate
test_features, test_labels = generate_synthetic_threat_data(n_samples=200)
metrics = model.subcortical.evaluate_threat_detector(test_features, test_labels)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2f}")
```

### RL Agent with Emotional Rewards

```python
from src.rl_agent import EmotionalRLAgent
import gym

# Create environment
env = gym.make('CartPole-v1')

# Create agent with HARNA
agent = EmotionalRLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    harna_model=model,
    emotional_weight=0.5
)

# Training loop
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
```

### Compare Emotion Theories

```python
from src.theories import JamesLangeTheory, CannonBardTheory, SchachterSingerTheory

theories = {
    'James-Lange': JamesLangeTheory(),
    'Cannon-Bard': CannonBardTheory(),
    'Schachter-Singer': SchachterSingerTheory()
}

stimulus_features = generate_random_stimulus()
context = create_context(social=True)

for name, theory in theories.items():
    response = theory.generate_emotion(stimulus_features, context)
    print(f"{name}: {response.emotion_label} (valence: {response.valence:+.2f})")
```

## Key Concepts

### Dual-Process Architecture
- **Bottom-up (Subcortical)**: Fast, automatic threat detection
- **Top-down (Cortical)**: Slower, deliberate appraisal

### Scherer's Appraisal Dimensions
1. **Novelty**: How unexpected is the event?
2. **Intrinsic Pleasantness**: Is it inherently pleasant/unpleasant?
3. **Goal Relevance**: Does it matter to my goals?
4. **Coping Potential**: Can I handle this?
5. **Norm Compatibility**: Does it align with norms?

### Regulation Strategies
- **Reappraisal**: Change interpretation (most effective, high effort)
- **Suppression**: Inhibit expression (moderate effectiveness, can backfire)
- **Distraction**: Redirect attention (quick, temporary)
- **Acceptance**: Acknowledge without changing (low effort)

### Individual Differences
- **Neuroticism**: Increases threat sensitivity, negative bias
- **Resilience**: Improves coping potential
- **Emotional Intelligence**: Enhances regulation effectiveness
- **Alexithymia**: Impairs emotion recognition and regulation

### Multi-Timescale Processing
- **Millisecond**: Orienting responses, threat detection
- **Second**: Appraisal processes
- **Minute**: Mood regulation
- **Hour**: Affective styles

## Troubleshooting

### Import Errors
Ensure you're in the project directory:
```bash
cd harna
python examples/basic_emotion_generation.py
```

### PyTorch Issues
Install PyTorch for your system:
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# With CUDA (if available)
pip install torch torchvision
```

### Memory Issues
Reduce network sizes in model initialization:
```python
# Smaller networks
model.subcortical.threat_detector = ThreatDetector(
    input_dim=512,
    hidden_dims=[128, 64]  # Smaller than default
)
```

## Next Steps

1. **Train on Real Data**: Replace synthetic data with actual fMRI/physiological data
2. **Extend Theories**: Implement additional emotion theories
3. **Add Neurofeedback**: Integrate real-time biosignal processing
4. **Clinical Applications**: Model specific disorders (depression, anxiety)
5. **Social Emotions**: Extend to multi-agent emotional interactions

## Citation

```bibtex
@software{harna,
  title={HARNA: Hierarchical Appraisal-Regulation Neural Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/harna}
}
```
