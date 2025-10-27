# HARNA: Hierarchical Appraisal-Regulation Neural Architecture

A multi-layered computational model integrating cognitive appraisal theory with affective neuroscience, implementing competing theories of emotion and testing them against real neural and behavioral data.

## Theoretical Background

HARNA synthesizes multiple theoretical frameworks in emotion science:

### Appraisal Theory
Emotions arise from cognitive evaluations (appraisals) of events along multiple dimensions:
- **Goal Relevance**: Does this event matter to my goals?
- **Coping Potential**: Can I handle this situation?
- **Norm Compatibility**: Does this align with social/moral norms?
- **Novelty/Familiarity**: How unexpected is this?
- **Intrinsic Pleasantness**: Is this inherently pleasant/unpleasant?

### Dual-Process Architecture
- **Bottom-up pathway**: Fast, automatic subcortical processing (amygdala-mediated)
- **Top-down pathway**: Slower, deliberate cortical appraisal (prefrontal cortex)

### Predictive Processing
Emotions as interoceptive predictions about bodily states, with prediction errors driving learning and updating.

### Emotion Regulation
Active modulation of emotional responses through:
- **Reappraisal**: Changing interpretation of the situation
- **Suppression**: Inhibiting emotional expression
- **Distraction**: Redirecting attention away from emotional stimuli

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HARNA Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Sensory Input   │────────▶│  Feature         │         │
│  │  (Visual, Audio, │         │  Extraction      │         │
│  │   Interoceptive) │         └────────┬─────────┘         │
│  └──────────────────┘                  │                    │
│                                        │                    │
│         ┌──────────────────────────────┴─────────┐         │
│         │                                         │         │
│         ▼                                         ▼         │
│  ┌──────────────┐                      ┌──────────────┐    │
│  │  BOTTOM-UP   │                      │   TOP-DOWN   │    │
│  │  Subcortical │                      │  Prefrontal  │    │
│  │   Pathway    │                      │   Appraisal  │    │
│  │              │                      │              │    │
│  │ • Amygdala   │                      │ • Goal Rel.  │    │
│  │ • Threat Det │                      │ • Coping     │    │
│  │ • Fast Route │                      │ • Norms      │    │
│  └──────┬───────┘                      └──────┬───────┘    │
│         │                                     │             │
│         └──────────────┬──────────────────────┘             │
│                        │                                    │
│                        ▼                                    │
│              ┌──────────────────┐                          │
│              │  Emotion State   │                          │
│              │  Integration     │                          │
│              └────────┬─────────┘                          │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         │             │             │                      │
│         ▼             ▼             ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │Predictive│  │Regulation│  │    RL    │                │
│  │Processing│  │ Modules  │  │  Agent   │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                             │
│  ┌────────────────────────────────────────────┐           │
│  │  Individual Differences Engine              │           │
│  │  (Neuroticism, Alexithymia, Resilience)    │           │
│  └────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Bottom-Up Subcortical Pathway
- **Amygdala-like threat detection**: Deep neural network trained on affective stimuli
- **Temporal precedence constraints**: Fast processing (< 100ms)
- **Automatic arousal modulation**: Direct influence on physiological responses
- **Salience detection**: Identifies emotionally relevant stimuli

### 2. Top-Down Prefrontal Appraisal
- **Scherer's Component Process Model**: Multi-dimensional appraisal
- **Goal-directed evaluation**: Context-dependent appraisal
- **Cognitive reappraisal**: Flexible interpretation of events
- **Executive control**: Modulation of emotional responses

### 3. Predictive Processing Framework
- **Interoceptive predictions**: Forward models of bodily states
- **Prediction error signals**: Drive learning and updating
- **Precision weighting**: Context-dependent reliability
- **Active inference**: Action selection to minimize prediction error

### 4. Reinforcement Learning Integration
- **Intrinsic emotional rewards**: Positive emotions as rewards
- **Emotional costs**: Negative emotions as penalties
- **Exploration-exploitation trade-off**: Modulated by emotional state
- **Policy learning**: Emotion-guided action selection

### 5. Emotion Regulation Modules
- **Reappraisal**: Cognitive restructuring of appraisals
- **Suppression**: Response inhibition
- **Distraction**: Attentional deployment
- **Situation selection**: Proactive emotion regulation

### 6. Individual Differences Engine
- **Personality traits**: Neuroticism, extraversion, conscientiousness
- **Emotional competencies**: Alexithymia, emotional intelligence
- **Resilience factors**: Stress tolerance, coping capacity
- **Pathway modulation**: Trait-dependent processing biases

### 7. Multi-Timescale Processing
- **Millisecond**: Orienting responses, threat detection
- **Second**: Appraisal processes, emotion generation
- **Minute**: Mood regulation, sustained emotional states
- **Hour**: Affective styles, long-term emotional patterns

### 8. Competing Emotion Theories
Implements multiple theoretical frameworks for comparison:

#### James-Lange Theory
Emotions follow from bodily responses
```
Stimulus → Bodily Response → Emotion
```

#### Cannon-Bard Theory
Emotions and bodily responses occur simultaneously
```
Stimulus → (Emotion + Bodily Response)
```

#### Schachter-Singer Two-Factor Theory
Emotions require both arousal and cognitive label
```
Stimulus → Arousal + Cognitive Label → Emotion
```

#### Constructionist Theory (Barrett)
Emotions constructed from core affect + conceptualization
```
Core Affect + Conceptualization → Emotion
```

## Installation

```bash
cd harna
pip install -r requirements.txt
```

## Usage

### Basic Emotion Generation

```python
from harna import HARNAModel, Stimulus
import numpy as np

# Create model
model = HARNAModel(
    theory='scherer',  # or 'james-lange', 'cannon-bard', etc.
    individual_traits={
        'neuroticism': 0.5,
        'alexithymia': 0.2,
        'resilience': 0.7
    }
)

# Present stimulus
stimulus = Stimulus(
    features=np.random.randn(512),  # Visual/audio features
    context={'social': True, 'threat_level': 0.3}
)

# Process through architecture
response = model.process(stimulus)

print(f"Emotion: {response.emotion_label}")
print(f"Valence: {response.valence:.2f}")
print(f"Arousal: {response.arousal:.2f}")
print(f"Appraisals: {response.appraisals}")
```

### Emotion Regulation

```python
from harna import RegulationStrategy

# Initial emotional response
response = model.process(negative_stimulus)
print(f"Initial emotion: {response.emotion_label}")

# Apply reappraisal
regulated_response = model.regulate(
    response,
    strategy=RegulationStrategy.REAPPRAISAL,
    reappraisal_content="This is a learning opportunity"
)

print(f"Regulated emotion: {regulated_response.emotion_label}")
print(f"Valence change: {regulated_response.valence - response.valence:.2f}")
```

### Reinforcement Learning with Emotional Rewards

```python
from harna import EmotionalRLAgent
import gym

# Create environment
env = gym.make('CartPole-v1')

# Create agent with emotional rewards
agent = EmotionalRLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    harna_model=model
)

# Train
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(state)
        
        # Environment step
        next_state, reward, done, _ = env.step(action)
        
        # Compute emotional reward
        emotional_reward = agent.compute_emotional_reward(state, action, next_state)
        
        # Combined reward
        total_reward = reward + 0.5 * emotional_reward
        
        # Update
        agent.update(state, action, total_reward, next_state, done)
        
        state = next_state
```

### Model Comparison

```python
from harna import ModelComparison
import pandas as pd

# Load empirical data (fMRI, behavioral)
data = pd.read_csv('emotion_dataset.csv')

# Compare theories
comparison = ModelComparison(
    theories=['james-lange', 'cannon-bard', 'schachter-singer', 'constructionist'],
    data=data
)

# Fit models
comparison.fit_all_models()

# Compare using BIC
results = comparison.compare_models(metric='bic')

print("Model Comparison Results:")
print(results)

# Best model
best_model = comparison.get_best_model()
print(f"\nBest fitting theory: {best_model}")

# Visualize
comparison.plot_comparison()
```

### Multi-Timescale Analysis

```python
from harna import MultiTimescaleAnalyzer

# Create analyzer
analyzer = MultiTimescaleAnalyzer(model)

# Simulate long-term emotional dynamics
timeline = analyzer.simulate(
    duration_hours=24,
    events=[
        {'time': 2.0, 'type': 'stressor', 'intensity': 0.7},
        {'time': 8.0, 'type': 'positive', 'intensity': 0.5},
        {'time': 16.0, 'type': 'stressor', 'intensity': 0.4}
    ]
)

# Analyze across timescales
analysis = analyzer.analyze_timescales(timeline)

print(f"Millisecond responses: {analysis['millisecond']['n_orienting']}")
print(f"Second-scale appraisals: {analysis['second']['mean_appraisal_time']}")
print(f"Minute-scale mood: {analysis['minute']['mood_stability']}")
print(f"Hour-scale affective style: {analysis['hour']['predominant_affect']}")

# Visualize
analyzer.plot_timescales(timeline)
```

## CLI Interface

```bash
# Run emotion generation demo
python harna_cli.py generate --stimulus "threatening image" --theory scherer

# Compare theories on dataset
python harna_cli.py compare --data dataset.csv --theories all --output results/

# Simulate regulation strategies
python harna_cli.py regulate --emotion anger --strategies reappraisal,suppression

# Train RL agent with emotional rewards
python harna_cli.py train-rl --env CartPole-v1 --episodes 1000

# Analyze individual differences
python harna_cli.py individual-diff --trait neuroticism --range 0,1 --steps 10
```

## Project Structure

```
harna/
├── README.md
├── requirements.txt
├── harna_cli.py
├── src/
│   ├── __init__.py
│   ├── model.py                    # Main HARNA model
│   ├── subcortical.py              # Bottom-up pathway
│   ├── appraisal.py                # Top-down appraisal (Scherer)
│   ├── predictive_processing.py   # Interoceptive predictions
│   ├── regulation.py               # Emotion regulation modules
│   ├── rl_agent.py                 # RL with emotional rewards
│   ├── individual_differences.py  # Trait modulation
│   ├── timescales.py              # Multi-timescale processing
│   ├── theories.py                # Competing emotion theories
│   ├── model_comparison.py        # Theory comparison tools
│   └── utils.py                   # Utilities
├── examples/
│   ├── basic_emotion_generation.py
│   ├── regulation_demo.py
│   ├── rl_training.py
│   └── theory_comparison.py
└── tests/
    ├── test_appraisal.py
    ├── test_regulation.py
    └── test_theories.py
```

## Scientific Background

### Key References

**Appraisal Theory:**
- Scherer, K. R. (2009). The dynamic architecture of emotion
- Lazarus, R. S. (1991). Emotion and adaptation
- Moors, A. (2009). Theories of emotion causation

**Dual-Process Models:**
- LeDoux, J. E. (1996). The emotional brain
- Pessoa, L. (2008). On the relationship between emotion and cognition

**Predictive Processing:**
- Barrett, L. F. (2017). How emotions are made
- Seth, A. K. (2013). Interoceptive inference

**Emotion Regulation:**
- Gross, J. J. (2015). Emotion regulation: Current status and future prospects
- Ochsner, K. N., & Gross, J. J. (2005). The cognitive control of emotion

**Individual Differences:**
- Costa, P. T., & McCrae, R. R. (1992). NEO PI-R
- Bagby, R. M., et al. (1994). Toronto Alexithymia Scale

## Applications

### Clinical Psychology
- Model emotional disorders (depression, anxiety, PTSD)
- Test intervention mechanisms (CBT, mindfulness)
- Predict treatment response

### Affective Neuroscience
- Generate testable predictions for fMRI/EEG studies
- Explain neural activation patterns
- Model brain-behavior relationships

### Human-Computer Interaction
- Affective computing systems
- Emotion-aware AI agents
- Personalized emotional support

### Education & Training
- Emotional intelligence training
- Stress management interventions
- Resilience building programs

## Future Extensions

- Integration with real-time fMRI neurofeedback
- Multi-agent social emotion modeling
- Cultural differences in appraisal patterns
- Developmental trajectories of emotion regulation
- Pharmacological intervention modeling

## Citation

```bibtex
@software{harna,
  title={HARNA: Hierarchical Appraisal-Regulation Neural Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/harna}
}
```

## License

MIT License
