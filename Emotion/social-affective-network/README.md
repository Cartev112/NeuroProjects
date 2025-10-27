# Social Affective Resonance Network (SARN)

A multi-agent system modeling emotional contagion, empathy, collective emotional dynamics, and Theory of Mind in social networks.

## Theoretical Background

SARN integrates multiple frameworks from social neuroscience and affective science:

### Emotional Contagion
The automatic tendency to synchronize emotions with others through:
- **Primitive contagion**: Automatic mimicry and feedback
- **Cognitive empathy**: Understanding others' emotional states
- **Affective empathy**: Sharing others' emotional experiences

### Theory of Mind (ToM)
The ability to attribute mental states to others:
- **First-order ToM**: "I think she is happy"
- **Second-order ToM**: "I think she thinks I am sad"
- **Recursive mentalizing**: Nested mental state attributions

### Mirror Neuron System
Neural mechanisms that map observed actions to internal representations:
- Action observation → Motor simulation
- Emotional expression → Emotional experience
- Shared representations for self and other

### Social Network Dynamics
Emotions propagate through social networks with:
- **Network topology**: Structure affects spread patterns
- **Homophily**: Similar individuals cluster together
- **Influence**: Some individuals are more influential
- **Thresholds**: Activation thresholds for emotional spread

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Social Affective Resonance Network              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Social Network Graph                      │    │
│  │  Nodes: Agents with emotional states               │    │
│  │  Edges: Social connections with weights            │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                          │
│         ┌─────────┴─────────┐                               │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌─────────────┐    ┌─────────────┐                       │
│  │   Mirror    │    │  Theory of  │                       │
│  │   Neuron    │    │    Mind     │                       │
│  │   System    │    │   Module    │                       │
│  └──────┬──────┘    └──────┬──────┘                       │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   │                                          │
│                   ▼                                          │
│         ┌──────────────────┐                                │
│         │  Emotion         │                                │
│         │  Recognition     │                                │
│         │  (Bayesian)      │                                │
│         └────────┬─────────┘                                │
│                  │                                           │
│         ┌────────┴────────┐                                 │
│         │                 │                                 │
│         ▼                 ▼                                 │
│  ┌──────────┐      ┌──────────┐                           │
│  │ Cultural │      │  Game    │                           │
│  │  Norms   │      │ Theory   │                           │
│  └──────────┘      └──────────┘                           │
│                                                             │
│  ┌────────────────────────────────────────────┐           │
│  │  Emotional Propagation (GNN)                │           │
│  │  • Contagion dynamics                       │           │
│  │  • Synchrony detection                      │           │
│  │  • Epidemiology simulation                  │           │
│  └────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Graph Neural Networks for Emotion Propagation
- **Message passing**: Emotions propagate through network edges
- **Attention mechanisms**: Weighted influence based on relationship strength
- **Temporal dynamics**: Time-evolving emotional states
- **Decay and amplification**: Emotions fade or intensify over time

### 2. Mirror Neuron Simulation
- **Action-to-emotion mapping**: Observed actions trigger emotional responses
- **Expression recognition**: Facial, vocal, and postural cues
- **Embodied simulation**: Internal recreation of observed states
- **Shared representations**: Common neural codes for self and other

### 3. Hierarchical Bayesian Emotion Recognition
- **Multimodal integration**: Face, voice, body, context
- **Prior distributions**: Cultural and individual expectations
- **Posterior inference**: Optimal emotion estimation
- **Uncertainty quantification**: Confidence in recognition

### 4. Theory of Mind Modules
- **Recursive mentalizing**: Nested belief representations
- **Perspective taking**: Simulate others' viewpoints
- **False belief reasoning**: Understand mismatched beliefs
- **Intentionality detection**: Infer goals and desires

### 5. Cultural Emotion Norms
- **Display rules**: Culture-specific expression patterns
- **Emotion concepts**: Language-dependent categorization
- **Social appropriateness**: Context-dependent norms
- **Collectivism vs. individualism**: Cultural value systems

### 6. Inter-Brain Synchrony Detection
- **Phase locking**: Coordinated neural oscillations
- **Coherence analysis**: Frequency-domain synchrony
- **Hyperscanning**: Simultaneous multi-brain recording
- **Predictive synchrony**: Leading and following patterns

### 7. Game-Theoretic Emotional Signaling
- **Strategic displays**: Emotions as communication
- **Cost-benefit analysis**: Effort vs. impact
- **Signaling games**: Honest vs. deceptive signals
- **Evolutionary stability**: Nash equilibria in emotional strategies

### 8. Attachment Style Modulation
- **Secure**: Balanced emotion regulation
- **Anxious**: Heightened emotional reactivity
- **Avoidant**: Suppressed emotional expression
- **Disorganized**: Inconsistent patterns

### 9. Emotional Epidemiology
- **Spread dynamics**: SIR-like models for emotions
- **Network interventions**: Targeted emotion regulation
- **Influencer identification**: High-centrality nodes
- **Outbreak prediction**: Early warning systems

## Installation

```bash
cd social-affective-network
pip install -r requirements.txt
```

## Usage

### Basic Emotional Contagion

```python
from sarn import SocialNetwork, Agent, EmotionPropagator
import networkx as nx

# Create social network
G = nx.watts_strogatz_graph(n=50, k=6, p=0.3)
network = SocialNetwork(G)

# Add agents with initial emotions
for node in G.nodes():
    agent = Agent(
        agent_id=node,
        initial_emotion={'valence': 0.0, 'arousal': 0.0},
        personality={'extraversion': 0.5, 'neuroticism': 0.5}
    )
    network.add_agent(agent)

# Create propagator
propagator = EmotionPropagator(network, decay_rate=0.1)

# Seed emotion in one agent
network.get_agent(0).set_emotion({'valence': 0.8, 'arousal': 0.7})

# Simulate propagation
for t in range(100):
    propagator.step()
    
# Analyze spread
analysis = propagator.analyze_spread()
print(f"Infection rate: {analysis['infection_rate']:.2%}")
print(f"Peak time: {analysis['peak_time']}")
```

### Theory of Mind Reasoning

```python
from sarn import TheoryOfMindModule, Agent

# Create agents
alice = Agent(agent_id='alice', initial_emotion={'valence': 0.5, 'arousal': 0.3})
bob = Agent(agent_id='bob', initial_emotion={'valence': -0.3, 'arousal': 0.6})

# Create ToM module
tom = TheoryOfMindModule(max_recursion=3)

# First-order: Alice thinks about Bob's emotion
alice_belief_about_bob = tom.infer_other_emotion(
    observer=alice,
    target=bob,
    observed_behavior={'facial_expression': 'frown', 'voice_tone': 'tense'}
)

print(f"Alice thinks Bob feels: {alice_belief_about_bob}")

# Second-order: Alice thinks about what Bob thinks about her
alice_meta_belief = tom.recursive_mentalizing(
    agent=alice,
    target=bob,
    depth=2
)

print(f"Alice thinks Bob thinks Alice feels: {alice_meta_belief}")
```

### Emotional Epidemiology Simulation

```python
from sarn import EmotionalEpidemiology, SocialNetwork
import networkx as nx

# Create realistic social network
G = nx.barabasi_albert_graph(n=1000, m=5)
network = SocialNetwork(G)

# Initialize epidemiology simulator
epi = EmotionalEpidemiology(
    network=network,
    emotion_type='anger',
    transmission_rate=0.3,
    recovery_rate=0.1
)

# Seed initial "infected" nodes
epi.seed_emotion(patient_zeros=[0, 10, 50], intensity=0.9)

# Run simulation
history = epi.simulate(steps=200)

# Identify influencers
influencers = epi.identify_influencers(top_k=10)
print("Top emotional influencers:", influencers)

# Test intervention
epi.apply_intervention(
    target_nodes=influencers,
    intervention_type='regulation',
    effectiveness=0.7
)

# Visualize spread
epi.plot_epidemic_curve()
epi.plot_network_snapshot(time=50)
```

### Mirror Neuron System

```python
from sarn import MirrorNeuronSystem

# Create mirror neuron system
mns = MirrorNeuronSystem(
    input_dim=512,  # Visual/motor features
    emotion_dim=64
)

# Observe action
observed_action = extract_action_features(video_frame)  # Your feature extractor

# Simulate internal emotional response
internal_emotion = mns.simulate(observed_action)

print(f"Observed action triggered emotion: {internal_emotion}")

# Train on action-emotion pairs
mns.train(
    action_features=training_actions,
    emotion_labels=training_emotions,
    epochs=50
)
```

### Cultural Emotion Norms

```python
from sarn import CulturalNorms

# Define cultural contexts
western_norms = CulturalNorms(
    culture='western',
    individualism=0.8,
    power_distance=0.3,
    display_rules={'anger': 'moderate', 'sadness': 'suppress'}
)

eastern_norms = CulturalNorms(
    culture='eastern',
    individualism=0.3,
    power_distance=0.7,
    display_rules={'anger': 'suppress', 'sadness': 'moderate'}
)

# Modulate emotional expression
raw_emotion = {'valence': -0.6, 'arousal': 0.8}  # Anger

western_expression = western_norms.modulate_expression(raw_emotion)
eastern_expression = eastern_norms.modulate_expression(raw_emotion)

print(f"Western expression intensity: {western_expression['intensity']:.2f}")
print(f"Eastern expression intensity: {eastern_expression['intensity']:.2f}")
```

### Inter-Brain Synchrony

```python
from sarn import SynchronyDetector

# Load dual-EEG data
eeg_person1 = load_eeg('person1.edf')
eeg_person2 = load_eeg('person2.edf')

# Create synchrony detector
detector = SynchronyDetector(
    sampling_rate=250,
    frequency_bands={'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
)

# Compute synchrony
synchrony = detector.compute_phase_locking(eeg_person1, eeg_person2)

print(f"Theta synchrony: {synchrony['theta']:.3f}")
print(f"Alpha synchrony: {synchrony['alpha']:.3f}")

# Detect moments of high synchrony
high_sync_moments = detector.detect_synchrony_events(
    eeg_person1, eeg_person2,
    threshold=0.7
)

print(f"Found {len(high_sync_moments)} high synchrony moments")
```

## CLI Interface

```bash
# Run emotional contagion simulation
python sarn_cli.py simulate --network-type small-world --n-nodes 100 --seed-emotion joy

# Analyze emotional influencers
python sarn_cli.py influencers --network data/facebook.gml --emotion anger

# Test Theory of Mind reasoning
python sarn_cli.py tom --scenario false-belief --recursion-depth 3

# Compare cultural norms
python sarn_cli.py culture --cultures western,eastern,african --emotion shame

# Run epidemiology simulation
python sarn_cli.py epidemic --network-size 1000 --emotion anxiety --intervention targeted

# Interactive demo
python sarn_cli.py demo
```

## Project Structure

```
social-affective-network/
├── README.md
├── requirements.txt
├── sarn_cli.py
├── src/
│   ├── __init__.py
│   ├── network.py              # Social network graph
│   ├── agent.py                # Individual agents
│   ├── propagation.py          # GNN emotion propagation
│   ├── mirror_neurons.py       # Mirror neuron system
│   ├── emotion_recognition.py  # Bayesian recognition
│   ├── theory_of_mind.py       # ToM modules
│   ├── cultural_norms.py       # Cultural modulation
│   ├── synchrony.py            # Inter-brain synchrony
│   ├── game_theory.py          # Strategic signaling
│   ├── attachment.py           # Attachment styles
│   ├── epidemiology.py         # Emotional epidemiology
│   └── utils.py                # Utilities
├── examples/
│   ├── contagion_demo.py
│   ├── tom_reasoning.py
│   ├── epidemic_simulation.py
│   └── cultural_comparison.py
└── tests/
    ├── test_propagation.py
    ├── test_tom.py
    └── test_epidemiology.py
```

## Scientific Background

### Key References

**Emotional Contagion:**
- Hatfield, E., et al. (1994). Emotional contagion
- Prochazkova, E., & Kret, M. E. (2017). Connecting minds and sharing emotions

**Theory of Mind:**
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind?
- Frith, C. D., & Frith, U. (2006). The neural basis of mentalizing

**Mirror Neurons:**
- Rizzolatti, G., & Craighero, L. (2004). The mirror-neuron system
- Gallese, V. (2003). The roots of empathy

**Social Networks:**
- Christakis, N. A., & Fowler, J. H. (2009). Connected
- Kramer, A. D., et al. (2014). Experimental evidence of massive-scale emotional contagion

**Cultural Psychology:**
- Mesquita, B., & Frijda, N. H. (1992). Cultural variations in emotions
- Kitayama, S., & Markus, H. R. (1994). Emotion and culture

**Hyperscanning:**
- Montague, P. R., et al. (2002). Hyperscanning
- Hasson, U., et al. (2012). Brain-to-brain coupling

## Applications

### Clinical Psychology
- Model depression as network phenomenon
- Identify vulnerable individuals in social networks
- Design network-based interventions

### Social Psychology
- Study emotional contagion in groups
- Understand empathy deficits (autism, psychopathy)
- Model crowd behavior and collective emotions

### Organizational Behavior
- Optimize team emotional dynamics
- Identify toxic emotional patterns
- Design emotionally intelligent organizations

### Public Health
- Track emotional well-being at population scale
- Predict mental health crises
- Design targeted interventions

### Digital Phenotyping
- Extract emotional dynamics from social media
- Validate computational models with real data
- Early detection of emotional disorders

## Novel Contributions

1. **Emotional Epidemiology**: SIR-like models for emotion spread
2. **Influencer Identification**: Network-based targeting
3. **Digital Phenotyping**: Social media → Emotional dynamics
4. **Pathological Networks**: Depression/anxiety as network states
5. **Cultural Priors**: Bayesian integration of cultural norms
6. **Game-Theoretic Signaling**: Strategic emotional displays
7. **Multi-Level Modeling**: Individual → Dyad → Group → Network

## Future Extensions

- Real-time social media monitoring
- VR/AR experiments with avatars
- Cross-cultural validation studies
- Longitudinal network tracking
- Integration with wearable biosensors
- Multi-modal emotion recognition (face + voice + text)

## Citation

```bibtex
@software{sarn,
  title={SARN: Social Affective Resonance Network},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sarn}
}
```

## License

MIT License
