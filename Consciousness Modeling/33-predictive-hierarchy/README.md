## Project 33: Predictive Processing Hierarchy Simulator with Metacognitive Layers

### Overview
Hierarchical Bayesian brain model implementing predictive coding, active inference, attention, metacognition, counterfactual reasoning, and global workspace broadcasting.

Core components:
- **Hierarchical Predictive Coding**: Multi-level architecture (sensory → perceptual → conceptual → metacognitive)
- **Active Inference**: Minimize prediction error through both perception and action
- **Precision-Weighted Attention**: Attention as modulation of prediction error precision
- **Recurrent Processing**: Multi-timescale dynamics (faster at lower levels, slower at higher levels)
- **Metacognition**: Self-modeling module that monitors internal states and estimates confidence
- **Counterfactual Reasoning**: "What-if" simulations by intervening on hidden states
- **Global Workspace**: Information broadcast when prediction errors exceed thresholds (access consciousness)
- **Psychophysical Experiments**: Change blindness, binocular rivalry, attentional blink

### Architecture

```
Level 3 (Conceptual/Metacognitive)
    ↕ prediction/error
Level 2 (Perceptual)
    ↕ prediction/error
Level 1 (Sensory)
    ↕ prediction/error
Observation (sensory input)
```

Each level:
- Generates top-down predictions
- Computes precision-weighted prediction errors
- Updates hidden states via error signals
- Adapts precision based on error statistics

### Install
```bash
pip install numpy scipy matplotlib
```

### Usage

**Basic hierarchy simulation:**
```bash
python pph_cli.py \
  --n_levels 3 \
  --input_dim 10 \
  --hidden_dims 16,12,8 \
  --n_steps 100 \
  --out_dir outputs/pph_basic
```

**With active inference:**
```bash
python pph_cli.py \
  --n_levels 3 --input_dim 10 --hidden_dims 16,12,8 \
  --active_inference \
  --n_steps 100 \
  --out_dir outputs/pph_active
```

**With metacognition:**
```bash
python pph_cli.py \
  --n_levels 3 --input_dim 10 --hidden_dims 16,12,8 \
  --metacognition \
  --n_steps 100 \
  --out_dir outputs/pph_meta
```

**With global workspace:**
```bash
python pph_cli.py \
  --n_levels 3 --input_dim 10 --hidden_dims 16,12,8 \
  --global_workspace \
  --n_steps 100 \
  --out_dir outputs/pph_gw
```

**Full system with psychophysics:**
```bash
python pph_cli.py \
  --n_levels 3 --input_dim 10 --hidden_dims 16,12,8 \
  --active_inference --metacognition --global_workspace --counterfactual \
  --change_blindness --rivalry --attentional_blink \
  --n_steps 100 \
  --out_dir outputs/pph_full
```

### Outputs

- **summary.json** — All experimental results and statistics
- **error_timecourse.png** — Total prediction error over time
- **confidence_timecourse.png** — Metacognitive confidence (if enabled)
- **broadcast_timecourse.png** — Global workspace broadcast strength (if enabled)

**Psychophysics results (in summary.json):**
- `change_blindness`: Detection rates with/without attention
- `rivalry`: Binocular rivalry switch rate
- `attentional_blink`: T2 detection by lag

### Methods

#### Predictive Coding
- Each level predicts activity at level below
- Prediction errors propagate bottom-up
- Hidden states updated to minimize precision-weighted errors
- Free energy = sum of precision-weighted squared errors

#### Active Inference
- Agent selects actions to minimize expected free energy
- Expected free energy = expected error + expected information gain
- Action policy updated via simplified RL

#### Attention
- Precision = inverse variance of prediction errors
- High precision → strong influence on learning
- Attention modulates precision (focus = increase precision)

#### Metacognition
- Meta-level monitors all hidden states
- Confidence estimated from precision and error magnitudes
- Self-modeling: predict future internal states

#### Global Workspace
- Prediction errors > threshold → broadcast to workspace
- Workspace = globally accessible information (conscious content)
- Decay over time unless refreshed

#### Counterfactual Reasoning
- Intervene on hidden states
- Simulate alternative scenarios
- Compare outcomes (errors)

### Psychophysical Phenomena

**Change Blindness**
- Changes detected when attended (high precision)
- Changes missed when unattended (low precision)
- Workspace difference indicates detection

**Binocular Rivalry**
- Two competing stimuli
- Alternating dominance based on prediction error
- Switch rate ~10-20% of steps

**Attentional Blink**
- T2 detection impaired at short lags (2-3 steps)
- Recovery at longer lags (8+ steps)
- Workspace occupied by T1 processing

### Notes
- Simplified implementation (full active inference requires variational message passing)
- Timescales increase exponentially up the hierarchy
- Precision adaptation is simplified (full model uses hierarchical Bayesian estimation)
- Psychophysics use synthetic stimuli (can be extended to real images)

### Roadmap
- Hierarchical Bayesian PCN with explicit prediction/error nodes
- Subject-specific priors for individual differences
- Integration with real psychophysical data
- 3D visualization of hierarchical dynamics
