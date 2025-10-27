# 🧠 Continuous Attractor Network Navigation

biologically-inspired neural navigation system where a mouse learns to find cheese using brain-like learning mechanisms!

![Sci-Fi Neural Navigation](https://img.shields.io/badge/Status-Learning-00FFCC?style=for-the-badge)
![Biologically Inspired](https://img.shields.io/badge/Bio--Inspired-Brain--Like-FF6B9D?style=for-the-badge)

## 🎯 What Does This Do?

Watch a virtual mouse **learn** to navigate to cheese using the same kind of neural mechanisms found in real animal brains! The mouse starts out clueless but gradually learns efficient paths through **reward-based learning** - just like how your brain learns!

## 🎮 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
python A_navigate_can.py
```

### Controls
- **Drag the speed slider** to speed up/slow down learning
- **R** - Reset and try a new episode
- **ESC** - Exit

---

## 🧪 How Does The Learning Work? (Simple Explanation)

Imagine you're in a dark room trying to find a light switch. At first, you wander randomly. But every time you get closer to the switch, your brain gives you a little "reward" feeling. Over time, your brain **strengthens the connections** between "sensing where the switch is" and "moving toward it."

This is **exactly** how our mouse learns!

### The Three Key Components:

#### 1. **👁️ The Sensory System (Ring Attractor Network)**
- The mouse has a "mental compass" - a ring of 36 neurons arranged in a circle
- When food is detected, neurons pointing in that direction "light up"
- **Think of it like**: A compass needle always pointing toward the cheese
- **Technical term**: This is a **Continuous Attractor Network (CAN)** with ring topology
- **Why it's brain-like**: Real animals have "head direction cells" in their brains that work exactly like this!

#### 2. **🎮 The Action System (Motor Output Layer)**
- 3 neurons control movement: Turn Left, Go Straight, Turn Right
- These neurons **start out random** - the mouse doesn't know what to do!
- **Connection**: Each motor neuron receives signals from all 36 compass neurons
- **The magic**: These connections are **weak and random at first**, but they **learn over time**

#### 3. **⚡ The Learning Mechanism (Reward-Modulated Hebbian Learning)**

This is where the brain-magic happens!

**The Learning Rule (in plain English):**
- When the mouse moves **closer** to the cheese → REWARD! ✨
- The mouse's brain asks: "Which neurons were active? What action did I take?"
- **Strengthen those connections!** "Neurons that fire together, wire together"
- **Technical term**: This is called **Hebbian Learning** (discovered by Donald Hebb in 1949)

**The Reward Signal (Dopamine-like):**
- In real brains, **dopamine** acts like a learning signal
- High dopamine = "That was good! Do more of that!"
- In our simulation, reward acts like dopamine: it **modulates** (controls) how much learning happens
- **Technical term**: **Reward-Modulated Hebbian Plasticity**

### 📊 What You're Seeing

#### **Ring Attractor Network (Top Right)**
- Bright yellow/cyan neurons = "Food is in this direction!"
- The "bump" of activity rotates as the mouse turns
- Shows direction **relative to where the mouse is facing**
- 0° (East) = straight ahead from mouse's perspective

#### **Motor Output Layer (Right)**
- Three colored neurons showing what action the network wants to take
- Size and glow indicate strength of activation
- **At first**: Random activation (the mouse is confused!)
- **After learning**: The correct action lights up when food is in the matching direction

#### **Learning Progress Plots (Bottom)**
Three graphs show the learning happening in real-time:

1. **Steps to Reach Food** - Shows how many moves it takes each episode
   - **High numbers at start** = mouse is wandering randomly
   - **Decreasing trend** = mouse is learning efficient paths!
   
2. **Learning Efficiency** - Smoothed average (5 episodes)
   - This is the **learning curve**
   - Going down = getting smarter!
   
3. **Rewards per Episode** - Total reward earned
   - Higher = mouse found food faster
   - Shows overall success

---

## 🔬 The Technical Details

### Network Architecture

```
Sensory Input → Ring Attractor Network → Motor Output → Action
                    (36 neurons)          (3 neurons)
                    
                          ↓
                    Reward Signal
                          ↓
                   Weight Updates (Learning)
```

### Key Algorithms

#### Ring Attractor Network
- **Topology**: Circular connectivity (neuron 0 connects back to neuron 35)
- **Recurrent connections**: Nearby neurons excite each other, distant neurons inhibit
- **Activation**: `activity = tanh(W_recurrent @ activity + W_input @ sensory_signal)`
- **Purpose**: Maintains a stable "bump" of activity representing direction

#### Hebbian Learning Rule
```
Δw = learning_rate × reward × (input_activity × output_activity)
```

Where:
- `Δw` = change in connection weight
- `reward` = positive when mouse gets closer to food
- `input_activity` = how active the sensory neurons were
- `output_activity` = how active the motor neurons were

This means: **"If input and output were both active together during a reward, strengthen their connection"**

#### Exploration vs. Exploitation
- **15% of the time**: Mouse takes a **random action** (exploration)
- **85% of the time**: Mouse follows the network's learned policy (exploitation)
- **Why?** Without exploration, the mouse might never discover good behaviors to reinforce!

### Biological Inspiration

| Component | Brain Equivalent | Location in Real Brains |
|-----------|------------------|-------------------------|
| Ring Attractor Network | Head Direction Cells | Postsubiculum, Entorhinal Cortex |
| Reward Signal | Dopamine Release | Ventral Tegmental Area (VTA) |
| Hebbian Learning | Synaptic Plasticity | Throughout cortex |
| Motor Neurons | Motor Commands | Primary Motor Cortex, Basal Ganglia |

---

## 🚀 Advanced Features

### Unstuck Mechanism
- If mouse gets stuck in a corner (no movement for 60 steps)
- **Automatic intervention**: Random sharp turn or reverse
- **Why?** Helps the mouse escape local minima during early learning
- **Technical term**: This prevents the agent from getting trapped in suboptimal policies

### Weight Clipping
- Prevents learned weights from exploding to infinity
- Keeps weights in range [-2, 2]
- **Ensures stability** during long training runs

---

## 📚 Want to Learn More?

### Key Concepts to Research:

1. **Continuous Attractor Networks** - How brains represent continuous variables like direction
2. **Hebbian Learning** - "Neurons that fire together, wire together"
3. **Reinforcement Learning** - Learning from rewards and punishments
4. **Head Direction Cells** - Real neurons in animal brains that encode direction
5. **Dopamine and Learning** - How reward signals modulate synaptic plasticity
6. **Spike-Timing-Dependent Plasticity (STDP)** - Timing-based Hebbian learning

### Suggested Reading:
- Hebb, D.O. (1949). "The Organization of Behavior"
- Schultz, W. (1998). "Predictive Reward Signal of Dopamine Neurons"
- Zhang, K. (1996). "Representation of spatial orientation by the intrinsic dynamics of the head-direction cell ensemble"

---

## 🎨 Why Does It Look So Cool?

Because **learning should be beautiful!** 

The sci-fi aesthetic isn't just for show - it makes the complex neural processes **visible and intuitive**:
- Glowing neurons show activation
- Color coding helps distinguish different components
- Real-time plots reveal the learning process
- The mouse and cheese make it relatable

---

## 🧬 The Big Picture

This simulation demonstrates that:

1. **Intelligence can emerge** from simple learning rules
2. **Biological learning** is elegant and powerful
3. **No backpropagation needed** - simpler, more brain-like learning works!
4. **Neural representations** (like the ring) make learning easier

This is how **real brains learn** - through local learning rules, reward signals, and neural representations. No giant training datasets, no GPUs computing gradients - just simple, powerful, **biological intelligence**.

---

## 🤝 Credits

Built with love for neuroscience, learning, and beautiful visualizations! 🧠✨

**Technologies**: Python, Pygame, NumPy, Biologically-Inspired Algorithms

---

**Enjoy watching your mouse become smarter!** 🐭🧀

