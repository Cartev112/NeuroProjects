# Learning Doc: Leaky Integrate-and-Fire (LIF) Neuron

This document provides a conceptual overview of the Leaky Integrate-and-Fire (LIF) neuron model, how this project simulates it, and how to interpret the results.

## 1. Core Concept: The LIF Neuron

The LIF model is a simple yet powerful abstraction of a biological neuron. It's a "point neuron" model, meaning it doesn't have complex spatial structures like dendrites or axons; it's treated as a single electrical unit. The model captures two fundamental aspects of a neuron's behavior:

1.  **Integration of Inputs:** Neurons receive signals from other neurons (or external stimuli). These signals, represented as an **input current**, cause the neuron's internal **membrane potential** (voltage) to change over time.
2.  **Firing a Spike:** When the membrane potential reaches a certain **threshold**, the neuron "fires" a spike (an action potential). After firing, the potential is reset, and the neuron enters a brief **refractory period** where it cannot fire again.

### The "Leaky" Part

A crucial feature is the "leak." If there's no input current, the neuron's membrane potential doesn't just stay where it is; it gradually "leaks" back down to its stable **resting potential**. This is analogous to a bucket with a small hole in it: if you stop pouring water in, the water level will slowly drop.

This behavior is described by a simple differential equation:

\[ \tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R \cdot I(t) \]

Where:
- \(V\) is the membrane potential.
- \(\tau_m\) (tau_m) is the membrane time constant, which determines how quickly the potential leaks.
- \(V_{\text{rest}}\) is the resting potential.
- \(R\) is the membrane resistance, which scales the effect of the input current.
- \(I(t)\) is the input current at time \(t\).

## 2. How the Simulation Works

The script `lif_simulator.py` simulates this model by numerically integrating the differential equation over small time steps (`dt`).

### Key Components in the Code

-   **`LIFNeuron` class:** This class manages the state of a single neuron (its current voltage `V`) and its configuration parameters (`LIFConfig`).
-   **`simulate` method:** This is the main loop. At each time step `t`:
    1.  It checks if the neuron is in a refractory period. If so, its voltage is held at the reset potential.
    2.  If not refractory, it calculates the change in voltage `dV` using the equation above.
    3.  **Noise:** To make the model more realistic, a small amount of random Gaussian noise is added to `dV` at each step. This represents the noisy nature of biological systems.
    4.  The voltage is updated: `V(t) = V(t-1) + dV`.
    5.  It checks if `V(t)` has crossed the threshold `V_th`.
    6.  **Spike Event:** If the threshold is crossed, a spike is recorded, the voltage is set to `V_reset`, and the refractory countdown begins.

### Input Current Protocols

The simulation allows for different types of input currents to be delivered to the neuron, controlled by the `--protocol` argument:
-   **`constant`:** A steady, continuous input. If this current is high enough (above the "rheobase"), the neuron will fire at a regular rate.
-   **`step`:** The current starts at one level and abruptly "steps" to another. This is useful for seeing how the neuron responds to sudden changes in stimulus.
-   **`sine`:** A sinusoidally varying current, mimicking rhythmic or oscillating input. This can cause the neuron to fire in bursts that are phase-locked to the input wave.

## 3. Interpreting the Output

The script generates a plot with two panels, sharing a time axis (in milliseconds).

### Top Panel: Membrane Potential

-   **Blue Line:** This is the core of the simulation—the neuron's membrane potential (`V`) over time. You can see it rise in response to input current and leak back down when the input is low.
-   **Red Dashed Line:** The firing threshold. When the blue line hits this line, a spike occurs.
-   **Red Dots:** These mark the exact moments when a spike was fired. Notice how the voltage immediately drops to the reset potential afterward.

### Bottom Panel: Input Current

-   **Green Line:** This shows the input current `I(t)` that was fed into the neuron at each time step. You can directly correlate the shape of this current with the behavior of the voltage in the top panel. For example, when the green line is high, the blue line's slope is steeper.

### Spike Metrics

After running, the script prints out basic statistics about the neuron's firing pattern:
-   **`spike_count`**: Total number of spikes.
-   **`firing_rate_hz`**: The average number of spikes per second (in Hertz).
-   **`isi_...`**: Statistics about the Inter-Spike Interval (the time between consecutive spikes), which is a key measure of a neuron's firing pattern.

By changing parameters like `--input-current`, `--noise-std`, and `--tau-m`, you can explore how different properties affect a neuron's computational behavior.
