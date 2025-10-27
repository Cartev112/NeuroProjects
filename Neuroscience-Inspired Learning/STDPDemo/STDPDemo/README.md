# STDP Demo

Two LIF neurons connected by a single excitatory synapse with pair-based STDP.
Plots membrane voltages and the synaptic weight over time to illustrate learning.

## Run

Default demo (2 s):

```powershell
python STDPDemo/stdp_demo.py --duration 2000 --save stdp_demo.png --no-show
```

Key options:

- `--pre-current`, `--post-current`: background currents for pre/post neurons
- `--w-init`, `--w-min`, `--w-max`: synaptic weight settings
- `--tau-syn`: synaptic current time constant
- `--tau-plus`, `--tau-minus`: STDP trace time constants
- `--a-plus`, `--a-minus`: LTP/LTD magnitudes
- `--noise-std`: membrane noise (mV/√ms)
- `--seed`: RNG seed

## Interpretation

- When pre spikes just before post, LTP increases the weight; post-before-pre triggers LTD, decreasing it.
- The weight trace shows how repeated relative timing shapes synaptic strength.

