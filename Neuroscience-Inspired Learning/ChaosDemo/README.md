# Chaos in Recurrent Spiking Networks

Random E/I recurrent LIF network exhibiting regimes from stable/asynchronous to irregular/chaotic as inhibitory strength (g) and synaptic scale (J) vary.

## Run

```powershell
python ChaosDemo/chaos_demo.py --N 800 --duration 2000 --g 4.0 --J 0.15 --save chaos.png --no-show
```

## Key parameters

- `--g`: inhibitory strength multiplier (inhib weight = -g * J)
- `--J`: excitatory synaptic weight magnitude
- `--p`: connection probability
- `--frac-exc`: fraction of excitatory neurons
- `--tau-m`, `--tau-syn`, `--refractory`, `--dt`: neuron/time constants
- `--I-ext`, `--noise-std`: external drive and membrane noise

## Outputs

- Raster plot (spikes over time)
- Population rate over time (binned)
- Single-neuron rate distribution
- Printed metrics:
  - Mean/excitatory/inhibitory firing rates
  - ISI CV (irregularity)
  - Average absolute pairwise correlation (binned)

## Interpreting regimes

- Lower g or J → stable/asynchronous: moderate rates, low CV, low correlations.
- Higher g or J → irregular/chaotic: broad rate distribution, CV ~ 1, fluctuating population rate, higher variability.
- Edge of chaos often emerges near a balance where inhibition just restrains excitation.


