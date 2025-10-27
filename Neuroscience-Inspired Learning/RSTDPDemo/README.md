# Reinforcement-Modulated STDP (R-STDP) Demo

Recurrent LIF network with eligibility-trace STDP modulated by a dopamine-like global reward signal. The network learns to regulate an output population's firing rate toward a context-dependent target using only local plasticity and a scalar reinforcement signal.

## Run

```powershell
python RSTDPDemo/rstdp_demo.py --N 120 --episodes 50 --episode-ms 1500 --save rstdp.png --no-show
```

## Key ideas

- STDP with pre/post traces forms eligibility (potential weight change) locally
- A global reward signal (dopamine) gates eligibility into actual weight updates
- Advantage shaping (reward minus baseline) stabilizes learning
- E/I constraints and weight caps prevent runaway excitation

## Options (highlights)

- Network: `--N`, `--frac-exc`, `--p`, `--J-init`, `--g`, `--tau-m`, `--tau-syn`, `--dt`, `--noise-std`, `--I-ext`
- Plasticity: `--tau-pre`, `--tau-post`, `--tau-e`, `--A-plus`, `--A-minus`, `--eta`, `--w-exc-max`, `--w-inh-min`, `--row-norm-max`
- Task: `--episodes`, `--episode-ms`, `--switch-ms`, `--out-frac`, `--target-low`, `--target-high`, `--tau-readout`, `--dopamine-gain`, `--reward-baseline-tau`

## Plots/outputs

- Spike raster (last episode)
- Output rate vs. target (last episode)
- Learning curves over episodes: dopamine advantage, mean absolute error
- Weight statistics (mean/std)

## Notes

- Balancing plasticity and stability: use small `--eta`, capped excitatory weights, and strong inhibition (`--g`) to avoid blow-up
- Reward shaping matters: the advantage (reward minus baseline) helps reduce variance and drift
- Extend to control: replace the context target with a control error (e.g., cart-pole angle); reward is error reduction over time


