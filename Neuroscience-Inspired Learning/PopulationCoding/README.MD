# Population Coding Demo (Poisson)

Encode a circular stimulus (orientation/direction) with Poisson neurons having von Mises-like tuning. Decode with Maximum Likelihood (grid) or Population Vector (double-angle for 180° orientation). Plots tuning curves, example counts, true vs estimated, and error stats.

## Run

Maximum Likelihood:

```powershell
python PopulationCodingDemo/population_demo.py --trials 500 --decoder ml --save pop_demo.png --no-show
```

Population Vector:

```powershell
python PopulationCodingDemo/population_demo.py --trials 500 --decoder pv --save pop_demo_pv.png --no-show
```

## Options

- `--neurons`: number of neurons (default 32)
- `--period`: stimulus periodicity in degrees (180 for orientation, 360 for direction)
- `--baseline`, `--max-rate`: firing rate range (Hz)
- `--kappa`: tuning sharpness
- `--duration`: observation window per trial (ms)
- `--decoder`: `ml` or `pv`
- `--ml-grid`: grid resolution for ML (deg)
- `--seed`: RNG seed

## Notes

- PV uses double-angle for 180° orientation; for 360° direction, it reduces to standard vector sum.
- Increasing duration or neuron count typically reduces decoding error.



