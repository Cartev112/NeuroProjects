## Project 13: P300 Speller Detection

### Overview
Detect P300 responses in an oddball speller paradigm using XDAWN + LDA or a compact CNN. Outputs AUC/accuracy and latency analysis.

### Instructions
Provide epochs `X` (N, C, T) and labels `y` (0/1). Run:
```bash
python p300_cli.py --data_npz /path/to/p300.npz --out_dir outputs/p300
```

### Learning: P300 Component
The P300 is a positive deflection around ~300 ms elicited by rare/target stimuli. Spatial filters like XDAWN enhance event-related activity, improving SNR before classification.

