## Project 16: MEG Sensor vs Source Decoding

### Overview
Compare decoding in sensor space vs source space (minimum-norm estimates) for a classification task.

### Instructions
Use BIDS MEG with events and forward/inverse solutions prepared or use MNE sample. Run:
```bash
python meg_decode_cli.py --bids_root /path --subject sub-01 --task faces --out_dir outputs/meg_decode
```

### Learning: Sensor vs Source Space
Source estimates project sensor data into cortical space using an inverse operator. Decoding in source space may improve spatial specificity but introduces regularization choices.

