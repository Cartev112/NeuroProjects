## Project 21: RSA – EEG/MEG vs DNN Representations

### Overview
Compute time-resolved neural RDMs and compare with DNN RDMs using correlation tests and permutation significance.

### Instructions
Provide neural features `X.npy` (N, T, F) and DNN features `Z.npy` (N, P). Run:
```bash
python rsa_cli.py --neural X.npy --dnn Z.npy --out_dir outputs/rsa
```

### Learning: Representational Similarity Analysis
RSA compares dissimilarity structures. Time-resolved RSA tracks when neural representations align with model representations.

