## Project 15: Graph Features on rs-fMRI Connectomes

### Overview
Compute graph-theoretic features from FC matrices and predict phenotype (e.g., age/sex) with ML baselines.

### Instructions
Provide FC matrices `corr.npy` (N, R, R) and labels `y.npy`. Run:
```bash
python graph_cli.py --corr corr.npy --labels y.npy --out_dir outputs/graph
```

### Learning: Graph Metrics
Features like degree, clustering, path length, modularity summarize network topology. Regularization and permutation testing assess significance.

