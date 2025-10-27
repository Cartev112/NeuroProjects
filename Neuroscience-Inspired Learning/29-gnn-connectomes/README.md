## Project 29: GNNs on Structural/Functional Connectomes

### Overview
Train a Graph Neural Network to predict phenotype from brain graphs (nodes=ROIs, edges=connectivity weights).

### Instructions
Provide adjacency tensors `A.npy` (N, R, R) and optional node features `F.npy` (R, D). Run:
```bash
python gnn_cli.py --A A.npy --labels y.npy --out_dir outputs/gnn
```

### Learning: Graph Learning on Connectomes
GNNs aggregate neighbor information to learn graph representations; careful normalization and regularization are needed to avoid overfitting.

