## Project 32: Integrated Information Architecture Mapper (IIT-Φ Dynamics)

### Overview
Multi-scale computational framework combining Integrated Information Theory (IIT), graph theory, topological data analysis, and dynamical systems to map consciousness landscapes.

Core components:
- **Φ Computation**: Integrated information via MIP (Minimum Information Partition) search
- **Graph Analysis**: Connectivity metrics, hub identification, rich-club coefficient
- **Topological Data Analysis**: Persistent homology to identify high-dimensional structures
- **Consciousness Landscape**: Basin of attraction mapping with perturbation analysis
- **Dynamical Systems**: Attractor identification and state transition dynamics

### Data
- **Connectivity matrix** `--connectivity`: `.npy` or `.npz` with shape `(N, N)` adjacency/connectivity
- **State sequence** `--states` (optional): `.npy` or `.npz` with shape `(T, N)` binary or continuous states

### Install
```bash
pip install numpy scipy matplotlib
```

### Usage

**Graph metrics only:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --graph_metrics \
  --out_dir outputs/iit_graph
```

**Φ profile from state sequence:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --states states.npy \
  --compute_phi \
  --out_dir outputs/iit_phi
```

**Topological complexity:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --states states.npy \
  --topology \
  --out_dir outputs/iit_topo
```

**Consciousness landscape with attractors:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --landscape \
  --n_attractors 100 \
  --grid_res 30 \
  --out_dir outputs/iit_landscape
```

**Perturbation analysis:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --perturbation \
  --perturb_nodes 0,1,2 \
  --perturb_strength 0.5 \
  --out_dir outputs/iit_perturb
```

**Full pipeline:**
```bash
python iit_cli.py \
  --connectivity A.npy \
  --states states.npy \
  --compute_phi --graph_metrics --topology --landscape --perturbation \
  --n_attractors 50 --grid_res 20 \
  --out_dir outputs/iit_full
```

### Outputs

- **summary.json** — All computed metrics and statistics
- **phi_profile.npy** — Φ values over time (if states provided)
- **phi_timecourse.png** — Φ time series plot
- **graph_metrics.npz** — Degree, clustering per node
- **rich_club.json** — Rich-club coefficients by degree
- **attractors.npz** — Identified attractors and basin sizes
- **consciousness_landscape.png** — 2D Φ landscape with attractors
- **perturbation_divergence.png** — Trajectory divergence after perturbation

### Methods

#### Integrated Information (Φ)
- Simplified IIT 3.0 implementation
- MIP search over bipartitions
- EMD-based distance between whole and partitioned distributions
- Proxy: entropy of transition probability distribution

#### Graph Metrics
- In/out degree, clustering coefficient
- Global efficiency (connectivity-based proxy)
- Modularity via Fiedler value (spectral partitioning)
- Hub identification (top 25% by degree)
- Rich-club coefficient

#### Topological Data Analysis
- Vietoris-Rips filtration on state point cloud
- Persistent homology (Betti numbers b0, b1, b2)
- Topological complexity = sum of persistent features

#### Consciousness Landscape
- Attractor search via random initialization
- Basin of attraction estimation (Monte Carlo)
- 2D PCA projection with Φ contours
- Perturbation analysis: trajectory divergence

### Notes
- Φ computation is simplified (full IIT requires cause-effect repertoires)
- TDA uses basic persistent homology (for production, use `ripser` or `gudhi`)
- Landscape visualization uses PCA for 2D projection
- Perturbation analysis tracks L2 divergence over time

### Roadmap
- Full IIT 3.0 with cause-effect structures
- Dynamic causal modeling for state transitions
- ML classifier trained on Φ-profiles
- Interactive 3D visualization (Plotly/VTK)
- Multi-state comparison (wake/sleep/anesthesia)
