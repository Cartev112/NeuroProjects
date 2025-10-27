# Geodesic Distances on Surfaces (Graph Shortest Paths Approximation)

This project computes approximate geodesic distances on triangular meshes using graph shortest paths (edge lengths as weights). It also supports multi-source distances and saving vertexwise distance maps.

## Quickstart

```bash
python geodesic_cli.py --mesh path/to/mesh.obj --source-vertex 0 --output distances.npy
```

Multi-source example:
```bash
python geodesic_cli.py --mesh path/to/mesh.obj --source-vertices 1 42 128 --output distances.npy
```

## Notes
- This is a graph-based approximation. For higher fidelity, consider the heat method or exact MMP.
- Ensure the mesh is manifold and connected for meaningful distances.
