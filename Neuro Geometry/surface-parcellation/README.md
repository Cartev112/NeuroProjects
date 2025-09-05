# Surface Parcellation via Geodesic Voronoi + Lloyd

This project performs geodesic Voronoi parcellation on a mesh using graph shortest paths as the geodesic proxy and runs one Lloyd iteration (seed relocation to geodesic centroid).

## Quickstart

```bash
python parcellation_cli.py --mesh path/to/mesh.obj --num-seeds 50 --output labels.npy
```

Provide custom seeds:
```bash
python parcellation_cli.py --mesh path/to/mesh.obj --seeds 1 42 128 --output labels.npy
```

Outputs:
- `labels.npy`: array of shape `(num_vertices,)` with parcel indices in `[0, K-1]`
- `seeds.npy`: final seed vertices
