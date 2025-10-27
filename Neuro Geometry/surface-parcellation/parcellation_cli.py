import argparse
import numpy as np
import trimesh
import networkx as nx


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Unsupported mesh type")
    if mesh.faces is None or mesh.vertices is None:
        raise ValueError("Mesh missing faces or vertices")
    if mesh.faces.shape[1] != 3:
        raise ValueError("Only triangular meshes are supported")
    return mesh


def build_graph(mesh: trimesh.Trimesh) -> nx.Graph:
    G = nx.Graph()
    coords = mesh.vertices.view(np.ndarray)
    G.add_nodes_from(range(coords.shape[0]))
    edges = set()
    for tri in mesh.faces.view(np.ndarray):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    for (a, b) in edges:
        w = float(np.linalg.norm(coords[a] - coords[b]))
        G.add_edge(a, b, weight=w)
    return G


def multi_source_dist(G: nx.Graph, seeds: list[int]) -> dict[int, float]:
    dist_map, _ = nx.multi_source_dijkstra(G, seeds, weight='weight')
    return dist_map


def assign_labels(num_vertices: int, dist_maps: list[dict[int, float]], seeds: list[int]) -> np.ndarray:
    labels = np.full(num_vertices, -1, dtype=int)
    dists = np.full((num_vertices, len(seeds)), np.inf, dtype=float)
    for s_idx, dmap in enumerate(dist_maps):
        for v, d in dmap.items():
            dists[v, s_idx] = d
    labels = np.argmin(dists, axis=1)
    return labels


def geodesic_centroid(G: nx.Graph, vertices: np.ndarray, members: np.ndarray) -> int:
    # Choose member minimizing sum of distances to others (1-median on graph)
    if members.size == 0:
        return int(np.random.randint(0, vertices.shape[0]))
    sub = list(members)
    # Precompute single-source shortest paths from an arbitrary seed subset center
    # Use heuristic: pick the member with smallest eccentricity in induced subgraph
    best_node = sub[0]
    best_sum = np.inf
    lengths = nx.multi_source_dijkstra_path_length(G, sub, weight='weight')
    # Evaluate sum of distances within cluster using lengths; fallback to argmin of distance to seeds
    for node in sub:
        total = 0.0
        for other in sub:
            if other == node:
                continue
            d = nx.dijkstra_path_length(G, node, other, weight='weight')
            total += d
        if total < best_sum:
            best_sum = total
            best_node = node
    return int(best_node)


def lloyd_iteration(G: nx.Graph, num_vertices: int, seeds: list[int]) -> tuple[np.ndarray, list[int]]:
    # Compute distances per seed
    dist_maps = [multi_source_dist(G, [s]) for s in seeds]
    labels = assign_labels(num_vertices, dist_maps, seeds)
    # Recompute seeds as geodesic centroids
    new_seeds: list[int] = []
    members_by_seed = [np.where(labels == si)[0] for si in range(len(seeds))]
    for members in members_by_seed:
        new_seed = geodesic_centroid(G, None, members)
        new_seeds.append(new_seed)
    return labels, new_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surface parcellation via geodesic Voronoi with one Lloyd iteration")
    p.add_argument('--mesh', required=True, type=str)
    p.add_argument('--num-seeds', type=int, default=None, help='Number of random seeds to initialize')
    p.add_argument('--seeds', type=int, nargs='*', default=None, help='Explicit seed vertex indices')
    p.add_argument('--output', type=str, required=True, help='Path to save labels (.npy)')
    p.add_argument('--save-seeds', type=str, default=None, help='Path to save final seeds (.npy)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args.mesh)
    G = build_graph(mesh)

    if args.seeds is None and args.num_seeds is None:
        raise SystemExit('Provide --seeds or --num-seeds')
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    else:
        n = mesh.vertices.shape[0]
        seeds = np.random.choice(n, size=int(args.num_seeds), replace=False).astype(int).tolist()

    labels, new_seeds = lloyd_iteration(G, mesh.vertices.shape[0], seeds)
    np.save(args.output, labels)
    if args.save_seeds is not None:
        np.save(args.save_seeds, np.array(new_seeds, dtype=int))
    print(f"Saved labels to {args.output}")
    if args.save_seeds is not None:
        print(f"Saved seeds to {args.save_seeds}")


if __name__ == '__main__':
    main()
