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
    # Add edges with Euclidean length as weight
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


def dijkstra_distances(G: nx.Graph, sources: list[int]) -> np.ndarray:
    # Multi-source by virtual super source technique
    # Here we run multi_source_dijkstra for efficiency
    dist = {n: np.inf for n in G.nodes}
    for s in sources:
        if s not in G:
            raise ValueError(f"Source {s} not a valid vertex index")
    dist_map, _ = nx.multi_source_dijkstra(G, sources, weight='weight')
    for n, d in dist_map.items():
        dist[n] = d
    # Convert to array
    num_nodes = G.number_of_nodes()
    out = np.full(num_nodes, np.inf, dtype=float)
    for n in range(num_nodes):
        out[n] = dist.get(n, np.inf)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approximate geodesic distances on a triangular mesh via graph shortest paths")
    p.add_argument('--mesh', required=True, type=str)
    p.add_argument('--source-vertex', type=int, default=None, help='Single source vertex index')
    p.add_argument('--source-vertices', type=int, nargs='*', default=None, help='Multiple source vertex indices')
    p.add_argument('--output', type=str, required=True, help='Path to save distances (.npy)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args.mesh)
    G = build_graph(mesh)
    sources = []
    if args.source_vertices is not None and len(args.source_vertices) > 0:
        sources.extend(args.source_vertices)
    if args.source_vertex is not None:
        sources.append(args.source_vertex)
    if len(sources) == 0:
        raise SystemExit("Provide --source-vertex or --source-vertices")
    d = dijkstra_distances(G, sources)
    np.save(args.output, d)
    print(f"Saved distances to {args.output}")


if __name__ == '__main__':
    main()
