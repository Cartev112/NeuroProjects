import argparse
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Unsupported mesh type")
    if mesh.faces is None or mesh.vertices is None:
        raise ValueError("Mesh missing faces or vertices")
    if mesh.faces.shape[1] != 3:
        raise ValueError("Only triangular meshes are supported")
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.rezero()
    return mesh


def estimate_curvature(vertices: np.ndarray, faces: np.ndarray, k_neighbors: int = 20):
    # Use PCA-based local frame and quadratic surface fit in tangent plane.
    V = vertices.view(np.ndarray)
    F = faces.view(np.ndarray)
    n_verts = V.shape[0]
    # Compute per-vertex normals
    mesh_tmp = trimesh.Trimesh(vertices=V, faces=F, process=False)
    normals = mesh_tmp.vertex_normals

    tree = cKDTree(V)
    k = min(k_neighbors, n_verts-1)

    k1 = np.zeros(n_verts)
    k2 = np.zeros(n_verts)

    for vid in range(n_verts):
        dists, idxs = tree.query(V[vid], k=k+1)
        idxs = idxs[1:]  # exclude self
        P = V[idxs]
        p0 = V[vid]

        # Build local frame (normal = n, tangent basis t1,t2)
        n = normals[vid]
        n = n / (np.linalg.norm(n) + 1e-12)
        # choose arbitrary t1 not parallel to n
        a = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        t1 = a - np.dot(a, n) * n
        t1 = t1 / (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(n, t1)

        # Project neighbors to local frame
        X = P - p0
        x = X @ t1
        y = X @ t2
        z = X @ n

        # Fit quadratic: z = 0.5*(a x^2 + 2b x y + c y^2) + d x + e y + f
        A = np.column_stack([0.5*x*x, x*y, 0.5*y*y, x, y, np.ones_like(x)])
        try:
            coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(6)
        a, b, c, _, _, _ = coeffs
        # Second fundamental form in local frame ~ [[a, b], [b, c]]
        H = np.array([[a, b], [b, c]])
        # Principal curvatures are eigenvalues of H
        w, _ = np.linalg.eigh(H)
        # Sort so that k1 >= k2
        kmax = float(np.max(w))
        kmin = float(np.min(w))
        k1[vid] = kmax
        k2[vid] = kmin

    H_mean = 0.5 * (k1 + k2)
    K_gauss = k1 * k2
    return k1, k2, H_mean, K_gauss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate principal curvatures on a triangular mesh")
    p.add_argument('--mesh', required=True, type=str)
    p.add_argument('--k-neighbors', type=int, default=20)
    p.add_argument('--output-dir', type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mesh = load_mesh(args.mesh)
    k1, k2, H, K = estimate_curvature(mesh.vertices, mesh.faces, k_neighbors=args.k_neighbors)
    np.save(os.path.join(args.output_dir, 'k1.npy'), k1)
    np.save(os.path.join(args.output_dir, 'k2.npy'), k2)
    np.save(os.path.join(args.output_dir, 'mean_curvature.npy'), H)
    np.save(os.path.join(args.output_dir, 'gaussian_curvature.npy'), K)
    print(f"Saved curvature maps to {args.output_dir}")


if __name__ == '__main__':
    main()
