import argparse
import os
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh


def load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type loaded from {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces. Provide a triangular surface mesh.")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices.")
    if mesh.faces.shape[1] != 3:
        raise ValueError("Only triangular meshes are supported.")
    return mesh


def compute_cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Compute cotangent Laplacian L and lumped vertex areas A (as 1D array).

    Returns
    -------
    L : csr_matrix shape (V, V)
        Symmetric cotangent Laplacian (negative-semidefinite)
    A : ndarray shape (V,)
        Lumped vertex areas (positive), used as the mass diagonal
    """
    v = vertices
    f = faces
    num_vertices = v.shape[0]

    i = f[:, 0]
    j = f[:, 1]
    k = f[:, 2]

    vi = v[i]
    vj = v[j]
    vk = v[k]

    normals = np.cross(vj - vi, vk - vi)
    tri_areas = 0.5 * np.linalg.norm(normals, axis=1)
    valid = tri_areas > 1e-16
    if not np.all(valid):
        vi = vi[valid]
        vj = vj[valid]
        vk = vk[valid]
        i = i[valid]
        j = j[valid]
        k = k[valid]
        normals = normals[valid]
        tri_areas = tri_areas[valid]

    l_ij2 = np.sum((vi - vj) ** 2, axis=1)
    l_jk2 = np.sum((vj - vk) ** 2, axis=1)
    l_ki2 = np.sum((vk - vi) ** 2, axis=1)

    cot_alpha = (l_jk2 + l_ki2 - l_ij2) / (4.0 * tri_areas)
    cot_beta = (l_ki2 + l_ij2 - l_jk2) / (4.0 * tri_areas)
    cot_gamma = (l_ij2 + l_jk2 - l_ki2) / (4.0 * tri_areas)

    cot_alpha = np.clip(cot_alpha, -1e6, 1e6)
    cot_beta = np.clip(cot_beta, -1e6, 1e6)
    cot_gamma = np.clip(cot_gamma, -1e6, 1e6)

    row = np.concatenate([i, j, j, k, k, i])
    col = np.concatenate([j, i, k, j, i, k])
    wts = 0.5 * np.concatenate([
        cot_gamma, cot_gamma,  # (i,j)
        cot_alpha, cot_alpha,  # (j,k)
        cot_beta, cot_beta     # (k,i)
    ])
    W = sp.coo_matrix((wts, (row, col)), shape=(num_vertices, num_vertices)).tocsr()
    diag = -np.array(W.sum(axis=1)).ravel()
    L = sp.diags(diag) + W

    def angles(a, b, c):
        ba = b - a
        ca = c - a
        cosang = np.sum(ba * ca, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(ca, axis=1))
        return np.arccos(np.clip(cosang, -1.0, 1.0))

    alpha = angles(vi, vj, vk)
    beta = angles(vj, vk, vi)
    gamma = angles(vk, vi, vj)
    obtuse = (alpha > np.pi/2) | (beta > np.pi/2) | (gamma > np.pi/2)

    Ai = np.zeros(num_vertices, dtype=np.float64)
    Aj = np.zeros(num_vertices, dtype=np.float64)
    Ak = np.zeros(num_vertices, dtype=np.float64)

    non_obtuse = ~obtuse
    lij = np.sqrt(l_ij2)
    ljk = np.sqrt(l_jk2)
    lki = np.sqrt(l_ki2)
    Ai_contrib = (lij**2 * cot_gamma + lki**2 * cot_beta) / 8.0
    Aj_contrib = (lij**2 * cot_gamma + ljk**2 * cot_alpha) / 8.0
    Ak_contrib = (lki**2 * cot_beta + ljk**2 * cot_alpha) / 8.0

    for idx in np.where(non_obtuse)[0]:
        Ai[i[idx]] += Ai_contrib[idx]
        Aj[j[idx]] += Aj_contrib[idx]
        Ak[k[idx]] += Ak_contrib[idx]

    for idx in np.where(obtuse)[0]:
        area = tri_areas[idx]
        if alpha[idx] > np.pi/2:
            Ai[i[idx]] += 0.5 * area
            Aj[j[idx]] += 0.25 * area
            Ak[k[idx]] += 0.25 * area
        elif beta[idx] > np.pi/2:
            Aj[j[idx]] += 0.5 * area
            Ai[i[idx]] += 0.25 * area
            Ak[k[idx]] += 0.25 * area
        else:
            Ak[k[idx]] += 0.5 * area
            Ai[i[idx]] += 0.25 * area
            Aj[j[idx]] += 0.25 * area

    A = Ai + Aj + Ak
    A[A <= 1e-16] = np.median(A[A > 0]) if np.any(A > 0) else 1.0

    return L.tocsr(), A


def solve_eigenpairs(L: sp.csr_matrix, mass_diag: np.ndarray, num_eigs: int) -> Tuple[np.ndarray, np.ndarray]:
    M = sp.diags(mass_diag)
    try:
        vals, vecs = spla.eigsh(L, k=num_eigs, M=M, sigma=0.0, which='LM')
    except Exception:
        vals, vecs = spla.eigsh(L, k=num_eigs, sigma=0.0, which='LM')
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def compute_hks(eigenvalues: np.ndarray, eigenvectors: np.ndarray, times: np.ndarray) -> np.ndarray:
    lam = np.maximum(eigenvalues, 0.0)
    phi2 = eigenvectors ** 2
    H = np.stack([np.exp(-t * lam) @ phi2.T for t in times], axis=1)
    return H


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laplace–Beltrami eigenpairs and HKS on a triangular mesh")
    parser.add_argument('--mesh', type=str, required=True, help='Path to mesh file (obj/ply/stl/off)')
    parser.add_argument('--num-eigs', type=int, default=100, help='Number of eigenpairs to compute')
    parser.add_argument('--num-times', type=int, default=10, help='Number of HKS time scales')
    parser.add_argument('--t-min', type=float, default=1e-4, help='Minimum HKS time scale')
    parser.add_argument('--t-max', type=float, default=1e-1, help='Maximum HKS time scale')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--save-evecs', dest='save_evecs', action='store_true')
    parser.add_argument('--no-save-evecs', dest='save_evecs', action='store_false')
    parser.set_defaults(save_evecs=True)
    parser.add_argument('--save-evals', dest='save_evals', action='store_true')
    parser.add_argument('--no-save-evals', dest='save_evals', action='store_false')
    parser.set_defaults(save_evals=True)
    parser.add_argument('--save-hks', dest='save_hks', action='store_true')
    parser.add_argument('--no-save-hks', dest='save_hks', action='store_false')
    parser.set_defaults(save_hks=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mesh = load_trimesh(args.mesh)
    L, A = compute_cotangent_weights(mesh.vertices.view(np.ndarray), mesh.faces.view(np.ndarray))

    num_eigs = min(args.num_eigs, mesh.vertices.shape[0] - 1) if mesh.vertices.shape[0] > 1 else args.num_eigs
    evals, evecs = solve_eigenpairs(L, A, num_eigs)

    times = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.num_times)
    hks = compute_hks(evals, evecs, times)

    if args.save_evals:
        np.save(os.path.join(args.output_dir, 'eigenvalues.npy'), evals)
    if args.save_evecs:
        np.save(os.path.join(args.output_dir, 'eigenvectors.npy'), evecs)
    if args.save_hks:
        np.save(os.path.join(args.output_dir, 'hks.npy'), hks)
    np.save(os.path.join(args.output_dir, 'times.npy'), times)

    print(f"Saved results to: {args.output_dir}")


if __name__ == '__main__':
    main()
