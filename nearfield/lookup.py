from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree


def _wrap_theta(theta_deg: np.ndarray) -> np.ndarray:
    t = np.asarray(theta_deg, dtype=np.float64)
    t = (t + 180.0) % 360.0 - 180.0
    t[t == 180.0] = -180.0
    return t


def _clamp_phi(phi_deg: np.ndarray) -> np.ndarray:
    p = np.asarray(phi_deg, dtype=np.float64)
    return np.clip(p, -90.0, 90.0)


def _rtp_features(rtp: np.ndarray) -> np.ndarray:
    if not isinstance(rtp, np.ndarray):
        raise ValueError("rtp must be a numpy ndarray")
    if rtp.ndim != 2 or rtp.shape[1] != 3:
        raise ValueError("rtp must have shape (K,3)")
    if not np.isfinite(rtp).all():
        raise ValueError("rtp contains NaN/inf")
    r = rtp[:, 0]
    if np.any(r <= 0):
        raise ValueError("All radii must be > 0")
    th = _wrap_theta(rtp[:, 1])
    ph = _clamp_phi(rtp[:, 2])
    f = np.empty_like(rtp, dtype=np.float64)
    f[:, 0] = np.log(r)
    f[:, 1] = th / 180.0
    f[:, 2] = ph / 90.0
    return f


@dataclass
class RTPKDTree:
    tree: KDTree
    index_map: np.ndarray  # maps augmented feature index -> original grid index
    features: np.ndarray   # augmented feature matrix used to build KDTree


def build_kdtree(rtp_grid: np.ndarray):
    """Return KD-tree over normalized features [log r, theta/180, phi/90] with angle wrapping handled.

    Strategy
    - Build base features F0 = [log r, th/180, ph/90]. To address azimuth wrap at ±180,
      augment dataset with duplicates whose theta feature shifted by ±2 (i.e., ±360°/180°).
    - Keep an index map so NN queries can be mapped back to the original rtp_grid rows.
    """
    F0 = _rtp_features(rtp_grid)

    # Create augmented features to handle wrap at -180/180 for theta.
    Fm = F0.copy(); Fm[:, 1] -= 2.0
    Fp = F0.copy(); Fp[:, 1] += 2.0
    F_aug = np.vstack([F0, Fm, Fp])
    idx_map = np.concatenate([
        np.arange(F0.shape[0], dtype=np.int64),
        np.arange(F0.shape[0], dtype=np.int64),
        np.arange(F0.shape[0], dtype=np.int64),
    ])

    tree = KDTree(F_aug, leaf_size=40, metric="euclidean")
    return RTPKDTree(tree=tree, index_map=idx_map, features=F_aug)


def nearest_codeword(
    tree,
    rtp_grid: np.ndarray,
    query_rtp: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and distances of nearest neighbors in the codebook grid.

    Parameters
    - tree: object returned by build_kdtree.
    - rtp_grid: (K,3) original grid. Used only for validation.
    - query_rtp: (Q,3) or (3,) array of [r, theta_deg, phi_deg].
    - k: number of neighbors to return.

    Returns
    - (indices, distances):
      - indices: (Q,k) int64 indices into rtp_grid rows.
      - distances: (Q,k) float64 KD-tree distances in feature space.
    """
    if not isinstance(tree, RTPKDTree):
        raise ValueError("tree must be the object returned by build_kdtree")
    if not isinstance(rtp_grid, np.ndarray) or rtp_grid.ndim != 2 or rtp_grid.shape[1] != 3:
        raise ValueError("rtp_grid must be (K,3)")

    q = np.asarray(query_rtp, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("query_rtp must be (Q,3) or (3,)")

    Fq = _rtp_features(q)
    dist, idx = tree.tree.query(Fq, k=k, return_distance=True)
    # Map augmented indices back to original grid indices
    mapped = tree.index_map[idx]
    return mapped, dist

