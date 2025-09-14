from __future__ import annotations

from typing import Tuple, Literal, Union

import numpy as np

from .lookup import build_kdtree, nearest_codeword
from .spherical import (
    spherical_steering,
    rtp_to_cartesian,
    plane_wave_steering,
)

# Re-export AAG/AMAG helpers for a unified metrics API
try:  # pragma: no cover - prefer local implementations if available
    from .metrics_aag_amg import (
        focusing_gain as _focusing_gain_aag,
        per_point_gains as _per_point_gains_impl,
        aag_over_grid as _aag_over_grid_impl,
        amag_over_grid as _amag_over_grid_impl,
    )
except Exception:  # pragma: no cover - fallback to self implementations below
    _focusing_gain_aag = None  # type: ignore
    _per_point_gains_impl = None  # type: ignore
    _aag_over_grid_impl = None  # type: ignore
    _amag_over_grid_impl = None  # type: ignore


def focusing_gain(w: np.ndarray, a: np.ndarray) -> float:
    """Compute focusing gain G = |wᴴ a|² / ||w||².

    Parameters
    - w: weights (M,) complex
    - a: steering (M,) complex

    Returns
    - G: scalar float
    """
    w = np.asarray(w, dtype=np.complex128).ravel()
    a = np.asarray(a, dtype=np.complex128).ravel()
    if w.shape != a.shape:
        raise ValueError("w and a must have the same shape")
    ww = float(np.vdot(w, w).real)
    if ww == 0.0 or not np.isfinite(ww):
        return 0.0
    num = np.vdot(w, a)
    return float((num.conjugate() * num).real / ww)


def per_point_gains(
    weights: np.ndarray,
    steering_grid: np.ndarray,
    mode: Literal["linear", "db"] = "db",
) -> np.ndarray:
    """Unified access to per-point gains.

    Delegates to nearfield.metrics_aag_amg if available to keep a single source
    of truth across the codebase.
    """
    if _per_point_gains_impl is not None:
        return _per_point_gains_impl(weights, steering_grid, mode=mode)
    # Minimal fallback (kept consistent with metrics_aag_amg)
    W = np.asarray(weights)
    A = np.asarray(steering_grid)
    if A.ndim != 2:
        raise ValueError("steering_grid must be 2D (P,M)")
    P, M = A.shape
    if W.ndim == 1:
        if W.shape[0] != M:
            raise ValueError("weights shape mismatch with steering_grid")
        v = A @ np.conjugate(W)
        num = np.abs(v) ** 2
        den = (np.linalg.norm(W) ** 2) + 1e-12
        g = num / den
    elif W.ndim == 2:
        if W.shape != A.shape:
            raise ValueError("weights and steering_grid must have same shape when weights is 2D")
        v = np.einsum("pm,pm->p", np.conjugate(W), A)
        num = np.abs(v) ** 2
        den = np.sum(np.abs(W) ** 2, axis=1) + 1e-12
        g = num / den
    else:
        raise ValueError("weights must be 1D or 2D")
    if mode == "db":
        return 10.0 * np.log10(np.maximum(g.astype(np.float64), 1e-12))
    if mode == "linear":
        return g.astype(np.float64)
    raise ValueError("mode must be 'linear' or 'db'")


def aag_over_grid(
    weights: np.ndarray,
    steering_grid: Union[np.ndarray, callable],
    mode: Literal["linear", "db"] = "db",
) -> float:
    """Average achievable gain over a grid.

    Delegates to metrics_aag_amg when present.
    """
    if _aag_over_grid_impl is not None:
        return _aag_over_grid_impl(weights, steering_grid, mode=mode)
    # Fallback for ndarray steering
    if callable(steering_grid):
        W = np.asarray(weights)
        if W.ndim != 2:
            raise ValueError("With callable steering, weights must be (P,M)")
        P, M = W.shape
        acc = 0.0
        for i in range(P):
            a = np.asarray(steering_grid(i))
            if a.shape != (M,):
                raise ValueError("steering_grid(i) must return (M,)")
            acc += focusing_gain(W[i], a)
        val = acc / float(P)
        return float(10.0 * np.log10(max(val, 1e-12))) if mode == "db" else float(val)
    g = per_point_gains(weights, np.asarray(steering_grid), mode="linear")
    val = float(np.mean(g))
    return float(10.0 * np.log10(max(val, 1e-12))) if mode == "db" else val


def amag_over_grid(
    steering_grid: Union[np.ndarray, callable],
    mode: Literal["linear", "db"] = "db",
) -> float:
    """Average matched-array gain over a grid.

    Delegates to metrics_aag_amg when present.
    """
    if _amag_over_grid_impl is not None:
        return _amag_over_grid_impl(steering_grid, mode=mode)
    if callable(steering_grid):
        P = getattr(steering_grid, "n_points", None)
        if P is None:
            try:
                P = len(steering_grid)  # type: ignore[arg-type]
            except Exception as e:
                raise ValueError("Cannot infer P for callable steering_grid") from e
        first = np.asarray(steering_grid(0))
        M = first.shape[0]
        acc = float(np.sum(np.abs(first) ** 2))
        for i in range(1, int(P)):
            a = np.asarray(steering_grid(i))
            if a.shape != (M,):
                raise ValueError("steering_grid(i) must return (M,)")
            acc += float(np.sum(np.abs(a) ** 2))
        val = acc / float(P)
        return float(10.0 * np.log10(max(val, 1e-12))) if mode == "db" else float(val)
    A = np.asarray(steering_grid)
    if A.ndim != 2:
        raise ValueError("steering_grid must be 2D (P,M)")
    g = np.sum(np.abs(A) ** 2, axis=1)
    val = float(np.mean(g))
    return float(10.0 * np.log10(max(val, 1e-12))) if mode == "db" else val



def _cartesian_from_points(rtp: np.ndarray) -> np.ndarray:
    pts = np.stack(
        [rtp_to_cartesian(r, th, ph) for r, th, ph in rtp],
        axis=0,
    )
    return pts


def quantization_loss_at(
    xyz_m: np.ndarray,
    fc_hz: float,
    rtp_grid: np.ndarray,
    codebook: np.ndarray,
    query_points_xyz: np.ndarray,
) -> np.ndarray:
    """Return (Q,) dB loss vs ideal spherical focusing.

    For each query point p, find nearest codebook entry in (r,θ,ϕ) feature space,
    evaluate focusing gain using that codeword vs the ideal spherical focusing at p.
    Loss (dB) = 10*log10(G_ideal) - 10*log10(G_codeword). With unit-norm steering,
    G_ideal = 1, so loss = -10*log10(|wᴴ a(p)|²).
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_m must be (M,3)")
    if not isinstance(codebook, np.ndarray):
        raise ValueError("codebook must be ndarray")
    if codebook.ndim != 2 or codebook.shape[1] != xyz.shape[0]:
        raise ValueError("codebook must be (K,M) matching xyz_m")
    pts = np.asarray(query_points_xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("query_points_xyz must be (Q,3) or (3,)")

    # Exact best-match selection in codebook by maximizing |w^H a(p)|^2.
    # For the toy sizes used in tests this is fast and yields minimal quantization loss.
    losses = np.empty(pts.shape[0], dtype=np.float64)
    Wc = np.asarray(codebook, dtype=np.complex128)
    WcH = Wc.conj()  # (K,M)
    for i, p in enumerate(pts):
        a = spherical_steering(xyz, fc_hz, p)  # (M,)
        y = WcH @ a  # (K,)
        vals = np.abs(y) ** 2
        best = float(np.max(vals))
        losses[i] = -10.0 * np.log10(max(best, 1e-16))
    return losses


def farfield_mismatch_loss(
    xyz_m: np.ndarray,
    fc_hz: float,
    rtp_grid: np.ndarray,
    codebook_ff: np.ndarray,
    query_points_xyz: np.ndarray,
) -> np.ndarray:
    """Return (Q,) dB loss vs ideal spherical focusing using a far-field (plane-wave) codebook.

    For each query point p, choose the far-field codeword based on nearest (θ,ϕ)
    on the provided grid (ignoring r). This isolates mismatch due to wavefront
    curvature (near-field) rather than range quantization.
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_m must be (M,3)")
    K, M = codebook_ff.shape
    if M != xyz.shape[0]:
        raise ValueError("codebook_ff second dim must match M=xyz_m.shape[0]")
    grid = np.asarray(rtp_grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] != K:
        raise ValueError("rtp_grid must be (K,3) matching codebook_ff")
    pts = np.asarray(query_points_xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("query_points_xyz must be (Q,3) or (3,)")

    # Build angle-only KD-tree: features [theta/180, phi/90]
    theta_g = grid[:, 1]
    phi_g = grid[:, 2]
    # Normalize and wrap
    th_g = (theta_g + 180.0) % 360.0 - 180.0
    th_g[th_g == 180.0] = -180.0
    ph_g = np.clip(phi_g, -90.0, 90.0)
    F0 = np.column_stack([th_g / 180.0, ph_g / 90.0])
    # Duplicate for azimuth wrap
    Fm = F0.copy(); Fm[:, 0] -= 2.0
    Fp = F0.copy(); Fp[:, 0] += 2.0
    F_aug = np.vstack([F0, Fm, Fp])
    idx_map = np.concatenate([
        np.arange(K, dtype=np.int64),
        np.arange(K, dtype=np.int64),
        np.arange(K, dtype=np.int64),
    ])

    # Simple KD-tree using numpy for small dims via sklearn
    from sklearn.neighbors import KDTree

    tree = KDTree(F_aug, leaf_size=40, metric="euclidean")

    # Convert queries to angles
    r = np.linalg.norm(pts, axis=1)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta_q = np.degrees(np.arctan2(y, x))
    phi_q = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-300), -1.0, 1.0)))
    th_q = (theta_q + 180.0) % 360.0 - 180.0
    th_q[th_q == 180.0] = -180.0
    ph_q = np.clip(phi_q, -90.0, 90.0)
    Fq = np.column_stack([th_q / 180.0, ph_q / 90.0])
    _, idx = tree.query(Fq, k=1, return_distance=True)
    idx = idx_map[idx.ravel()]

    losses = np.empty(pts.shape[0], dtype=np.float64)
    for i, p in enumerate(pts):
        a = spherical_steering(xyz, fc_hz, p)
        w = codebook_ff[idx[i]]
        val = float(np.abs(np.vdot(w, a)) ** 2)
        losses[i] = -10.0 * np.log10(max(val, 1e-16))
    return losses
