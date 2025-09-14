from __future__ import annotations

import numpy as np

from .spherical import C_MPS


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    if not isinstance(xyz, np.ndarray):
        raise ValueError("xyz_m must be a numpy ndarray")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_m must have shape (M,3)")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz_m contains NaN/inf")
    return np.asarray(xyz, dtype=np.float64)


def _validate_point(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size != 3:
        raise ValueError("p_xyz must have 3 entries")
    if not np.isfinite(p).all():
        raise ValueError("p_xyz contains NaN/inf")
    return p


def subcarrier_frequencies(fc_hz: float, bw_hz: float, n_sc: int) -> np.ndarray:
    """Return (K,) subcarrier frequencies in Hz centered at fc_hz.

    f_k = fc + (k - (K-1)/2) * Δf, where Δf = bw_hz / n_sc, k=0..K-1.
    """
    if not np.isfinite(fc_hz) or not np.isfinite(bw_hz):
        raise ValueError("fc_hz and bw_hz must be finite")
    if fc_hz <= 0 or bw_hz <= 0:
        raise ValueError("fc_hz and bw_hz must be positive")
    if not isinstance(n_sc, int) or n_sc < 1:
        raise ValueError("n_sc must be a positive integer")
    K = n_sc
    delta_f = bw_hz / float(K)
    k = np.arange(K, dtype=np.float64)
    f = fc_hz + (k - (K - 1) / 2.0) * delta_f
    return f


def spherical_steering_wideband(
    xyz_m: np.ndarray, p_xyz: np.ndarray, f_sc_hz: np.ndarray
) -> np.ndarray:
    """Return (K,M) steering across subcarriers for focus p (near-field delays).

    a_km = exp(-j * 2π f_k * τ_m), with τ_m = ||p - x_m|| / c.
    No normalization is applied (|a_m|=1).
    """
    xyz = _validate_xyz(xyz_m)
    p = _validate_point(p_xyz)
    f = np.asarray(f_sc_hz, dtype=np.float64).ravel()
    if f.ndim != 1 or f.size < 1 or not np.isfinite(f).all():
        raise ValueError("f_sc_hz must be a finite 1D array")

    dists = np.linalg.norm(p[None, :] - xyz, axis=1)  # (M,)
    tau = dists / C_MPS  # (M,)
    phase = -1j * 2.0 * np.pi * f[:, None] * tau[None, :]
    A = np.exp(phase).astype(np.complex128)
    return A

