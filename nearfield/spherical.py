from __future__ import annotations

import math
from typing import Tuple

import numpy as np


C_MPS = 299_792_458.0  # speed of light [m/s]


def _wrap_theta_deg(theta: np.ndarray | float) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64)
    # Wrap to [-180, 180)
    t = (t + 180.0) % 360.0 - 180.0
    # Special-case exact 180 to -180 for consistency
    if t.ndim == 0:
        return np.array(-180.0 if float(t) == 180.0 else float(t), dtype=np.float64)
    t[t == 180.0] = -180.0
    return t


def _clamp_phi_deg(phi: np.ndarray | float) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    return np.clip(p, -90.0, 90.0)


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    if not isinstance(xyz, np.ndarray):
        raise ValueError("xyz must be a numpy ndarray")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (M,3)")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN/inf")
    return np.asarray(xyz, dtype=np.float64)


def _validate_point(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size != 3:
        raise ValueError("p_xyz must have 3 entries")
    if not np.isfinite(p).all():
        raise ValueError("p_xyz contains NaN/inf")
    return p


def _wavenumber(fc_hz: float) -> float:
    if not np.isfinite(fc_hz) or fc_hz <= 0:
        raise ValueError("fc_hz must be positive and finite")
    return 2.0 * math.pi * (fc_hz / C_MPS)


def rtp_to_cartesian(r: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """Convert spherical to Cartesian: x=r cos(phi) cos(theta), y=r cos(phi) sin(theta), z=r sin(phi).

    Angles are in degrees; the conversion uses radians internally.

    Parameters
    - r: radius in meters (r > 0)
    - theta_deg: azimuth in degrees, expected in [-180, 180]
    - phi_deg: elevation in degrees, expected in [-90, 90]

    Returns
    - (3,) float64 Cartesian coordinate in meters.
    """
    if r <= 0 or not np.isfinite(r):
        raise ValueError("r must be positive and finite")
    th = np.deg2rad(float(_wrap_theta_deg(theta_deg)))
    ph = np.deg2rad(float(_clamp_phi_deg(phi_deg)))
    cph = np.cos(ph)
    x = r * cph * np.cos(th)
    y = r * cph * np.sin(th)
    z = r * np.sin(ph)
    return np.asarray([x, y, z], dtype=np.float64)


def _cartesian_to_rtp(p_xyz: np.ndarray) -> Tuple[float, float, float]:
    p = _validate_point(p_xyz)
    r = float(np.linalg.norm(p))
    if r <= 0:
        raise ValueError("Point at the origin is invalid (r>0 required)")
    x, y, z = p
    theta = math.degrees(math.atan2(y, x))
    phi = math.degrees(math.asin(z / r))
    theta = float(_wrap_theta_deg(theta))
    phi = float(_clamp_phi_deg(phi))
    return r, theta, phi


def spherical_steering(xyz_m: np.ndarray, fc_hz: float, p_xyz: np.ndarray) -> np.ndarray:
    """Spherical steering (phase-only) vector for focus point p.

    Equation
    - k = 2π/λ, with λ = c / f_c, c=299792458 m/s.
    - a_m(p) = exp(-j * k * ||p - x_m||)
    - Return unit-norm vector a ∈ C^M.

    Parameters
    - xyz_m: (M,3) array of element positions in meters.
    - fc_hz: carrier frequency in Hz (>0).
    - p_xyz: (3,) focus point in meters.

    Returns
    - a: (M,) complex128 unit-norm steering vector.
    """
    xyz = _validate_xyz(xyz_m)
    p = _validate_point(p_xyz)
    # Ensure uniqueness of elements (~ 1e-9 m)
    rounded = np.round(xyz / 1e-9).astype(np.int64)
    if np.unique(rounded, axis=0).shape[0] != xyz.shape[0]:
        raise ValueError("Element positions must be unique (>=1e-9 m apart)")

    k = _wavenumber(fc_hz)
    dists = np.linalg.norm(p[None, :] - xyz, axis=1)
    phases = -1j * k * dists
    a = np.exp(phases)
    # Normalize to unit l2-norm
    norm = float(np.linalg.norm(a))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Degenerate steering vector norm")
    a = a.astype(np.complex128) / norm
    return a


def _ttd_delays(xyz_m: np.ndarray, p_xyz: np.ndarray) -> np.ndarray:
    """Time delays τ_m = ||p - x_m||/c (hook for TTD). Not used for wideband here.

    Parameters
    - xyz_m: (M,3) array
    - p_xyz: (3,) point
    Returns
    - (M,) float64 delays in seconds
    """
    xyz = _validate_xyz(xyz_m)
    p = _validate_point(p_xyz)
    dists = np.linalg.norm(p[None, :] - xyz, axis=1)
    return dists / C_MPS


def plane_wave_steering(
    xyz_m: np.ndarray, fc_hz: float, theta_deg: float, phi_deg: float
) -> np.ndarray:
    """Far-field (plane-wave) steering vector for direction (θ,ϕ).

    - û(θ,ϕ) = [cosϕ cosθ, cosϕ sinθ, sinϕ].
    - a_m = exp(-j * k * x_m^T û)
    - Return unit-norm vector a ∈ C^M.
    """
    xyz = _validate_xyz(xyz_m)
    k = _wavenumber(fc_hz)
    th = np.deg2rad(float(_wrap_theta_deg(theta_deg)))
    ph = np.deg2rad(float(_clamp_phi_deg(phi_deg)))
    u = np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], dtype=np.float64)
    phase = -1j * k * (xyz @ u)
    a = np.exp(phase)
    norm = float(np.linalg.norm(a))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Degenerate steering vector norm")
    return a.astype(np.complex128) / norm


def spherical_codebook(
    xyz_m: np.ndarray, fc_hz: float, rtp_grid: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """Build spherical codebook over an (r,θ,ϕ) grid.

    Parameters
    - xyz_m: (M,3) array of element positions (m).
    - fc_hz: carrier frequency (Hz).
    - rtp_grid: (K,3) array with columns [r, theta_deg, phi_deg].
    - chunk: number of grid points per batch (to limit memory).

    Returns
    - codebook: (K, M) complex128 array with unit-norm steering vectors.

    Notes
    - Processing is chunked over K to reduce peak memory.
    - For very large K×M, consider streaming to HDF5 externally.
    """
    xyz = _validate_xyz(xyz_m)
    if not isinstance(rtp_grid, np.ndarray):
        raise ValueError("rtp_grid must be a numpy ndarray")
    if rtp_grid.ndim != 2 or rtp_grid.shape[1] != 3:
        raise ValueError("rtp_grid must have shape (K,3)")
    if not np.isfinite(rtp_grid).all():
        raise ValueError("rtp_grid contains NaN/inf")
    K = rtp_grid.shape[0]
    M = xyz.shape[0]
    out = np.empty((K, M), dtype=np.complex128)

    for start in range(0, K, int(max(1, chunk))):
        end = min(K, start + int(max(1, chunk)))
        sub = rtp_grid[start:end]
        # Convert to Cartesian points
        pts = np.stack(
            [rtp_to_cartesian(r, th, ph) for r, th, ph in sub],
            axis=0,
        )  # (B,3)
        # Compute steering per point
        for i in range(pts.shape[0]):
            out[start + i] = spherical_steering(xyz, fc_hz, pts[i])

    return out
