import numpy as np
from typing import Tuple
import math

SPEED_OF_LIGHT = 299_792_458.0


def make_theta_phi_grid(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Return (P,2) mesh flattened with columns [theta_deg, phi_deg].

    Theta is wrapped to [-180,180), Phi is clamped to [-90,90].
    """
    th = np.asarray(theta_deg, dtype=np.float64)
    ph = np.asarray(phi_deg, dtype=np.float64)
    if not np.isfinite(th).all() or not np.isfinite(ph).all():
        raise ValueError("theta/phi contain NaN/inf")
    # Wrap and clamp
    th = (th + 180.0) % 360.0 - 180.0
    th[th == 180.0] = -180.0
    ph = np.clip(ph, -90.0, 90.0)
    TH, PH = np.meshgrid(th, ph, indexing="xy")
    return np.stack([TH.reshape(-1), PH.reshape(-1)], axis=1)


def make_radius_grid(r_vals: np.ndarray, theta_deg: float, phi_deg: float) -> np.ndarray:
    """Return (P,1) radii for a fixed angular slice (helper/placeholder)."""
    r = np.asarray(r_vals, dtype=np.float64).reshape(-1, 1)
    return r


def _rtp_to_xyz(r: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    ph = math.radians(float(phi_deg))
    cph = math.cos(ph)
    return np.array([
        r * cph * math.cos(th),
        r * cph * math.sin(th),
        r * math.sin(ph),
    ], dtype=np.float64)


def _spherical_steering(xyz_m: np.ndarray, p_xyz: np.ndarray, fc_hz: float) -> np.ndarray:
    lam = SPEED_OF_LIGHT / float(fc_hz)
    k = 2.0 * math.pi / lam
    d = np.linalg.norm(p_xyz[None, :] - xyz_m, axis=1)
    a = np.exp(-1j * k * d)
    a = a / (np.linalg.norm(a) + 1e-12)
    return a.astype(np.complex128)


def build_steering_on_angular_slice(
    xyz_m: np.ndarray,
    fc_hz: float,
    r_fixed_m: float,
    theta_phi_grid: np.ndarray,
) -> np.ndarray:
    """Return (P,M) spherical steering at fixed radius over (theta,phi).

    Validates inputs; wraps theta and clamps phi to safe ranges.
    """
    if not np.isfinite(fc_hz) or fc_hz <= 0:
        raise ValueError("fc_hz must be positive")
    if not np.isfinite(r_fixed_m) or r_fixed_m <= 0:
        raise ValueError("r_fixed_m must be positive")
    grid = np.asarray(theta_phi_grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("theta_phi_grid must be (P,2)")
    if not np.isfinite(grid).all():
        raise ValueError("theta_phi_grid contains NaN/inf")
    # Wrap and clamp per row
    th = (grid[:, 0] + 180.0) % 360.0 - 180.0
    th[th == 180.0] = -180.0
    ph = np.clip(grid[:, 1], -90.0, 90.0)
    P = grid.shape[0]
    # Convert to XYZ (P,3)
    pts = np.stack([
        _rtp_to_xyz(r_fixed_m, float(t), float(p)) for t, p in zip(th, ph)
    ], axis=0)
    # Compute distances (P,M)
    lam = SPEED_OF_LIGHT / float(fc_hz)
    k = 2.0 * math.pi / lam
    diff = pts[:, None, :] - xyz_m[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    A = np.exp(-1j * k * d)
    # Normalize rows to unit norm
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    A = (A / norms).astype(np.complex128)
    return A


def build_steering_on_radial_slice(
    xyz_m: np.ndarray,
    fc_hz: float,
    r_vals: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> np.ndarray:
    """Return (P,M) steering over radii at fixed angles.

    Theta is wrapped to [-180,180), Phi is clamped to [-90,90]. Radii must be >0.
    """
    if not np.isfinite(fc_hz) or fc_hz <= 0:
        raise ValueError("fc_hz must be positive")
    r = np.asarray(r_vals, dtype=np.float64).reshape(-1)
    if (r <= 0).any() or not np.isfinite(r).all():
        raise ValueError("r_vals must be positive and finite")
    th = (float(theta_deg) + 180.0) % 360.0 - 180.0
    if th == 180.0:
        th = -180.0
    ph = float(np.clip(phi_deg, -90.0, 90.0))
    pts = np.stack([
        _rtp_to_xyz(float(rv), th, ph) for rv in r
    ], axis=0)
    lam = SPEED_OF_LIGHT / float(fc_hz)
    k = 2.0 * math.pi / lam
    diff = pts[:, None, :] - xyz_m[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    A = np.exp(-1j * k * d)
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    A = (A / norms).astype(np.complex128)
    return A
