from __future__ import annotations

import numpy as np


def make_rtp_grid(
    r_min: float,
    r_max: float,
    r_step: float,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Return (K,3) array with columns [r, theta_deg, phi_deg].

    Parameters
    - r_min, r_max, r_step: range for radius r in meters (r > 0), inclusive of
      endpoints according to step. Must satisfy r_min > 0, r_max > r_min, r_step > 0.
    - theta_deg: 1D array of azimuth angles in degrees, expected in [-180, 180].
    - phi_deg: 1D array of elevation angles in degrees, expected in [-90, 90].

    Returns
    - rtp: (K,3) float64 array with rows [r, theta_deg, phi_deg]. K = Nr * Ntheta * Nphi.
    """
    if any(not isinstance(arr, np.ndarray) for arr in (theta_deg, phi_deg)):
        raise ValueError("theta_deg and phi_deg must be numpy ndarrays")
    theta = np.asarray(theta_deg, dtype=np.float64).ravel()
    phi = np.asarray(phi_deg, dtype=np.float64).ravel()

    if not np.isfinite(theta).all() or not np.isfinite(phi).all():
        raise ValueError("theta_deg or phi_deg contains NaN/inf")

    if r_min <= 0 or r_max <= 0 or r_step <= 0:
        raise ValueError("r must be positive and step > 0")
    if r_max < r_min:
        raise ValueError("r_max must be >= r_min")

    # Build radii inclusively with step; use rounding to avoid cumulative FP error.
    n_steps = int(np.floor((r_max - r_min) / r_step + 1e-12))
    radii = r_min + r_step * np.arange(n_steps + 1, dtype=np.float64)
    if radii.size == 0 or radii[-1] < r_max - 1e-12:
        # Add r_max if not included by step
        radii = np.append(radii, r_max)

    RR, TT, PP = np.meshgrid(radii, theta, phi, indexing="ij")
    rtp = np.column_stack([RR.ravel(), TT.ravel(), PP.ravel()])
    return rtp

