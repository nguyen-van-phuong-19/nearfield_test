from __future__ import annotations

import numpy as np

from ..wideband import _validate_xyz, _validate_point
from ..spherical import C_MPS


def design_ttd_delays(xyz_m: np.ndarray, p_xyz: np.ndarray) -> np.ndarray:
    """Return (M,) delays d_m (seconds) to align at p.

    d_m = τ_m = ||p - x_m|| / c.
    """
    xyz = _validate_xyz(xyz_m)
    p = _validate_point(p_xyz)
    dists = np.linalg.norm(p[None, :] - xyz, axis=1)
    return (dists / C_MPS).astype(np.float64)


def weights_over_band_ttd(d_sec: np.ndarray, f_sc_hz: np.ndarray) -> np.ndarray:
    """Return (K,M) frequency-dependent weights: exp(-j*2π f_k d_m)."""
    d = np.asarray(d_sec, dtype=np.float64).ravel()
    f = np.asarray(f_sc_hz, dtype=np.float64).ravel()
    if d.ndim != 1 or f.ndim != 1:
        raise ValueError("d_sec and f_sc_hz must be 1D arrays")
    phase = -1j * 2.0 * np.pi * f[:, None] * d[None, :]
    return np.exp(phase).astype(np.complex128)


def fractional_delay_fir_lagrange(order: int, delay_samples: float) -> np.ndarray:
    """Stub for time-domain fractional-delay FIR (Lagrange interpolation).

    Parameters
    - order: non-negative integer FIR order (filter length = order+1).
    - delay_samples: desired fractional delay in samples.

    Returns
    - h: (order+1,) filter taps. Not used in wideband evaluation here.
    """
    if order < 0 or int(order) != order:
        raise ValueError("order must be a non-negative integer")
    N = int(order)
    n = np.arange(N + 1, dtype=np.float64)
    # Lagrange interpolation kernel
    h = np.ones(N + 1, dtype=np.float64)
    for i in range(N + 1):
        for m in range(N + 1):
            if m == i:
                continue
            h[i] *= (delay_samples - m) / (i - m)
    return h

