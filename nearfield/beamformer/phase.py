from __future__ import annotations

import numpy as np

from ..wideband import _validate_xyz, _validate_point
from ..spherical import C_MPS


def design_phase_shifter_weights(
    xyz_m: np.ndarray, fc_hz: float, p_xyz: np.ndarray
) -> np.ndarray:
    """Return (M,) complex weights designed at fc (frequency-flat).

    w_m = exp(-j * 2π fc * τ_m), τ_m = ||p - x_m|| / c.
    """
    if not np.isfinite(fc_hz) or fc_hz <= 0:
        raise ValueError("fc_hz must be positive")
    xyz = _validate_xyz(xyz_m)
    p = _validate_point(p_xyz)
    tau = np.linalg.norm(p[None, :] - xyz, axis=1) / C_MPS
    w = np.exp(-1j * 2.0 * np.pi * fc_hz * tau)
    return w.astype(np.complex128)


def weights_over_band_phase_shifter(w_fc: np.ndarray, n_sc: int) -> np.ndarray:
    """Broadcast PS weights across subcarriers → (K,M)."""
    w = np.asarray(w_fc, dtype=np.complex128).ravel()
    if not isinstance(n_sc, int) or n_sc < 1:
        raise ValueError("n_sc must be a positive integer")
    return np.tile(w[None, :], (n_sc, 1))

