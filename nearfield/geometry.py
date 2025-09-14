from __future__ import annotations

import numpy as np


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    if not isinstance(xyz, np.ndarray):
        raise ValueError("xyz must be a numpy ndarray")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (M,3)")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or inf")
    return np.asarray(xyz, dtype=np.float64)


def _ensure_unique(xyz: np.ndarray, tol: float = 1e-9) -> None:
    # Check minimum pairwise spacing >= tol using a hash on rounded coords.
    rounded = np.round(xyz / tol).astype(np.int64)
    uniq = np.unique(rounded, axis=0)
    if uniq.shape[0] != xyz.shape[0]:
        raise ValueError("Element positions must be unique (>=1e-9 m apart)")


def make_array(
    layout: str,
    num_x: int | None = None,
    num_y: int | None = None,
    dx: float = 0.5e-3,
    dy: float = 0.5e-3,
    custom_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Return (M,3) element coordinates (meters), origin at phase center.

    Parameters
    - layout: one of {"upa", "ula", "custom"}.
      - "ula": Uniform Linear Array along +x with y=z=0, centered at origin.
      - "upa": Uniform Planar Array on x-y plane (z=0), centered at origin.
      - "custom": Use `custom_xyz` positions (m); they will be re-centered.
    - num_x: number of elements along x (required for ULA/UPA).
    - num_y: number along y (required for UPA; ignored for ULA).
    - dx, dy: element spacings in meters.
    - custom_xyz: (M,3) array for custom layout.

    Returns
    - xyz: (M,3) float64 array of element positions in meters.

    Notes
    - Phase center is the mean of element coordinates (re-centered to 0).
    - Ensures unique element positions to >= 1e-9 m.
    """
    layout = str(layout).lower()

    if layout not in {"ula", "upa", "custom"}:
        raise ValueError("layout must be one of {'ula','upa','custom'}")

    if layout == "custom":
        if custom_xyz is None:
            raise ValueError("custom_xyz is required when layout='custom'")
        xyz = _validate_xyz(custom_xyz)
        # Re-center to phase center (mean)
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        _ensure_unique(xyz)
        return xyz

    if num_x is None or (layout == "upa" and num_y is None):
        raise ValueError("num_x (and num_y for UPA) must be provided")

    if not (isinstance(num_x, int) and num_x >= 1):
        raise ValueError("num_x must be a positive integer")
    if layout == "upa":
        if not (isinstance(num_y, int) and num_y >= 1):
            raise ValueError("num_y must be a positive integer for UPA")

    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    if layout == "ula":
        # Positions along x, centered at origin
        xs = (np.arange(num_x, dtype=np.float64) - (num_x - 1) / 2.0) * dx
        xyz = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    else:  # UPA
        xs = (np.arange(num_x, dtype=np.float64) - (num_x - 1) / 2.0) * dx
        ys = (np.arange(num_y, dtype=np.float64) - (num_y - 1) / 2.0) * dy
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        xyz = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])

    xyz = _validate_xyz(xyz)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)  # recentre (phase center)
    _ensure_unique(xyz)
    return xyz

