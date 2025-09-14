import numpy as np
from nearfield.heatmaps import (
    make_theta_phi_grid,
    build_steering_on_angular_slice,
    build_steering_on_radial_slice,
)


SPEED_OF_LIGHT = 299_792_458.0


def make_array(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    M = nx * ny
    xyz = np.zeros((M, 3), dtype=np.float64)
    xyz[:, 0] = xv.reshape(-1)
    xyz[:, 1] = yv.reshape(-1)
    return xyz


def test_angular_and_radial_shapes():
    nx = ny = 4
    lam = SPEED_OF_LIGHT / 28e9
    dx = dy = 0.5 * lam
    xyz = make_array(nx, ny, dx, dy)
    theta = np.linspace(-30, 30, 7)
    phi = np.linspace(-15, 15, 5)
    grid = make_theta_phi_grid(theta, phi)
    A = build_steering_on_angular_slice(xyz, 28e9, 5.0, grid)
    assert A.shape == (grid.shape[0], xyz.shape[0])
    # Radial slice
    r = np.linspace(2.0, 8.0, 10)
    B = build_steering_on_radial_slice(xyz, 28e9, r, 0.0, 0.0)
    assert B.shape == (r.shape[0], xyz.shape[0])
    # Gains should be finite
    assert np.isfinite(np.sum(np.abs(A) ** 2))
    assert np.isfinite(np.sum(np.abs(B) ** 2))

