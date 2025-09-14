import numpy as np
import pytest

from nearfield.metrics import focusing_gain as fg_metrics
from nearfield.metrics_aag_amg import focusing_gain as fg_aag, aag_over_grid, amag_over_grid, per_point_gains
from nearfield.heatmaps import make_theta_phi_grid, build_steering_on_angular_slice


def test_focusing_gain_agrees():
    w = np.array([1.0, 0.0], dtype=np.complex128)
    a = np.array([1.0, 0.0], dtype=np.complex128)
    g1 = fg_metrics(w, a)
    g2 = fg_aag(w, a)
    assert np.isclose(g1, g2)


def test_amag_ge_aag_linear():
    rng = np.random.default_rng(123)
    P, M = 16, 6
    A = rng.standard_normal((P, M)) + 1j * rng.standard_normal((P, M))
    A = (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)).astype(np.complex128)
    w = A[0].copy()
    aag = aag_over_grid(w, A, mode="linear")
    amag = amag_over_grid(A, mode="linear")
    assert amag >= aag - 1e-12
    g = per_point_gains(w, A, mode="linear")
    assert np.all(np.isfinite(g))
    assert np.all(g <= 1.0 + 1e-9)


def test_angle_wrap_and_clamp_no_nans():
    # Build a grid with angles beyond nominal ranges
    theta = np.array([-200.0, -190.0, -180.0, 180.0, 190.0, 200.0])
    phi = np.array([-120.0, -100.0, 0.0, 100.0, 120.0])
    grid = make_theta_phi_grid(theta, phi)
    # Simple array
    M = 8
    xyz = np.zeros((M, 3), dtype=np.float64)
    xyz[:, 0] = np.linspace(-0.05, 0.05, M)
    A = build_steering_on_angular_slice(xyz, 28e9, 5.0, grid)
    assert A.shape == (grid.shape[0], M)
    assert np.isfinite(np.sum(np.abs(A) ** 2))


@pytest.mark.skipif(pytest.importorskip("plotly") is None, reason="plotly not available")
def test_plotting_smoke():
    from nearfield.plotting_interactive import heatmap_theta_phi, surface_theta_phi, line_radial_slice

    th = np.linspace(-30, 30, 7)
    ph = np.linspace(-15, 15, 5)
    data = np.random.default_rng(0).standard_normal(th.size * ph.size)
    fig1 = heatmap_theta_phi(th, ph, data, "demo")
    assert hasattr(fig1, "to_dict")
    fig2 = surface_theta_phi(th, ph, data, "demo3d")
    assert hasattr(fig2, "to_dict")
    r = np.linspace(1.0, 3.0, 10)
    v = np.random.default_rng(0).standard_normal(r.size)
    fig3 = line_radial_slice(r, v, "demo line")
    assert hasattr(fig3, "to_dict")

