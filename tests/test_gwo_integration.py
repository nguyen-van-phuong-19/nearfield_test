import numpy as np

from nearfield.optim.gwo import gwo_minimize
from nearfield.optim.adapters import make_objective


def _make_ula(M: int, dx: float) -> np.ndarray:
    xs = (np.arange(M, dtype=np.float64) - (M - 1) / 2.0) * dx
    xyz = np.zeros((M, 3), dtype=np.float64)
    xyz[:, 0] = xs
    return xyz


def test_gwo_adapter_gain_improves_over_random():
    rng = np.random.default_rng(0)
    M = 8
    dx = 0.005
    xyz = _make_ula(M, dx)
    p = np.array([0.0, 0.0, 2.0], dtype=np.float64)  # 2 m on boresight
    fc = 28e9
    # Use small K to keep test fast
    freqs = np.linspace(fc - 0.5e9, fc + 0.5e9, 11)

    func, bounds, post = make_objective("Max Gain @ focus", None, {"xyz_m": xyz, "p_xyz": p, "freqs_hz": freqs})

    # Baseline: random phases
    x0 = rng.uniform(-np.pi, np.pi, size=M)
    baseline = -float(func(x0))  # maximize mean gain -> negate objective

    # Optimize with GWO
    x_best, f_best, info = gwo_minimize(func, bounds, n_agents=25, n_iter=80, seed=123)
    best_metric = -float(f_best)

    # Should improve over random baseline (allow small tolerance)
    assert best_metric >= baseline - 1e-6

    # Deterministic seed
    x_best2, f_best2, info2 = gwo_minimize(func, bounds, n_agents=25, n_iter=80, seed=123)
    assert np.allclose(x_best, x_best2)
    assert f_best == f_best2
    assert info["hist"] == info2["hist"]

