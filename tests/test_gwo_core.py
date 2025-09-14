import numpy as np

from nearfield.optim.gwo import gwo_minimize


def test_gwo_sphere_improves_and_deterministic():
    D = 10
    bounds = [(-5.0, 5.0)] * D

    def sphere(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return float(np.sum(X * X))
        return np.sum(X * X, axis=1)

    x1, f1, info1 = gwo_minimize(sphere, bounds, n_agents=30, n_iter=100, seed=42)
    # Must improve over initial best (hist[0])
    assert f1 <= info1["hist"][0] + 1e-12

    # Deterministic with same seed
    x2, f2, info2 = gwo_minimize(sphere, bounds, n_agents=30, n_iter=100, seed=42)
    assert np.allclose(x1, x2)
    assert f1 == f2
    assert info1["hist"] == info2["hist"]


def test_gwo_handles_nan_and_clamps():
    D = 6
    bounds = [(-1.0, 1.0)] * D

    def nan_sphere(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return float(np.sum(X * X))
        out = np.sum(X * X, axis=1)
        # Inject a NaN in the first sample to ensure +inf handling
        if out.size > 0:
            out[0] = np.nan
        return out

    x_best, f_best, info = gwo_minimize(nan_sphere, bounds, n_agents=20, n_iter=50, seed=123, clamp=True)
    assert np.isfinite(f_best)
    assert np.all(x_best >= -1.0 - 1e-12) and np.all(x_best <= 1.0 + 1e-12)

