import numpy as np
from nearfield.metrics_aag_amg import focusing_gain, aag_over_grid, amag_over_grid, per_point_gains


def test_focusing_gain_simple():
    w = np.array([1.0, 0.0], dtype=np.complex128)
    a = np.array([1.0, 0.0], dtype=np.complex128)
    g = focusing_gain(w, a)
    assert np.isclose(g, 1.0)


def test_amag_ge_aag():
    rng = np.random.default_rng(0)
    P, M = 32, 8
    A = rng.standard_normal((P, M)) + 1j * rng.standard_normal((P, M))
    # Normalize rows to unit norm
    A = (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)).astype(np.complex128)
    # Use a fixed mismatched weight for AAG
    w = A[0].copy()
    aag = aag_over_grid(w, A, mode="linear")
    amag = amag_over_grid(A, mode="linear")
    assert amag >= aag - 1e-12
    # Per-point are finite and <= 1 for normalized steering
    g = per_point_gains(w, A, mode="linear")
    assert np.all(np.isfinite(g))
    assert np.all(g <= 1.0 + 1e-9)

