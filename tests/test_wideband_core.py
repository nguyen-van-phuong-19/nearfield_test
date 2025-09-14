import numpy as np

from nearfield.geometry import make_array
from nearfield.spherical import C_MPS
from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband


def test_subcarrier_frequencies_symmetric_and_shape():
    fc = 10e9
    bw = 200e6
    K = 128
    f = subcarrier_frequencies(fc, bw, K)
    assert f.shape == (K,)
    assert np.isclose(np.mean(f), fc, atol=1e-6)
    df = np.diff(f)
    assert np.allclose(df, df[0])


def test_spherical_steering_wideband_phases_follow_tau():
    fc = 28e9
    bw = 100e6
    K = 16
    f = subcarrier_frequencies(fc, bw, K)
    lam = C_MPS / fc
    xyz = make_array("ula", num_x=2, dx=lam / 2.0)
    p = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    A = spherical_steering_wideband(xyz, p, f)  # (K,2)
    d = np.linalg.norm(p[None, :] - xyz, axis=1)
    tau = d / C_MPS
    # Relative phase between element 2 and 1 at each subcarrier
    rel = np.angle(A[:, 1] * np.conj(A[:, 0]))
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    expected = -2.0 * np.pi * f * (tau[1] - tau[0])
    expected = (expected + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(rel, expected, atol=1e-3)

