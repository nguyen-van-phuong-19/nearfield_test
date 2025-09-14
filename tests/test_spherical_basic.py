import numpy as np

from nearfield.geometry import make_array
from nearfield.spherical import spherical_steering, C_MPS


def test_spherical_phase_and_norm():
    # 2x1 ULA at fc=3.5e9, dx=lambda/2
    fc = 3.5e9
    lam = C_MPS / fc
    xyz = make_array("ula", num_x=2, dx=lam / 2.0)

    # Point on boresight at r=10 m (along +z)
    p = np.array([0.0, 0.0, 10.0], dtype=np.float64)

    a = spherical_steering(xyz, fc, p)

    # Distances
    d = np.linalg.norm(p[None, :] - xyz, axis=1)
    k = 2.0 * np.pi * (fc / C_MPS)
    # Relative phase predicted by distance difference
    dphi = k * (d[1] - d[0])
    # Actual relative phase from steering vector
    rel = np.angle(a[1] * np.conj(a[0]))
    # Wrap both to [-pi,pi]
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    assert abs(rel - dphi) < 1e-3

    # Norm close to 1
    norm = np.linalg.norm(a)
    assert abs(norm - 1.0) <= 1e-6

