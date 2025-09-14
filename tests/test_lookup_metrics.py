import numpy as np

from nearfield.geometry import make_array
from nearfield.grids import make_rtp_grid
from nearfield.lookup import build_kdtree, nearest_codeword
from nearfield.metrics import quantization_loss_at, farfield_mismatch_loss
from nearfield.spherical import spherical_codebook, plane_wave_steering, C_MPS


def test_kdtree_on_grid_exact():
    theta = np.arange(-180.0, 180.0, 45.0)
    phi = np.arange(-45.0, 45.0 + 1e-9, 15.0)
    rtp = make_rtp_grid(1.0, 2.0, 0.5, theta, phi)
    tree = build_kdtree(rtp)
    inds, dist = nearest_codeword(tree, rtp, rtp, k=1)
    inds = inds.ravel()
    assert np.array_equal(inds, np.arange(rtp.shape[0]))
    assert np.allclose(dist, 0.0)


def test_quantization_loss_small_on_dense_grid():
    rng = np.random.default_rng(0)
    fc = 10e9
    lam = C_MPS / fc
    # Small UPA for speed
    xyz = make_array("upa", num_x=8, num_y=8, dx=lam / 2.0, dy=lam / 2.0)

    theta = np.arange(-40.0, 40.0 + 1e-9, 4.0)
    phi = np.arange(-20.0, 20.0 + 1e-9, 4.0)
    rtp = make_rtp_grid(5.0, 6.0, 0.25, theta, phi)
    cb = spherical_codebook(xyz, fc, rtp, chunk=1024)

    # Random points near the grid coverage
    Q = 50
    th = rng.uniform(-40.0, 40.0, size=Q)
    ph = rng.uniform(-20.0, 20.0, size=Q)
    r = rng.uniform(5.0, 6.0, size=Q)
    th_r = np.deg2rad(th); ph_r = np.deg2rad(ph)
    x = r * np.cos(ph_r) * np.cos(th_r)
    y = r * np.cos(ph_r) * np.sin(th_r)
    z = r * np.sin(ph_r)
    pts = np.column_stack([x, y, z])

    losses = quantization_loss_at(xyz, fc, rtp, cb, pts)
    assert np.all(losses <= 0.3 + 1e-9)


def test_farfield_mismatch_large_near():
    # Use higher frequency and larger array to expose near-field mismatch
    fc = 28e9
    lam = C_MPS / fc
    xyz = make_array("upa", num_x=16, num_y=16, dx=lam / 2.0, dy=lam / 2.0)

    theta = np.arange(-60.0, 60.0 + 1e-9, 10.0)
    phi = np.arange(-30.0, 30.0 + 1e-9, 10.0)
    # r is irrelevant for FF codebook but needed for indexing; keep a single radius
    rtp = make_rtp_grid(0.8, 1.2, 0.4, theta, phi)

    # Build far-field codebook at the grid directions
    K = rtp.shape[0]
    M = xyz.shape[0]
    cb_ff = np.empty((K, M), dtype=np.complex128)
    for i, (r, th, ph) in enumerate(rtp):
        cb_ff[i] = plane_wave_steering(xyz, fc, th, ph)

    # Random near points (well inside Rayleigh distance for this aperture)
    rng = np.random.default_rng(1)
    Q = 40
    th = rng.uniform(-50.0, 50.0, size=Q)
    ph = rng.uniform(-25.0, 25.0, size=Q)
    r = rng.uniform(0.6, 1.0, size=Q)
    th_r = np.deg2rad(th); ph_r = np.deg2rad(ph)
    x = r * np.cos(ph_r) * np.cos(th_r)
    y = r * np.cos(ph_r) * np.sin(th_r)
    z = r * np.sin(ph_r)
    pts = np.column_stack([x, y, z])

    losses = farfield_mismatch_loss(xyz, fc, rtp, cb_ff, pts)
    # Near-field mismatch should be significant
    assert np.median(losses) >= 3.0
