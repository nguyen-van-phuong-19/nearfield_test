from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from nearfield.geometry import make_array
from nearfield.grids import make_rtp_grid
from nearfield.spherical import spherical_codebook, plane_wave_steering
from nearfield.metrics import quantization_loss_at, farfield_mismatch_loss


def main() -> None:
    rng = np.random.default_rng(1234)
    c = 299_792_458.0
    fc = 28e9
    lam = c / fc

    # Array (moderate size to run fast)
    nx = ny = 16
    dx = dy = lam / 2.0
    xyz = make_array("upa", num_x=nx, num_y=ny, dx=dx, dy=dy)

    # Angular grid (coarse)
    theta = np.arange(-60.0, 60.0 + 1e-9, 5.0)
    phi = np.arange(-30.0, 30.0 + 1e-9, 5.0)
    rtp = make_rtp_grid(3.0, 10.0, 1.0, theta, phi)

    # Spherical codebook
    cb = spherical_codebook(xyz, fc, rtp, chunk=1024)

    # Far-field codebook over same (θ,ϕ) grid, ignoring r
    # Build by taking the first radius for each (θ,ϕ) tuple to match indexing.
    K = rtp.shape[0]
    codebook_ff = np.empty_like(cb)
    for i, (r, th, ph) in enumerate(rtp):
        codebook_ff[i] = plane_wave_steering(xyz, fc, th, ph)

    # Random query points in front of array
    Q = 200
    # Sample angles within the grid coverage
    th_q = rng.uniform(-60.0, 60.0, size=Q)
    ph_q = rng.uniform(-30.0, 30.0, size=Q)
    r_q = rng.uniform(3.0, 10.0, size=Q)
    # Convert to Cartesian
    th_r = np.deg2rad(th_q)
    ph_r = np.deg2rad(ph_q)
    x = r_q * np.cos(ph_r) * np.cos(th_r)
    y = r_q * np.cos(ph_r) * np.sin(th_r)
    z = r_q * np.sin(ph_r)
    pts = np.column_stack([x, y, z])

    # Evaluate losses
    qloss = quantization_loss_at(xyz, fc, rtp, cb, pts)
    mismatch = farfield_mismatch_loss(xyz, fc, rtp, codebook_ff, pts)

    # Plot histograms
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].hist(qloss, bins=20, color="tab:blue", alpha=0.8)
    ax[0].set_title("Quantization loss (spherical vs ideal)")
    ax[0].set_xlabel("Loss (dB)")
    ax[0].set_ylabel("Count")

    ax[1].hist(mismatch, bins=20, color="tab:orange", alpha=0.8)
    ax[1].set_title("Far-field mismatch loss")
    ax[1].set_xlabel("Loss (dB)")

    plt.show()


if __name__ == "__main__":
    main()

