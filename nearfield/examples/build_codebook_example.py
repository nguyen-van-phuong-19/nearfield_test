from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from nearfield.geometry import make_array
from nearfield.spherical import spherical_steering, rtp_to_cartesian
from nearfield.grids import make_rtp_grid


def main() -> None:
    # Parameters
    c = 299_792_458.0
    fc = 28e9
    lam = c / fc
    nx = ny = 32
    dx = dy = lam / 2.0

    # Array
    xyz = make_array("upa", num_x=nx, num_y=ny, dx=dx, dy=dy)

    # Grid: r=3..20 m, step 0.5; theta=-60:5:60; phi=-30:5:30
    theta = np.arange(-60.0, 60.0 + 1e-9, 5.0)
    phi = np.arange(-30.0, 30.0 + 1e-9, 5.0)
    rtp = make_rtp_grid(3.0, 20.0, 0.5, theta, phi)
    K, M = rtp.shape[0], xyz.shape[0]

    out_path = Path("nearfield_codebook_32x32_28Ghz.h5")
    print(f"Writing to {out_path.resolve()}")

    # Stream to HDF5 to limit memory
    with h5py.File(out_path, "w") as f:
        f.create_dataset("xyz_m", data=xyz)
        f.create_dataset("rtp_grid", data=rtp)
        d_cb = f.create_dataset("codebook", shape=(K, M), dtype=np.complex128, chunks=(1024, M))
        f.attrs["fc_hz"] = float(fc)
        start = time.time()
        batch = 2048
        n_done = 0
        for s in range(0, K, batch):
            e = min(K, s + batch)
            pts = np.stack([rtp_to_cartesian(r, th, ph) for r, th, ph in rtp[s:e]], axis=0)
            for i in range(pts.shape[0]):
                d_cb[s + i] = spherical_steering(xyz, fc, pts[i])
            n_done += (e - s)
            if (s // batch) % 10 == 0:
                elapsed = time.time() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(f"Built {n_done}/{K} entries at {rate:,.0f} entries/s", flush=True)

    elapsed = time.time() - start
    rate = K / elapsed if elapsed > 0 else 0.0
    print(f"Done. {K} entries in {elapsed:.1f}s -> {rate:,.0f} entries/s")


if __name__ == "__main__":
    main()

