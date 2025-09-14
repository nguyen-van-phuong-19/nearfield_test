import os
import tempfile

import numpy as np

from nearfield.geometry import make_array
from nearfield.grids import make_rtp_grid
from nearfield.spherical import spherical_codebook, C_MPS
from nearfield.codebook_io import (
    save_codebook_h5,
    load_codebook_h5,
    save_codebook_json,
    load_codebook_json,
)


def test_hdf5_roundtrip_and_attrs():
    fc = 10e9
    lam = C_MPS / fc
    xyz = make_array("upa", num_x=2, num_y=2, dx=lam / 2.0, dy=lam / 2.0)
    theta = np.array([-10.0, 0.0, 10.0])
    phi = np.array([-5.0, 0.0, 5.0])
    rtp = make_rtp_grid(1.0, 2.0, 1.0, theta, phi)
    cb = spherical_codebook(xyz, fc, rtp, chunk=32)

    attrs = {"name": "unit-test", "version": 1}

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.h5")
        save_codebook_h5(path, xyz, fc, rtp, cb, attrs=attrs)
        obj = load_codebook_h5(path)

        assert np.allclose(obj["xyz_m"], xyz)
        assert obj["fc_hz"] == fc
        assert np.allclose(obj["rtp_grid"], rtp)
        assert np.allclose(obj["codebook"], cb)
        # Attributes preserved
        for k, v in attrs.items():
            assert k in obj["attrs"]


def test_json_grid_roundtrip():
    theta = np.array([-30.0, 0.0, 30.0])
    phi = np.array([-10.0, 0.0, 10.0])
    rtp = make_rtp_grid(2.0, 3.0, 0.5, theta, phi)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "grid.json")
        save_codebook_json(path, rtp)
        g2 = load_codebook_json(path)
        assert np.allclose(rtp, g2)

