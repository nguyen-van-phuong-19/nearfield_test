from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np


def save_codebook_h5(
    path: str,
    xyz_m: np.ndarray,
    fc_hz: float,
    rtp_grid: np.ndarray,
    codebook: np.ndarray,
    attrs: dict | None = None,
) -> None:
    """Save codebook, geometry, and metadata to HDF5.

    Datasets
    - xyz_m: (M,3) float64
    - rtp_grid: (K,3) float64
    - codebook: (K,M) complex128
    Attributes
    - fc_hz: float64
    - Any additional attrs items are saved as attributes on the file root.
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    grid = np.asarray(rtp_grid, dtype=np.float64)
    cb = np.asarray(codebook, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        f.create_dataset("xyz_m", data=xyz)
        f.create_dataset("rtp_grid", data=grid)
        f.create_dataset("codebook", data=cb)
        f.attrs["fc_hz"] = float(fc_hz)
        if attrs is not None:
            for k, v in attrs.items():
                # h5py supports many types; convert unsupported to JSON string
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = json.dumps(v)


def load_codebook_h5(path: str) -> dict:
    """Load HDF5 codebook and return a dict with keys: xyz_m, fc_hz, rtp_grid, codebook, attrs."""
    with h5py.File(path, "r") as f:
        xyz = np.array(f["xyz_m"], dtype=np.float64)
        grid = np.array(f["rtp_grid"], dtype=np.float64)
        cb = np.array(f["codebook"], dtype=np.complex128)
        fc = float(f.attrs["fc_hz"]) if "fc_hz" in f.attrs else None
        attrs = {k: f.attrs[k] for k in f.attrs.keys() if k != "fc_hz"}
    return {"xyz_m": xyz, "fc_hz": fc, "rtp_grid": grid, "codebook": cb, "attrs": attrs}


def save_codebook_json(path: str, rtp_grid: np.ndarray) -> None:
    """Save only the grid to JSON with shape metadata for portability."""
    grid = np.asarray(rtp_grid, dtype=np.float64)
    obj = {
        "shape": list(grid.shape),
        "data": grid.reshape(-1).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_codebook_json(path: str) -> np.ndarray:
    """Load grid saved by save_codebook_json."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    shape = tuple(int(x) for x in obj["shape"])
    data = np.array(obj["data"], dtype=np.float64)
    if data.size != int(np.prod(shape)):
        raise ValueError("JSON grid size mismatch")
    return data.reshape(shape)

