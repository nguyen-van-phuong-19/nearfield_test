from __future__ import annotations

from typing import Tuple, Callable, List, Dict, Any

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


def _wideband_steering(xyz_m: np.ndarray, p_xyz: np.ndarray, f_hz: float) -> np.ndarray:
    tau = np.linalg.norm(p_xyz[None, :] - xyz_m, axis=1) / SPEED_OF_LIGHT
    a = np.exp(-1j * 2.0 * np.pi * float(f_hz) * tau)
    a = a / (np.linalg.norm(a) + 1e-12)
    return a.astype(np.complex128)


def _build_A(xyz_m: np.ndarray, p_xyz: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    K = int(freqs_hz.size)
    M = int(xyz_m.shape[0])
    A = np.zeros((K, M), dtype=np.complex128)
    for k, fk in enumerate(freqs_hz):
        A[k, :] = _wideband_steering(xyz_m, p_xyz, float(fk))
    return A


def _gains_for_weights(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compute per-frequency gains for batch of weights.

    W: (B,M) complex; A: (K,M) complex steering rows.
    Returns G: (B,K) real gains.
    """
    if W.ndim == 1:
        W = W[None, :]
    B, M = W.shape
    K = int(A.shape[0])
    Wn2 = np.maximum(np.sum(np.abs(W) ** 2, axis=1, keepdims=True), 1e-12)  # (B,1)
    H = W.conj() @ A.T  # (B,K)
    G = (np.abs(H) ** 2) / Wn2  # (B,K)
    return G.real.astype(np.float64)


def make_objective(name: str, cfg, data_ctx: Dict[str, Any]) -> Tuple[Callable[[np.ndarray], float | np.ndarray], List[Tuple[float, float]], Callable[[np.ndarray], Dict[str, Any]]]:
    """Return (func, bounds, postproc) for given target.

    Known names:
      - 'Max Gain @ focus' | 'beam_gain_max' | 'gain_max'
      - 'Min Gain Flatness' | 'flatness_min'
      - 'Custom (advanced)' -> same as gain_max
    data_ctx must provide: 'xyz_m': (M,3), 'p_xyz': (3,), 'freqs_hz': (K,)
    """
    xyz = np.asarray(data_ctx.get("xyz_m"), dtype=np.float64)
    p = np.asarray(data_ctx.get("p_xyz"), dtype=np.float64)
    freqs = np.asarray(data_ctx.get("freqs_hz"), dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("data_ctx['xyz_m'] must have shape (M,3)")
    if p.size != 3:
        raise ValueError("data_ctx['p_xyz'] must be length 3")
    if freqs.size < 1:
        raise ValueError("data_ctx['freqs_hz'] must be non-empty")

    A = _build_A(xyz, p, freqs)  # (K,M)
    M = int(xyz.shape[0])

    # Bounds: per-element phase [-pi, pi]
    bounds = [(-np.pi, np.pi) for _ in range(M)]

    canonical = (name or "").strip().lower()
    if canonical in {"max gain @ focus", "beam_gain_max", "gain_max"}:
        mode = "gain_max"
    elif canonical in {"min gain flatness", "flatness_min"}:
        mode = "flatness_min"
    else:
        mode = "gain_max"

    def obj(X: np.ndarray):
        # X: (D,) or (B,D) phases
        X = np.asarray(X, dtype=np.float64)
        # build complex weights (phase-only)
        if X.ndim == 1:
            W = np.exp(1j * X)
            G = _gains_for_weights(W, A)
            if mode == "gain_max":
                val = -float(np.mean(G))
            else:
                gdb = 10.0 * np.log10(np.maximum(G, 1e-12))
                val = float(np.max(gdb) - np.min(gdb))
            return val
        else:
            W = np.exp(1j * X)
            G = _gains_for_weights(W, A)  # (B,K)
            if mode == "gain_max":
                vals = -np.mean(G, axis=1)
            else:
                gdb = 10.0 * np.log10(np.maximum(G, 1e-12))  # (B,K)
                vals = np.max(gdb, axis=1) - np.min(gdb, axis=1)
            vals = np.asarray(vals, dtype=np.float64)
            vals[~np.isfinite(vals)] = np.inf
            return vals

    def postproc(x_best: np.ndarray) -> Dict[str, Any]:
        x_best = np.asarray(x_best, dtype=np.float64)
        w = np.exp(1j * x_best)
        w = w / (np.linalg.norm(w) + 1e-12)
        return {
            "weights": w.astype(np.complex128),
            "phases": x_best.astype(np.float64),
            "mode": mode,
        }

    return obj, bounds, postproc

