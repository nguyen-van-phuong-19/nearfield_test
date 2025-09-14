from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple, List, Dict, Any

import numpy as np


def _validate_bounds(bounds: Iterable[Tuple[float, float]]) -> np.ndarray:
    try:
        b = np.asarray(list(bounds), dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Invalid bounds: {e}")
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must be iterable of (low, high) pairs with shape (D,2)")
    if b.shape[0] == 0:
        raise ValueError("bounds cannot be empty")
    lo = b[:, 0]
    hi = b[:, 1]
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("bounds contain non-finite values")
    if np.any(hi <= lo):
        raise ValueError("each bounds pair must satisfy high > low")
    return b


def _eval_func(func: Callable[[np.ndarray], float | np.ndarray], X: np.ndarray) -> np.ndarray:
    """Evaluate func on batch X (B,D) robustly. Returns (B,) float64, inf on NaN/err."""
    B = X.shape[0]
    try:
        y = func(X)
        y_arr = np.asarray(y, dtype=np.float64)
        # Accept vectorized output matching batch size
        if y_arr.ndim == 1 and y_arr.shape[0] == B:
            out = y_arr
        else:
            # Not vectorized -> force error to jump to loop fallback
            raise TypeError("non-vectorized output shape")
    except Exception:
        # Scalar evaluation loop
        out = np.empty(B, dtype=np.float64)
        for i in range(B):
            try:
                yi = float(func(np.asarray(X[i], dtype=np.float64)))
            except Exception:
                yi = np.inf
            if not np.isfinite(yi):
                yi = np.inf
            out[i] = yi
        return out
    # Guard NaN/Inf
    out[~np.isfinite(out)] = np.inf
    return out


def gwo_minimize(
    func: Callable[[np.ndarray], float | np.ndarray],
    bounds: Iterable[Tuple[float, float]],
    n_agents: int = 30,
    n_iter: int = 200,
    seed: int | None = None,
    w_alpha: float = 0.5,
    w_beta: float = 0.3,
    w_delta: float = 0.2,
    clamp: bool = True,
    early_stop_patience: int | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Minimize black-box func(x) within box bounds using Grey Wolf Optimizer.

    Returns (x_best, f_best, info) where info contains:
      - 'hist': list of best f per iteration
      - 'rng_state': RNG state dict
      - 'n_evals': total objective evaluations
      - 'iterations': total iterations completed
      - 'cancelled': bool if a cancel event was observed
    """
    b = _validate_bounds(bounds)
    D = b.shape[0]
    n_agents = int(max(4, n_agents))
    n_iter = int(max(1, n_iter))

    rng = np.random.default_rng(seed)
    lo = b[:, 0]
    hi = b[:, 1]
    span = hi - lo

    # Initialize population uniformly in bounds
    X = (rng.random((n_agents, D), dtype=np.float64) * span) + lo
    # Evaluate initial fitness
    f = _eval_func(func, X)
    n_evals = int(f.size)

    # Track alpha/beta/delta (best three)
    order = np.argsort(f)
    a_idx, b_idx, d_idx = int(order[0]), int(order[1]), int(order[2])
    X_alpha = X[a_idx].copy()
    X_beta = X[b_idx].copy()
    X_delta = X[d_idx].copy()
    f_alpha = float(f[a_idx])
    f_beta = float(f[b_idx])
    f_delta = float(f[d_idx])

    hist: List[float] = [f_alpha]
    best_f_prev = f_alpha
    no_improve = 0
    cancelled = False

    # Optional cancel event attached to func by caller (e.g., Tk GUI)
    cancel_event = getattr(func, "__cancel_event__", None) or getattr(func, "_cancel_event", None) or getattr(func, "cancel_flag", None)

    for t in range(n_iter):
        # Linear decrease of a from 2 -> 0
        a = 2.0 - 2.0 * (t / max(1.0, float(n_iter)))

        # Random coefficients per agent/dimension
        r1 = rng.random((n_agents, D), dtype=np.float64)
        r2 = rng.random((n_agents, D), dtype=np.float64)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2

        r1 = rng.random((n_agents, D), dtype=np.float64)
        r2 = rng.random((n_agents, D), dtype=np.float64)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2

        r1 = rng.random((n_agents, D), dtype=np.float64)
        r2 = rng.random((n_agents, D), dtype=np.float64)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2

        # Broadcast leaders
        Xa = X_alpha[None, :]
        Xb = X_beta[None, :]
        Xd = X_delta[None, :]

        X1 = Xa - A1 * np.abs(C1 * Xa - X)
        X2 = Xb - A2 * np.abs(C2 * Xb - X)
        X3 = Xd - A3 * np.abs(C3 * Xd - X)

        X_new = (w_alpha * X1) + (w_beta * X2) + (w_delta * X3)

        if clamp:
            # Clip to bounds for each dimension
            X_new = np.minimum(np.maximum(X_new, lo[None, :]), hi[None, :])

        # Evaluate
        f_new = _eval_func(func, X_new)
        n_evals += int(f_new.size)

        # Replace current population and update leaders
        X = X_new
        f = f_new
        order = np.argsort(f)
        a_idx, b_idx, d_idx = int(order[0]), int(order[1]), int(order[2])
        if f[a_idx] < f_alpha:
            X_alpha = X[a_idx].copy()
            f_alpha = float(f[a_idx])
        if f[b_idx] < f_beta or (b_idx != a_idx and f[b_idx] < f_beta):
            X_beta = X[b_idx].copy()
            f_beta = float(f[b_idx])
        if f[d_idx] < f_delta or (d_idx != a_idx and d_idx != b_idx and f[d_idx] < f_delta):
            X_delta = X[d_idx].copy()
            f_delta = float(f[d_idx])

        hist.append(f_alpha)

        # Early stopping on no improvement
        if early_stop_patience is not None and early_stop_patience > 0:
            if f_alpha + 1e-12 < best_f_prev:
                best_f_prev = f_alpha
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(early_stop_patience):
                    break

        # Cooperative progress logging every ~10% (keep light)
        if (t % max(1, n_iter // 10)) == 0:
            try:
                print(f"[GWO] iter={t}/{n_iter} best={f_alpha:.6g}")
            except Exception:
                pass

        # Cancellation
        try:
            if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
                cancelled = True
                break
        except Exception:
            pass

    info: Dict[str, Any] = {
        "hist": hist,
        "rng_state": getattr(rng.bit_generator, "state", {}),
        "n_evals": n_evals,
        "iterations": len(hist) - 1,
        "cancelled": cancelled,
    }
    return X_alpha.astype(np.float64), float(f_alpha), info

