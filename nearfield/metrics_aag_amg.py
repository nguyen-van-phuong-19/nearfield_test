import numpy as np
from typing import Literal, Optional, Callable, Union

try:  # optional acceleration, not required
    from numba import njit  # type: ignore

    HAS_NUMBA = True
except Exception:  # pragma: no cover - optional
    HAS_NUMBA = False


def focusing_gain(w: np.ndarray, a: np.ndarray) -> float:
    """|w^H a|^2 / ||w||^2, scalar.

    Parameters
    ----------
    w : np.ndarray
        Weight vector (M,)
    a : np.ndarray
        Steering vector (M,)
    """
    w = np.asarray(w)
    a = np.asarray(a)
    num = np.abs(np.vdot(w, a)) ** 2
    den = (np.linalg.norm(w) ** 2) + 1e-12
    return float(num / den)


def _to_db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def per_point_gains(
    weights: np.ndarray,
    steering_grid: np.ndarray,
    mode: Literal["linear", "db"] = "db",
) -> np.ndarray:
    """Return (P,) gains for plotting/stats.

    weights: (M,) or (P,M)
    steering_grid: (P,M)
    """
    W = np.asarray(weights)
    A = np.asarray(steering_grid)
    if A.ndim != 2:
        raise ValueError("steering_grid must be 2D (P,M)")
    P, M = A.shape

    if W.ndim == 1:
        if W.shape[0] != M:
            raise ValueError("weights shape mismatch with steering_grid")
        # v = A @ conj(W)
        v = A @ np.conjugate(W)
        num = np.abs(v) ** 2
        den = (np.linalg.norm(W) ** 2) + 1e-12
        g = num / den
    elif W.ndim == 2:
        if W.shape != A.shape:
            raise ValueError("weights and steering_grid must have same shape when weights is 2D")
        v = np.einsum("pm,pm->p", np.conjugate(W), A)
        num = np.abs(v) ** 2
        den = np.sum(np.abs(W) ** 2, axis=1) + 1e-12
        g = num / den
    else:
        raise ValueError("weights must be 1D or 2D")

    if mode == "db":
        return _to_db(g.astype(np.float64))
    if mode == "linear":
        return g.astype(np.float64)
    raise ValueError("mode must be 'linear' or 'db'")


def aag_over_grid(
    weights: np.ndarray,
    steering_grid: Union[np.ndarray, Callable[[int], np.ndarray]],
    mode: Literal["linear", "db"] = "db",
) -> float:
    """Return scalar AAG. Accepts vectorized steering or callback for memory-light mode.

    Parameters
    ----------
    weights : np.ndarray
        (M,) or (P,M) if per-point weights are pre-selected.
    steering_grid : np.ndarray | callable
        (P,M) array or callable p_idx -> (M,) steering vector. When a callable
        is provided, weights must be 2D with shape (P,M) to infer P.
    mode : {"linear","db"}
        Output scale.
    """
    if callable(steering_grid):
        # Memory-light path: require per-point weights to infer P
        W = np.asarray(weights)
        if W.ndim != 2:
            raise ValueError("With callable steering, weights must be (P,M) to infer P")
        P, M = W.shape
        acc = 0.0
        for p in range(P):
            a = np.asarray(steering_grid(p))
            if a.shape != (M,):
                raise ValueError("steering_grid(p) must return shape (M,)")
            acc += focusing_gain(W[p], a)
        val = acc / float(P)
        return float(_to_db(val)) if mode == "db" else float(val)
    else:
        g = per_point_gains(weights, np.asarray(steering_grid), mode="linear")
        val = float(np.mean(g))
        return float(_to_db(val)) if mode == "db" else val


def amag_over_grid(
    steering_grid: Union[np.ndarray, Callable[[int], np.ndarray]],
    mode: Literal["linear", "db"] = "db",
) -> float:
    """Return scalar AMAG using ideal w = a/||a|| per point.

    For steering vectors a_p (rows), the ideal gain equals ||a_p||^2.
    Accepts (P,M) array or callable p_idx -> (M,). If a callable is provided,
    it must be possible to infer P from steering_grid.n_points or len(weights)
    in the caller; since weights are not passed here, prefer array input when
    using AMAG.
    """
    if callable(steering_grid):
        # Try to infer number of points
        P = getattr(steering_grid, "n_points", None)
        if P is None:
            try:
                P = len(steering_grid)  # type: ignore[arg-type]
            except Exception as e:  # pragma: no cover - defensive
                raise ValueError("Cannot infer P for callable steering_grid in AMAG") from e
        acc = 0.0
        first = np.asarray(steering_grid(0))
        M = first.shape[0]
        acc += float(np.sum(np.abs(first) ** 2))
        for p in range(1, int(P)):
            a = np.asarray(steering_grid(p))
            if a.shape != (M,):
                raise ValueError("steering_grid(p) must return shape (M,)")
            acc += float(np.sum(np.abs(a) ** 2))
        val = acc / float(P)
        return float(_to_db(val)) if mode == "db" else float(val)
    else:
        A = np.asarray(steering_grid)
        if A.ndim != 2:
            raise ValueError("steering_grid must be 2D (P,M)")
        g = np.sum(np.abs(A) ** 2, axis=1)
        val = float(np.mean(g))
        return float(_to_db(val)) if mode == "db" else val
