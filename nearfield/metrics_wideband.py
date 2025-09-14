from __future__ import annotations

import numpy as np

from .spherical import rtp_to_cartesian, C_MPS


def beampattern_gain_spectrum(W_km: np.ndarray, A_km: np.ndarray) -> np.ndarray:
    """Return (K,) gain vs frequency: |w_k^H a_k|^2 / ||w_k||^2.

    W_km: (K,M) complex weights per subcarrier
    A_km: (K,M) complex steering per subcarrier
    """
    W = np.asarray(W_km, dtype=np.complex128)
    A = np.asarray(A_km, dtype=np.complex128)
    if W.shape != A.shape or W.ndim != 2:
        raise ValueError("W_km and A_km must have same (K,M) shape")
    K, M = W.shape
    # y_k = sum_m conj(W_km) * A_km
    y = np.sum(np.conjugate(W) * A, axis=1)
    num = np.abs(y) ** 2
    den = np.sum(np.abs(W) ** 2, axis=1)
    den = np.maximum(den, 1e-16)
    return (num / den).astype(np.float64)


def gain_flatness_db(gain_k: np.ndarray) -> float:
    """Return peak-to-peak variation in dB across band."""
    g = np.asarray(gain_k, dtype=np.float64).ravel()
    g = np.maximum(g, 1e-16)
    gdb = 10.0 * np.log10(g)
    return float(np.max(gdb) - np.min(gdb))


def _steering_for_angles_at_freq(
    xyz_m: np.ndarray, r_m: float, theta_deg: np.ndarray, phi_deg: float, f_hz: float
) -> np.ndarray:
    # Build steering for many angles at a single frequency using near-field delays
    th = np.asarray(theta_deg, dtype=np.float64).ravel()
    ph = float(phi_deg)
    # Points for each theta (P,3)
    P = th.size
    pts = np.stack([rtp_to_cartesian(r_m, t, ph) for t in th], axis=0)
    # Distances to each element (P,M)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    d = np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=2)
    tau = d / C_MPS
    return np.exp(-1j * 2.0 * np.pi * f_hz * tau)


def beam_squint_deg(
    xyz_m: np.ndarray,
    fc_hz: float,
    W_km: np.ndarray,
    f_sc_hz: np.ndarray,
    r_fixed_m: float,
) -> float:
    """Estimate effective pointing drift (deg) vs frequency on a fixed-radius arc.

    Scan θ∈[-60,60] at φ=0° and find argmax of gain for each subcarrier.
    Return (max angle - min angle) as squint magnitude (deg).
    """
    W = np.asarray(W_km, dtype=np.complex128)
    f = np.asarray(f_sc_hz, dtype=np.float64).ravel()
    if W.ndim != 2 or f.ndim != 1 or W.shape[0] != f.size:
        raise ValueError("W_km must be (K,M) and f_sc_hz (K,)")
    th_grid = np.linspace(-60.0, 60.0, 241)
    phi0 = 0.0
    K, M = W.shape
    peak_angles = np.empty(K, dtype=np.float64)
    for k in range(K):
        A_angles = _steering_for_angles_at_freq(xyz_m, r_fixed_m, th_grid, phi0, f[k])  # (P,M)
        # gain(θ) = |w_k^H a(θ)|^2 / ||w_k||^2
        wk = W[k]
        y = A_angles @ np.conjugate(wk)
        num = np.abs(y) ** 2
        den = max(float(np.vdot(wk, wk).real), 1e-16)
        g = num / den
        peak_angles[k] = th_grid[int(np.argmax(g))]
    return float(np.max(peak_angles) - np.min(peak_angles))


def achievable_rate_bpshz(
    gain_k: np.ndarray, noise_psd_w_hz: float, subcarrier_bw_hz: float
) -> float:
    """Shannon-like rate sum per Hz over subcarriers (no waterfilling).

    Returns mean_k log2(1 + SNR_k), where SNR_k = gain_k / (N0 * Δf).
    """
    g = np.asarray(gain_k, dtype=np.float64).ravel()
    if not np.isfinite(noise_psd_w_hz) or noise_psd_w_hz <= 0:
        raise ValueError("noise_psd_w_hz must be positive")
    if not np.isfinite(subcarrier_bw_hz) or subcarrier_bw_hz <= 0:
        raise ValueError("subcarrier_bw_hz must be positive")
    snr = g / (noise_psd_w_hz * subcarrier_bw_hz)
    return float(np.mean(np.log2(1.0 + snr)))


def evm_percent(W_km: np.ndarray, A_km: np.ndarray) -> float:
    """Simple EVM proxy from mismatch across K.

    Compute normalized correlation ρ_k = |w_k^H a_k| / (||w_k|| ||a_k||).
    EVM% = 100 * sqrt(mean_k(1 - ρ_k^2)).
    """
    W = np.asarray(W_km, dtype=np.complex128)
    A = np.asarray(A_km, dtype=np.complex128)
    if W.shape != A.shape or W.ndim != 2:
        raise ValueError("W_km and A_km must have same (K,M) shape")
    num = np.abs(np.sum(np.conjugate(W) * A, axis=1))
    den = np.sqrt(np.sum(np.abs(W) ** 2, axis=1) * np.sum(np.abs(A) ** 2, axis=1))
    den = np.maximum(den, 1e-16)
    rho = num / den
    evm = np.sqrt(np.mean(1.0 - np.clip(rho * rho, 0.0, 1.0))) * 100.0
    return float(evm)

