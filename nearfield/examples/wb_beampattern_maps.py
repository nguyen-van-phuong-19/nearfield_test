from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from nearfield.geometry import make_array
from nearfield.spherical import rtp_to_cartesian
from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband
from nearfield.beamformer.phase import design_phase_shifter_weights, weights_over_band_phase_shifter
from nearfield.beamformer.ttd import design_ttd_delays, weights_over_band_ttd
from nearfield.metrics_wideband import beampattern_gain_spectrum
from nearfield.metrics_wideband import _steering_for_angles_at_freq


def main() -> None:
    c = 299_792_458.0
    fc = 28e9
    bw = 400e6
    K = 128
    lam = c / fc
    nx = ny = 16
    dx = dy = lam / 2.0
    xyz = make_array("upa", num_x=nx, num_y=ny, dx=dx, dy=dy)

    r_fixed = 6.0
    p = rtp_to_cartesian(r_fixed, 0.0, 0.0)

    f = subcarrier_frequencies(fc, bw, K)
    A = spherical_steering_wideband(xyz, p, f)

    w_fc = design_phase_shifter_weights(xyz, fc, p)
    W_ps = weights_over_band_phase_shifter(w_fc, K)
    d_sec = design_ttd_delays(xyz, p)
    W_ttd = weights_over_band_ttd(d_sec, f)

    # Choose low/mid/high subcarriers
    ks = [0, K // 2, K - 1]
    labels = ["low", "mid", "high"]
    th_grid = np.linspace(-60, 60, 241)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
    for j, k in enumerate(ks):
        A_angles = _steering_for_angles_at_freq(xyz, r_fixed, th_grid, 0.0, f[k])
        # PS map
        wk = W_ps[k]
        y = A_angles @ np.conjugate(wk)
        g = np.abs(y) ** 2 / max(float(np.vdot(wk, wk).real), 1e-16)
        axes[0, j].plot(th_grid, 10 * np.log10(g + 1e-16))
        axes[0, j].set_title(f"PS {labels[j]}")
        axes[0, j].grid(True, ls=":")
        # TTD map
        wk2 = W_ttd[k]
        y2 = A_angles @ np.conjugate(wk2)
        g2 = np.abs(y2) ** 2 / max(float(np.vdot(wk2, wk2).real), 1e-16)
        axes[1, j].plot(th_grid, 10 * np.log10(g2 + 1e-16), color="tab:orange")
        axes[1, j].set_title(f"TTD {labels[j]}")
        axes[1, j].grid(True, ls=":")

    for ax in axes[:, 0]:
        ax.set_ylabel("Gain (dB)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Azimuth (deg)")
    plt.show()


if __name__ == "__main__":
    main()

