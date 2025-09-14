from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from nearfield.geometry import make_array
from nearfield.spherical import rtp_to_cartesian
from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband
from nearfield.beamformer.phase import (
    design_phase_shifter_weights,
    weights_over_band_phase_shifter,
)
from nearfield.beamformer.ttd import design_ttd_delays, weights_over_band_ttd
from nearfield.metrics_wideband import (
    beampattern_gain_spectrum,
    gain_flatness_db,
    beam_squint_deg,
    achievable_rate_bpshz,
)


def main() -> None:
    c = 299_792_458.0
    fc = 28e9
    bw = 400e6
    K = 256
    lam = c / fc
    nx = ny = 32
    dx = dy = lam / 2.0
    xyz = make_array("upa", num_x=nx, num_y=ny, dx=dx, dy=dy)

    # Focus: r=8 m, theta=0, phi=0
    p = rtp_to_cartesian(8.0, 0.0, 0.0)

    f = subcarrier_frequencies(fc, bw, K)
    A = spherical_steering_wideband(xyz, p, f)

    # PS design at fc (flat over band)
    w_fc = design_phase_shifter_weights(xyz, fc, p)
    W_ps = weights_over_band_phase_shifter(w_fc, K)

    # TTD design across band
    d_sec = design_ttd_delays(xyz, p)
    W_ttd = weights_over_band_ttd(d_sec, f)

    gain_ps = beampattern_gain_spectrum(W_ps, A)
    gain_ttd = beampattern_gain_spectrum(W_ttd, A)

    flat_ps = gain_flatness_db(gain_ps)
    flat_ttd = gain_flatness_db(gain_ttd)
    squint_ps = beam_squint_deg(xyz, fc, W_ps, f, r_fixed_m=8.0)
    squint_ttd = beam_squint_deg(xyz, fc, W_ttd, f, r_fixed_m=8.0)

    sub_bw = bw / K
    N0 = 1e-17
    rate_ps = achievable_rate_bpshz(gain_ps, N0, sub_bw)
    rate_ttd = achievable_rate_bpshz(gain_ttd, N0, sub_bw)

    print(
        f"flatness_PS_dB={flat_ps:.2f}, flatness_TTD_dB={flat_ttd:.2f}, "
        f"squint_PS_deg={squint_ps:.2f}, squint_TTD_deg={squint_ttd:.2f}, "
        f"rate_PS={rate_ps:.3f} bps/Hz, rate_TTD={rate_ttd:.3f} bps/Hz"
    )

    # Plot gain vs frequency
    f_off = (f - fc) / 1e6
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(f_off, 10 * np.log10(gain_ps + 1e-16), label="PS")
    plt.plot(f_off, 10 * np.log10(gain_ttd + 1e-16), label="TTD")
    plt.xlabel("Frequency offset (MHz)")
    plt.ylabel("Gain (dB)")
    plt.title("Beampattern gain vs frequency")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

