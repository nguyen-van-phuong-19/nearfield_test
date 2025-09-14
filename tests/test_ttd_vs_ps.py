import numpy as np

from nearfield.geometry import make_array
from nearfield.spherical import rtp_to_cartesian, C_MPS
from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband
from nearfield.beamformer.phase import design_phase_shifter_weights, weights_over_band_phase_shifter
from nearfield.beamformer.ttd import design_ttd_delays, weights_over_band_ttd
from nearfield.metrics_wideband import (
    beampattern_gain_spectrum,
    gain_flatness_db,
    achievable_rate_bpshz,
)


def test_ttd_beats_ps_nearfield():
    fc = 28e9
    bw = 2.0e9
    K = 128
    lam = C_MPS / fc
    xyz = make_array("upa", num_x=64, num_y=64, dx=lam / 2.0, dy=lam / 2.0)
    p = rtp_to_cartesian(0.6, 15.0, 0.0)

    f = subcarrier_frequencies(fc, bw, K)
    A = spherical_steering_wideband(xyz, p, f)

    w_fc = design_phase_shifter_weights(xyz, fc, p)
    W_ps = weights_over_band_phase_shifter(w_fc, K)
    d = design_ttd_delays(xyz, p)
    W_ttd = weights_over_band_ttd(d, f)

    gain_ps = beampattern_gain_spectrum(W_ps, A)
    gain_ttd = beampattern_gain_spectrum(W_ttd, A)

    flat_ps = gain_flatness_db(gain_ps)
    flat_ttd = gain_flatness_db(gain_ttd)

    assert flat_ps - flat_ttd >= 3.0 - 1e-9

    sub_bw = bw / K
    N0 = 1e-17
    rate_ps = achievable_rate_bpshz(gain_ps, N0, sub_bw)
    rate_ttd = achievable_rate_bpshz(gain_ttd, N0, sub_bw)

    assert rate_ttd >= 0.99 * rate_ps
