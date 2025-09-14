import numpy as np

from nearfield.geometry import make_array
from nearfield.spherical import rtp_to_cartesian, C_MPS
from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband
from nearfield.beamformer.phase import design_phase_shifter_weights, weights_over_band_phase_shifter
from nearfield.beamformer.ttd import design_ttd_delays, weights_over_band_ttd
from nearfield.metrics_wideband import (
    beampattern_gain_spectrum,
    evm_percent,
    beam_squint_deg,
)


def test_beampattern_gain_exact_match_constant():
    fc = 20e9
    bw = 100e6
    K = 32
    lam = C_MPS / fc
    xyz = make_array("ula", num_x=8, dx=lam / 2.0)
    p = rtp_to_cartesian(4.0, 0.0, 0.0)
    f = subcarrier_frequencies(fc, bw, K)
    A = spherical_steering_wideband(xyz, p, f)
    W = A.copy()  # exact match per subcarrier
    g = beampattern_gain_spectrum(W, A)
    assert np.allclose(g, g[0])
    assert np.all(g > 0)


def test_evm_and_squint_behavior():
    fc = 28e9
    bw = 200e6
    K = 96
    lam = C_MPS / fc
    xyz = make_array("upa", num_x=12, num_y=12, dx=lam / 2.0, dy=lam / 2.0)
    p = rtp_to_cartesian(1.2, 15.0, 0.0)
    f = subcarrier_frequencies(fc, bw, K)
    A = spherical_steering_wideband(xyz, p, f)

    # PS vs TTD
    w_fc = design_phase_shifter_weights(xyz, fc, p)
    W_ps = weights_over_band_phase_shifter(w_fc, K)
    d = design_ttd_delays(xyz, p)
    W_ttd = weights_over_band_ttd(d, f)

    evm_ps = evm_percent(W_ps, A)
    evm_ttd = evm_percent(W_ttd, A)
    assert evm_ttd < evm_ps

    squint_ps = beam_squint_deg(xyz, fc, W_ps, f, r_fixed_m=1.2)
    squint_ttd = beam_squint_deg(xyz, fc, W_ttd, f, r_fixed_m=1.2)
    assert squint_ps >= squint_ttd - 1e-9
    assert squint_ttd <= 0.5 + 1e-9
