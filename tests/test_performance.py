"""Performance metric related tests."""

from optimized_nearfield_system import create_system_with_presets


def test_aag_mag_relationship():
    simulator = create_system_with_presets("small_test")
    positions = [(5.0, 0.0, 50.0), (-3.0, 4.0, 60.0)]

    beta = simulator.average_phase_beamforming_optimized(positions)
    aag, mag = simulator.compute_aag_mag_batch(beta, positions)

    assert isinstance(aag, float) and isinstance(mag, float)
    assert aag >= mag

