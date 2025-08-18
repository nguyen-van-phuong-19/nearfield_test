"""Simulator level behavioural tests."""

from optimized_nearfield_system import create_system_with_presets


def test_compute_array_gain_accepts_flattened_beta():
    simulator = create_system_with_presets("small_test")
    position = (5.0, 0.0, 50.0)

    beta = simulator.far_field_beamforming([position])

    gain_matrix = simulator.compute_array_gain_optimized(beta, position)
    gain_flat = simulator.compute_array_gain_optimized(beta.flatten(), position)

    assert gain_matrix == gain_flat

