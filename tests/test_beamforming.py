"""Tests for beamforming helper methods."""

import numpy as np

from optimized_nearfield_system import create_system_with_presets


def test_beamforming_outputs_have_correct_shape():
    simulator = create_system_with_presets("small_test")
    positions = [(5.0, 0.0, 50.0), (0.0, 5.0, 60.0)]

    beta_far = simulator.far_field_beamforming(positions)
    beta_avg = simulator.average_phase_beamforming_optimized(positions)

    assert beta_far.shape == (simulator.params.M, simulator.params.N)
    assert beta_avg.shape == (simulator.params.M, simulator.params.N)

    # Far-field phases are all zeros whereas average-phase should not be
    assert np.all(beta_far == 0)
    assert np.any(beta_avg != 0)

