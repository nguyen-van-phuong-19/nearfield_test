"""Compare basic beamforming strategies for the LIS-UAV system.

The script contrasts far-field and average-phase beamforming for a small
test array and prints out the resulting average and minimum array gains.
"""

from optimized_nearfield_system import create_system_with_presets


def main() -> None:
    simulator = create_system_with_presets("small_test")

    # Two users located at different positions around the LIS
    positions = [(5.0, 0.0, 50.0), (-3.0, 4.0, 60.0)]

    # Far-field beamforming (all zeros)
    beta_far = simulator.far_field_beamforming(positions)
    aag_far, mag_far = simulator.compute_aag_mag_batch(beta_far, positions)

    # Average-phase beamforming
    beta_avg = simulator.average_phase_beamforming_optimized(positions)
    aag_avg, mag_avg = simulator.compute_aag_mag_batch(beta_avg, positions)

    print("Far-field beamforming:   AAG = {:.2f}, MAG = {:.2f}".format(aag_far, mag_far))
    print("Average phase beamforming: AAG = {:.2f}, MAG = {:.2f}".format(aag_avg, mag_avg))


if __name__ == "__main__":  # pragma: no cover - example script
    main()

