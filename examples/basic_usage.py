"""Basic usage demonstration for the near-field beamforming simulator.

This example shows how to create a small test system, design a simple
beamforming vector and evaluate the resulting average and minimum array
gains for a single user.
"""

from optimized_nearfield_system import create_system_with_presets


def main() -> None:
    """Run a minimal simulation using default helper functions."""
    # Create a small test system (16x16 array) for a quick demonstration
    simulator = create_system_with_presets("small_test")

    # Single user located off boresight
    positions = [(5.0, 0.0, 50.0)]

    # Design beamforming phases using the average phase approach
    beta = simulator.average_phase_beamforming_optimized(positions)

    # Evaluate performance for the configured user
    aag, mag = simulator.compute_aag_mag_batch(beta, positions)

    print(f"Average Array Gain: {aag:.2f}")
    print(f"Minimum Array Gain: {mag:.2f}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()

