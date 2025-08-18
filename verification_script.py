"""Simple verification script for the near-field beamforming project.

The goal is to ensure that the helper constructors return objects with
expected attributes. This script can be executed manually to perform a
quick sanity check of the installation.
"""

from optimized_nearfield_system import (
    create_simulation_config,
    create_system_with_presets,
)


def verify() -> None:
    simulator = create_system_with_presets("small_test")
    config = create_simulation_config("fast")

    assert simulator.params.M == 16 and simulator.params.N == 16
    assert len(config.z_values) == 10
    assert config.num_users_list == [5]

    print("System and configuration verified successfully.")


if __name__ == "__main__":  # pragma: no cover - helper script
    verify()

