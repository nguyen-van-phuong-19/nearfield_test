"""Utilities for safe random parameter selection.

All random choices respect the valid domains used across the project
so that simulations remain within expected bounds.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np

# Valid presets and modes should stay in-sync with
# optimized_nearfield_system.create_system_with_presets/create_simulation_config
VALID_PRESETS = ["standard", "high_freq", "large_array", "small_test"]
VALID_MODES = ["fast", "standard", "comprehensive"]


def random_basic_params(users_min: int = 1, users_max: int = 20) -> Dict[str, object]:
    """Pick a valid random preset, mode, and user count.

    Args:
        users_min: Minimum users (inclusive, >=1).
        users_max: Maximum users (inclusive).

    Returns:
        Dict with keys: preset(str), mode(str), users(int)
    """
    users_min = max(1, int(users_min))
    users_max = max(users_min, int(users_max))
    return {
        "preset": random.choice(VALID_PRESETS),
        "mode": random.choice(VALID_MODES),
        "users": random.randint(users_min, users_max),
    }


def apply_random_to_config(
    config,
    *,
    randomize_users: bool = True,
    randomize_ranges: bool = False,
    users_min: int = 1,
    users_max: int = 20,
):
    """Optionally randomize SimulationConfig fields while staying in bounds.

    - Users: sets ``num_users_list`` to a single random integer in [users_min, users_max].
    - Ranges: adjusts ``z_values`` length and span within [0.1, 200] and keeps
      x/y ranges within [-10, 10].
    """
    if randomize_users:
        users_min = max(1, int(users_min))
        users_max = max(users_min, int(users_max))
        config.num_users_list = [random.randint(users_min, users_max)]

    if randomize_ranges:
        # Choose z range and number of points within sensible bounds
        z_min = round(random.uniform(0.1, 20.0), 1)
        z_max = round(random.uniform(80.0, 200.0), 1)
        if z_max <= z_min:
            z_min, z_max = 0.1, 200.0
        num_points = random.choice([10, 12, 15, 20, 24, 30])
        config.z_values = np.linspace(z_min, z_max, num_points)

        # Keep x/y ranges within original rule-of-thumb box
        x_lo = round(random.uniform(-10.0, -2.0), 1)
        x_hi = round(random.uniform(2.0, 10.0), 1)
        y_lo = round(random.uniform(-10.0, -2.0), 1)
        y_hi = round(random.uniform(2.0, 10.0), 1)
        config.x_range = (min(x_lo, -2.0), max(x_hi, 2.0))
        config.y_range = (min(y_lo, -2.0), max(y_hi, 2.0))

    return config

