"""Near-field (spherical-wave) beamforming utilities.

Public API intentionally minimal and functional. See module docstrings
for detailed equations, units, and references.
"""

from .geometry import make_array
from .grids import make_rtp_grid
from .spherical import (
    rtp_to_cartesian,
    spherical_steering,
    plane_wave_steering,
    spherical_codebook,
)
from .lookup import build_kdtree, nearest_codeword
from .metrics import (
    focusing_gain,
    quantization_loss_at,
    farfield_mismatch_loss,
)
from .codebook_io import (
    save_codebook_h5,
    load_codebook_h5,
    save_codebook_json,
    load_codebook_json,
)
from .wideband import (
    subcarrier_frequencies,
    spherical_steering_wideband,
)
from .beamformer.phase import (
    design_phase_shifter_weights,
    weights_over_band_phase_shifter,
)
from .beamformer.ttd import (
    design_ttd_delays,
    weights_over_band_ttd,
)
from .metrics_wideband import (
    beampattern_gain_spectrum,
    gain_flatness_db,
    beam_squint_deg,
    achievable_rate_bpshz,
    evm_percent,
)

__all__ = [
    "make_array",
    "make_rtp_grid",
    "rtp_to_cartesian",
    "spherical_steering",
    "plane_wave_steering",
    "spherical_codebook",
    "build_kdtree",
    "nearest_codeword",
    "focusing_gain",
    "quantization_loss_at",
    "farfield_mismatch_loss",
    "save_codebook_h5",
    "load_codebook_h5",
    "save_codebook_json",
    "load_codebook_json",
    "subcarrier_frequencies",
    "spherical_steering_wideband",
    "design_phase_shifter_weights",
    "weights_over_band_phase_shifter",
    "design_ttd_delays",
    "weights_over_band_ttd",
    "beampattern_gain_spectrum",
    "gain_flatness_db",
    "beam_squint_deg",
    "achievable_rate_bpshz",
    "evm_percent",
]
