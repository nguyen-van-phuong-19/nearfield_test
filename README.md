Near‑Field Multi‑Beamforming (LIS–UAV) — Run & Parameter Guide
================================================================

This project provides an optimized near‑field beamforming simulator with a desktop GUI for running AAG/AMAG experiments, including very large scenarios (up to 1000 users) and 5G/6G/7G presets.

1) Get the Project
-------------------

Clone from GitHub and enter the folder:

```
git clone https://github.com/nguyen-van-phuong-19/nearfield_test
cd nearfield_test
```

2) Environment & Install
------------------------

Requirements (tested): Python 3.9+, 8 GB RAM recommended for large runs.

Create a virtual environment and install packages:

```
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3) Start the GUI
-----------------

```
python main_simulation_gui.py
```

The GUI has three tabs:
- Activity: live logs
- Plots: embedded figures (full size with both scrollbars)
- Summary: per‑scenario statistics (AAG/AMAG per method)

4) Configure Parameters (left panel)
------------------------------------

Core controls:
- Preset (hardware/frequency):
  - `standard` (6 GHz), `5g_sub6` (3.5 GHz), `5g_mmwave` (28 GHz),
    `6g_subthz` (~100 GHz), `7g_thz` (300 GHz), `large_array`, `small_test`.
- Mode (scenario):
  - `standard`, `comprehensive`, or `massive` (large square, up to 1000 users).
- Users: 1–1000 (overrides the mode’s list with a single value).
- Randomize on Run: quickly picks a valid random preset/mode/users.

JSON profile support:
- Check “Use JSON profile” and select a profile from `config/simulation_configs.json`.
  - Randomized profiles (with `"randomize": true`) draw values from ranges/choices.

Advanced Overrides (optional):
- Enable Overrides and set any of:
  - `z_min`, `z_max`, `#z points`, `#realizations/z`
  - `x_min`, `x_max`, `y_min`, `y_max` (user area)
  - `n_jobs` (‑1 = automatic)

Run selection:
- Check “Complete Simulation (AAG/AMAG)” and click Run. Results appear in Plots/Summary.

Recommended large run:
- Preset: `5g_mmwave` or `6g_subthz`
- Mode: `massive`
- Users: 500–1000

5) JSON Configuration (optional)
--------------------------------

Edit `config/simulation_configs.json` to add fixed or randomized profiles. Example (already included):

```jsonc
"randomized_quick": {
  "description": "Quick randomized ranges",
  "randomize": true,
  "users_choices": [1, 3, 5, 10, 20, 50],
  "z_min_range": [0.1, 20.0],
  "z_max_range": [80.0, 200.0],
  "num_z_points_choices": [10, 12, 15, 20, 24, 30],
  "num_realizations_choices": [10, 20, 30, 50, 80, 100],
  "x_range": [-10, 10],
  "y_range": [-10, 10],
  "n_jobs": -1
}
```

6) Run from Code (advanced)
----------------------------

Using presets/modes:

```
from optimized_nearfield_system import create_system_with_presets, create_simulation_config

sim = create_system_with_presets("5g_mmwave")
cfg = create_simulation_config("massive")
cfg.num_users_list = [1000]
results = sim.run_optimized_simulation(cfg)
sim.plot_comprehensive_results(results, save_dir="my_results")
```

Using JSON profiles:

```
from config_loader import ConfigManager
from optimized_nearfield_system import create_system_with_presets

cfgm = ConfigManager()
cfg = cfgm.get_simulation_config("randomized_quick")
sim = create_system_with_presets("6g_subthz")
results = sim.run_optimized_simulation(cfg)
```

7) Tips & Troubleshooting
-------------------------

- Large runs are compute heavy. Reduce `#realizations` and `#z points` if needed.
- Plots tab shows full‑size figures; use horizontal/vertical scrollbars to pan.
- Use “Cancel Run” before closing the window during heavy runs.
- Missing packages? `pip install -r requirements.txt` (numpy, matplotlib, scipy, joblib, psutil, pillow).
- Windows: ensure your Python includes Tkinter (comes with the standard installer).

Project URL: https://github.com/nguyen-van-phuong-19/nearfield_test

