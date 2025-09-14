import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import sys
import queue
from typing import List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import os
import multiprocessing as mp
import numpy as np
import math
import traceback

from random_params import VALID_PRESETS, VALID_MODES, random_basic_params
from config_loader import ConfigManager

# Demo/experiment entry points
from demo_script import (
    demo_basic_functionality,
    demo_parameter_analysis,
    demo_fast_simulation,
    demo_comparison_analysis,
    demo_performance_benchmark,
    demo_gui_error_check,
)
from research_workflow import run_random_quick_experiment
from optimized_nearfield_system import create_system_with_presets, create_simulation_config
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime

# Optional HDF5 support
try:  # pragma: no cover - optional
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional
    h5py = None
try:  # pragma: no cover - optional
    import plotly.io as pio  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional
    pio = None
    go = None

# Nearfield helpers for advanced maps
try:
    from nearfield.heatmaps import (
        make_theta_phi_grid,
        build_steering_on_angular_slice,
        build_steering_on_radial_slice,
    )
    from nearfield.metrics_aag_amg import per_point_gains
    from nearfield.plotting_interactive import (
        heatmap_theta_phi as plotly_heatmap_theta_phi,
        surface_theta_phi as plotly_surface_theta_phi,
        line_radial_slice as plotly_line_radial_slice,
    )
except Exception:
    # Modules may not be present during partial installs; GUI still loads.
    make_theta_phi_grid = None  # type: ignore
    build_steering_on_angular_slice = None  # type: ignore
    build_steering_on_radial_slice = None  # type: ignore
    per_point_gains = None  # type: ignore
    plotly_heatmap_theta_phi = None  # type: ignore
    plotly_surface_theta_phi = None  # type: ignore
    plotly_line_radial_slice = None  # type: ignore

# Optional optimizer (GWO)
try:
    from nearfield.optim.gwo import gwo_minimize  # type: ignore
    from nearfield.optim.adapters import make_objective  # type: ignore
except Exception:  # pragma: no cover - optional
    gwo_minimize = None  # type: ignore
    make_objective = None  # type: ignore

# --- Physical constants ---
SPEED_OF_LIGHT = 299_792_458.0


# ============= Minimal math/helpers for near-field experiments =============
def rtp_to_xyz(r: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """Convert spherical (r, theta_deg=azimuth, phi_deg=elevation) to XYZ in meters.

    theta: azimuth in degrees, phi: elevation in degrees.
    x = r cos(phi) cos(theta); y = r cos(phi) sin(theta); z = r sin(phi)
    """
    th = math.radians(float(theta_deg))
    ph = math.radians(float(phi_deg))
    cph = math.cos(ph)
    return np.array([
        r * cph * math.cos(th),
        r * cph * math.sin(th),
        r * math.sin(ph),
    ], dtype=np.float64)


def make_array(layout: str, num_x: int, num_y: int, dx: float, dy: float) -> np.ndarray:
    """Create array element positions for UPA/ULA centered at origin.

    layout: 'upa' or 'ula'
    Returns array of shape (M, 3)
    """
    layout = (layout or "upa").lower()
    num_x = int(max(1, num_x))
    num_y = int(max(1, num_y))
    dx = float(dx)
    dy = float(dy)
    if dx <= 0 or dy <= 0:
        raise ValueError("dx/dy must be > 0")
    if layout not in {"upa", "ula"}:
        raise ValueError("layout must be 'upa' or 'ula'")
    if layout == "ula":
        num_y = 1
        dy = 1.0  # unused

    xs = (np.arange(num_x, dtype=np.float64) - (num_x - 1) / 2.0) * dx
    ys = (np.arange(num_y, dtype=np.float64) - (num_y - 1) / 2.0) * (dy if layout == "upa" else 0.0)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    M = num_x * num_y
    xyz = np.zeros((M, 3), dtype=np.float64)
    xyz[:, 0] = xv.reshape(-1)
    xyz[:, 1] = yv.reshape(-1)
    # z=0 plane
    # Validate uniqueness
    if np.unique(np.round(xyz, 9), axis=0).shape[0] != xyz.shape[0]:
        raise ValueError("Array has duplicate element positions")
    return xyz


def _spherical_steering_vector(xyz_m: np.ndarray, p_xyz: np.ndarray, fc_hz: float) -> np.ndarray:
    """Near-field narrowband steering at point p for carrier fc (complex, unit-norm)."""
    lam = SPEED_OF_LIGHT / float(fc_hz)
    k = 2.0 * math.pi / lam
    d = np.linalg.norm(p_xyz[None, :] - xyz_m, axis=1)
    a = np.exp(-1j * k * d)
    # normalize to unit norm
    a = a / (np.linalg.norm(a) + 1e-12)
    return a.astype(np.complex128)


def _far_field_steering_vector(xyz_m: np.ndarray, theta_deg: float, phi_deg: float, fc_hz: float) -> np.ndarray:
    """Plane-wave steering for direction (theta, phi) at fc (unit-norm)."""
    lam = SPEED_OF_LIGHT / float(fc_hz)
    k = 2.0 * math.pi / lam
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    u = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)], dtype=np.float64)
    phase = -k * (xyz_m @ u)
    a = np.exp(1j * phase)
    a = a / (np.linalg.norm(a) + 1e-12)
    return a.astype(np.complex128)


def _wideband_steering_vector(xyz_m: np.ndarray, p_xyz: np.ndarray, f_hz: float) -> np.ndarray:
    """Wideband steering per subcarrier frequency f."""
    tau = np.linalg.norm(p_xyz[None, :] - xyz_m, axis=1) / SPEED_OF_LIGHT
    a = np.exp(-1j * 2.0 * math.pi * float(f_hz) * tau)
    a = a / (np.linalg.norm(a) + 1e-12)
    return a.astype(np.complex128)


def _gain(w: np.ndarray, a: np.ndarray) -> float:
    """Focusing gain |w^H a|^2 / ||w||^2"""
    num = np.abs(np.vdot(w, a)) ** 2
    den = (np.linalg.norm(w) ** 2) + 1e-12
    return float(num / den)


# ============= Config/Controller for experiments =============
@dataclass
class RunConfig:
    """Aggregated configuration for experiments and overrides."""
    # Global overrides
    n_jobs: int = -1
    seed: Optional[int] = None
    log_level: str = "INFO"

    # Experiment selections
    do_build_codebook: bool = False
    do_eval_codebook: bool = False
    do_wideband_ttd_vs_phase: bool = False

    # Array params (shared)
    layout: str = "upa"
    num_x: int = 8
    num_y: int = 8
    dx_m: float = 0.005
    dy_m: float = 0.005

    # Spherical Codebook params
    fc_hz: float = 28e9
    r_min: float = 1.0
    r_max: float = 10.0
    r_step: float = 0.5
    theta_start: float = -60.0
    theta_stop: float = 60.0
    theta_step: float = 5.0
    phi_start: float = -20.0
    phi_stop: float = 20.0
    phi_step: float = 5.0
    chunk: int = 2048
    out_codebook_path: str | None = None

    # Evaluate Codebook Mismatch
    Q: int = 200
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 1.0
    z_max: float = 5.0
    compare_ff: bool = False
    in_codebook_path: str | None = None
    save_eval_csv: bool = False
    save_eval_png: bool = False
    save_eval_html: bool = False

    # Wideband TTD vs Phase
    wb_fc_hz: float = 28e9
    wb_bw_hz: float = 1e9
    wb_n_sc: int = 64
    focus_r: float = 2.0
    focus_theta_deg: float = 0.0
    focus_phi_deg: float = 0.0
    save_wb_png: bool = False
    save_wb_json: bool = False
    save_wb_html: bool = False

    # Advanced AAG/AMAG maps
    do_adv_maps: bool = False
    adv_map_type: str = "Angular 2D (θ–ϕ @ r)"  # or "Radial Slice (r @ θ,ϕ)"
    adv_fc_hz: float = 28e9
    adv_layout: str = "upa"
    adv_num_x: int = 8
    adv_num_y: int = 8
    adv_dx_m: float = 0.005
    adv_dy_m: float = 0.005
    adv_r_fixed_m: float = 5.0
    adv_theta_start: float = -60.0
    adv_theta_stop: float = 60.0
    adv_theta_step: float = 5.0
    adv_phi_start: float = -20.0
    adv_phi_stop: float = 20.0
    adv_phi_step: float = 5.0
    adv_r_min: float = 1.0
    adv_r_max: float = 10.0
    adv_r_step: float = 0.5
    adv_theta_deg: float = 0.0
    adv_phi_deg: float = 0.0
    adv_weighting: str = "Ideal (AMAG)"  # or "Codebook-selected (AAG)"
    adv_compare_ff: bool = False
    adv_use_inmem_cb: bool = True
    adv_cb_path: str | None = None
    adv_save_html: bool = True
    adv_save_csv: bool = False
    adv_save_png: bool = False
    adv_chunk: int = 2048

    # Optimizer (optional)
    use_gwo: bool = False
    gwo_n_agents: int = 30
    gwo_n_iter: int = 200
    gwo_seed: Optional[int] = None
    gwo_patience: Optional[int] = None
    gwo_target: str = "Max Gain @ focus"

    # Computed/optional
    callbacks: dict[str, Callable[..., Any]] = field(default_factory=dict)
    cancel_flag: Optional[threading.Event] = None


class ExperimentController:
    """Controller to orchestrate experiment runs and GUI callbacks."""
    def __init__(self, gui: "MainSimulationGUI") -> None:
        self.gui = gui
        self.cancel_flag = threading.Event()
        self._last_codebook_result: Optional[dict[str, Any]] = None

    def read_ui(self) -> RunConfig:
        cfg = RunConfig()
        # Advanced overrides (experiments)
        try:
            cfg.n_jobs = int(self.gui.exp_njobs_var.get())
            seed_val = self.gui.exp_seed_var.get().strip()
            cfg.seed = int(seed_val) if seed_val else None
            cfg.log_level = self.gui.exp_loglevel_var.get() or "INFO"
        except Exception:
            pass

        # Selections
        cfg.do_build_codebook = bool(self.gui.exp_build_cb_var.get())
        cfg.do_eval_codebook = bool(self.gui.exp_eval_cb_var.get())
        cfg.do_wideband_ttd_vs_phase = bool(self.gui.exp_wideband_cb_var.get())
        cfg.do_adv_maps = bool(self.gui.exp_adv_maps_cb_var.get())

        # Shared array params from spherical and wideband panels (use spherical panel as base)
        try:
            cfg.layout = (self.gui.sc_layout_var.get() or "upa").lower()
            cfg.num_x = int(self.gui.sc_numx_var.get())
            cfg.num_y = int(self.gui.sc_numy_var.get())
            cfg.dx_m = float(self.gui.sc_dx_var.get())
            cfg.dy_m = float(self.gui.sc_dy_var.get())
        except Exception:
            pass

        # Spherical Codebook params
        try:
            cfg.fc_hz = float(self.gui.sc_fc_var.get())
            cfg.r_min = float(self.gui.sc_rmin_var.get())
            cfg.r_max = float(self.gui.sc_rmax_var.get())
            cfg.r_step = float(self.gui.sc_rstep_var.get())
            cfg.theta_start = float(self.gui.sc_th_start_var.get())
            cfg.theta_stop = float(self.gui.sc_th_stop_var.get())
            cfg.theta_step = float(self.gui.sc_th_step_var.get())
            cfg.phi_start = float(self.gui.sc_ph_start_var.get())
            cfg.phi_stop = float(self.gui.sc_ph_stop_var.get())
            cfg.phi_step = float(self.gui.sc_ph_step_var.get())
            cfg.chunk = int(self.gui.sc_chunk_var.get())
            out_path = self.gui.sc_out_path_var.get().strip()
            cfg.out_codebook_path = out_path or None
        except Exception:
            pass

        # Evaluate Codebook
        try:
            cfg.Q = int(self.gui.ev_Q_var.get())
            cfg.x_min = float(self.gui.ev_xmin_var.get())
            cfg.x_max = float(self.gui.ev_xmax_var.get())
            cfg.y_min = float(self.gui.ev_ymin_var.get())
            cfg.y_max = float(self.gui.ev_ymax_var.get())
            cfg.z_min = float(self.gui.ev_zmin_var.get())
            cfg.z_max = float(self.gui.ev_zmax_var.get())
            cfg.compare_ff = bool(self.gui.ev_compare_ff_var.get())
            in_path = self.gui.ev_in_path_var.get().strip()
            cfg.in_codebook_path = in_path or None
            cfg.save_eval_csv = bool(self.gui.ev_save_csv_var.get())
            cfg.save_eval_png = bool(self.gui.ev_save_png_var.get())
            cfg.save_eval_html = bool(self.gui.ev_save_html_var.get())
        except Exception:
            pass

        # Advanced maps
        try:
            cfg.adv_map_type = self.gui.adv_map_type_var.get()
            cfg.adv_fc_hz = float(self.gui.adv_fc_var.get())
            cfg.adv_layout = (self.gui.adv_layout_var.get() or "upa").lower()
            cfg.adv_num_x = int(self.gui.adv_numx_var.get())
            cfg.adv_num_y = int(self.gui.adv_numy_var.get())
            cfg.adv_dx_m = float(self.gui.adv_dx_var.get())
            cfg.adv_dy_m = float(self.gui.adv_dy_var.get())
            cfg.adv_r_fixed_m = float(self.gui.adv_rfixed_var.get())
            cfg.adv_theta_start = float(self.gui.adv_th_start_var.get())
            cfg.adv_theta_stop = float(self.gui.adv_th_stop_var.get())
            cfg.adv_theta_step = float(self.gui.adv_th_step_var.get())
            cfg.adv_phi_start = float(self.gui.adv_ph_start_var.get())
            cfg.adv_phi_stop = float(self.gui.adv_ph_stop_var.get())
            cfg.adv_phi_step = float(self.gui.adv_ph_step_var.get())
            cfg.adv_r_min = float(self.gui.adv_rmin_var.get())
            cfg.adv_r_max = float(self.gui.adv_rmax_var.get())
            cfg.adv_r_step = float(self.gui.adv_rstep_var.get())
            cfg.adv_theta_deg = float(self.gui.adv_theta_var.get())
            cfg.adv_phi_deg = float(self.gui.adv_phi_var.get())
            cfg.adv_weighting = self.gui.adv_weighting_var.get()
            cfg.adv_compare_ff = bool(self.gui.adv_compare_ff_var.get())
            cfg.adv_use_inmem_cb = bool(self.gui.adv_use_inmem_cb_var.get())
            cb_path = self.gui.adv_cb_path_var.get().strip()
            cfg.adv_cb_path = cb_path or None
            cfg.adv_save_html = bool(self.gui.adv_save_html_var.get())
            cfg.adv_save_csv = bool(self.gui.adv_save_csv_var.get())
            cfg.adv_save_png = bool(self.gui.adv_save_png_var.get())
            cfg.adv_chunk = int(self.gui.adv_chunk_var.get())
        except Exception:
            pass

        # Optimizer (optional)
        try:
            cfg.use_gwo = bool(self.gui.opt_use_gwo_var.get())
            cfg.gwo_n_agents = int(self.gui.opt_agents_var.get())
            cfg.gwo_n_iter = int(self.gui.opt_iters_var.get())
            tgt = self.gui.opt_target_var.get()
            if tgt:
                cfg.gwo_target = tgt
            seed_val = (self.gui.opt_seed_var.get() or "").strip()
            cfg.gwo_seed = int(seed_val) if seed_val else cfg.seed
            pat = (self.gui.opt_patience_var.get() or "").strip()
            cfg.gwo_patience = int(pat) if pat else None
        except Exception:
            pass

        # Wideband
        try:
            cfg.wb_fc_hz = float(self.gui.wb_fc_var.get())
            cfg.wb_bw_hz = float(self.gui.wb_bw_var.get())
            cfg.wb_n_sc = int(self.gui.wb_nsc_var.get())
            cfg.focus_r = float(self.gui.wb_r_var.get())
            cfg.focus_theta_deg = float(self.gui.wb_theta_var.get())
            cfg.focus_phi_deg = float(self.gui.wb_phi_var.get())
            cfg.save_wb_png = bool(self.gui.wb_save_png_var.get())
            cfg.save_wb_json = bool(self.gui.wb_save_json_var.get())
            cfg.save_wb_html = bool(self.gui.wb_save_html_var.get())
        except Exception:
            pass

        # Callbacks
        cfg.callbacks = {
            "log": self.gui._append_log,
            "status": self.gui._schedule_status,
            "progress": self.gui._set_progress_fraction,
            "plots_clear": self.gui._clear_plots,
            "add_fig": self.gui._add_figure,
            "set_summary": self.gui._set_summary_text,
        }
        cfg.cancel_flag = self.cancel_flag
        return cfg

    # Expose last codebook to evaluation step
    def get_last_codebook(self) -> Optional[dict]:
        return self._last_codebook_result

    def set_last_codebook(self, res: dict | None) -> None:
        self._last_codebook_result = res


# ============= Experiment run functions =============
def run_build_spherical_codebook(cfg: RunConfig) -> dict:
    """
    Build spherical codebook over (r,theta,phi) grid for the specified array and fc.
    Returns a dict with keys: {'xyz_m','fc_hz','rtp_grid','codebook','attrs'}.
    Write to HDF5 if an output path is provided.
    Stream logs to Activity tab; push a small progress bar (% grid processed).
    """
    log = cfg.callbacks.get("log", lambda s: None)
    status = cfg.callbacks.get("status", lambda s: None)
    progress = cfg.callbacks.get("progress", lambda f=None: None)

    # Validation
    if cfg.fc_hz <= 0 or cfg.r_min <= 0 or cfg.r_step <= 0 or cfg.wb_n_sc < 0:
        raise ValueError("Invalid numeric parameters (check fc, r_min, r_step)")
    if cfg.num_x < 1 or cfg.num_y < 1 or cfg.dx_m <= 0 or cfg.dy_m <= 0:
        raise ValueError("Invalid array parameters")

    rng = np.random.default_rng(cfg.seed)
    _ = rng  # reserved for future use

    log("Preparing array and grid...\n")
    xyz_m = make_array(cfg.layout, cfg.num_x, cfg.num_y, cfg.dx_m, cfg.dy_m)
    r_vals = np.arange(cfg.r_min, cfg.r_max + 0.5 * cfg.r_step, cfg.r_step, dtype=np.float64)
    th_vals = np.arange(cfg.theta_start, cfg.theta_stop + 0.5 * cfg.theta_step, cfg.theta_step, dtype=np.float64)
    ph_vals = np.arange(cfg.phi_start, cfg.phi_stop + 0.5 * cfg.phi_step, cfg.phi_step, dtype=np.float64)
    grid = np.array([(r, th, ph) for r in r_vals for th in th_vals for ph in ph_vals], dtype=np.float64)
    N = grid.shape[0]
    M = xyz_m.shape[0]
    log(f"Grid points: {N} | Elements: {M}\n")

    # Allocate codebook
    codebook = np.zeros((N, M), dtype=np.complex128)
    chunk = max(1, int(cfg.chunk))
    status("Building spherical codebook...")
    progress(0.0)
    t0 = time.perf_counter()
    for i0 in range(0, N, chunk):
        if cfg.cancel_flag and cfg.cancel_flag.is_set():
            log("Cancelled during codebook build.\n")
            break
        i1 = min(N, i0 + chunk)
        for i in range(i0, i1):
            r, th, ph = grid[i]
            p = rtp_to_xyz(float(r), float(th), float(ph))
            a = _spherical_steering_vector(xyz_m, p, cfg.fc_hz)
            codebook[i, :] = a
        progress(i1 / N)
    dt = time.perf_counter() - t0
    status(f"Codebook built in {dt:.2f}s")
    progress(None)

    attrs = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fc_hz": float(cfg.fc_hz),
        "layout": cfg.layout,
        "num_x": int(cfg.num_x),
        "num_y": int(cfg.num_y),
        "dx_m": float(cfg.dx_m),
        "dy_m": float(cfg.dy_m),
        "r_min": float(cfg.r_min),
        "r_max": float(cfg.r_max),
        "r_step": float(cfg.r_step),
        "theta_start": float(cfg.theta_start),
        "theta_stop": float(cfg.theta_stop),
        "theta_step": float(cfg.theta_step),
        "phi_start": float(cfg.phi_start),
        "phi_stop": float(cfg.phi_stop),
        "phi_step": float(cfg.phi_step),
        "seed": int(cfg.seed) if cfg.seed is not None else None,
    }
    res = {
        "xyz_m": xyz_m,
        "fc_hz": float(cfg.fc_hz),
        "rtp_grid": grid,
        "codebook": codebook,
        "attrs": attrs,
    }

    # Save if requested
    if cfg.out_codebook_path:
        if h5py is None:
            log("h5py not available; skipping HDF5 save.\n")
        else:
            status("Writing HDF5...")
            try:
                with h5py.File(cfg.out_codebook_path, "w") as hf:
                    hf.create_dataset("xyz_m", data=xyz_m)
                    hf.create_dataset("fc_hz", data=np.array([cfg.fc_hz], dtype=np.float64))
                    hf.create_dataset("rtp_grid", data=grid)
                    hf.create_dataset("codebook", data=codebook)
                    for k, v in attrs.items():
                        if v is not None:
                            hf.attrs[k] = v
                log(f"Saved codebook to {cfg.out_codebook_path}\n")
            except Exception as e:  # pragma: no cover - filesystem
                log(f"HDF5 save error: {e}\n")
    return res


def run_eval_codebook_mismatch(cfg: RunConfig, codebook_res: Optional[dict] = None) -> dict:
    """
    Load or reuse an in-memory spherical codebook. Sample Q query points in the region,
    compute (1) quantization loss vs ideal spherical focusing and
    (2) optional far-field mismatch loss.
    Plot two histograms to the Plots tab and a small text summary to the Summary tab.
    Save CSV of per-point losses if toggled. Return summary stats dict.
    """
    log = cfg.callbacks.get("log", lambda s: None)
    status = cfg.callbacks.get("status", lambda s: None)
    add_fig = cfg.callbacks.get("add_fig", lambda f: None)
    plots_clear = cfg.callbacks.get("plots_clear", lambda: None)
    set_summary = cfg.callbacks.get("set_summary", lambda s: None)

    rng = np.random.default_rng(cfg.seed)
    # Load or reuse
    if codebook_res is None and cfg.in_codebook_path:
        if h5py is None:
            raise RuntimeError("h5py not available to load codebook")
        status("Loading codebook from HDF5...")
        with h5py.File(cfg.in_codebook_path, "r") as hf:
            xyz_m = hf["xyz_m"][...]
            fc_hz = float(hf["fc_hz"][...][0]) if "fc_hz" in hf else float(hf.attrs.get("fc_hz", cfg.fc_hz))
            rtp_grid = hf["rtp_grid"][...]
            codebook = hf["codebook"][...]
        codebook_res = {"xyz_m": xyz_m, "fc_hz": fc_hz, "rtp_grid": rtp_grid, "codebook": codebook, "attrs": {}}
        log(f"Loaded codebook: {cfg.in_codebook_path}\n")
    if codebook_res is None:
        raise RuntimeError("No in-memory codebook and no HDF5 path provided.")

    xyz_m = np.array(codebook_res["xyz_m"])  # (M,3)
    fc_hz = float(codebook_res.get("fc_hz", cfg.fc_hz))
    codebook = np.array(codebook_res["codebook"])  # (N,M)

    # Sample Q query points
    Q = int(max(1, cfg.Q))
    xs = rng.uniform(cfg.x_min, cfg.x_max, size=Q)
    ys = rng.uniform(cfg.y_min, cfg.y_max, size=Q)
    zs = rng.uniform(cfg.z_min, cfg.z_max, size=Q)
    P = np.stack([xs, ys, zs], axis=1)

    # Evaluate losses
    losses_q_db = np.zeros(Q, dtype=np.float64)
    losses_ff_db = np.zeros(Q, dtype=np.float64) if cfg.compare_ff else None
    log("Evaluating codebook mismatch...\n")
    for i in range(Q):
        if cfg.cancel_flag and cfg.cancel_flag.is_set():
            log("Cancelled during evaluation.\n")
            break
        p = P[i]
        a = _spherical_steering_vector(xyz_m, p, fc_hz)
        # Best codebook match by inner product magnitude
        sims = np.abs(codebook @ np.conjugate(a))  # (N,)
        best = np.argmax(sims)
        c = codebook[best]
        Gq = _gain(c, a)
        losses_q_db[i] = -10.0 * np.log10(max(Gq, 1e-12))
        if cfg.compare_ff and losses_ff_db is not None:
            # Direction from origin to p
            r = float(np.linalg.norm(p))
            if r <= 0:
                losses_ff_db[i] = 0.0
            else:
                # derive angles
                # phi = arcsin(z/r), theta = atan2(y,x)
                phi_deg = math.degrees(math.asin(float(p[2] / r)))
                theta_deg = math.degrees(math.atan2(float(p[1]), float(p[0])))
                ff = _far_field_steering_vector(xyz_m, theta_deg, phi_deg, fc_hz)
                Gff = _gain(ff, a)
                losses_ff_db[i] = -10.0 * np.log10(max(Gff, 1e-12))

    # Plot histograms
    fig_mpl = None
    try:
        plots_clear()
        fig_mpl = plot_histograms(losses_q_db, losses_ff_db)
        add_fig(fig_mpl)
    except Exception as e:
        log(f"Plot error: {e}\n")

    # Summary
    def _stats(x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
            "q95": float(np.quantile(x, 0.95)),
        }

    summary = {"quantization_loss_db": _stats(losses_q_db)}
    if losses_ff_db is not None:
        summary["farfield_loss_db"] = _stats(losses_ff_db)

    # Text summary
    lines = ["Codebook Evaluation Summary\n"]
    q = summary["quantization_loss_db"]
    lines.append(
        f"- Quantization loss dB: mean={q['mean']:.2f}, median={q['median']:.2f}, std={q['std']:.2f}, 95th={q['q95']:.2f}"
    )
    if "farfield_loss_db" in summary:
        ff = summary["farfield_loss_db"]
        lines.append(
            f"- Far-field mismatch dB: mean={ff['mean']:.2f}, median={ff['median']:.2f}, std={ff['std']:.2f}, 95th={ff['q95']:.2f}"
        )
    set_summary("\n".join(lines))

    # Optional CSV/PNG save
    if cfg.save_eval_csv:
        try:
            out_csv = os.path.join(os.getcwd(), f"codebook_eval_{int(time.time())}.csv")
            with open(out_csv, "w", encoding="utf-8") as f:
                if losses_ff_db is None:
                    f.write("loss_quant_db\n")
                    for v in losses_q_db:
                        f.write(f"{float(v):.6f}\n")
                else:
                    f.write("loss_quant_db,loss_farfield_db\n")
                    for v1, v2 in zip(losses_q_db, losses_ff_db):
                        f.write(f"{float(v1):.6f},{float(v2):.6f}\n")
            log(f"Saved CSV: {out_csv}\n")
        except Exception as e:  # pragma: no cover - filesystem
            log(f"CSV save error: {e}\n")
    if cfg.save_eval_png and fig_mpl is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            out_png = os.path.join(os.getcwd(), f"codebook_eval_{int(time.time())}.png")
            fig_mpl.savefig(out_png, dpi=150, bbox_inches="tight")
            log(f"Saved PNG: {out_png}\n")
        except Exception as e:  # pragma: no cover - filesystem
            log(f"PNG save error: {e}\n")

    # Optional HTML (interactive) save via Plotly
    if cfg.save_eval_html:
        try:
            import plotly.graph_objects as go  # type: ignore
            import plotly.io as pio  # type: ignore
            bins = 30
            figly = go.Figure()
            figly.add_trace(go.Histogram(x=losses_q_db, nbinsx=bins, name='Quantization loss', opacity=0.75))
            if losses_ff_db is not None:
                figly.add_trace(go.Histogram(x=losses_ff_db, nbinsx=bins, name='Far-field mismatch', opacity=0.75))
            figly.update_layout(barmode='overlay', xaxis_title='Loss (dB)', yaxis_title='Count', legend_title_text='Series')
            out_html = os.path.join(os.getcwd(), f"codebook_eval_{int(time.time())}.html")
            pio.write_html(figly, file=out_html, include_plotlyjs='cdn', full_html=True, auto_open=False)
            log(f"Saved interactive HTML: {out_html}\n")
        except Exception as e:
            log(f"HTML save error: {e}\n")

    return summary


def run_wideband_compare_ttd_vs_phase(cfg: RunConfig) -> dict:
    """
    Build wideband spherical steering across subcarriers (K = n_sc).
    Design frequency-flat phase-shifter weights at fc and TTD delays per element.
    Compute gain spectrum per subcarrier, gain flatness (dB), beam squint (deg),
    and achievable rate. Plot gain-vs-frequency (TTD vs PS) and print summary.
    Return dict with metrics and arrays for optional saving.
    """
    log = cfg.callbacks.get("log", lambda s: None)
    status = cfg.callbacks.get("status", lambda s: None)
    plots_clear = cfg.callbacks.get("plots_clear", lambda: None)
    add_fig = cfg.callbacks.get("add_fig", lambda f: None)
    set_summary = cfg.callbacks.get("set_summary", lambda s: None)

    # Validation
    if cfg.wb_fc_hz <= 0 or cfg.wb_bw_hz <= 0 or cfg.wb_n_sc < 4:
        raise ValueError("Invalid wideband params (check fc, bw, n_sc>=4)")
    if cfg.num_x < 1 or cfg.num_y < 1 or cfg.dx_m <= 0 or cfg.dy_m <= 0:
        raise ValueError("Invalid array parameters")

    xyz_m = make_array(cfg.layout, cfg.num_x, cfg.num_y, cfg.dx_m, cfg.dy_m)
    p = rtp_to_xyz(cfg.focus_r, cfg.focus_theta_deg, cfg.focus_phi_deg)

    # Frequencies
    K = int(cfg.wb_n_sc)
    f0 = float(cfg.wb_fc_hz)
    bw = float(cfg.wb_bw_hz)
    freqs = np.linspace(f0 - bw / 2.0, f0 + bw / 2.0, K)

    # PS weights designed at fc (baseline)
    a0 = _spherical_steering_vector(xyz_m, p, f0)
    w_ps = a0 / (np.linalg.norm(a0) + 1e-12)

    # Optional GWO optimizer for phase-only weights across band
    if cfg.use_gwo and make_objective is not None and gwo_minimize is not None:
        try:
            log("Starting GWO optimizer for phase-only wideband weights...\n")
            func, bounds, post = make_objective(cfg.gwo_target, cfg, {
                "xyz_m": xyz_m,
                "p_xyz": p,
                "freqs_hz": np.asarray(freqs, dtype=np.float64),
            })
            # Attach cancel event for cooperative cancellation
            try:
                setattr(func, "__cancel_event__", cfg.cancel_flag)
            except Exception:
                pass
            x_best, f_best, info = gwo_minimize(
                func,
                bounds,
                n_agents=int(max(4, cfg.gwo_n_agents)),
                n_iter=int(max(1, cfg.gwo_n_iter)),
                seed=cfg.gwo_seed,
                clamp=True,
                early_stop_patience=cfg.gwo_patience,
            )
            log(f"GWO finished: iters={info.get('iterations')} best={f_best:.6f} evals={info.get('n_evals')}\n")
            # Log brief history
            try:
                hist = info.get('hist', [])
                for i, v in enumerate(hist):
                    if (i % max(1, int(len(hist) / 10))) == 0:
                        log(f"  iter={i}: best={float(v):.6f}\n")
            except Exception:
                pass
            sol = post(x_best)
            w_ps = np.asarray(sol.get("weights", w_ps))
        except Exception as e:
            log(f"GWO optimizer error (continuing with baseline weights): {e}\n")

    # TTD weights: delays to align at point p
    tau = np.linalg.norm(p[None, :] - xyz_m, axis=1) / SPEED_OF_LIGHT  # (M,)

    gains_ps = np.zeros(K, dtype=np.float64)
    gains_ttd = np.zeros(K, dtype=np.float64)
    for k, fk in enumerate(freqs):
        if cfg.cancel_flag and cfg.cancel_flag.is_set():
            log("Cancelled during wideband comparison.\n")
            break
        a_k = _wideband_steering_vector(xyz_m, p, float(fk))
        # PS fixed weights
        gains_ps[k] = _gain(w_ps, a_k)
        # TTD frequency-dependent weights
        w_ttd = np.exp(-1j * 2.0 * math.pi * float(fk) * tau)
        w_ttd = w_ttd / (np.linalg.norm(w_ttd) + 1e-12)
        gains_ttd[k] = _gain(w_ttd, a_k)

    # Convert to dB
    g_ps_db = 10.0 * np.log10(np.maximum(gains_ps, 1e-12))
    g_ttd_db = 10.0 * np.log10(np.maximum(gains_ttd, 1e-12))
    flat_ps = float(np.max(g_ps_db) - np.min(g_ps_db))
    flat_ttd = float(np.max(g_ttd_db) - np.min(g_ttd_db))

    # Beam squint estimation along azimuth slice at fixed elevation (phi=focus_phi)
    def _estimate_squint(weights: np.ndarray, phi_deg: float) -> float:
        thetas = np.linspace(cfg.focus_theta_deg - 60.0, cfg.focus_theta_deg + 60.0, 121)
        # Peak theta at center frequency
        best0 = 0.0
        best_t0 = thetas[0]
        ff0 = [_far_field_steering_vector(xyz_m, float(t), float(phi_deg), f0) for t in thetas]
        for t, a_ff in zip(thetas, ff0):
            val = _gain(weights, a_ff)
            if val > best0:
                best0 = val
                best_t0 = float(t)
        # Now measure at band edges
        edges = [freqs[0], freqs[-1]]
        t_edges = []
        for fk in edges:
            best = 0.0
            best_t = thetas[0]
            ffk = [_far_field_steering_vector(xyz_m, float(t), float(phi_deg), float(fk)) for t in thetas]
            for t, a_ff in zip(thetas, ffk):
                val = _gain(weights, a_ff)
                if val > best:
                    best = val
                    best_t = float(t)
            t_edges.append(best_t)
        # Average absolute drift relative to center
        return float(np.mean(np.abs(np.array(t_edges) - best_t0)))

    squint_ps = _estimate_squint(w_ps, cfg.focus_phi_deg)
    # For TTD, evaluate at center frequency weights across edges (expect near zero)
    w_ttd_center = np.exp(-1j * 2.0 * math.pi * f0 * tau)
    w_ttd_center = w_ttd_center / (np.linalg.norm(w_ttd_center) + 1e-12)
    squint_ttd = _estimate_squint(w_ttd_center, cfg.focus_phi_deg)

    # Plot spectrum
    fig_mpl = None
    try:
        plots_clear()
        fig_mpl = plot_gain_spectrum(freqs, g_ps_db, g_ttd_db)
        add_fig(fig_mpl)
    except Exception as e:
        log(f"Plot error: {e}\n")

    # Achievable rate (normalized SNR=1)
    rate_ps = float(np.mean(np.log2(1.0 + gains_ps)))
    rate_ttd = float(np.mean(np.log2(1.0 + gains_ttd)))

    summary_lines = [
        "Wideband Comparison Summary\n",
        f"- flatness_PS_dB: {flat_ps:.2f}",
        f"- flatness_TTD_dB: {flat_ttd:.2f}",
        f"- squint_PS_deg: {squint_ps:.2f}",
        f"- squint_TTD_deg: {squint_ttd:.2f}",
        f"- rate_PS: {rate_ps:.3f}",
        f"- rate_TTD: {rate_ttd:.3f}",
    ]
    set_summary("\n".join(summary_lines))

    res = {
        "freqs_hz": freqs,
        "gain_ps_db": g_ps_db,
        "gain_ttd_db": g_ttd_db,
        "flatness_ps_db": flat_ps,
        "flatness_ttd_db": flat_ttd,
        "squint_ps_deg": squint_ps,
        "squint_ttd_deg": squint_ttd,
        "rate_ps": rate_ps,
        "rate_ttd": rate_ttd,
    }

    # Optional saves
    try:
        if cfg.save_wb_png and fig_mpl is not None:
            out_png = os.path.join(os.getcwd(), f"wideband_ttd_vs_ps_{int(time.time())}.png")
            fig_mpl.savefig(out_png, dpi=150, bbox_inches="tight")
            log(f"Saved PNG: {out_png}\n")
        if cfg.save_wb_json:
            out_json = os.path.join(os.getcwd(), f"wideband_ttd_vs_ps_{int(time.time())}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}, f, indent=2)
            log(f"Saved JSON: {out_json}\n")
        if cfg.save_wb_html:
            try:
                import plotly.graph_objects as go  # type: ignore
                import plotly.io as pio  # type: ignore
                figly = go.Figure()
                figly.add_trace(go.Scatter(x=freqs/1e9, y=g_ps_db, mode='lines', name='PS'))
                figly.add_trace(go.Scatter(x=freqs/1e9, y=g_ttd_db, mode='lines', name='TTD'))
                figly.update_layout(xaxis_title='Frequency (GHz)', yaxis_title='Gain (dB)', legend_title_text='Series')
                out_html = os.path.join(os.getcwd(), f"wideband_ttd_vs_ps_{int(time.time())}.html")
                pio.write_html(figly, file=out_html, include_plotlyjs='cdn', full_html=True, auto_open=False)
                log(f"Saved interactive HTML: {out_html}\n")
            except Exception as e:
                log(f"HTML save error: {e}\n")
    except Exception as e:  # pragma: no cover - filesystem
        log(f"Save error: {e}\n")

    return res


def run_advanced_aag_amg(cfg: RunConfig, codebook_res: Optional[dict]) -> dict:
    """Compute AAG/AMAG maps and export interactive Plotly visuals.

    Returns summary dict with stats and file paths.
    """
    log = cfg.callbacks.get("log", lambda s: None)
    status = cfg.callbacks.get("status", lambda s: None)
    set_summary = cfg.callbacks.get("set_summary", lambda s: None)
    progress = cfg.callbacks.get("progress", lambda f=None: None)

    if make_theta_phi_grid is None or per_point_gains is None:
        raise RuntimeError("nearfield modules not available for advanced maps")

    # Build array and validate
    if cfg.adv_fc_hz <= 0 or cfg.adv_dx_m <= 0 or cfg.adv_dy_m <= 0 or cfg.adv_num_x < 1 or cfg.adv_num_y < 1:
        raise ValueError("Invalid array or carrier parameters for advanced maps")
    xyz = make_array(cfg.adv_layout, cfg.adv_num_x, cfg.adv_num_y, cfg.adv_dx_m, cfg.adv_dy_m)

    files: list[str] = []
    rng = np.random.default_rng(cfg.seed)
    _ = rng

    def _save_fig(fig, stem: str) -> None:
        try:
            ts = int(time.time())
            if cfg.adv_save_html and pio is not None:
                out_html = os.path.join(os.getcwd(), f"{stem}_{ts}.html")
                pio.write_html(fig, file=out_html, include_plotlyjs='cdn', full_html=True, auto_open=False)
                files.append(out_html)
                log(f"Saved HTML: {out_html}\n")
            if cfg.adv_save_png and pio is not None:
                try:
                    out_png = os.path.join(os.getcwd(), f"{stem}_{ts}.png")
                    pio.write_image(fig, out_png, scale=2)
                    files.append(out_png)
                    log(f"Saved PNG: {out_png}\n")
                except Exception as e:
                    log(f"PNG export skipped (install kaleido): {e}\n")
        except Exception as e:
            log(f"Figure save error: {e}\n")

    # Angular 2D map
    if cfg.adv_map_type.startswith("Angular 2D"):
        status("Computing angular 2D map...")
        th = np.arange(cfg.adv_theta_start, cfg.adv_theta_stop + 0.5 * cfg.adv_theta_step, cfg.adv_theta_step, dtype=np.float64)
        ph = np.arange(cfg.adv_phi_start, cfg.adv_phi_stop + 0.5 * cfg.adv_phi_step, cfg.adv_phi_step, dtype=np.float64)
        if th.size == 0 or ph.size == 0:
            raise ValueError("Empty theta/phi range")
        grid = make_theta_phi_grid(th, ph)
        P = grid.shape[0]
        chunk = max(1, int(cfg.adv_chunk))
        progress(0.0)
        values_db = np.zeros(P, dtype=np.float64)
        baseline_db = np.zeros(P, dtype=np.float64) if cfg.adv_compare_ff else None

        # Prepare codebook if needed
        C = None
        if cfg.adv_weighting.startswith("Codebook"):
            if cfg.adv_use_inmem_cb and codebook_res is not None:
                C = np.array(codebook_res.get("codebook"))
            elif cfg.adv_cb_path:
                if h5py is None:
                    raise RuntimeError("h5py not available to load codebook")
                with h5py.File(cfg.adv_cb_path, "r") as hf:
                    C = hf["codebook"][...]
            else:
                messagebox.showerror("Codebook required", "Select 'Use in-memory codebook' or choose a codebook HDF5 file.")
                return {}
            if C is None:
                messagebox.showerror("Codebook required", "Codebook not found or failed to load.")
                return {}

        # Compute in chunks for responsiveness
        for i0 in range(0, P, chunk):
            if cfg.cancel_flag and cfg.cancel_flag.is_set():
                log("Cancelled advanced maps computation.\n")
                break
            i1 = min(P, i0 + chunk)
            sub = grid[i0:i1]
            A = build_steering_on_angular_slice(xyz, cfg.adv_fc_hz, cfg.adv_r_fixed_m, sub)
            if cfg.adv_weighting.startswith("Ideal"):
                vals = np.sum(np.abs(A) ** 2, axis=1)
                values_db[i0:i1] = 10.0 * np.log10(np.maximum(vals, 1e-12))
            else:
                # Codebook-selected per point
                sims = A @ np.conjugate(C.T)  # (p,Nc)
                idx = np.argmax(np.abs(sims), axis=1)
                W = C[idx]
                values_db[i0:i1] = per_point_gains(W, A, mode="db")
            if cfg.adv_compare_ff:
                # Far-field weights for same (theta,phi)
                Wff = np.stack([
                    _far_field_steering_vector(xyz, float(t), float(p), cfg.adv_fc_hz) for t, p in sub
                ], axis=0)
                baseline_db[i0:i1] = per_point_gains(Wff, A, mode="db")
            progress(i1 / P)
        progress(None)

        # Optional loss vs ideal when codebook used
        loss_db = None
        if cfg.adv_weighting.startswith("Codebook"):
            amag_db = np.zeros_like(values_db)
            # For normalized steering rows, AMAG is 0 dB
            # But compute explicitly for robustness
            grid_full = build_steering_on_angular_slice(xyz, cfg.adv_fc_hz, cfg.adv_r_fixed_m, grid)
            amag_vals = np.sum(np.abs(grid_full) ** 2, axis=1)
            amag_db = 10.0 * np.log10(np.maximum(amag_vals, 1e-12))
            loss_db = amag_db - values_db

        # Plotly figures and save
        if plotly_heatmap_theta_phi is not None:
            title = "Gain map (dB): " + ("AMAG" if cfg.adv_weighting.startswith("Ideal") else "AAG (codebook)")
            Z = values_db.reshape(ph.size, th.size)
            fig = plotly_heatmap_theta_phi(th, ph, Z, title)
            _save_fig(fig, "adv_map")
            if loss_db is not None:
                fig_loss = plotly_heatmap_theta_phi(th, ph, loss_db.reshape(ph.size, th.size), "Loss (AMAG - AAG) dB")
                _save_fig(fig_loss, "adv_map_loss")
            if baseline_db is not None:
                fig_ff = plotly_heatmap_theta_phi(th, ph, baseline_db.reshape(ph.size, th.size), "Far-field baseline (dB)")
                _save_fig(fig_ff, "adv_map_farfield")
        else:
            log("Plotly not available; skipping interactive heatmap.\n")

        # Summary stats
        def stats(x: np.ndarray) -> dict:
            return {
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "std": float(np.std(x)),
                "p50": float(np.percentile(x, 50)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
            }

        summary = {"values_db": stats(values_db)}
        if loss_db is not None:
            summary["loss_db"] = stats(loss_db)
        if baseline_db is not None:
            summary["farfield_db"] = stats(baseline_db)

        # CSV write
        if cfg.adv_save_csv:
            try:
                out_csv = os.path.join(os.getcwd(), f"adv_map_{int(time.time())}.csv")
                with open(out_csv, "w", encoding="utf-8") as f:
                    header = "theta_deg,phi_deg,value_db"
                    if baseline_db is not None:
                        header += ",baseline_db"
                    if loss_db is not None:
                        header += ",loss_db"
                    f.write(header + "\n")
                    for (t, p), v, i in zip(grid, values_db, range(P)):
                        row = f"{float(t):.6f},{float(p):.6f},{float(v):.6f}"
                        if baseline_db is not None:
                            row += f",{float(baseline_db[i]):.6f}"
                        if loss_db is not None:
                            row += f",{float(loss_db[i]):.6f}"
                        f.write(row + "\n")
                files.append(out_csv)
                log(f"Saved CSV: {out_csv}\n")
            except Exception as e:
                log(f"CSV save error: {e}\n")

        set_summary("Advanced AAG/AMAG map complete. See exported HTML/CSV for interactive view.")

        return {"files": files, "theta": th, "phi": ph, "values_db": values_db}

    # Radial slice
    elif cfg.adv_map_type.startswith("Radial Slice"):
        status("Computing radial slice map...")
        r = np.arange(cfg.adv_r_min, cfg.adv_r_max + 0.5 * cfg.adv_r_step, cfg.adv_r_step, dtype=np.float64)
        if r.size == 0:
            raise ValueError("Empty r range")
        A = build_steering_on_radial_slice(xyz, cfg.adv_fc_hz, r, cfg.adv_theta_deg, cfg.adv_phi_deg)
        if cfg.adv_weighting.startswith("Ideal"):
            vals = np.sum(np.abs(A) ** 2, axis=1)
            values_db = 10.0 * np.log10(np.maximum(vals, 1e-12))
        else:
            C = None
            if cfg.adv_use_inmem_cb and codebook_res is not None:
                C = np.array(codebook_res.get("codebook"))
            elif cfg.adv_cb_path:
                if h5py is None:
                    raise RuntimeError("h5py not available to load codebook")
                with h5py.File(cfg.adv_cb_path, "r") as hf:
                    C = hf["codebook"][...]
            else:
                messagebox.showerror("Codebook required", "Select 'Use in-memory codebook' or choose a codebook HDF5 file.")
                return {}
            sims = A @ np.conjugate(C.T)
            idx = np.argmax(np.abs(sims), axis=1)
            W = C[idx]
            values_db = per_point_gains(W, A, mode="db")

        baseline_db = None
        if cfg.adv_compare_ff:
            Wff = np.stack([
                _far_field_steering_vector(xyz, float(cfg.adv_theta_deg), float(cfg.adv_phi_deg), cfg.adv_fc_hz)
                for _ in range(r.size)
            ], axis=0)
            baseline_db = per_point_gains(Wff, A, mode="db")

        if plotly_line_radial_slice is not None:
            fig = plotly_line_radial_slice(r, values_db, "Radial slice gain (dB)")
            _save_fig(fig, "adv_radial")
            if baseline_db is not None:
                if go is not None:
                    fig2 = go.Figure(fig)
                    fig2.add_scatter(x=r, y=baseline_db, mode='lines+markers', name='Far-field')
                    _save_fig(fig2, "adv_radial_baseline")
        else:
            log("Plotly not available; skipping interactive line plot.\n")

        if cfg.adv_save_csv:
            try:
                out_csv = os.path.join(os.getcwd(), f"adv_radial_{int(time.time())}.csv")
                with open(out_csv, "w", encoding="utf-8") as f:
                    header = "r_m,value_db"
                    if baseline_db is not None:
                        header += ",baseline_db"
                    f.write(header + "\n")
                    for i, rv in enumerate(r):
                        row = f"{float(rv):.6f},{float(values_db[i]):.6f}"
                        if baseline_db is not None:
                            row += f",{float(baseline_db[i]):.6f}"
                        f.write(row + "\n")
                files.append(out_csv)
                log(f"Saved CSV: {out_csv}\n")
            except Exception as e:
                log(f"CSV save error: {e}\n")

        set_summary("Advanced AAG/AMAG radial slice complete. See exported HTML/CSV for interactive view.")
        return {"files": files, "r": r, "values_db": values_db}

    else:
        raise ValueError("Unknown advanced map type")


# ============= Plot helpers =============
def plot_histograms(loss_q_db: np.ndarray, loss_ff_db_or_none: Optional[np.ndarray]) -> Any:
    import matplotlib.pyplot as plt  # type: ignore
    # Build consistent bins across both arrays
    bins = 30
    data = loss_q_db if loss_ff_db_or_none is None else np.concatenate([loss_q_db, loss_ff_db_or_none])
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0
    bins_edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    h1, _ = np.histogram(loss_q_db, bins=bins_edges)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
    l1, = ax.plot(centers, h1, drawstyle='steps-mid', label="Quantization loss")
    l2 = None
    if loss_ff_db_or_none is not None:
        h2, _ = np.histogram(loss_ff_db_or_none, bins=bins_edges)
        l2, = ax.plot(centers, h2, drawstyle='steps-mid', label="Far-field mismatch")
    ax.set_xlabel("Loss (dB)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc='best', fancybox=True, shadow=True)
    # Enable toggling series by clicking legend
    lines = [l for l in [l1, l2] if l is not None]
    leg_lines = leg.get_lines()
    # Some backends return more legend lines than plotted; slice
    leg_pickables = leg_lines[: len(lines)]
    for ll in leg_pickables:
        ll.set_picker(True)
        ll.set_pickradius(5)

    def on_pick(event):
        if event.artist in leg_pickables:
            idx = leg_pickables.index(event.artist)
            line = lines[idx]
            vis = not line.get_visible()
            line.set_visible(vis)
            event.artist.set_alpha(1.0 if vis else 0.2)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.tight_layout()
    return fig


def plot_gain_spectrum(freqs_hz: np.ndarray, gain_ps_db: np.ndarray, gain_ttd_db: np.ndarray) -> Any:
    import matplotlib.pyplot as plt  # type: ignore
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=120)
    f_ghz = freqs_hz / 1e9
    l1, = ax.plot(f_ghz, gain_ps_db, label="Phase Shifter (PS)")
    l2, = ax.plot(f_ghz, gain_ttd_db, label="True-Time-Delay (TTD)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Gain (dB)")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc='best', fancybox=True, shadow=True)
    # Toggle by clicking legend entries
    lines = [l1, l2]
    leg_lines = leg.get_lines()[:2]
    for ll in leg_lines:
        ll.set_picker(True)
        ll.set_pickradius(5)

    def on_pick(event):
        if event.artist in leg_lines:
            idx = leg_lines.index(event.artist)
            line = lines[idx]
            vis = not line.get_visible()
            line.set_visible(vis)
            event.artist.set_alpha(1.0 if vis else 0.2)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.tight_layout()
    return fig


class _ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        ttk.Label(tw, text=self.text, padding=(8, 4)).pack()

    def _hide(self, _e=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ScrollablePane(ttk.Frame):
    """Scrollable vertical pane using Canvas + ttk.Scrollbar.

    - Child content goes into self.body (a ttk.Frame)
    - update_scrollregion() recalculates scrollregion and toggles scrollbar state
    - Handles mouse wheel for Windows/macOS and Button-4/5 for Linux
    """
    def __init__(self, parent, width: int = 480, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")

        # Inner content frame
        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # Keep scrollregion and width in sync
        self.body.bind("<Configure>", lambda e: self._on_body_configure())
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))

        try:
            self.update_idletasks()
            self.canvas.configure(width=int(width))
        except Exception:
            pass

        # Wheel bindings
        self._bind_wheel()

    def update_scrollregion(self) -> None:
        try:
            self.update_idletasks()
            bbox = self.canvas.bbox("all")
            if not bbox:
                self.canvas.configure(scrollregion=(0, 0, 0, 0))
                self.vbar.state(["disabled"])  # no content
                return
            self.canvas.configure(scrollregion=bbox)
            content_h = max(0, bbox[3] - bbox[1])
            view_h = max(1, self.canvas.winfo_height())
            if content_h <= view_h:
                self.vbar.state(["disabled"])
                self.canvas.yview_moveto(0.0)
            else:
                self.vbar.state(["!disabled"])  # enable
        except Exception:
            pass

    def _on_body_configure(self) -> None:
        try:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.update_scrollregion()
        except Exception:
            pass

    def _bind_wheel(self) -> None:
        root = self.winfo_toplevel()

        def wheel_handler(event):
            try:
                # Only scroll when pointer is within our body
                w = root.winfo_containing(event.x_root, event.y_root)
                if not self._is_descendant_of_body(w):
                    return
                delta = 0
                if hasattr(event, 'num') and event.num in (4, 5):
                    delta = -1 if event.num == 4 else 1
                else:
                    d = int(getattr(event, 'delta', 0))
                    if d != 0:
                        delta = -1 if d > 0 else 1
                if delta:
                    self.canvas.yview_scroll(delta, "units")
                    return "break"
            except Exception:
                return

        root.bind_all("<MouseWheel>", wheel_handler, add=True)
        root.bind_all("<Button-4>", wheel_handler, add=True)
        root.bind_all("<Button-5>", wheel_handler, add=True)

    def _is_descendant_of_body(self, widget) -> bool:
        try:
            while widget is not None:
                if widget is self.body:
                    return True
                widget = widget.master
        except Exception:
            pass
        return False

class MainSimulationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Near-Field Simulator - Main Console")
        root.geometry("1280x820")
        root.minsize(1024, 640)

        self.running = False
        self._setup_styles()
        # Experiments controller
        self.exp_ctrl = ExperimentController(self)
        # A small flag to track if progress bar is in determinate mode
        self._progress_determinate = False

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Simulation Console", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Select tests and parameters, then Run", style="Subtitle.TLabel").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Separator(outer).pack(fill=tk.X, pady=(6, 8))

        # Main split
        paned = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: controls inside scrollable pane
        self._pane_left = ScrollablePane(paned, width=500)
        # Ensure the inner body expands horizontally
        try:
            self._pane_left.body.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
        controls = ttk.Labelframe(self._pane_left.body, text="Controls", padding=(12, 10))
        # Place the controls frame inside the scrollable body's grid
        controls.grid(row=0, column=0, sticky="nwe")
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)

        # Parameters
        ttk.Label(controls, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 4))
        self.preset_var = tk.StringVar(value="standard")
        self.preset_cb = ttk.Combobox(controls, textvariable=self.preset_var, values=VALID_PRESETS, state="readonly")
        self.preset_cb.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        ttk.Label(controls, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.mode_var = tk.StringVar(value="fast")
        self.mode_cb = ttk.Combobox(controls, textvariable=self.mode_var, values=VALID_MODES, state="readonly")
        self.mode_cb.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Users:").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.users_var = tk.IntVar(value=5)
        self.users_spin = ttk.Spinbox(controls, from_=1, to=1000, textvariable=self.users_var, width=10)
        self.users_spin.grid(row=2, column=1, sticky="w", pady=4)

        self.randomize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Randomize on Run", variable=self.randomize_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))

        # JSON profile selection
        ttk.Separator(controls).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        ttk.Label(controls, text="JSON Profile:").grid(row=5, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 4))
        self.use_json_var = tk.BooleanVar(value=False)
        self.cfg_manager = ConfigManager()
        self.json_profiles = sorted(list(self.cfg_manager.sim_configs.keys()))
        self.json_profile_var = tk.StringVar(value=self.json_profiles[0] if self.json_profiles else "")
        self.json_cb = ttk.Combobox(controls, textvariable=self.json_profile_var, values=self.json_profiles, state="readonly")
        self.json_cb.grid(row=5, column=1, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(controls, text="Use JSON profile", variable=self.use_json_var).grid(row=6, column=0, columnspan=2, sticky=tk.W)

        # Advanced overrides (legacy complete simulation)
        ttk.Separator(controls).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        self.adv_var = tk.BooleanVar(value=False)
        adv = ttk.Labelframe(controls, text="Advanced Overrides", padding=(8, 6))
        adv.grid(row=8, column=0, columnspan=2, sticky="ew")
        ttk.Checkbutton(adv, text="Enable Overrides", variable=self.adv_var).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        # z range & points
        ttk.Label(adv, text="z_min").grid(row=1, column=0, sticky=tk.W, padx=(0,4))
        ttk.Label(adv, text="z_max").grid(row=1, column=2, sticky=tk.W, padx=(8,4))
        self.zmin_var = tk.DoubleVar(value=1.0)
        self.zmax_var = tk.DoubleVar(value=200.0)
        ttk.Entry(adv, textvariable=self.zmin_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(adv, textvariable=self.zmax_var, width=8).grid(row=1, column=3, sticky=tk.W)
        ttk.Label(adv, text="z points").grid(row=2, column=0, sticky=tk.W, padx=(0,4))
        ttk.Label(adv, text="realizations per z").grid(row=2, column=2, sticky=tk.W, padx=(8,4))
        self.nz_var = tk.IntVar(value=30)
        self.nreal_var = tk.IntVar(value=100)
        ttk.Entry(adv, textvariable=self.nz_var, width=8).grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(adv, textvariable=self.nreal_var, width=8).grid(row=2, column=3, sticky=tk.W)
        # x/y ranges & n_jobs
        ttk.Label(adv, text="x_min").grid(row=3, column=0, sticky=tk.W, padx=(0,4))
        ttk.Label(adv, text="x_max").grid(row=3, column=2, sticky=tk.W, padx=(8,4))
        self.xmin_var = tk.DoubleVar(value=-100.0)
        self.xmax_var = tk.DoubleVar(value=100.0)
        ttk.Entry(adv, textvariable=self.xmin_var, width=8).grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(adv, textvariable=self.xmax_var, width=8).grid(row=3, column=3, sticky=tk.W)
        ttk.Label(adv, text="y_min").grid(row=4, column=0, sticky=tk.W, padx=(0,4))
        ttk.Label(adv, text="y_max").grid(row=4, column=2, sticky=tk.W, padx=(8,4))
        self.ymin_var = tk.DoubleVar(value=-100.0)
        self.ymax_var = tk.DoubleVar(value=100.0)
        ttk.Entry(adv, textvariable=self.ymin_var, width=8).grid(row=4, column=1, sticky=tk.W)
        ttk.Entry(adv, textvariable=self.ymax_var, width=8).grid(row=4, column=3, sticky=tk.W)
        ttk.Label(adv, text="n_jobs (-1=auto)").grid(row=5, column=0, sticky=tk.W, padx=(0,4))
        self.njobs_var = tk.IntVar(value=-1)
        ttk.Entry(adv, textvariable=self.njobs_var, width=8).grid(row=5, column=1, sticky=tk.W)

        # Experiments (advanced)
        ttk.Separator(controls).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        self._build_experiments_section(controls, row=10)

        # Tests selection
        tests_frame = ttk.Labelframe(controls, text="Select Tests", padding=(10, 8))
        tests_frame.grid(row=11, column=0, columnspan=2, sticky="nsew")
        tests_frame.grid_columnconfigure(0, weight=1)
        self.test_vars = {
            "Basic Functionality": tk.BooleanVar(value=False),
            "Parameter Analysis": tk.BooleanVar(value=False),
            "Complete Simulation (AAG/AMAG)": tk.BooleanVar(value=True),
            "Comparison Analysis": tk.BooleanVar(value=False),
            "Performance Benchmark": tk.BooleanVar(value=False),
            "GUI Smoke Test": tk.BooleanVar(value=False),
            "Random Quick Experiment": tk.BooleanVar(value=False),
        }
        r = 0
        for name, var in self.test_vars.items():
            ttk.Checkbutton(tests_frame, text=name, variable=var).grid(row=r, column=0, sticky=tk.W, pady=2)
            r += 1
        # Select all / none
        btn_row = ttk.Frame(tests_frame)
        btn_row.grid(row=r, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Clear", command=self._clear_all).pack(side=tk.LEFT, padx=(6, 0))

        # Run/Stop buttons
        action_row = ttk.Frame(controls)
        action_row.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=1)
        action_row.grid_columnconfigure(2, weight=1)
        self.run_btn = ttk.Button(action_row, text="Run", style="Accent.TButton", command=self.run_selected)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cancel_btn = ttk.Button(action_row, text="Cancel Run", command=lambda: self._cancel_run())
        self.cancel_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.cancel_btn.state(["disabled"])  # enabled during a run
        self.stop_btn = ttk.Button(action_row, text="Close", command=root.destroy)
        self.stop_btn.grid(row=0, column=2, sticky="ew")

        # Right: notebook (Activity / Plots / Summary) and status
        right = ttk.Frame(paned, padding=(4, 4))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Activity tab
        self.activity_tab = ttk.Frame(self.nb)
        self.nb.add(self.activity_tab, text="Activity")
        self.log = tk.Text(self.activity_tab, height=20, wrap="word")
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(self.activity_tab, orient="vertical", command=self.log.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=yscroll.set, state="disabled")

        # Plots tab (scrollable with both axes) using a proper grid layout so bars span the area
        self.plots_tab = ttk.Frame(self.nb)
        self.nb.add(self.plots_tab, text="Plots")
        self.plots_tab.grid_columnconfigure(0, weight=1)
        self.plots_tab.grid_rowconfigure(0, weight=1)
        self.plots_canvas = tk.Canvas(self.plots_tab, highlightthickness=0)
        self.plots_vscroll = ttk.Scrollbar(self.plots_tab, orient="vertical", command=self.plots_canvas.yview)
        self.plots_hscroll = ttk.Scrollbar(self.plots_tab, orient="horizontal", command=self.plots_canvas.xview)
        self.plots_canvas.configure(yscrollcommand=self.plots_vscroll.set, xscrollcommand=self.plots_hscroll.set)
        self.plots_canvas.grid(row=0, column=0, sticky="nsew")
        self.plots_vscroll.grid(row=0, column=1, sticky="ns")
        self.plots_hscroll.grid(row=1, column=0, sticky="ew")
        # Inner container attached to canvas (native figure sizes; scrolling handles overflow)
        self.plots_container = ttk.Frame(self.plots_canvas)
        self._plots_window = self.plots_canvas.create_window((0, 0), window=self.plots_container, anchor="nw")
        self.plots_container.bind(
            "<Configure>",
            lambda e: self.plots_canvas.configure(scrollregion=self.plots_canvas.bbox("all")),
        )
        self._fig_canvases = []
        self._plot_toolbar = None  # Single toolbar instance for Plots tab

        # Summary tab
        self.summary_tab = ttk.Frame(self.nb)
        self.nb.add(self.summary_tab, text="Summary")
        self.summary_text = tk.Text(self.summary_tab, wrap="word")
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sum_scroll = ttk.Scrollbar(self.summary_tab, orient="vertical", command=self.summary_text.yview)
        sum_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(yscrollcommand=sum_scroll.set, state="disabled")

        status = ttk.Frame(right)
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=200) 
        self.progress.pack(side=tk.LEFT) 
        self.status_var = tk.StringVar(value="Ready.") 
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.RIGHT) 

        # Add panes: make right expand; left has a minimum width
        paned.add(self._pane_left, weight=0)
        paned.add(right, weight=1)
        try:
            paned.pane(self._pane_left, minsize=420)
        except Exception:
            pass

        # Ensure left scroll region is up to date
        try:
            self._pane_left.update_scrollregion()
        except Exception:
            pass

        # Closing guard and protocol handler
        self._closing = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Queue to capture printed output from worker threads
        self._msg_queue: queue.Queue = queue.Queue()
        self._after_poll_id = self.root.after(100, self._poll_queue)

        # Tooltips
        for w, t in [
            (self.preset_cb, "Predefined system configuration"),
            (self.mode_cb, "Simulation depth/speed"),
            (self.users_spin, "Number of users (1-200)"),
            (self.run_btn, "Run selected tests with parameters"),
        ]:
            _ToolTip(w, t)

    # ============= Actions =============
    def _select_all(self) -> None:
        for v in self.test_vars.values():
            v.set(True)

    def _clear_all(self) -> None:
        for v in self.test_vars.values():
            v.set(False)

    def run_selected(self) -> None:
        if self.running:
            self._append_log("A run is already in progress.\n")
            return
        selected: List[str] = [k for k, v in self.test_vars.items() if v.get()]
        if not selected:
            # Maybe experiments are selected
            if not (self.exp_build_cb_var.get() or self.exp_eval_cb_var.get() or self.exp_wideband_cb_var.get()):
                self._append_log("No tests selected.\n")
                return

        preset = self.preset_var.get() or "standard"
        mode = self.mode_var.get() or "fast"
        users = int(self.users_var.get() or 5)

        if self.randomize_var.get():
            rp = random_basic_params()
            preset, mode, users = rp["preset"], rp["mode"], int(rp["users"])
            self.preset_var.set(preset)
            self.mode_var.set(mode)
            self.users_var.set(users)

        self._set_running(True)
        self._append_log(f"Starting run with preset={preset}, mode={mode}, users={users}\n")

        def worker():
            last_output = None
            try:
                # Redirect stdout/stderr to GUI log via queue
                prev_out, prev_err = sys.stdout, sys.stderr
                class _QueueWriter:
                    def __init__(self, put_func):
                        self._put = put_func
                    def write(self, s):
                        if s:
                            try:
                                self._put(("log_chunk", s))
                            except Exception:
                                pass
                    def flush(self):
                        pass
                sys.stdout = sys.stderr = _QueueWriter(self._msg_queue.put)
                # Prevent matplotlib windows from blocking the Tk GUI
                try:
                    import matplotlib.pyplot as plt  # noqa: WPS433
                    _orig_show = plt.show
                    plt.show = lambda *a, **k: None
                except Exception:
                    _orig_show = None
                # Cancellation event shared with simulation processes
                self._stop_event = mp.Event()
                # Clear experiments cancel flag for this run
                try:
                    self.exp_ctrl.cancel_flag.clear()
                except Exception:
                    pass
                # Build experiments selection
                exp_selected = []
                if self.exp_build_cb_var.get():
                    exp_selected.append("Build Spherical Codebook")
                if self.exp_eval_cb_var.get():
                    exp_selected.append("Evaluate Codebook Mismatch")
                if self.exp_wideband_cb_var.get():
                    exp_selected.append("Wideband: TTD vs Phase")
                if self.exp_adv_maps_cb_var.get():
                    exp_selected.append("Advanced AAG/AMAG Maps")

                # If Complete Simulation selected, run original pipeline unchanged
                run_only_selected_tests = ("Complete Simulation (AAG/AMAG)" in selected)

                for name in (selected if run_only_selected_tests else (selected + exp_selected)):
                    t0 = time.perf_counter()
                    if self._closing:
                        break
                    self._schedule_status(f"Running: {name}")
                    self._append_log(f"\n=== {name} ===\n")
                    if name == "Basic Functionality":
                        demo_basic_functionality()
                    elif name == "Parameter Analysis":
                        demo_parameter_analysis()
                    
                    elif name == "Complete Simulation (AAG/AMAG)":
                        # Build config from JSON profile or mode, then apply overrides if requested
                        sim = create_system_with_presets(preset)
                        profile = None
                        if self.use_json_var.get() and self.json_profiles:
                            profile = self.json_profile_var.get()
                            self._append_log(f"Using JSON profile: {profile}\n")
                            cfg = self.cfg_manager.get_simulation_config(profile)
                        else:
                            mode_sel = self.mode_var.get() or "comprehensive"
                            self._append_log(f"Running complete simulation with mode={mode_sel}\n")
                            cfg = create_simulation_config(mode_sel)
                        # Override users
                        if users is not None:
                            cfg.num_users_list = [users]
                        # Apply advanced overrides
                        if self.adv_var.get():
                            try:
                                zmin = float(self.zmin_var.get())
                                zmax = float(self.zmax_var.get())
                                nz = max(2, int(self.nz_var.get()))
                                if zmax <= zmin:
                                    zmax = zmin + 1.0
                                cfg.z_values = np.linspace(zmin, zmax, nz)
                                cfg.num_realizations = max(1, int(self.nreal_var.get()))
                                cfg.x_range = (float(self.xmin_var.get()), float(self.xmax_var.get()))
                                cfg.y_range = (float(self.ymin_var.get()), float(self.ymax_var.get()))
                                cfg.n_jobs = int(self.njobs_var.get())
                            except Exception as e:
                                self._append_log(f"Override parse error: {e}\n")
                        res = sim.run_optimized_simulation(cfg)
                        # Save results bundle (figures + pickle + config json)
                        try:
                            out_dir = f"my_results/full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            os.makedirs(out_dir, exist_ok=True)
                            sim.plot_comprehensive_results(res, save_dir=out_dir)
                            sim.save_results(res, f"{out_dir}/simulation_results.pkl")
                            cfg_json = {
                                "preset": preset,
                                "source": ("json_profile" if profile else "mode"),
                                "profile_or_mode": profile or (self.mode_var.get() or "comprehensive"),
                                "num_users_list": list(getattr(cfg, 'num_users_list', [])),
                                "z_values": list(map(float, getattr(cfg, 'z_values', []))),
                                "num_realizations": int(getattr(cfg, 'num_realizations', 0)),
                                "x_range": tuple(getattr(cfg, 'x_range', ())),
                                "y_range": tuple(getattr(cfg, 'y_range', ())),
                                "n_jobs": int(getattr(cfg, 'n_jobs', -1)),
                            }
                            with open(f"{out_dir}/config.json", "w", encoding="utf-8") as f:
                                json.dump(cfg_json, f, indent=2)
                            last_output = out_dir
                            self._append_log(f"Results saved to: {out_dir}\n")
                        except Exception as e:
                            self._append_log(f"Save error: {e}\n")
                        # Show results and plots in tabs
                        try:
                            if not self._closing:
                                self.root.after(0, lambda p=preset, r=res: self._update_results_view(p, r, None))
                        except Exception:
                            pass
                    elif name == "Build Spherical Codebook":
                        # Experiments controller path
                        try:
                            cfg = self.exp_ctrl.read_ui()
                            # Wire callbacks
                            res = run_build_spherical_codebook(cfg)
                            # Cache codebook for subsequent steps
                            self.exp_ctrl.set_last_codebook(res)
                        except Exception as e:
                            tb = traceback.format_exc()
                            self._append_log(f"Experiment error: {e}\n{tb}\n")
                    elif name == "Evaluate Codebook Mismatch":
                        try:
                            cfg = self.exp_ctrl.read_ui()
                            # Use in-memory if available
                            cb = self.exp_ctrl.get_last_codebook()
                            if cb is None and not cfg.in_codebook_path:
                                messagebox.showerror("Codebook required", "No in-memory codebook found and no HDF5 path provided.")
                            else:
                                _res = run_eval_codebook_mismatch(cfg, codebook_res=cb)
                        except Exception as e:
                            tb = traceback.format_exc()
                            self._append_log(f"Experiment error: {e}\n{tb}\n")
                    elif name == "Wideband: TTD vs Phase":
                        try:
                            cfg = self.exp_ctrl.read_ui()
                            _res = run_wideband_compare_ttd_vs_phase(cfg)
                        except Exception as e:
                            tb = traceback.format_exc()
                            self._append_log(f"Experiment error: {e}\n{tb}\n")
                    elif name == "Advanced AAG/AMAG Maps":
                        try:
                            cfg = self.exp_ctrl.read_ui()
                            _res = run_advanced_aag_amg(cfg, self.exp_ctrl.get_last_codebook())
                        except Exception as e:
                            tb = traceback.format_exc()
                            self._append_log(f"Experiment error: {e}\n{tb}\n")
                    elif name == "Comparison Analysis":
                        demo_comparison_analysis()
                    elif name == "Performance Benchmark":
                        demo_performance_benchmark()
                    elif name == "GUI Smoke Test":
                        demo_gui_error_check()
                    elif name == "Random Quick Experiment":
                        _res, rp = run_random_quick_experiment()
                        if self._closing:
                            break
                        self._append_log(f"Random params: {rp}\n")
                    dt = time.perf_counter() - t0
                    if self._closing:
                        break
                    self._append_log(f"{name} completed in {dt:.2f}s\n")
                if last_output:
                    if not self._closing:
                        self._append_log(f"\nLatest output directory: {last_output}\n")
            except Exception as e:
                if not self._closing:
                    self._append_log(f"Error: {e}\n")
            finally:
                # Restore stdout/stderr
                try:
                    sys.stdout = prev_out
                    sys.stderr = prev_err
                except Exception:
                    pass
                # Restore matplotlib show if we patched it
                try:
                    if _orig_show is not None:
                        import matplotlib.pyplot as plt  # noqa: WPS433
                        plt.show = _orig_show
                except Exception:
                    pass
                if not self._closing:
                    self._schedule_finish()

        threading.Thread(target=worker, daemon=True).start()

    # ============= Helpers =============
    def _append_log(self, text: str) -> None:
        def _do():
            self.log.configure(state="normal")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")
        if not self._closing:
            self.root.after(0, _do)

    def _set_running(self, running: bool) -> None:
        self.running = running
        def _do():
            try:
                if running:
                    if self._progress_determinate:
                        # Reset to indeterminate for a new run
                        self.progress.configure(mode="indeterminate")
                        self._progress_determinate = False
                    self.progress.start(10)
                    self.status_var.set("Running...")
                    self.run_btn.state(["disabled"]) 
                    self.cancel_btn.state(["!disabled"]) 
                else:
                    self.progress.stop()
                    self.status_var.set("Ready.")
                    self.run_btn.state(["!disabled"]) 
                    self.cancel_btn.state(["disabled"]) 
            except Exception:
                pass
        self.root.after(0, _do)

    def _schedule_finish(self) -> None:
        if not self._closing:
            self.root.after(0, lambda: self._set_running(False))

    def _schedule_status(self, text: str) -> None:
        if not self._closing:
            self.root.after(0, lambda: self.status_var.set(text))

    def _setup_styles(self) -> None:
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        PRIMARY = "#1f77b4"
        SUB = "#5a5a5a"
        s.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=PRIMARY)
        s.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground=SUB)
        s.configure("Accent.TButton", padding=6, foreground="#fff", background=PRIMARY)
        s.map("Accent.TButton", background=[("active", "#16629a"), ("disabled", "#9bbbd3")])

    # ============= Results rendering =============
    def _update_results_view(self, preset: str, results, out_dir: str | None) -> None:
        """Render figures and text summary in the Plots and Summary tabs."""
        # Recreate scrollable area if it was destroyed for any reason
        try:
            exists = self.plots_container.winfo_exists()
        except Exception:
            exists = False
        if not exists:
            # Rebuild the scrollable plots area
            for w in list(self.plots_tab.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass
            self.plots_canvas = tk.Canvas(self.plots_tab, highlightthickness=0)
            self.plots_scroll = ttk.Scrollbar(self.plots_tab, orient="vertical", command=self.plots_canvas.yview)
            self.plots_canvas.configure(yscrollcommand=self.plots_scroll.set)
            self.plots_container = ttk.Frame(self.plots_canvas)
            self.plots_container.bind(
                "<Configure>",
                lambda e: self.plots_canvas.configure(scrollregion=self.plots_canvas.bbox("all")),
            )
            self._plots_window = self.plots_canvas.create_window((0, 0), window=self.plots_container, anchor="nw")
            self.plots_canvas.bind(
                "<Configure>",
                lambda e: self.plots_canvas.itemconfigure(self._plots_window, width=e.width),
            )
            self.plots_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.plots_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            # Reset toolbar handle on rebuild
            self._plot_toolbar = None
        # Clear previous canvases (only children within container, not the scroll widgets)
        for w in list(self.plots_container.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._fig_canvases.clear()
        # Ensure any old toolbar is dropped
        try:
            if self._plot_toolbar is not None:
                self._plot_toolbar.destroy()
                self._plot_toolbar = None
        except Exception:
            self._plot_toolbar = None

        # Build figures using a simulator with same preset
        try:
            import matplotlib.pyplot as plt  # type: ignore
            orig_show = plt.show
            plt.show = lambda *a, **k: None
        except Exception:
            orig_show = None
        figs = []
        try:
            sim = create_system_with_presets(preset)
            figs = sim.plot_comprehensive_results(results, save_dir=None)
        except Exception as e:
            self._append_log(f"Plot error: {e}\n")
        finally:
            try:
                if orig_show is not None:
                    import matplotlib.pyplot as plt  # type: ignore
                    plt.show = orig_show
            except Exception:
                pass

        first_canvas = None
        for i, fig in enumerate(figs):
            canvas = FigureCanvasTkAgg(fig, master=self.plots_container)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(side=tk.TOP, anchor='nw', pady=(2, 0))
            # Keep the first canvas as the toolbar target
            if i == 0:
                first_canvas = canvas
            # Subtle spacing between figures
            if i < len(figs) - 1:
                try:
                    sep = ttk.Separator(self.plots_container, orient='horizontal')
                    sep.pack(fill='x', pady=(2, 6))
                except Exception:
                    pass
            self._fig_canvases.append(canvas)
        # Create a single toolbar (bound to first canvas) if figures exist
        try:
            if first_canvas is not None:
                self._plot_toolbar = NavigationToolbar2Tk(first_canvas, self.plots_container)
                self._plot_toolbar.update()
        except Exception:
            pass

        # Update textual summary
        self._set_summary_text(self._build_summary_text(results))
        # Switch to plots tab to make it visible
        try:
            if not self._closing:
                self.nb.select(self.plots_tab)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._closing = True
        try:
            if hasattr(self, '_stop_event') and self._stop_event is not None:
                self._stop_event.set()
        except Exception:
            pass
        try:
            if hasattr(self, '_after_poll_id') and self._after_poll_id is not None:
                self.root.after_cancel(self._after_poll_id)
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # Poll queued log text produced by worker thread stdout/stderr
    def _poll_queue(self) -> None:
        if self._closing:
            return
        try:
            while True:
                item = self._msg_queue.get_nowait()
                kind = item[0]
                if kind == "log_chunk":
                    _, chunk = item
                    self._append_log(chunk)
        except queue.Empty:
            pass
        if not self._closing:
            try:
                if self.root.winfo_exists():
                    self._after_poll_id = self.root.after(100, self._poll_queue)
            except Exception:
                pass

    def _cancel_run(self) -> None:
        try:
            if hasattr(self, '_stop_event') and self._stop_event is not None:
                self._stop_event.set()
                self._append_log("Cancelling run...\n")
                self._schedule_status("Cancelling...")
            # Experiments cancel flag
            if hasattr(self, 'exp_ctrl') and self.exp_ctrl is not None:
                self.exp_ctrl.cancel_flag.set()
        except Exception:
            pass

    def _build_summary_text(self, simulation_results) -> str:
        import numpy as np
        lines = []
        try:
            all_results = simulation_results.get('all_results', {})
            lines.append("Simulation Summary\n")
            for user_key, user_data in all_results.items():
                num_users = user_data.get('num_users')
                z_values = user_data.get('z_values')
                num_realizations = user_data.get('num_realizations')
                results = user_data.get('results', {})
                method_names = user_data.get('method_names', [])
                lines.append(f"- Scenario: {num_users} users, z points={len(z_values)}, realizations/z={num_realizations}")
                best_method = None
                best_aag = -1
                for m in method_names:
                    aag_vals = results.get(m, {}).get('aag', [])
                    mag_vals = results.get(m, {}).get('mag', [])
                    aag_mean = float(np.mean(aag_vals)) if len(aag_vals) else 0.0
                    amag_mean = float(np.mean(mag_vals)) if len(mag_vals) else 0.0
                    lines.append(f"  • {m}: AAG={aag_mean:.2f}, AMAG={amag_mean:.2f}")
                    if aag_mean > best_aag:
                        best_aag = aag_mean
                        best_method = m
                if best_method:
                    lines.append(f"  → Best by AAG: {best_method} ({best_aag:.2f})")
                lines.append("")
        except Exception as e:
            lines.append(f"(Summary unavailable: {e})")
        return "\n".join(lines)

    def _set_summary_text(self, text: str) -> None:
        try:
            self.summary_text.configure(state="normal")
            self.summary_text.delete("1.0", tk.END)
            if text:
                self.summary_text.insert(tk.END, text)
            self.summary_text.configure(state="disabled")
        except Exception:
            pass

    # ============= Plot helpers for experiments =============
    def _clear_plots(self) -> None:
        try:
            for w in list(self.plots_container.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass
            self._fig_canvases.clear()
            # Remove the single toolbar if present
            try:
                if self._plot_toolbar is not None:
                    self._plot_toolbar.destroy()
            except Exception:
                pass
            self._plot_toolbar = None
        except Exception:
            pass

    def _add_figure(self, fig) -> None:
        try:
            canvas = FigureCanvasTkAgg(fig, master=self.plots_container)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(side=tk.TOP, anchor='nw', pady=(2, 0))
            # Maintain a single toolbar: replace existing toolbar to bind to latest canvas
            try:
                if self._plot_toolbar is not None:
                    self._plot_toolbar.destroy()
                self._plot_toolbar = NavigationToolbar2Tk(canvas, self.plots_container)
                self._plot_toolbar.update()
            except Exception:
                pass
            # Subtle spacing after figure
            try:
                sep = ttk.Separator(self.plots_container, orient='horizontal')
                sep.pack(fill='x', pady=(2, 6))
            except Exception:
                pass
            self._fig_canvases.append(canvas)
            try:
                if not self._closing:
                    self.nb.select(self.plots_tab)
            except Exception:
                pass
        except Exception:
            pass

    def _set_progress_fraction(self, frac: Optional[float] = None) -> None:
        """Set determinate fraction [0,1]; if None revert to indeterminate."""
        try:
            if frac is None:
                if self._progress_determinate:
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(10)
                    self._progress_determinate = False
                return
            frac = max(0.0, min(1.0, float(frac)))
            if not self._progress_determinate:
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=100.0)
                self._progress_determinate = True
            self.progress['value'] = 100.0 * frac
        except Exception:
            pass

    # ============= Experiments UI =============
    def _build_experiments_section(self, parent: ttk.Frame, row: int) -> None:
        wrapper = ttk.Labelframe(parent, text="Experiments (advanced)", padding=(10, 8))
        wrapper.grid(row=row, column=0, columnspan=2, sticky="nsew")
        wrapper.grid_columnconfigure(0, weight=1)

        # Primary checkboxes
        cb_row = ttk.Frame(wrapper)
        cb_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.exp_build_cb_var = tk.BooleanVar(value=False)
        self.exp_eval_cb_var = tk.BooleanVar(value=False)
        self.exp_wideband_cb_var = tk.BooleanVar(value=False)
        self.exp_adv_maps_cb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cb_row, text="Build Spherical Codebook", variable=self.exp_build_cb_var).pack(side=tk.LEFT)
        ttk.Checkbutton(cb_row, text="Evaluate Codebook Mismatch", variable=self.exp_eval_cb_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(cb_row, text="Wideband: TTD vs Phase", variable=self.exp_wideband_cb_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(cb_row, text="Advanced AAG/AMAG Maps", variable=self.exp_adv_maps_cb_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(cb_row, text="Auto-fit", width=8, command=self._auto_fit_experiments_sections).pack(side=tk.RIGHT)

        # Collapsible panels helper
        def make_fold(parent_frame: ttk.Frame, title: str, r_index: int):
            header = ttk.Frame(parent_frame)
            header.grid(row=r_index, column=0, sticky="ew")
            header.grid_columnconfigure(0, weight=1)
            var = tk.BooleanVar(value=True)
            btn = ttk.Button(header, text="▼ " + title, width=32)
            btn.grid(row=0, column=0, sticky=tk.W)
            body = ttk.Frame(parent_frame)
            body.grid(row=r_index + 1, column=0, sticky="ew", pady=(4, 8))
            def toggle():
                if var.get():
                    var.set(False)
                    btn.configure(text="► " + title)
                    body.grid_remove()
                else:
                    var.set(True)
                    btn.configure(text="▼ " + title)
                    body.grid()
            btn.configure(command=toggle)
            return body

        r = 1
        # 1) Spherical Codebook panel
        sc_body = make_fold(wrapper, "Spherical Codebook", r)
        try:
            sc_body.bind("<Map>", lambda e: self._after_layout_refresh())
            sc_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2
        # Array layout/size
        ttk.Label(sc_body, text="Array layout").grid(row=0, column=0, sticky=tk.W)
        self.sc_layout_var = tk.StringVar(value="upa")
        ttk.Combobox(sc_body, textvariable=self.sc_layout_var, values=["upa", "ula"], width=8, state="readonly").grid(row=0, column=1, sticky=tk.W)
        ttk.Label(sc_body, text="num_x").grid(row=0, column=2, sticky=tk.W, padx=(6, 0))
        self.sc_numx_var = tk.IntVar(value=8)
        ttk.Entry(sc_body, textvariable=self.sc_numx_var, width=6).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(sc_body, text="num_y").grid(row=0, column=4, sticky=tk.W, padx=(6, 0))
        self.sc_numy_var = tk.IntVar(value=8)
        ttk.Entry(sc_body, textvariable=self.sc_numy_var, width=6).grid(row=0, column=5, sticky=tk.W)
        ttk.Label(sc_body, text="dx (m)").grid(row=1, column=0, sticky=tk.W)
        self.sc_dx_var = tk.DoubleVar(value=0.005)
        ttk.Entry(sc_body, textvariable=self.sc_dx_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(sc_body, text="dy (m)").grid(row=1, column=2, sticky=tk.W, padx=(6, 0))
        self.sc_dy_var = tk.DoubleVar(value=0.005)
        ttk.Entry(sc_body, textvariable=self.sc_dy_var, width=8).grid(row=1, column=3, sticky=tk.W)
        # Frequencies and grid
        ttk.Label(sc_body, text="fc_hz").grid(row=2, column=0, sticky=tk.W)
        self.sc_fc_var = tk.DoubleVar(value=28e9)
        ttk.Entry(sc_body, textvariable=self.sc_fc_var, width=14).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(sc_body, text="r_min").grid(row=3, column=0, sticky=tk.W)
        self.sc_rmin_var = tk.DoubleVar(value=1.0)
        ttk.Entry(sc_body, textvariable=self.sc_rmin_var, width=10).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(sc_body, text="r_max").grid(row=3, column=2, sticky=tk.W, padx=(6, 0))
        self.sc_rmax_var = tk.DoubleVar(value=5.0)
        ttk.Entry(sc_body, textvariable=self.sc_rmax_var, width=10).grid(row=3, column=3, sticky=tk.W)
        ttk.Label(sc_body, text="r_step").grid(row=3, column=4, sticky=tk.W, padx=(6, 0))
        self.sc_rstep_var = tk.DoubleVar(value=0.5)
        ttk.Entry(sc_body, textvariable=self.sc_rstep_var, width=10).grid(row=3, column=5, sticky=tk.W)
        ttk.Label(sc_body, text="theta start/stop/step (deg)").grid(row=4, column=0, sticky=tk.W, columnspan=2)
        self.sc_th_start_var = tk.DoubleVar(value=-60.0)
        self.sc_th_stop_var = tk.DoubleVar(value=60.0)
        self.sc_th_step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(sc_body, textvariable=self.sc_th_start_var, width=8).grid(row=4, column=2, sticky=tk.W)
        ttk.Entry(sc_body, textvariable=self.sc_th_stop_var, width=8).grid(row=4, column=3, sticky=tk.W)
        ttk.Entry(sc_body, textvariable=self.sc_th_step_var, width=8).grid(row=4, column=5, sticky=tk.W)
        ttk.Label(sc_body, text="phi start/stop/step (deg)").grid(row=5, column=0, sticky=tk.W, columnspan=2)
        self.sc_ph_start_var = tk.DoubleVar(value=-20.0)
        self.sc_ph_stop_var = tk.DoubleVar(value=20.0)
        self.sc_ph_step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(sc_body, textvariable=self.sc_ph_start_var, width=8).grid(row=5, column=2, sticky=tk.W)
        ttk.Entry(sc_body, textvariable=self.sc_ph_stop_var, width=8).grid(row=5, column=3, sticky=tk.W)
        ttk.Entry(sc_body, textvariable=self.sc_ph_step_var, width=8).grid(row=5, column=5, sticky=tk.W)
        ttk.Label(sc_body, text="chunk").grid(row=6, column=0, sticky=tk.W)
        self.sc_chunk_var = tk.IntVar(value=2048)
        ttk.Entry(sc_body, textvariable=self.sc_chunk_var, width=10).grid(row=6, column=1, sticky=tk.W)
        # Output path
        ttk.Label(sc_body, text="Output HDF5").grid(row=7, column=0, sticky=tk.W)
        self.sc_out_path_var = tk.StringVar(value="")
        ttk.Entry(sc_body, textvariable=self.sc_out_path_var, width=32).grid(row=7, column=1, columnspan=3, sticky="ew")
        ttk.Button(sc_body, text="Browse...", command=lambda: self._choose_file(self.sc_out_path_var, save=True, defname="codebook.h5", filetypes=[("HDF5","*.h5")])).grid(row=7, column=4, sticky=tk.W)

        # 2) Evaluate Codebook Mismatch
        ev_body = make_fold(wrapper, "Evaluate Codebook Mismatch", r)
        try:
            ev_body.bind("<Map>", lambda e: self._after_layout_refresh())
            ev_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2
        ttk.Label(ev_body, text="Q").grid(row=0, column=0, sticky=tk.W)
        self.ev_Q_var = tk.IntVar(value=200)
        ttk.Entry(ev_body, textvariable=self.ev_Q_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(ev_body, text="x_min/x_max").grid(row=1, column=0, sticky=tk.W)
        self.ev_xmin_var = tk.DoubleVar(value=-2.0)
        self.ev_xmax_var = tk.DoubleVar(value=2.0)
        ttk.Entry(ev_body, textvariable=self.ev_xmin_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(ev_body, textvariable=self.ev_xmax_var, width=8).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(ev_body, text="y_min/y_max").grid(row=2, column=0, sticky=tk.W)
        self.ev_ymin_var = tk.DoubleVar(value=-2.0)
        self.ev_ymax_var = tk.DoubleVar(value=2.0)
        ttk.Entry(ev_body, textvariable=self.ev_ymin_var, width=8).grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(ev_body, textvariable=self.ev_ymax_var, width=8).grid(row=2, column=2, sticky=tk.W)
        ttk.Label(ev_body, text="z_min/z_max").grid(row=3, column=0, sticky=tk.W)
        self.ev_zmin_var = tk.DoubleVar(value=1.0)
        self.ev_zmax_var = tk.DoubleVar(value=5.0)
        ttk.Entry(ev_body, textvariable=self.ev_zmin_var, width=8).grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(ev_body, textvariable=self.ev_zmax_var, width=8).grid(row=3, column=2, sticky=tk.W)
        self.ev_compare_ff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ev_body, text="Compare with plane-wave codebook (FF)", variable=self.ev_compare_ff_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
        ttk.Label(ev_body, text="Input codebook HDF5").grid(row=5, column=0, sticky=tk.W)
        self.ev_in_path_var = tk.StringVar(value="")
        ttk.Entry(ev_body, textvariable=self.ev_in_path_var, width=32).grid(row=5, column=1, columnspan=2, sticky="ew")
        ttk.Button(ev_body, text="Browse...", command=lambda: self._choose_file(self.ev_in_path_var, save=False, filetypes=[("HDF5","*.h5")])).grid(row=5, column=3, sticky=tk.W)
        self.ev_save_csv_var = tk.BooleanVar(value=False)
        self.ev_save_png_var = tk.BooleanVar(value=False)
        self.ev_save_html_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ev_body, text="Save CSV", variable=self.ev_save_csv_var).grid(row=6, column=0, sticky=tk.W)
        ttk.Checkbutton(ev_body, text="Save PNG", variable=self.ev_save_png_var).grid(row=6, column=1, sticky=tk.W)
        ttk.Checkbutton(ev_body, text="Save HTML (interactive)", variable=self.ev_save_html_var).grid(row=6, column=2, sticky=tk.W)

        # 3) Wideband: TTD vs Phase
        wb_body = make_fold(wrapper, "Wideband: TTD vs Phase", r)
        try:
            wb_body.bind("<Map>", lambda e: self._after_layout_refresh())
            wb_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2
        ttk.Label(wb_body, text="fc_hz").grid(row=0, column=0, sticky=tk.W)
        self.wb_fc_var = tk.DoubleVar(value=28e9)
        ttk.Entry(wb_body, textvariable=self.wb_fc_var, width=14).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(wb_body, text="bw_hz").grid(row=0, column=2, sticky=tk.W)
        self.wb_bw_var = tk.DoubleVar(value=1e9)
        ttk.Entry(wb_body, textvariable=self.wb_bw_var, width=14).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(wb_body, text="n_sc").grid(row=0, column=4, sticky=tk.W)
        self.wb_nsc_var = tk.IntVar(value=64)
        ttk.Entry(wb_body, textvariable=self.wb_nsc_var, width=8).grid(row=0, column=5, sticky=tk.W)
        # Focus point r,theta,phi
        ttk.Label(wb_body, text="Focus r, theta, phi (deg)").grid(row=1, column=0, sticky=tk.W, columnspan=2)
        self.wb_r_var = tk.DoubleVar(value=2.0)
        self.wb_theta_var = tk.DoubleVar(value=0.0)
        self.wb_phi_var = tk.DoubleVar(value=0.0)
        ttk.Entry(wb_body, textvariable=self.wb_r_var, width=10).grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(wb_body, textvariable=self.wb_theta_var, width=10).grid(row=1, column=3, sticky=tk.W)
        ttk.Entry(wb_body, textvariable=self.wb_phi_var, width=10).grid(row=1, column=5, sticky=tk.W)
        self._wb_xyz_label = ttk.Label(wb_body, text="XYZ: (—)")
        self._wb_xyz_label.grid(row=2, column=0, columnspan=4, sticky=tk.W)
        ttk.Button(wb_body, text="Use r,θ,ϕ→XYZ", command=self._update_wb_xyz_preview).grid(row=1, column=6, sticky=tk.W, padx=(6, 0))
        # Array layout same as spherical
        ttk.Label(wb_body, text="Array layout").grid(row=3, column=0, sticky=tk.W)
        # Use same variables sc_layout_var/numx/numy/dx/dy for simplicity
        ttk.Label(wb_body, textvariable=self.sc_layout_var).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(wb_body, text="num_x/num_y").grid(row=3, column=2, sticky=tk.W)
        ttk.Label(wb_body, textvariable=self.sc_numx_var).grid(row=3, column=3, sticky=tk.W)
        ttk.Label(wb_body, textvariable=self.sc_numy_var).grid(row=3, column=4, sticky=tk.W)
        ttk.Label(wb_body, text="dx/dy (m)").grid(row=3, column=5, sticky=tk.W)
        ttk.Label(wb_body, textvariable=self.sc_dx_var).grid(row=3, column=6, sticky=tk.W)
        ttk.Label(wb_body, textvariable=self.sc_dy_var).grid(row=3, column=7, sticky=tk.W)
        # Output options
        self.wb_save_png_var = tk.BooleanVar(value=False)
        self.wb_save_json_var = tk.BooleanVar(value=False)
        self.wb_save_html_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wb_body, text="Save plots (PNG)", variable=self.wb_save_png_var).grid(row=4, column=0, sticky=tk.W)
        ttk.Checkbutton(wb_body, text="Save summary (JSON)", variable=self.wb_save_json_var).grid(row=4, column=1, sticky=tk.W)
        ttk.Checkbutton(wb_body, text="Save HTML (interactive)", variable=self.wb_save_html_var).grid(row=4, column=2, sticky=tk.W)

        # Optimizer (optional) panel
        opt_body = make_fold(wrapper, "Optimizer (optional)", r)
        try:
            opt_body.bind("<Map>", lambda e: self._after_layout_refresh())
            opt_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2
        self.opt_use_gwo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_body, text="Use GWO optimizer", variable=self.opt_use_gwo_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(opt_body, text="n_agents").grid(row=1, column=0, sticky=tk.W)
        self.opt_agents_var = tk.IntVar(value=30)
        ttk.Entry(opt_body, textvariable=self.opt_agents_var, width=10).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(opt_body, text="n_iter").grid(row=1, column=2, sticky=tk.W, padx=(8, 0))
        self.opt_iters_var = tk.IntVar(value=200)
        ttk.Entry(opt_body, textvariable=self.opt_iters_var, width=10).grid(row=1, column=3, sticky=tk.W)
        ttk.Label(opt_body, text="seed").grid(row=2, column=0, sticky=tk.W)
        self.opt_seed_var = tk.StringVar(value="")
        ttk.Entry(opt_body, textvariable=self.opt_seed_var, width=10).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(opt_body, text="patience").grid(row=2, column=2, sticky=tk.W, padx=(8, 0))
        self.opt_patience_var = tk.StringVar(value="")
        ttk.Entry(opt_body, textvariable=self.opt_patience_var, width=10).grid(row=2, column=3, sticky=tk.W)
        ttk.Label(opt_body, text="target").grid(row=3, column=0, sticky=tk.W)
        self.opt_target_var = tk.StringVar(value="Max Gain @ focus")
        ttk.Combobox(opt_body, textvariable=self.opt_target_var, values=["Max Gain @ focus", "Min Gain Flatness", "Custom (advanced)"], state="readonly").grid(row=3, column=1, columnspan=3, sticky="ew")

        # Advanced overrides for experiments
        adv_body = make_fold(wrapper, "Advanced overrides", r)
        try:
            adv_body.bind("<Map>", lambda e: self._after_layout_refresh())
            adv_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2
        ttk.Label(adv_body, text="n_jobs").grid(row=0, column=0, sticky=tk.W)
        self.exp_njobs_var = tk.IntVar(value=-1)
        ttk.Entry(adv_body, textvariable=self.exp_njobs_var, width=10).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(adv_body, text="seed").grid(row=0, column=2, sticky=tk.W)
        self.exp_seed_var = tk.StringVar(value="")
        ttk.Entry(adv_body, textvariable=self.exp_seed_var, width=10).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(adv_body, text="log level").grid(row=0, column=4, sticky=tk.W)
        self.exp_loglevel_var = tk.StringVar(value="INFO")
        ttk.Combobox(adv_body, textvariable=self.exp_loglevel_var, values=["DEBUG","INFO","WARNING","ERROR"], width=10, state="readonly").grid(row=0, column=5, sticky=tk.W)

        # Advanced AAG/AMAG maps panel
        maps_body = make_fold(wrapper, "Advanced AAG/AMAG Maps", r)
        try:
            maps_body.bind("<Map>", lambda e: self._after_layout_refresh())
            maps_body.bind("<Unmap>", lambda e: self._after_layout_refresh())
        except Exception:
            pass
        r += 2

        # Registry for Auto-fit
        try:
            self._fold_sections = {
                "Spherical Codebook": sc_body,
                "Evaluate Codebook Mismatch": ev_body,
                "Wideband: TTD vs Phase": wb_body,
                "Optimizer (optional)": opt_body,
                "Advanced overrides": adv_body,
                "Advanced AAG/AMAG Maps": maps_body,
            }
        except Exception:
            pass
        # Array
        ttk.Label(maps_body, text="Array").grid(row=0, column=0, sticky=tk.W)
        self.adv_layout_var = tk.StringVar(value="upa")
        ttk.Combobox(maps_body, textvariable=self.adv_layout_var, values=["upa","ula"], width=8, state="readonly").grid(row=0, column=1, sticky=tk.W)
        ttk.Label(maps_body, text="num_x").grid(row=0, column=2, sticky=tk.W)
        self.adv_numx_var = tk.IntVar(value=16)
        ttk.Entry(maps_body, textvariable=self.adv_numx_var, width=6).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(maps_body, text="num_y").grid(row=0, column=4, sticky=tk.W)
        self.adv_numy_var = tk.IntVar(value=16)
        ttk.Entry(maps_body, textvariable=self.adv_numy_var, width=6).grid(row=0, column=5, sticky=tk.W)
        ttk.Label(maps_body, text="dx (m)").grid(row=1, column=0, sticky=tk.W)
        self.adv_dx_var = tk.DoubleVar(value=0.005)
        ttk.Entry(maps_body, textvariable=self.adv_dx_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(maps_body, text="dy (m)").grid(row=1, column=2, sticky=tk.W)
        self.adv_dy_var = tk.DoubleVar(value=0.005)
        ttk.Entry(maps_body, textvariable=self.adv_dy_var, width=8).grid(row=1, column=3, sticky=tk.W)
        # Carrier and map type
        ttk.Label(maps_body, text="fc_hz").grid(row=2, column=0, sticky=tk.W)
        self.adv_fc_var = tk.DoubleVar(value=28e9)
        ttk.Entry(maps_body, textvariable=self.adv_fc_var, width=14).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(maps_body, text="Map Type").grid(row=2, column=2, sticky=tk.W)
        self.adv_map_type_var = tk.StringVar(value="Angular 2D (θ–ϕ @ r)")
        ttk.Combobox(maps_body, textvariable=self.adv_map_type_var, values=["Angular 2D (θ–ϕ @ r)", "Radial Slice (r @ θ,ϕ)"], state="readonly").grid(row=2, column=3, columnspan=3, sticky="ew")
        # Angular 2D params
        ttk.Label(maps_body, text="r_fixed (m)").grid(row=3, column=0, sticky=tk.W)
        self.adv_rfixed_var = tk.DoubleVar(value=5.0)
        ttk.Entry(maps_body, textvariable=self.adv_rfixed_var, width=10).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(maps_body, text="θ start/stop/step").grid(row=4, column=0, sticky=tk.W)
        self.adv_th_start_var = tk.DoubleVar(value=-60.0)
        self.adv_th_stop_var = tk.DoubleVar(value=60.0)
        self.adv_th_step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(maps_body, textvariable=self.adv_th_start_var, width=8).grid(row=4, column=1, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_th_stop_var, width=8).grid(row=4, column=2, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_th_step_var, width=8).grid(row=4, column=3, sticky=tk.W)
        ttk.Label(maps_body, text="ϕ start/stop/step").grid(row=5, column=0, sticky=tk.W)
        self.adv_ph_start_var = tk.DoubleVar(value=-20.0)
        self.adv_ph_stop_var = tk.DoubleVar(value=20.0)
        self.adv_ph_step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(maps_body, textvariable=self.adv_ph_start_var, width=8).grid(row=5, column=1, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_ph_stop_var, width=8).grid(row=5, column=2, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_ph_step_var, width=8).grid(row=5, column=3, sticky=tk.W)
        # Radial slice params
        ttk.Label(maps_body, text="r_min/max/step").grid(row=6, column=0, sticky=tk.W)
        self.adv_rmin_var = tk.DoubleVar(value=1.0)
        self.adv_rmax_var = tk.DoubleVar(value=10.0)
        self.adv_rstep_var = tk.DoubleVar(value=0.5)
        ttk.Entry(maps_body, textvariable=self.adv_rmin_var, width=8).grid(row=6, column=1, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_rmax_var, width=8).grid(row=6, column=2, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_rstep_var, width=8).grid(row=6, column=3, sticky=tk.W)
        ttk.Label(maps_body, text="θ, ϕ (deg)").grid(row=6, column=4, sticky=tk.W)
        self.adv_theta_var = tk.DoubleVar(value=0.0)
        self.adv_phi_var = tk.DoubleVar(value=0.0)
        ttk.Entry(maps_body, textvariable=self.adv_theta_var, width=8).grid(row=6, column=5, sticky=tk.W)
        ttk.Entry(maps_body, textvariable=self.adv_phi_var, width=8).grid(row=6, column=6, sticky=tk.W)
        # Weighting
        ttk.Label(maps_body, text="Weighting").grid(row=7, column=0, sticky=tk.W)
        self.adv_weighting_var = tk.StringVar(value="Ideal (AMAG)")
        ttk.Combobox(maps_body, textvariable=self.adv_weighting_var, values=["Ideal (AMAG)", "Codebook-selected (AAG)"], state="readonly").grid(row=7, column=1, columnspan=2, sticky="ew")
        self.adv_compare_ff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(maps_body, text="Compare Far-field", variable=self.adv_compare_ff_var).grid(row=7, column=3, sticky=tk.W)
        # Codebook
        self.adv_use_inmem_cb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(maps_body, text="Use in-memory codebook", variable=self.adv_use_inmem_cb_var).grid(row=8, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(maps_body, text="Codebook HDF5").grid(row=8, column=2, sticky=tk.W)
        self.adv_cb_path_var = tk.StringVar(value="")
        ttk.Entry(maps_body, textvariable=self.adv_cb_path_var, width=28).grid(row=8, column=3, columnspan=3, sticky="ew")
        ttk.Button(maps_body, text="Browse...", command=lambda: self._choose_file(self.adv_cb_path_var, save=False, filetypes=[("HDF5","*.h5")])).grid(row=8, column=6, sticky=tk.W)
        # Output
        self.adv_save_html_var = tk.BooleanVar(value=True)
        self.adv_save_csv_var = tk.BooleanVar(value=False)
        self.adv_save_png_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(maps_body, text="Save HTML (interactive)", variable=self.adv_save_html_var).grid(row=9, column=0, sticky=tk.W)
        ttk.Checkbutton(maps_body, text="Save CSV", variable=self.adv_save_csv_var).grid(row=9, column=1, sticky=tk.W)
        ttk.Checkbutton(maps_body, text="Save PNG", variable=self.adv_save_png_var).grid(row=9, column=2, sticky=tk.W)
        ttk.Label(maps_body, text="chunk").grid(row=9, column=3, sticky=tk.W)
        self.adv_chunk_var = tk.IntVar(value=2048)
        ttk.Entry(maps_body, textvariable=self.adv_chunk_var, width=10).grid(row=9, column=4, sticky=tk.W)

    def _choose_file(self, var: tk.StringVar, save: bool, defname: str | None = None, filetypes: list | None = None) -> None:
        try:
            if save:
                path = filedialog.asksaveasfilename(defaultextension=".h5", initialfile=(defname or ""), filetypes=filetypes or [("HDF5","*.h5"),("All","*.*")])
            else:
                path = filedialog.askopenfilename(filetypes=filetypes or [("HDF5","*.h5"),("All","*.*")])
            if path:
                var.set(path)
        except Exception:
            pass

    # ----- Layout helpers for scrollable left panel -----
    def _after_layout_refresh(self) -> None:
        try:
            if hasattr(self, "_pane_left") and self._pane_left is not None:
                self._pane_left.update_idletasks()
                self._pane_left.update_scrollregion()
        except Exception:
            pass

    def _auto_fit_experiments_sections(self) -> None:
        """Collapse sections not selected; expand selected ones."""
        try:
            mapping = {
                "Spherical Codebook": bool(self.exp_build_cb_var.get()),
                "Evaluate Codebook Mismatch": bool(self.exp_eval_cb_var.get()),
                "Wideband: TTD vs Phase": bool(self.exp_wideband_cb_var.get()),
                "Advanced AAG/AMAG Maps": bool(self.exp_adv_maps_cb_var.get()),
                # Show optimizer only when relevant experiments are on
                "Optimizer (optional)": bool(self.exp_wideband_cb_var.get() or self.exp_adv_maps_cb_var.get()),
            }
            folds = getattr(self, "_fold_sections", {})
            for title, desired in mapping.items():
                body = folds.get(title)
                if not body:
                    continue
                try:
                    if desired:
                        body.grid()
                    else:
                        body.grid_remove()
                except Exception:
                    pass
            self._after_layout_refresh()
        except Exception:
            pass

    def _update_wb_xyz_preview(self) -> None:
        try:
            p = rtp_to_xyz(float(self.wb_r_var.get()), float(self.wb_theta_var.get()), float(self.wb_phi_var.get()))
            self._wb_xyz_label.configure(text=f"XYZ: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m")
        except Exception as e:
            self._wb_xyz_label.configure(text=f"XYZ: (error: {e})")


def main():
    root = tk.Tk()
    app = MainSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
