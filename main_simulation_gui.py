import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import sys
import queue
from typing import List, Optional
from dataclasses import dataclass
import json
import os
import multiprocessing as mp
import numpy as np

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime


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


class MainSimulationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Near-Field Simulator – Main Console")
        root.geometry("1280x820")
        root.minsize(1024, 640)

        self.running = False
        self._setup_styles()

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

        # Left: controls
        controls = ttk.Labelframe(paned, text="Controls", padding=(12, 10))
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

        # Advanced overrides
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

        # Tests selection
        tests_frame = ttk.Labelframe(controls, text="Select Tests", padding=(10, 8))
        tests_frame.grid(row=9, column=0, columnspan=2, sticky="nsew")
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

        # ========== Experiments (advanced) ==========
        exp_frame = ttk.Labelframe(controls, text="Experiments (advanced)", padding=(10, 8))
        exp_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        exp_frame.grid_columnconfigure(0, weight=1)

        self.exp_build_cb_var = tk.BooleanVar(value=False)
        self.exp_eval_cb_var = tk.BooleanVar(value=False)
        self.exp_wb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(exp_frame, text="Build Spherical Codebook", variable=self.exp_build_cb_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(exp_frame, text="Evaluate Codebook Mismatch", variable=self.exp_eval_cb_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(exp_frame, text="Wideband: TTD vs Phase", variable=self.exp_wb_var).grid(row=2, column=0, sticky=tk.W)

        # Collapsible panels helpers
        def _make_toggle_row(parent, text, row):
            frm = ttk.Frame(parent)
            frm.grid(row=row, column=0, sticky="ew", pady=(6, 2))
            btn = ttk.Button(frm, text="-", width=2)
            lbl = ttk.Label(frm, text=text, font=("Segoe UI", 10, "bold"))
            btn.pack(side=tk.LEFT)
            lbl.pack(side=tk.LEFT, padx=(6, 0))
            return btn

        # (3.1) Spherical Codebook panel
        toggle_build = _make_toggle_row(exp_frame, "Spherical Codebook", 3)
        build_panel = ttk.Frame(exp_frame)
        build_panel.grid(row=4, column=0, sticky="ew")
        for c in range(6):
            build_panel.grid_columnconfigure(c, weight=1)

        ttk.Label(build_panel, text="Array layout").grid(row=0, column=0, sticky=tk.W)
        self.sc_layout_var = tk.StringVar(value="upa")
        ttk.Combobox(build_panel, textvariable=self.sc_layout_var, values=["upa", "ula"], state="readonly", width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(build_panel, text="num_x").grid(row=0, column=2, sticky=tk.W)
        self.sc_numx_var = tk.IntVar(value=32)
        ttk.Entry(build_panel, textvariable=self.sc_numx_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(build_panel, text="num_y").grid(row=0, column=4, sticky=tk.W)
        self.sc_numy_var = tk.IntVar(value=32)
        ttk.Entry(build_panel, textvariable=self.sc_numy_var, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(build_panel, text="dx (m)").grid(row=1, column=0, sticky=tk.W)
        self.sc_dx_var = tk.DoubleVar(value=0.5e-3)
        ttk.Entry(build_panel, textvariable=self.sc_dx_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(build_panel, text="dy (m)").grid(row=1, column=2, sticky=tk.W)
        self.sc_dy_var = tk.DoubleVar(value=0.5e-3)
        ttk.Entry(build_panel, textvariable=self.sc_dy_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(build_panel, text="fc_hz").grid(row=2, column=0, sticky=tk.W)
        self.sc_fc_var = tk.DoubleVar(value=28e9)
        ttk.Entry(build_panel, textvariable=self.sc_fc_var, width=12).grid(row=2, column=1, sticky="w")

        ttk.Label(build_panel, text="r_min").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(build_panel, text="r_max").grid(row=3, column=2, sticky=tk.W)
        ttk.Label(build_panel, text="r_step").grid(row=3, column=4, sticky=tk.W)
        self.sc_rmin_var = tk.DoubleVar(value=3.0)
        self.sc_rmax_var = tk.DoubleVar(value=20.0)
        self.sc_rstep_var = tk.DoubleVar(value=0.5)
        ttk.Entry(build_panel, textvariable=self.sc_rmin_var, width=8).grid(row=3, column=1, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_rmax_var, width=8).grid(row=3, column=3, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_rstep_var, width=8).grid(row=3, column=5, sticky="w")

        ttk.Label(build_panel, text="theta start/stop/step").grid(row=4, column=0, sticky=tk.W)
        self.sc_tstart_var = tk.DoubleVar(value=-60.0)
        self.sc_tstop_var = tk.DoubleVar(value=60.0)
        self.sc_tstep_var = tk.DoubleVar(value=5.0)
        ttk.Entry(build_panel, textvariable=self.sc_tstart_var, width=8).grid(row=4, column=1, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_tstop_var, width=8).grid(row=4, column=3, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_tstep_var, width=8).grid(row=4, column=5, sticky="w")

        ttk.Label(build_panel, text="phi start/stop/step").grid(row=5, column=0, sticky=tk.W)
        self.sc_pstart_var = tk.DoubleVar(value=-30.0)
        self.sc_pstop_var = tk.DoubleVar(value=30.0)
        self.sc_pstep_var = tk.DoubleVar(value=5.0)
        ttk.Entry(build_panel, textvariable=self.sc_pstart_var, width=8).grid(row=5, column=1, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_pstop_var, width=8).grid(row=5, column=3, sticky="w")
        ttk.Entry(build_panel, textvariable=self.sc_pstep_var, width=8).grid(row=5, column=5, sticky="w")

        ttk.Label(build_panel, text="chunk").grid(row=6, column=0, sticky=tk.W)
        self.sc_chunk_var = tk.IntVar(value=2048)
        ttk.Entry(build_panel, textvariable=self.sc_chunk_var, width=8).grid(row=6, column=1, sticky="w")

        ttk.Label(build_panel, text="Output HDF5").grid(row=7, column=0, sticky=tk.W)
        self.sc_out_path_var = tk.StringVar(value=os.path.abspath("codebook.h5"))
        ttk.Entry(build_panel, textvariable=self.sc_out_path_var, width=32).grid(row=7, column=1, columnspan=3, sticky="ew")
        ttk.Button(build_panel, text="Browse...", command=lambda: self._choose_file(self.sc_out_path_var, save=True, defaultextension=".h5", filetypes=[("HDF5","*.h5")])).grid(row=7, column=4, columnspan=2, sticky="w")

        # (3.2) Evaluate Codebook Mismatch
        toggle_eval = _make_toggle_row(exp_frame, "Evaluate Codebook Mismatch", 5)
        eval_panel = ttk.Frame(exp_frame)
        eval_panel.grid(row=6, column=0, sticky="ew")
        for c in range(6):
            eval_panel.grid_columnconfigure(c, weight=1)

        ttk.Label(eval_panel, text="Q (queries)").grid(row=0, column=0, sticky=tk.W)
        self.ev_q_var = tk.IntVar(value=200)
        ttk.Entry(eval_panel, textvariable=self.ev_q_var, width=8).grid(row=0, column=1, sticky="w")

        # region: x/y/z min/max
        ttk.Label(eval_panel, text="x_min/x_max").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(eval_panel, text="y_min/y_max").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(eval_panel, text="z_min/z_max").grid(row=3, column=0, sticky=tk.W)
        self.ev_xmin = tk.DoubleVar(value=-5.0)
        self.ev_xmax = tk.DoubleVar(value=5.0)
        self.ev_ymin = tk.DoubleVar(value=-5.0)
        self.ev_ymax = tk.DoubleVar(value=5.0)
        self.ev_zmin = tk.DoubleVar(value=2.0)
        self.ev_zmax = tk.DoubleVar(value=10.0)
        ttk.Entry(eval_panel, textvariable=self.ev_xmin, width=8).grid(row=1, column=1, sticky="w")
        ttk.Entry(eval_panel, textvariable=self.ev_xmax, width=8).grid(row=1, column=2, sticky="w")
        ttk.Entry(eval_panel, textvariable=self.ev_ymin, width=8).grid(row=2, column=1, sticky="w")
        ttk.Entry(eval_panel, textvariable=self.ev_ymax, width=8).grid(row=2, column=2, sticky="w")
        ttk.Entry(eval_panel, textvariable=self.ev_zmin, width=8).grid(row=3, column=1, sticky="w")
        ttk.Entry(eval_panel, textvariable=self.ev_zmax, width=8).grid(row=3, column=2, sticky="w")

        self.ev_compare_ff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(eval_panel, text="Compare with plane-wave baseline", variable=self.ev_compare_ff_var).grid(row=4, column=0, columnspan=3, sticky=tk.W)

        ttk.Label(eval_panel, text="Input HDF5 (optional)").grid(row=5, column=0, sticky=tk.W)
        self.ev_in_path_var = tk.StringVar(value="")
        ttk.Entry(eval_panel, textvariable=self.ev_in_path_var, width=32).grid(row=5, column=1, columnspan=3, sticky="ew")
        ttk.Button(eval_panel, text="Browse...", command=lambda: self._choose_file(self.ev_in_path_var, save=False, defaultextension=".h5", filetypes=[("HDF5","*.h5")])).grid(row=5, column=4, columnspan=2, sticky="w")

        self.ev_save_csv_var = tk.BooleanVar(value=False)
        self.ev_save_png_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(eval_panel, text="Save CSV", variable=self.ev_save_csv_var).grid(row=6, column=0, sticky=tk.W)
        ttk.Checkbutton(eval_panel, text="Save PNG", variable=self.ev_save_png_var).grid(row=6, column=1, sticky=tk.W)

        # (3.3) Wideband: TTD vs Phase
        toggle_wb = _make_toggle_row(exp_frame, "Wideband: TTD vs Phase", 7)
        wb_panel = ttk.Frame(exp_frame)
        wb_panel.grid(row=8, column=0, sticky="ew")
        for c in range(8):
            wb_panel.grid_columnconfigure(c, weight=1)

        ttk.Label(wb_panel, text="fc_hz").grid(row=0, column=0, sticky=tk.W)
        self.wb_fc_var = tk.DoubleVar(value=28e9)
        ttk.Entry(wb_panel, textvariable=self.wb_fc_var, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(wb_panel, text="bw_hz").grid(row=0, column=2, sticky=tk.W)
        self.wb_bw_var = tk.DoubleVar(value=400e6)
        ttk.Entry(wb_panel, textvariable=self.wb_bw_var, width=12).grid(row=0, column=3, sticky="w")
        ttk.Label(wb_panel, text="n_sc").grid(row=0, column=4, sticky=tk.W)
        self.wb_nsc_var = tk.IntVar(value=256)
        ttk.Entry(wb_panel, textvariable=self.wb_nsc_var, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(wb_panel, text="Focus r,θ,ϕ").grid(row=1, column=0, sticky=tk.W)
        self.wb_r_var = tk.DoubleVar(value=8.0)
        self.wb_th_var = tk.DoubleVar(value=0.0)
        self.wb_ph_var = tk.DoubleVar(value=0.0)
        ttk.Entry(wb_panel, textvariable=self.wb_r_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Entry(wb_panel, textvariable=self.wb_th_var, width=8).grid(row=1, column=2, sticky="w")
        ttk.Entry(wb_panel, textvariable=self.wb_ph_var, width=8).grid(row=1, column=3, sticky="w")
        ttk.Button(wb_panel, text="r,θ,ϕ → XYZ", command=self._wb_rtp_to_xyz).grid(row=1, column=4, sticky="w")
        ttk.Label(wb_panel, text="x,y,z").grid(row=1, column=5, sticky=tk.E)
        self.wb_x_var = tk.DoubleVar(value=0.0)
        self.wb_y_var = tk.DoubleVar(value=0.0)
        self.wb_z_var = tk.DoubleVar(value=8.0)
        ttk.Entry(wb_panel, textvariable=self.wb_x_var, width=8, state="readonly").grid(row=1, column=6, sticky="w")
        ttk.Entry(wb_panel, textvariable=self.wb_y_var, width=8, state="readonly").grid(row=1, column=7, sticky="w")
        ttk.Entry(wb_panel, textvariable=self.wb_z_var, width=8, state="readonly").grid(row=1, column=8 if 8 < wb_panel.grid_size()[0] else 7, sticky="w")

        ttk.Label(wb_panel, text="Array layout").grid(row=2, column=0, sticky=tk.W)
        self.wb_layout_var = tk.StringVar(value="upa")
        ttk.Combobox(wb_panel, textvariable=self.wb_layout_var, values=["upa", "ula"], state="readonly", width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(wb_panel, text="num_x").grid(row=2, column=2, sticky=tk.W)
        self.wb_numx_var = tk.IntVar(value=32)
        ttk.Entry(wb_panel, textvariable=self.wb_numx_var, width=8).grid(row=2, column=3, sticky="w")
        ttk.Label(wb_panel, text="num_y").grid(row=2, column=4, sticky=tk.W)
        self.wb_numy_var = tk.IntVar(value=32)
        ttk.Entry(wb_panel, textvariable=self.wb_numy_var, width=8).grid(row=2, column=5, sticky="w")
        ttk.Label(wb_panel, text="dx (m)").grid(row=3, column=0, sticky=tk.W)
        self.wb_dx_var = tk.DoubleVar(value=0.5e-3)
        ttk.Entry(wb_panel, textvariable=self.wb_dx_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(wb_panel, text="dy (m)").grid(row=3, column=2, sticky=tk.W)
        self.wb_dy_var = tk.DoubleVar(value=0.5e-3)
        ttk.Entry(wb_panel, textvariable=self.wb_dy_var, width=10).grid(row=3, column=3, sticky="w")

        self.wb_save_png_var = tk.BooleanVar(value=False)
        self.wb_save_json_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wb_panel, text="Save PNG", variable=self.wb_save_png_var).grid(row=4, column=0, sticky=tk.W)
        ttk.Checkbutton(wb_panel, text="Save JSON", variable=self.wb_save_json_var).grid(row=4, column=1, sticky=tk.W)

        # Collapse/expand behavior
        def _toggle(panel: ttk.Frame, btn: ttk.Button):
            if panel.winfo_ismapped():
                panel.grid_remove()
                btn.configure(text="▶")
            else:
                panel.grid()
                btn.configure(text="▼")
        toggle_build.configure(command=lambda: _toggle(build_panel, toggle_build))
        toggle_eval.configure(command=lambda: _toggle(eval_panel, toggle_eval))
        toggle_wb.configure(command=lambda: _toggle(wb_panel, toggle_wb))

        # Advanced fold-out
        adv2 = ttk.Labelframe(controls, text="Advanced", padding=(8, 6))
        adv2.grid(row=11, column=0, columnspan=2, sticky="ew")
        ttk.Label(adv2, text="random_seed").grid(row=0, column=0, sticky=tk.W)
        self.adv_seed_var = tk.IntVar(value=1234)
        ttk.Entry(adv2, textvariable=self.adv_seed_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(adv2, text="n_jobs").grid(row=0, column=2, sticky=tk.W)
        self.adv_jobs_var = tk.IntVar(value=-1)
        ttk.Entry(adv2, textvariable=self.adv_jobs_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(adv2, text="log_level").grid(row=0, column=4, sticky=tk.W)
        self.adv_loglvl_var = tk.StringVar(value="INFO")
        ttk.Combobox(adv2, textvariable=self.adv_loglvl_var, values=["INFO", "DEBUG"], state="readonly", width=8).grid(row=0, column=5, sticky="w")

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

        paned.add(controls, weight=400)
        paned.add(right, weight=600)

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
            (exp_frame, "Advanced experiments: codebooks and wideband TTD vs PS"),
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
            self._append_log("No legacy tests selected. Will run experiments if any are checked.\n")

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
                # Legacy tests first
                for name in selected:
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
                # Experiments (advanced)
                tasks_to_run = []
                if self.exp_build_cb_var.get():
                    tasks_to_run.append("Build Spherical Codebook")
                if self.exp_eval_cb_var.get():
                    tasks_to_run.append("Evaluate Codebook Mismatch")
                if self.exp_wb_var.get():
                    tasks_to_run.append("Wideband: TTD vs Phase")
                if tasks_to_run:
                    self._append_log("\n=== Experiments (advanced) ===\n")
                latest_ck = None
                for task in tasks_to_run:
                    if self._closing:
                        break
                    self._schedule_status(f"Running: {task}")
                    t0 = time.perf_counter()
                    try:
                        cfg = self._read_run_config(task)
                        if task == "Build Spherical Codebook":
                            res = run_build_spherical_codebook(cfg, progress_cb=self._progress_update, is_cancelled=self._is_cancelled)
                            latest_ck = res
                            self._latest_codebook = res
                        elif task == "Evaluate Codebook Mismatch":
                            cfg.in_memory_codebook = getattr(self, "_latest_codebook", None)
                            res = run_eval_codebook_mismatch(cfg, progress_cb=self._progress_update, is_cancelled=self._is_cancelled)
                        elif task == "Wideband: TTD vs Phase":
                            res = run_wideband_compare_ttd_vs_phase(cfg, progress_cb=self._progress_update, is_cancelled=self._is_cancelled)
                        else:
                            res = None
                        figs = res.get("figs", []) if isinstance(res, dict) else []
                        if figs:
                            self._show_figures(figs)
                        summary = res.get("summary_text") if isinstance(res, dict) else None
                        if summary:
                            self._set_summary_text(summary)
                        self._append_log(f"{task} done.\n")
                    except Exception as e:
                        self._append_log(f"{task} error: {e}\n")
                    finally:
                        dt = time.perf_counter() - t0
                        self._append_log(f"{task} completed in {dt:.2f}s\n")
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

    def _progress_update(self, frac: float | None = None, text: Optional[str] = None) -> None:
        """Thread-safe progress update. frac in [0,1] or None to switch to indeterminate."""
        if self._closing:
            return
        def _do():
            try:
                if frac is None:
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(10)
                else:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100.0, value=max(0.0, min(1.0, float(frac))) * 100.0)
                if text is not None:
                    self.status_var.set(text)
            except Exception:
                pass
        self.root.after(0, _do)

    def _is_cancelled(self) -> bool:
        try:
            return bool(getattr(self, '_stop_event', None) and self._stop_event.is_set())
        except Exception:
            return False

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
        # Clear previous canvases (only children within container, not the scroll widgets)
        for w in list(self.plots_container.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._fig_canvases.clear()

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

        for fig in figs:
            canvas = FigureCanvasTkAgg(fig, master=self.plots_container)
            canvas.draw()
            # Pack without expansion to preserve native size; scrolling shows overflow
            canvas.get_tk_widget().pack(side=tk.TOP, anchor='nw', pady=(2, 6))
            self._fig_canvases.append(canvas)

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

    def _choose_file(self, var: tk.StringVar, save: bool, defaultextension: str, filetypes):
        try:
            if save:
                path = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
            else:
                path = filedialog.askopenfilename(filetypes=filetypes)
            if path:
                var.set(path)
        except Exception:
            pass

    def _wb_rtp_to_xyz(self) -> None:
        try:
            from nearfield.spherical import rtp_to_cartesian
            p = rtp_to_cartesian(float(self.wb_r_var.get()), float(self.wb_th_var.get()), float(self.wb_ph_var.get()))
            self.wb_x_var.set(float(p[0])); self.wb_y_var.set(float(p[1])); self.wb_z_var.set(float(p[2]))
        except Exception as e:
            self._append_log(f"rtp->xyz error: {e}\n")

    def _show_figures(self, figs: List) -> None:
        # Create scrollable area if missing
        try:
            exists = self.plots_container.winfo_exists()
        except Exception:
            exists = False
        if not exists:
            try:
                self.plots_canvas = tk.Canvas(self.plots_tab, highlightthickness=0)
                self.plots_vscroll = ttk.Scrollbar(self.plots_tab, orient="vertical", command=self.plots_canvas.yview)
                self.plots_hscroll = ttk.Scrollbar(self.plots_tab, orient="horizontal", command=self.plots_canvas.xview)
                self.plots_canvas.configure(yscrollcommand=self.plots_vscroll.set, xscrollcommand=self.plots_hscroll.set)
                self.plots_canvas.grid(row=0, column=0, sticky="nsew")
                self.plots_vscroll.grid(row=0, column=1, sticky="ns")
                self.plots_hscroll.grid(row=1, column=0, sticky="ew")
                self.plots_container = ttk.Frame(self.plots_canvas)
                self._plots_window = self.plots_canvas.create_window((0, 0), window=self.plots_container, anchor="nw")
                self.plots_container.bind("<Configure>", lambda e: self.plots_canvas.configure(scrollregion=self.plots_canvas.bbox("all")))
            except Exception:
                return
        # Clear and add new
        for w in list(self.plots_container.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._fig_canvases.clear()
        try:
            for fig in figs:
                canvas = FigureCanvasTkAgg(fig, master=self.plots_container)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, anchor='nw', pady=(2, 6))
                self._fig_canvases.append(canvas)
            # switch to plots tab
            self.nb.select(self.plots_tab)
        except Exception:
            pass

    def _cancel_run(self) -> None:
        try:
            if hasattr(self, '_stop_event') and self._stop_event is not None:
                self._stop_event.set()
                self._append_log("Cancelling run...\n")
                self._schedule_status("Cancelling...")
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


@dataclass
class RunConfig:
    random_seed: int = 1234
    n_jobs: int = -1
    log_level: str = "INFO"
    # codebook build
    layout: str = "upa"
    num_x: int = 32
    num_y: int = 32
    dx: float = 0.5e-3
    dy: float = 0.5e-3
    fc_hz: float = 28e9
    r_min: float = 3.0
    r_max: float = 20.0
    r_step: float = 0.5
    theta_range: tuple = (-60.0, 60.0, 5.0)
    phi_range: tuple = (-30.0, 30.0, 5.0)
    chunk: int = 2048
    out_h5: Optional[str] = None
    # eval mismatch
    Q: int = 200
    region: tuple = (-5.0, 5.0, -5.0, 5.0, 2.0, 10.0)
    compare_ff: bool = False
    in_h5: Optional[str] = None
    save_csv: bool = False
    save_png: bool = False
    in_memory_codebook: Optional[dict] = None
    # wideband
    wb_fc_hz: float = 28e9
    wb_bw_hz: float = 400e6
    wb_n_sc: int = 256
    focus_rtp: tuple = (8.0, 0.0, 0.0)
    wb_layout: str = "upa"
    wb_num_x: int = 32
    wb_num_y: int = 32
    wb_dx: float = 0.5e-3
    wb_dy: float = 0.5e-3
    wb_save_png: bool = False
    wb_save_json: bool = False


def run_build_spherical_codebook(cfg: RunConfig, progress_cb=None, is_cancelled=None) -> dict:
    from nearfield.geometry import make_array
    from nearfield.grids import make_rtp_grid
    from nearfield.spherical import spherical_codebook
    from nearfield.codebook_io import save_codebook_h5

    rng = np.random.default_rng(cfg.random_seed)
    layout = cfg.layout
    xyz = make_array(layout, num_x=cfg.num_x, num_y=cfg.num_y if layout == "upa" else None, dx=cfg.dx, dy=cfg.dy)
    t0, t1, tstep = cfg.theta_range
    p0, p1, pstep = cfg.phi_range
    theta = np.arange(float(t0), float(t1) + 1e-12, float(tstep))
    phi = np.arange(float(p0), float(p1) + 1e-12, float(pstep))
    rtp = make_rtp_grid(cfg.r_min, cfg.r_max, cfg.r_step, theta, phi)
    K = rtp.shape[0]
    M = xyz.shape[0]
    print(f"Building spherical codebook: K={K}, M={M}, chunk={cfg.chunk}")
    out = np.empty((K, M), dtype=np.complex128)
    if progress_cb:
        progress_cb(0.0, text="Building codebook...")
    built = 0
    for s in range(0, K, max(1, int(cfg.chunk))):
        if is_cancelled and is_cancelled():
            print("Cancelled.")
            break
        e = min(K, s + int(cfg.chunk))
        out[s:e] = spherical_codebook(xyz, cfg.fc_hz, rtp[s:e], chunk=e - s)
        built = e
        if progress_cb:
            progress_cb(built / max(1, K), text=f"Codebook {built}/{K}")
    attrs = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": int(cfg.random_seed),
    }
    result = {"xyz_m": xyz, "fc_hz": float(cfg.fc_hz), "rtp_grid": rtp, "codebook": out, "attrs": attrs, "figs": []}
    if cfg.out_h5:
        try:
            save_codebook_h5(cfg.out_h5, xyz, cfg.fc_hz, rtp, out, attrs=attrs)
            print(f"Saved HDF5: {cfg.out_h5}")
        except Exception as e:
            print(f"Save error: {e}")
    return result


def run_eval_codebook_mismatch(cfg: RunConfig, progress_cb=None, is_cancelled=None) -> dict:
    import matplotlib.pyplot as plt
    from nearfield.codebook_io import load_codebook_h5
    from nearfield.metrics import quantization_loss_at, farfield_mismatch_loss
    from nearfield.spherical import plane_wave_steering

    rng = np.random.default_rng(cfg.random_seed)
    data = None
    if cfg.in_memory_codebook is not None:
        data = cfg.in_memory_codebook
    elif cfg.in_h5:
        data = load_codebook_h5(cfg.in_h5)
    else:
        raise ValueError("No codebook available. Build first or provide HDF5 path.")
    xyz = data["xyz_m"]; fc = data["fc_hz"]; rtp = data["rtp_grid"]; cb = data["codebook"]
    Q = int(cfg.Q)
    x = rng.uniform(cfg.region[0], cfg.region[1], size=Q)
    y = rng.uniform(cfg.region[2], cfg.region[3], size=Q)
    z = rng.uniform(cfg.region[4], cfg.region[5], size=Q)
    pts = np.column_stack([x, y, z])
    if progress_cb:
        progress_cb(0.2, text="Computing quantization loss...")
    qloss = quantization_loss_at(xyz, fc, rtp, cb, pts)
    mismatch = None
    if cfg.compare_ff:
        if progress_cb:
            progress_cb(0.5, text="Computing far-field mismatch...")
        K = rtp.shape[0]
        M = xyz.shape[0]
        cb_ff = np.empty((K, M), dtype=np.complex128)
        for i, (r, th, ph) in enumerate(rtp):
            cb_ff[i] = plane_wave_steering(xyz, fc, th, ph)
        mismatch = farfield_mismatch_loss(xyz, fc, rtp, cb_ff, pts)
    if progress_cb:
        progress_cb(0.8, text="Rendering plots...")
    figs = []
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax[0].hist(qloss, bins=30, color="tab:blue", alpha=0.85)
        ax[0].set_title("Quantization Loss (dB)"); ax[0].set_xlabel("dB"); ax[0].set_ylabel("Count")
        if mismatch is not None:
            ax[1].hist(mismatch, bins=30, color="tab:orange", alpha=0.85)
            ax[1].set_title("Far-field mismatch (dB)"); ax[1].set_xlabel("dB")
        else:
            ax[1].axis('off')
        figs.append(fig)
    except Exception as e:
        print(f"Plot error: {e}")
    def _stats(arr):
        return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)), float(np.percentile(arr, 95))
    m1 = _stats(qloss)
    summ_lines = ["Codebook Mismatch Summary\n", f"Quantization loss: mean={m1[0]:.3f} dB, median={m1[1]:.3f} dB, std={m1[2]:.3f} dB, 95th={m1[3]:.3f} dB"]
    if mismatch is not None:
        m2 = _stats(mismatch)
        summ_lines.append(f"Far-field mismatch: mean={m2[0]:.3f} dB, median={m2[1]:.3f} dB, std={m2[2]:.3f} dB, 95th={m2[3]:.3f} dB")
    summary_text = "\n".join(summ_lines)
    if cfg.save_csv:
        try:
            path = f"mismatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(path, 'w', encoding='utf-8') as f:
                f.write("qloss_db\n"); f.writelines(f"{v}\n" for v in qloss)
                if mismatch is not None:
                    f.write("mismatch_db\n"); f.writelines(f"{v}\n" for v in mismatch)
            print(f"Saved CSV: {path}")
        except Exception as e:
            print(f"CSV save error: {e}")
    if cfg.save_png and figs:
        try:
            figs[0].savefig("mismatch_hist.png", dpi=120)
            print("Saved PNG: mismatch_hist.png")
        except Exception as e:
            print(f"PNG save error: {e}")
    if progress_cb:
        progress_cb(1.0, text="Done")
    return {"figs": figs, "summary_text": summary_text}


def run_wideband_compare_ttd_vs_phase(cfg: RunConfig, progress_cb=None, is_cancelled=None) -> dict:
    import matplotlib.pyplot as plt
    from nearfield.geometry import make_array
    from nearfield.spherical import rtp_to_cartesian
    from nearfield.wideband import subcarrier_frequencies, spherical_steering_wideband
    from nearfield.beamformer.phase import design_phase_shifter_weights, weights_over_band_phase_shifter
    from nearfield.beamformer.ttd import design_ttd_delays, weights_over_band_ttd
    from nearfield.metrics_wideband import (
        beampattern_gain_spectrum,
        gain_flatness_db,
        beam_squint_deg,
        achievable_rate_bpshz,
    )

    if progress_cb:
        progress_cb(0.0, text="Preparing wideband compare...")
    layout = cfg.wb_layout
    xyz = make_array(layout, num_x=cfg.wb_num_x, num_y=cfg.wb_num_y if layout == "upa" else None, dx=cfg.wb_dx, dy=cfg.wb_dy)
    f = subcarrier_frequencies(cfg.wb_fc_hz, cfg.wb_bw_hz, cfg.wb_n_sc)
    p = rtp_to_cartesian(*cfg.focus_rtp)
    if progress_cb:
        progress_cb(0.2, text="Computing steering across band...")
    A = spherical_steering_wideband(xyz, p, f)
    if is_cancelled and is_cancelled():
        print("Cancelled.")
        return {"figs": [], "summary_text": "Cancelled."}
    if progress_cb:
        progress_cb(0.4, text="Designing PS/TTD weights...")
    w_fc = design_phase_shifter_weights(xyz, cfg.wb_fc_hz, p)
    W_ps = weights_over_band_phase_shifter(w_fc, f.size)
    d = design_ttd_delays(xyz, p)
    W_ttd = weights_over_band_ttd(d, f)
    if progress_cb:
        progress_cb(0.6, text="Evaluating performance metrics...")
    g_ps = beampattern_gain_spectrum(W_ps, A)
    g_ttd = beampattern_gain_spectrum(W_ttd, A)
    flat_ps = gain_flatness_db(g_ps)
    flat_ttd = gain_flatness_db(g_ttd)
    squint_ps = beam_squint_deg(xyz, cfg.wb_fc_hz, W_ps, f, r_fixed_m=cfg.focus_rtp[0])
    squint_ttd = beam_squint_deg(xyz, cfg.wb_fc_hz, W_ttd, f, r_fixed_m=cfg.focus_rtp[0])
    rate_ps = achievable_rate_bpshz(g_ps, noise_psd_w_hz=1e-17, subcarrier_bw_hz=cfg.wb_bw_hz / f.size)
    rate_ttd = achievable_rate_bpshz(g_ttd, noise_psd_w_hz=1e-17, subcarrier_bw_hz=cfg.wb_bw_hz / f.size)
    if progress_cb:
        progress_cb(0.8, text="Rendering plot...")
    figs = []
    try:
        f_off = (f - cfg.wb_fc_hz) / 1e6
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
        ax.plot(f_off, 10 * np.log10(g_ps + 1e-16), label="PS")
        ax.plot(f_off, 10 * np.log10(g_ttd + 1e-16), label="TTD")
        ax.set_xlabel("Frequency offset (MHz)"); ax.set_ylabel("Gain (dB)"); ax.set_title("Wideband beampattern gain")
        ax.grid(True, ls=":"); ax.legend()
        figs.append(fig)
        if cfg.wb_save_png:
            fig.savefig("wb_gain.png", dpi=120)
    except Exception as e:
        print(f"Plot error: {e}")
    summary = (
        f"Wideband Compare (TTD vs PS)\n"
        f"flatness_PS_dB={flat_ps:.2f}, flatness_TTD_dB={flat_ttd:.2f}\n"
        f"squint_PS_deg={squint_ps:.2f}, squint_TTD_deg={squint_ttd:.2f}\n"
        f"rate_PS={rate_ps:.3f} bps/Hz, rate_TTD={rate_ttd:.3f} bps/Hz\n"
    )
    if cfg.wb_save_json:
        try:
            path = "wb_summary.json"
            with open(path, 'w', encoding='utf-8') as fjs:
                json.dump({
                    "flatness_PS_dB": flat_ps,
                    "flatness_TTD_dB": flat_ttd,
                    "squint_PS_deg": squint_ps,
                    "squint_TTD_deg": squint_ttd,
                    "rate_PS_bpsHz": rate_ps,
                    "rate_TTD_bpsHz": rate_ttd,
                }, fjs, indent=2)
            print(f"Saved JSON: {path}")
        except Exception as e:
            print(f"JSON save error: {e}")
    if progress_cb:
        progress_cb(1.0, text="Done")
    return {"figs": figs, "summary_text": summary}


def _read_codebook_ranges(self) -> tuple:
    return (
        float(self.sc_rmin_var.get()),
        float(self.sc_rmax_var.get()),
        float(self.sc_rstep_var.get()),
        (float(self.sc_tstart_var.get()), float(self.sc_tstop_var.get()), float(self.sc_tstep_var.get())),
        (float(self.sc_pstart_var.get()), float(self.sc_pstop_var.get()), float(self.sc_pstep_var.get())),
    )


def _read_eval_region(self) -> tuple:
    return (
        float(self.ev_xmin.get()), float(self.ev_xmax.get()),
        float(self.ev_ymin.get()), float(self.ev_ymax.get()),
        float(self.ev_zmin.get()), float(self.ev_zmax.get()),
    )


def _read_run_config(self, task_name: str) -> RunConfig:
    seed = int(self.adv_seed_var.get())
    n_jobs = int(self.adv_jobs_var.get())
    lvl = self.adv_loglvl_var.get() or "INFO"
    cfg = RunConfig(random_seed=seed, n_jobs=n_jobs, log_level=lvl)
    if task_name == "Build Spherical Codebook":
        rmin, rmax, rstep, tr, pr = _read_codebook_ranges(self)
        cfg.layout = self.sc_layout_var.get() or "upa"
        cfg.num_x = int(self.sc_numx_var.get())
        cfg.num_y = int(self.sc_numy_var.get())
        cfg.dx = float(self.sc_dx_var.get()); cfg.dy = float(self.sc_dy_var.get())
        cfg.fc_hz = float(self.sc_fc_var.get())
        cfg.r_min = float(rmin); cfg.r_max = float(rmax); cfg.r_step = float(rstep)
        cfg.theta_range = tr; cfg.phi_range = pr
        cfg.chunk = int(self.sc_chunk_var.get())
        cfg.out_h5 = self.sc_out_path_var.get() or None
    elif task_name == "Evaluate Codebook Mismatch":
        cfg.Q = int(self.ev_q_var.get())
        cfg.region = _read_eval_region(self)
        cfg.compare_ff = bool(self.ev_compare_ff_var.get())
        cfg.in_h5 = self.ev_in_path_var.get() or None
        cfg.save_csv = bool(self.ev_save_csv_var.get())
        cfg.save_png = bool(self.ev_save_png_var.get())
    elif task_name == "Wideband: TTD vs Phase":
        cfg.wb_fc_hz = float(self.wb_fc_var.get()); cfg.wb_bw_hz = float(self.wb_bw_var.get()); cfg.wb_n_sc = int(self.wb_nsc_var.get())
        cfg.focus_rtp = (float(self.wb_r_var.get()), float(self.wb_th_var.get()), float(self.wb_ph_var.get()))
        cfg.wb_layout = self.wb_layout_var.get() or "upa"
        cfg.wb_num_x = int(self.wb_numx_var.get()); cfg.wb_num_y = int(self.wb_numy_var.get())
        cfg.wb_dx = float(self.wb_dx_var.get()); cfg.wb_dy = float(self.wb_dy_var.get())
        cfg.wb_save_png = bool(self.wb_save_png_var.get()); cfg.wb_save_json = bool(self.wb_save_json_var.get())
    return cfg


def main():
    root = tk.Tk()
    app = MainSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

