import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import queue
from typing import List

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
import json
import multiprocessing as mp
import os


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
        ttk.Label(adv, text="#z points").grid(row=2, column=0, sticky=tk.W, padx=(0,4))
        ttk.Label(adv, text="#realizations/z").grid(row=2, column=2, sticky=tk.W, padx=(8,4))
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
        tests_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
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
        action_row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
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


def main():
    root = tk.Tk()
    app = MainSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
