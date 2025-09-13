import tkinter as tk
from tkinter import ttk
import time
import threading
import queue
from random_params import random_basic_params, VALID_PRESETS, VALID_MODES
from optimized_nearfield_system import create_system_with_presets, create_simulation_config
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class _ToolTip:
    """Lightweight tooltip for Tk/ttk widgets."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwin = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tipwin or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tw, text=self.text, style="Status.TLabel", padding=(8, 4))
        label.pack()

    def _hide(self, _event=None):
        tw = self.tipwin
        self.tipwin = None
        if tw is not None:
            tw.destroy()


class SimulationGUI:
    """Simple GUI wrapper to run simulations and display plots."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        # Shorter, balanced window title
        root.title("Near-Field Simulator")
        # Set a harmonious default size and minimums
        root.geometry("1200x760")
        root.minsize(980, 600)

        self.running = False
        self._closing = False

        # Theming and styles
        self._setup_styles()

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main_frame, style="Header.TFrame", padding=(8, 6))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Near-Field Simulation",
            style="Title.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="LIS-UAV Multi-beamforming",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(6, 8))

        # Split layout: left controls, right plots
        paned = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Input controls (left panel)
        control_frame = ttk.Labelframe(paned, text="Controls", padding=(10, 8), style="Card.TLabelframe")
        # Make inputs expand nicely
        control_frame.grid_columnconfigure(0, weight=0)
        control_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, 4))
        self.preset_var = tk.StringVar(value="standard")
        self.preset_cb = ttk.Combobox(control_frame, textvariable=self.preset_var, values=VALID_PRESETS, state="readonly")
        self.preset_cb.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        ttk.Label(control_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.mode_var = tk.StringVar(value="fast")
        self.mode_cb = ttk.Combobox(control_frame, textvariable=self.mode_var, values=VALID_MODES, state="readonly")
        self.mode_cb.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(control_frame, text="Users:").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.users_var = tk.IntVar(value=5)
        self.users_spin = ttk.Spinbox(control_frame, from_=1, to=200, textvariable=self.users_var, width=8)
        self.users_spin.grid(row=2, column=1, sticky="w", pady=4)

        # Randomize option
        self.randomize_var = tk.BooleanVar(value=False)
        self.randomize_cb = ttk.Checkbutton(control_frame, text="Randomize on Run", variable=self.randomize_var)
        self.randomize_cb.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))

        # Buttons row
        btn_row = ttk.Frame(control_frame)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(btn_row, text="▶ Run Simulation", style="Accent.TButton", command=self.run_simulation)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        # Normalize label text across encodings
        try:
            self.run_btn.configure(text="Run Simulation")
        except Exception:
            pass
        self.clear_btn = ttk.Button(btn_row, text="✕ Clear Plots", command=self.clear_plots)
        self.clear_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        try:
            self.clear_btn.configure(text="Clear Plots")
        except Exception:
            pass

        # Area for plots and summary (right panel)
        self.plot_frame = ttk.Frame(paned, padding=(2, 2))
        # Notebook with two tabs: Plots and Summary
        self.notebook = ttk.Notebook(self.plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.plots_tab = ttk.Frame(self.notebook)
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_tab, text="Plots")
        self.notebook.add(self.summary_tab, text="Summary")
        # Scrollable plots area
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
        # Summary text area with scrollbar
        self.summary_text = tk.Text(self.summary_tab, wrap="word", height=12)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.summary_scroll = ttk.Scrollbar(self.summary_tab, orient="vertical", command=self.summary_text.yview)
        self.summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(yscrollcommand=self.summary_scroll.set, state="disabled")

        # Add panels to the paned window with proportional weights
        paned.add(control_frame, weight=382)  # ~38.2%
        paned.add(self.plot_frame, weight=618)  # ~61.8%

        # Set initial sash position to golden-ratio split
        root.update_idletasks()
        try:
            total_w = paned.winfo_width() or root.winfo_width()
            paned.sashpos(0, int(total_w * 0.382))
        except Exception:
            pass

        # Status bar with progress
        status = ttk.Frame(main_frame, padding=(2, 4))
        status.pack(fill=tk.X, pady=(8, 0))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        # Tooltips for better interaction
        self._add_tooltip(self.preset_cb, "Choose a predefined system configuration")
        self._add_tooltip(self.mode_cb, "Select simulation depth/speed")
        self._add_tooltip(self.users_spin, "Number of users (1-200)")
        self._add_tooltip(self.run_btn, "Run the simulation with current settings")

        # Thread-safe message queue and poller
        self._msg_queue: queue.Queue = queue.Queue()
        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run_simulation(self) -> None:
        # Optionally randomize inputs while respecting valid choices
        if self.running:
            self._set_status("A simulation is already running...")
            return
        if self.randomize_var.get():
            rp = random_basic_params()
            # Reflect randomized values in the UI
            self.preset_var.set(rp["preset"])
            self.mode_var.set(rp["mode"])
            self.users_var.set(int(rp["users"]))

        preset = self.preset_var.get() or "standard"
        mode = self.mode_var.get() or "fast"
        users = int(self.users_var.get() or 5)

        self._set_running(True)
        self._set_status(f"Running: preset={preset}, mode={mode}, users={users}")
        start = time.perf_counter()

        def worker():
            try:
                simulator = create_system_with_presets(preset)
                config = create_simulation_config(mode)
                config.num_users_list = [users]
                results = simulator.run_optimized_simulation(config)
            except Exception as e:
                # Push error back to UI thread
                try:
                    self._msg_queue.put(("error", str(e)))
                except Exception:
                    pass
                return
            elapsed = time.perf_counter() - start
            try:
                self._msg_queue.put(("done", simulator, results, elapsed))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_simulation_done(self, simulator, results, elapsed_s: float) -> None:
        self.clear_plots()
        # Plot in the UI thread for safety (suppress external windows)
        try:
            import matplotlib.pyplot as plt  # type: ignore
            _orig_show = plt.show
            plt.show = lambda *a, **k: None
        except Exception:
            _orig_show = None
        figures = simulator.plot_comprehensive_results(results)
        # Show plots in the Plots tab (scrollable container)
        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=self.plots_container)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(2, 6))
        # Restore plt.show
        try:
            if _orig_show is not None:
                import matplotlib.pyplot as plt  # type: ignore
                plt.show = _orig_show
        except Exception:
            pass
        # Build and show summary text
        try:
            summary = self._build_summary_text(results)
            self._set_summary_text(summary)
        except Exception:
            pass
        self._set_status(f"Completed in {elapsed_s:.2f}s")
        self._set_running(False)

    def _on_simulation_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")
        self._set_running(False)

    def clear_plots(self) -> None:
        for widget in self.plots_container.winfo_children():
            widget.destroy()
        self._set_summary_text("")

    def _set_running(self, running: bool) -> None:
        self.running = running
        try:
            if running:
                self.progress.start(10)
            else:
                self.progress.stop()
        except Exception:
            pass
        # Enable/disable controls
        new_state = "disabled" if running else "!disabled"
        for w in (self.preset_cb, self.mode_cb, self.users_spin, self.randomize_cb, self.run_btn, self.clear_btn):
            try:
                if new_state == "disabled":
                    w.state(["disabled"])
                else:
                    w.state(["!disabled"])
            except Exception:
                try:
                    w.configure(state=("disabled" if running else "normal"))
                except Exception:
                    pass

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    # -------------------- Styling and tooltips --------------------
    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            # Use a modern, flat theme when available
            style.theme_use("clam")
        except Exception:
            pass

        PRIMARY = "#1f77b4"  # matplotlib blue
        ACCENT = "#ff7f0e"   # matplotlib orange
        BG = "#f7f8fb"
        FG = "#222222"
        SUB = "#5a5a5a"

        style.configure("Header.TFrame", background=BG)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=PRIMARY, background=BG)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground=SUB, background=BG)
        style.configure("Status.TLabel", font=("Segoe UI", 9), foreground=SUB)

        style.configure("Card.TLabelframe", background="#ffffff")
        style.configure("Card.TLabelframe.Label", foreground=PRIMARY)

        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=6, foreground="#ffffff", background=PRIMARY)
        style.map(
            "Accent.TButton",
            background=[("active", "#16629a"), ("disabled", "#9bbbd3")],
            foreground=[("disabled", "#eaeaea")],
        )

    def _add_tooltip(self, widget, text: str) -> None:
        tip = _ToolTip(widget, text)
        # Keep a reference to avoid GC
        if not hasattr(self, "_tooltips"):
            self._tooltips = []
        self._tooltips.append(tip)

    # -------------------- Results summary helpers --------------------
    def _build_summary_text(self, simulation_results) -> str:
        import numpy as np  # local import to avoid top-level dependency surprises
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

    def _poll_queue(self) -> None:
        if self._closing:
            return
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                kind = msg[0]
                if kind == "error":
                    _, message = msg
                    self._on_simulation_error(message)
                elif kind == "done":
                    _, simulator, results, elapsed = msg
                    self._on_simulation_done(simulator, results, elapsed)
        except queue.Empty:
            pass
        # Schedule next poll
        self.root.after(100, self._poll_queue)

    def _on_close(self) -> None:
        # Mark closing to stop scheduling
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    root = tk.Tk()
    gui = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

