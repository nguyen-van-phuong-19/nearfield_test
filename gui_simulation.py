import tkinter as tk
from tkinter import ttk
from random_params import random_basic_params
from optimized_nearfield_system import create_system_with_presets, create_simulation_config
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimulationGUI:
    """Simple GUI wrapper to run simulations and display plots."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        # Shorter, balanced window title
        root.title("Near-Field Simulator")
        # Set a harmonious default size and minimums
        root.geometry("1200x740")
        root.minsize(900, 580)

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Split layout: left controls, right plots
        paned = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Input controls (left panel)
        control_frame = ttk.Frame(paned, padding=(6, 6))
        # Make inputs expand nicely
        control_frame.grid_columnconfigure(0, weight=0)
        control_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, 4))
        self.preset_var = tk.StringVar(value="standard")
        ttk.Entry(control_frame, textvariable=self.preset_var).grid(row=0, column=1, sticky="ew", pady=(0, 4))

        ttk.Label(control_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.mode_var = tk.StringVar(value="fast")
        ttk.Entry(control_frame, textvariable=self.mode_var).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(control_frame, text="Users:").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.users_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.users_var).grid(row=2, column=1, sticky="ew", pady=4)

        # Randomize option
        self.randomize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Randomize on Run", variable=self.randomize_var).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0)
        )

        run_btn = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        run_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # Area for plots (right panel)
        self.plot_frame = ttk.Frame(paned)

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

    def run_simulation(self) -> None:
        # Optionally randomize inputs while respecting valid choices
        if self.randomize_var.get():
            rp = random_basic_params()
            # Reflect randomized values in the UI
            self.preset_var.set(rp["preset"])
            self.mode_var.set(rp["mode"])
            self.users_var.set(str(rp["users"]))

        preset = self.preset_var.get() or "standard"
        mode = self.mode_var.get() or "fast"
        users = self.users_var.get().strip()

        simulator = create_system_with_presets(preset)
        config = create_simulation_config(mode)
        if users:
            try:
                config.num_users_list = [int(users)]
            except ValueError:
                pass

        results = simulator.run_optimized_simulation(config)
        figures = simulator.plot_comprehensive_results(results)

        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def main() -> None:
    root = tk.Tk()
    gui = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
