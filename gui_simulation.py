import tkinter as tk
from tkinter import ttk
from optimized_nearfield_system import create_system_with_presets, create_simulation_config
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimulationGUI:
    """Simple GUI wrapper to run simulations and display plots."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Near-Field Simulation GUI")

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W)
        self.preset_var = tk.StringVar(value="standard")
        ttk.Entry(control_frame, textvariable=self.preset_var, width=20).grid(row=0, column=1)

        ttk.Label(control_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="fast")
        ttk.Entry(control_frame, textvariable=self.mode_var, width=20).grid(row=1, column=1)

        ttk.Label(control_frame, text="Users:").grid(row=2, column=0, sticky=tk.W)
        self.users_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.users_var, width=20).grid(row=2, column=1)

        run_btn = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        run_btn.grid(row=3, column=0, columnspan=2, pady=5)

        # Area for plots
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def run_simulation(self) -> None:
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
