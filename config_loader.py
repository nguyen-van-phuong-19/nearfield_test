 # config_loader.py
import json
import re
import numpy as np
import random
from optimized_nearfield_system import SystemParameters, SimulationConfig

class ConfigManager:
    """Quản lý cấu hình mô phỏng từ các file JSON."""

    def __init__(self, config_dir="config"):
        """Khởi tạo trình quản lý cấu hình.

        Args:
            config_dir: Thư mục chứa các file cấu hình.
        """
        self.config_dir = config_dir
        # Use comment-tolerant loaders to support // and /* */ in JSON files
        self.system_configs = self._load_system_configs_any()
        self.sim_configs = self._load_simulation_configs_any()

    # -------- comment-tolerant JSON helpers --------
    def _parse_json_with_comments(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # remove /* ... */
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        # strip // comments per line
        text = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
        # remove trailing commas before ] or }
        text = re.sub(r",(\s*[\]}])", r"\1", text)
        return json.loads(text)

    def _load_system_configs_any(self):
        try:
            return self._parse_json_with_comments(f"{self.config_dir}/system_params.json")
        except FileNotFoundError:
            print("Warning: system_params.json not found, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: failed to parse system_params.json: {e}; using defaults")
            return {}

    def _load_simulation_configs_any(self):
        try:
            return self._parse_json_with_comments(f"{self.config_dir}/simulation_configs.json")
        except FileNotFoundError:
            print("Warning: simulation_configs.json not found, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: failed to parse simulation_configs.json: {e}; using defaults")
            return {}
    
    def load_system_configs(self):
        """Load system parameters từ JSON"""
        try:
            with open(f"{self.config_dir}/system_params.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: system_params.json not found, using defaults")
            return {}
    
    def load_simulation_configs(self):
        """Load simulation configs từ JSON"""
        try:
            with open(f"{self.config_dir}/simulation_configs.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: simulation_configs.json not found, using defaults")
            return {}
    
    def get_system_params(self, config_name):
        """Tạo ``SystemParameters`` từ tên cấu hình.

        Args:
            config_name: Tên cấu hình hệ thống cần lấy.

        Returns:
            Đối tượng ``SystemParameters``.
        """
        if config_name not in self.system_configs:
            raise ValueError(f"Unknown system config: {config_name}")

        config = self.system_configs[config_name]
        return SystemParameters(
            M=config["M"],
            N=config["N"],
            lambda_=config["lambda"],
            frequency=config["frequency"]
        )
    
    def get_simulation_config(self, config_name):
        """Tạo ``SimulationConfig`` từ tên cấu hình.

        Args:
            config_name: Tên cấu hình mô phỏng cần lấy.

        Returns:
            Đối tượng ``SimulationConfig``.
        """
        if config_name not in self.sim_configs:
            raise ValueError(f"Unknown simulation config: {config_name}")

        config = self.sim_configs[config_name]

        # Optional randomization controls in JSON
        if config.get("randomize"):
            seed = config.get("seed")
            if seed is not None:
                try:
                    random.seed(int(seed))
                    np.random.seed(int(seed))
                except Exception:
                    pass

            # Users
            if "users_choices" in config:
                users = random.choice(config["users_choices"])  # type: ignore
            elif "users_range" in config:
                lo, hi = config["users_range"]
                users = random.randint(int(lo), int(hi))
            else:
                # fallback: pick a single from fixed list
                users = random.choice(config.get("num_users_list", [5]))

            # z range
            z_min = float(config.get("z_min", 0.1))
            z_max = float(config.get("z_max", 200.0))
            if "z_min_range" in config:
                a, b = config["z_min_range"]
                z_min = float(random.uniform(a, b))
            if "z_max_range" in config:
                a, b = config["z_max_range"]
                z_max = float(random.uniform(a, b))
            if z_max <= z_min:
                z_max = z_min + 1.0

            # number of z points
            if "num_z_points_choices" in config:
                num_z_points = int(random.choice(config["num_z_points_choices"]))
            elif "num_z_points_range" in config:
                a, b = config["num_z_points_range"]
                num_z_points = int(random.randint(int(a), int(b)))
            else:
                num_z_points = int(config.get("num_z_points", 10))
            num_z_points = max(2, num_z_points)

            # realizations per z
            if "num_realizations_choices" in config:
                num_realizations = int(random.choice(config["num_realizations_choices"]))
            elif "num_realizations_range" in config:
                a, b = config["num_realizations_range"]
                num_realizations = int(random.randint(int(a), int(b)))
            else:
                num_realizations = int(config.get("num_realizations", 20))
            num_realizations = max(1, num_realizations)

            # x/y ranges (use provided; ensure ordering)
            x_lo, x_hi = config.get("x_range", [-10, 10])
            y_lo, y_hi = config.get("y_range", [-10, 10])

            return SimulationConfig(
                num_users_list=[int(users)],
                z_values=np.linspace(z_min, z_max, num_z_points),
                num_realizations=num_realizations,
                x_range=(min(x_lo, x_hi), max(x_lo, x_hi)),
                y_range=(min(y_lo, y_hi), max(y_lo, y_hi)),
                n_jobs=int(config.get("n_jobs", -1)),
            )

        # Fixed (non-randomized) config path
        return SimulationConfig(
            num_users_list=config["num_users_list"],
            z_values=np.linspace(config["z_min"], config["z_max"], config["num_z_points"]),
            num_realizations=config["num_realizations"],
            x_range=tuple(config["x_range"]),
            y_range=tuple(config["y_range"])
        )
    
    def list_available_configs(self):
        """List tất cả configs có sẵn"""
        print("Available System Configs:")
        for name, config in self.system_configs.items():
            print(f"  {name}: {config.get('description', 'No description')}")
        
        print("\nAvailable Simulation Configs:")
        for name, config in self.sim_configs.items():
            print(f"  {name}: {len(config['num_users_list'])} user scenarios, "
                  f"{config['num_z_points']} z-points, {config['num_realizations']} realizations")
