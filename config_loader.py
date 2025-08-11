 # config_loader.py
import json
import numpy as np
from optimized_nearfield_system import SystemParameters, SimulationConfig

class ConfigManager:
    """Quản lý cấu hình từ files"""
    
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.system_configs = self.load_system_configs()
        self.sim_configs = self.load_simulation_configs()
    
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
        """Tạo SystemParameters từ config name"""
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
        """Tạo SimulationConfig từ config name"""
        if config_name not in self.sim_configs:
            raise ValueError(f"Unknown simulation config: {config_name}")
        
        config = self.sim_configs[config_name]
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
