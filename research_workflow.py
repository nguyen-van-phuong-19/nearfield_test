 # research_workflow.py
"""
Workflow chuẩn cho việc thực hiện nghiên cứu với simulator
"""
import os
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

from optimized_nearfield_system import OptimizedNearFieldBeamformingSimulator
from config_loader import ConfigManager

class ResearchWorkflow:
    """
    Class quản lý workflow nghiên cứu hoàn chỉnh
    """
    
    def __init__(self, project_name, base_dir="research_projects"):
        self.project_name = project_name
        self.project_dir = f"{base_dir}/{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_project_directory()
        self.config_manager = ConfigManager()
        
        # Logging
        self.log_file = f"{self.project_dir}/simulation.log"
        self.log("Initialized research project: " + project_name)
    
    def setup_project_directory(self):
        """Tạo cấu trúc thư mục project"""
        dirs_to_create = [
            self.project_dir,
            f"{self.project_dir}/results",
            f"{self.project_dir}/plots",
            f"{self.project_dir}/data",
            f"{self.project_dir}/config",
            f"{self.project_dir}/logs"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Project directory created: {self.project_dir}")
    
    def log(self, message):
        """Ghi log với timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_parameter_study(self, param_ranges, base_config="standard"):
        """
        Chạy parameter study với nhiều giá trị khác nhau
        
        Args:
            param_ranges: dict với format {'param_name': [values_to_test]}
            base_config: tên config cơ sở
        """
        self.log("Starting parameter study")
        self.log(f"Base config: {base_config}")
        self.log(f"Parameters to study: {list(param_ranges.keys())}")
        
        results = {}
        total_runs = sum(len(values) for values in param_ranges.values())
        current_run = 0
        
        for param_name, param_values in param_ranges.items():
            param_results = {}
            
            for param_value in param_values:
                current_run += 1
                self.log(f"Run {current_run}/{total_runs}: {param_name}={param_value}")
                
                # Tạo modified system parameters
                base_params = self.config_manager.get_system_params(base_config)
                modified_params = self.modify_param(base_params, param_name, param_value)
                
                # Chạy simulation
                simulator = OptimizedNearFieldBeamformingSimulator(modified_params)
                config = self.config_manager.get_simulation_config("quick_test")
                
                start_time = time.time()
                sim_results = simulator.run_optimized_simulation(config)
                elapsed_time = time.time() - start_time
                
                # Lưu results
                param_results[param_value] = {
                    'simulation_results': sim_results,
                    'elapsed_time': elapsed_time,
                    'system_params': modified_params
                }
                
                self.log(f"Completed in {elapsed_time:.2f}s")
            
            results[param_name] = param_results
        
        # Lưu toàn bộ kết quả
        self.save_results(results, "parameter_study_results.pkl")
        self.plot_parameter_study_results(results)
        
        return results
    
    def modify_param(self, base_params, param_name, new_value):
        """Modify một parameter trong SystemParameters"""
        from dataclasses import replace
        return replace(base_params, **{param_name: new_value})
    
    def run_comparative_study(self, system_configs, sim_config="standard_sim"):
        """
        Chạy comparative study giữa các system configurations
        """
        self.log("Starting comparative study")
        self.log(f"System configs: {system_configs}")
        
        results = {}
        
        for i, config_name in enumerate(system_configs):
            self.log(f"Running config {i+1}/{len(system_configs)}: {config_name}")
            
            # Setup simulator
            system_params = self.config_manager.get_system_params(config_name)
            simulator = OptimizedNearFieldBeamformingSimulator(system_params)
            
            # Setup simulation config
            config = self.config_manager.get_simulation_config(sim_config)
            
            # Run simulation
            start_time = time.time()
            sim_results = simulator.run_optimized_simulation(config)
            elapsed_time = time.time() - start_time
            
            results[config_name] = {
                'simulation_results': sim_results,
                'elapsed_time': elapsed_time,
                'system_params': system_params
            }
            
            self.log(f"Config {config_name} completed in {elapsed_time:.2f}s")
        
        # Generate comparative plots
        self.plot_comparative_results(results)
        self.save_results(results, "comparative_study_results.pkl")
        
        return results
    
    def plot_parameter_study_results(self, results):
        """Vẽ kết quả parameter study"""
        for param_name, param_results in results.items():
            param_values = list(param_results.keys())
            
            # Extract metrics
            aag_values = []
            amag_values = []
            
            for param_val in param_values:
                sim_results = param_results[param_val]['simulation_results']
                all_results = sim_results['all_results']
                
                # Get results cho user scenario đầu tiên
                first_user_key = list(all_results.keys())[0]
                user_results = all_results[first_user_key]['results']
                
                # Tính mean AAG và AMAG cho Grouped (2x2)
                if 'Grouped (2x2)' in user_results:
                    aag = np.mean(user_results['Grouped (2x2)']['aag'])
                    amag = np.mean(user_results['Grouped (2x2)']['mag'])
                else:
                    aag = amag = 0
                
                aag_values.append(aag)
                amag_values.append(amag)
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(param_values, aag_values, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('Average Array Gain')
            ax1.set_title(f'AAG vs {param_name}')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(param_values, amag_values, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Average Minimum Array Gain')
            ax2.set_title(f'AMAG vs {param_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.project_dir}/plots/param_study_{param_name}.png", dpi=300)
            plt.show()
    
    def plot_comparative_results(self, results):
        """Vẽ kết quả comparative study"""
        config_names = list(results.keys())
        
        # Extract performance metrics
        aag_by_config = {}
        amag_by_config = {}
        time_by_config = {}
        
        for config_name in config_names:
            sim_results = results[config_name]['simulation_results']
            all_results = sim_results['all_results']
            
            # Get results cho user scenario đầu tiên
            first_user_key = list(all_results.keys())[0]
            user_results = all_results[first_user_key]['results']
            
            aag_by_config[config_name] = {}
            amag_by_config[config_name] = {}
            
            for method in user_results.keys():
                aag_by_config[config_name][method] = np.mean(user_results[method]['aag'])
                amag_by_config[config_name][method] = np.mean(user_results[method]['mag'])
            
            time_by_config[config_name] = results[config_name]['elapsed_time']
        
        # Plot comparison
        methods = list(user_results.keys())
        x = np.arange(len(methods))
        width = 0.25
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # AAG comparison
        for i, config_name in enumerate(config_names):
            aag_vals = [aag_by_config[config_name][method] for method in methods]
            ax1.bar(x + i*width, aag_vals, width, label=config_name, alpha=0.8)
        
        ax1.set_xlabel('Beamforming Methods')
        ax1.set_ylabel('Average Array Gain')
        ax1.set_title('AAG Comparison Across Configurations')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AMAG comparison
        for i, config_name in enumerate(config_names):
            amag_vals = [amag_by_config[config_name][method] for method in methods]
            ax2.bar(x + i*width, amag_vals, width, label=config_name, alpha=0.8)
        
        ax2.set_xlabel('Beamforming Methods')
        ax2.set_ylabel('Average Minimum Array Gain')
        ax2.set_title('AMAG Comparison Across Configurations')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(methods, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Execution time comparison
        configs = list(time_by_config.keys())
        times = list(time_by_config.values())
        ax3.bar(configs, times, color='skyblue', alpha=0.8)
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Simulation Time Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # System parameters comparison (example: array size)
        array_sizes = []
        for config_name in config_names:
            params = results[config_name]['system_params']
            array_sizes.append(params.M * params.N)
        
        ax4.bar(configs, array_sizes, color='lightcoral', alpha=0.8)
        ax4.set_ylabel('Total Array Elements')
        ax4.set_title('Array Size Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.project_dir}/plots/comparative_study.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results, filename):
        """Lưu results vào file"""
        filepath = f"{self.project_dir}/data/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        self.log(f"Results saved to: {filepath}")
    
    def generate_report(self, results):
        """Tạo báo cáo tóm tắt"""
        report_path = f"{self.project_dir}/research_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Research Report: {self.project_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Project: {self.project_name}\n")
            f.write(f"- Total configurations tested: {len(results)}\n")
            
            f.write("\n## Results Summary\n")
            for config_name, result in results.items():
                if 'elapsed_time' in result:
                    f.write(f"- {config_name}: Completed in {result['elapsed_time']:.2f}s\n")
            
            f.write(f"\n## Files Generated\n")
            f.write(f"- Project directory: {self.project_dir}\n")
            f.write("- Plots: Available in plots/ directory\n")
            f.write("- Data: Available in data/ directory\n")
            f.write("- Logs: Available in simulation.log\n")
        
        self.log(f"Report generated: {report_path}")
        return report_path

# Usage example
def example_research_workflow():
    """Ví dụ sử dụng research workflow"""
    
    # Khởi tạo workflow
    workflow = ResearchWorkflow("LIS_Size_Study")
    
    # Parameter study: ảnh hưởng của kích thước array
    param_ranges = {
        'M': [16, 32, 48],  # Array height
        'N': [16, 32, 48],  # Array width
    }
    
    results = workflow.run_parameter_study(param_ranges)
    workflow.generate_report(results)
    
    return workflow, results
