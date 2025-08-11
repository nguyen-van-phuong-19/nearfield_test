 # batch_processor.py
"""
Batch processing cho multiple scenarios và automated experiments
"""
import itertools
import json
from pathlib import Path

class BatchProcessor:
    """Xử lý batch nhiều scenarios tự động"""
    
    def __init__(self, base_output_dir="batch_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
    def create_parameter_grid(self, param_dict):
        """
        Tạo grid tất cả combinations của parameters
        
        Args:
            param_dict: {'param_name': [list_of_values]}
        
        Returns:
            List of parameter combinations
        """
        keys = param_dict.keys()
        values = param_dict.values()
        combinations = []
        
        for combo in itertools.product(*values):
            param_set = dict(zip(keys, combo))
            combinations.append(param_set)
        
        return combinations
    
    def run_batch_simulation(self, param_grid, base_config="standard"):
        """
        Chạy batch simulation cho tất cả parameter combinations
        """
        total_runs = len(param_grid)
        print(f"Starting batch processing: {total_runs} parameter combinations")
        
        all_results = {}
        
        for i, params in enumerate(param_grid):
            print(f"\n--- Batch Run {i+1}/{total_runs} ---")
            print(f"Parameters: {params}")
            
            # Create unique identifier for this run
            run_id = "_".join([f"{k}{v}" for k, v in params.items()])
            
            try:
                # Setup simulator with modified parameters
                base_params = self.config_manager.get_system_params(base_config)
                modified_params = self.modify_multiple_params(base_params, params)
                
                simulator = OptimizedNearFieldBeamformingSimulator(modified_params)
                config = self.config_manager.get_simulation_config("quick_test")
                
                # Run simulation
                start_time = time.time()
                results = simulator.run_optimized_simulation(config)
                elapsed = time.time() - start_time
                
                # Store results
                all_results[run_id] = {
                    'parameters': params,
                    'results': results,
                    'elapsed_time': elapsed,
                    'status': 'success'
                }
                
                print(f"✓ Completed in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                all_results[run_id] = {
                    'parameters': params,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save batch results
        self.save_batch_results(all_results)
        self.generate_batch_summary(all_results)
        
        return all_results
