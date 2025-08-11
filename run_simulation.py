 from optimized_nearfield_system import create_system_with_presets, create_simulation_config

# Tạo simulator với preset chuẩn
simulator = create_system_with_presets("standard")

# Tạo cấu hình simulation nhanh
config = create_simulation_config("fast")

# Chạy simulation
results = simulator.run_optimized_simulation(config)

# Vẽ kết quả
simulator.plot_comprehensive_results(results, save_dir="my_results")
