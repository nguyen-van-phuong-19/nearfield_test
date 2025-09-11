# NF-BF-LIS-UAV
Near-Field Multi-beamforming for Multi-User LIS-Aided UAV Relay Systems
Show Image
Show Image
Show Image
Hệ thống mô phỏng tiên tiến cho Near-Field Multi-beamforming trong môi trường LIS-UAV, được phát triển dựa trên nghiên cứu "Near-Field Multi-beamforming for Multi-User LIS-Aided UAV Relay Systems" với các cải tiến về hiệu suất và khả năng mở rộng.
📋 Mục lục

Tổng quan
Tính năng chính
Cài đặt
Hướng dẫn sử dụng
Ví dụ
Cấu trúc dự án
API Reference
Benchmarks
Đóng góp
Roadmap
Liên hệ

🌟 Tổng quan
Dự án này cung cấp một framework mô phỏng hoàn chỉnh cho việc nghiên cứu và phát triển các kỹ thuật beamforming trong vùng near-field cho hệ thống Large Intelligent Surface (LIS) hỗ trợ bởi UAV relay. Hệ thống được tối ưu hóa cho hiệu suất cao với khả năng parallel processing và các thuật toán beamforming tiên tiến.
🎯 Mục tiêu chính

Nghiên cứu khoa học: Cung cấp platform để nghiên cứu near-field beamforming
Mô phỏng chính xác: Triển khai các thuật toán dựa trên lý thuyết vững chắc
Hiệu suất cao: Tối ưu hóa cho tính toán song song và xử lý dữ liệu lớn
Mở rộng linh hoạt: Hỗ trợ tích hợp NOMA, RSMA và các kỹ thuật mới

🔬 Ứng dụng

Nghiên cứu 6G wireless communications
Thiết kế hệ thống LIS/RIS
Tối ưu hóa UAV communications
Phát triển thuật toán beamforming mới

✨ Tính năng chính
🚀 Core Features

Multiple Beamforming Methods: Far-field, Average Phase, Grouped Near-Field
High-Performance Computing: Parallel processing với joblib và multiprocessing
Comprehensive Metrics: AAG, MAG, AMAG với statistical analysis
Flexible Configuration: JSON-based config system với presets
Advanced Visualization: Professional plots với customizable styling

🔧 Technical Features

Vectorized Computations: NumPy optimized cho large arrays
Memory Management: Smart caching và garbage collection
Error Handling: Robust error handling với fallback methods
Progress Monitoring: Real-time progress tracking và system monitoring
Result Management: Automated saving/loading với pickle support

📊 Analysis Tools

Parameter Studies: Automated parameter sweeps
Comparative Analysis: Multi-configuration comparisons
Statistical Analysis: CDF analysis và performance metrics
Performance Benchmarking: System performance evaluation

🛠 Cài đặt
Yêu cầu hệ thống
bash# Minimum requirements
Python >= 3.8
RAM >= 4GB (8GB+ khuyến nghị cho simulations lớn)
CPU cores >= 2 (4+ khuyến nghị)
Disk space >= 2GB cho results
Cài đặt nhanh
bash# Clone repository (nếu có)
git clone https://github.com/your-repo/nearfield-beamforming.git
cd nearfield-beamforming

# Tạo virtual environment (khuyến nghị)
python -m venv nearfield_env
source nearfield_env/bin/activate  # Linux/Mac
# nearfield_env\Scripts\activate    # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt thủ công
pip install numpy>=1.20.0 matplotlib>=3.3.0 scipy>=1.7.0 joblib>=1.0.0
Kiểm tra cài đặt
bash# Chạy verification script
python verification_script.py

# Hoặc test installation
python -c "
from optimized_nearfield_system import create_system_with_presets
simulator = create_system_with_presets('small_test')
print('✅ Installation successful!')
"
🚀 Hướng dẫn sử dụng
Quick Start
pythonfrom optimized_nearfield_system import (
    create_system_with_presets, 
    create_simulation_config
)

# 1. Tạo simulator với preset
simulator = create_system_with_presets("standard")  # 32x32 LIS, 6GHz

# 2. Tạo cấu hình simulation
config = create_simulation_config("fast")  # Quick test config

# 3. Chạy simulation
results = simulator.run_optimized_simulation(config)

# 4. Vẽ kết quả
simulator.plot_comprehensive_results(results, save_dir="my_results")

# 5. Lưu kết quả
simulator.save_results(results, "simulation_results.pkl")

### Chạy mô phỏng tương tác

Bạn có thể chạy `run_simulation.py` để thiết lập tham số trước khi mô phỏng:

```bash
python run_simulation.py
```

Chương trình sẽ hiển thị các lựa chọn:

1. Dùng tham số đã lưu  
2. Dùng tham số mặc định  
3. Tùy chỉnh tham số theo hướng dẫn

Các tham số tùy chỉnh được lưu trong `config/user_params.json` và tự động sử dụng cho lần chạy sau nếu bạn không nhập giá trị mới.

### Giao diện GUI

Nếu muốn chạy mô phỏng và xem kết quả trực tiếp trên giao diện đồ họa, sử dụng:

```bash
python gui_simulation.py
```

Ứng dụng GUI cho phép nhập nhanh các tham số mô phỏng và hiển thị ngay các đồ thị kết quả.
Advanced Usage
pythonfrom optimized_nearfield_system import (
    SystemParameters,
    SimulationConfig,
    OptimizedNearFieldBeamformingSimulator
)
import numpy as np

# Custom system parameters
params = SystemParameters(
    M=64,                    # 64x64 LIS array
    N=64,
    lambda_=0.01,           # mmWave at 30GHz
    frequency=30e9
)

# Custom simulation config  
config = SimulationConfig(
    num_users_list=[5, 10, 15],
    z_values=np.logspace(0, 2, 20),  # Logarithmic spacing
    num_realizations=100,
    x_range=(-15, 15),
    y_range=(-15, 15),
    n_jobs=8                 # 8 parallel processes
)

# Run simulation
simulator = OptimizedNearFieldBeamformingSimulator(params)
results = simulator.run_optimized_simulation(config)
💡 Ví dụ
Ví dụ 1: So sánh cấu hình LIS khác nhau
pythonfrom optimized_nearfield_system import create_system_with_presets
import matplotlib.pyplot as plt

# Test different LIS sizes
configurations = ["small_test", "standard", "large_array"]
results = {}

for config_name in configurations:
    print(f"Testing {config_name}...")
    simulator = create_system_with_presets(config_name)
    
    # Test specific scenario
    user_positions = [(5, 5, 50), (-3, 7, 30), (0, 0, 80)]
    beta = simulator.grouped_beamforming_optimized(user_positions, group_size=2)
    aag, mag = simulator.compute_aag_mag_batch(beta, user_positions)
    
    results[config_name] = {"AAG": aag, "MAG": mag}
    print(f"  AAG: {aag:.1f}, MAG: {mag:.1f}")

# Plot comparison
configs = list(results.keys())
aag_values = [results[config]["AAG"] for config in configs]
mag_values = [results[config]["MAG"] for config in configs]

plt.figure(figsize=(10, 6))
x = range(len(configs))
plt.bar(x, aag_values, alpha=0.7, label='AAG')
plt.bar(x, mag_values, alpha=0.7, label='MAG')
plt.xlabel('LIS Configuration')
plt.ylabel('Array Gain')
plt.title('Performance Comparison Across LIS Configurations')
plt.xticks(x, configs)
plt.legend()
plt.show()
Ví dụ 2: Parameter Study
pythonfrom optimized_nearfield_system import SystemParameters, OptimizedNearFieldBeamformingSimulator
import numpy as np

# Study effect of wavelength
wavelengths = [0.01, 0.025, 0.05, 0.1]  # Different frequencies
results = {}

for lambda_ in wavelengths:
    params = SystemParameters(M=32, N=32, lambda_=lambda_)
    simulator = OptimizedNearFieldBeamformingSimulator(params)
    
    # Fixed test scenario
    positions = [(0, 0, 50)]  # Single user on boresight
    beta = simulator.grouped_beamforming_optimized(positions, group_size=4)
    aag, mag = simulator.compute_aag_mag_batch(beta, positions)
    
    freq_ghz = 3e8 / lambda_ / 1e9
    results[freq_ghz] = {"AAG": aag, "MAG": mag, "d_F1": simulator.d_F1}
    
    print(f"λ={lambda_}m ({freq_ghz:.1f}GHz): AAG={aag:.1f}, d_F1={simulator.d_F1:.1f}m")
Ví dụ 3: Demo script tương tác
python# Chạy demo tương tác
python demo_script.py

# Chạy một demo cụ thể bằng tùy chọn --demo
python demo_script.py --demo basic      # Demo tính năng cơ bản
python demo_script.py --demo gui        # Kiểm tra lỗi GUI
python demo_script.py --demo all        # Chạy toàn bộ demos
📁 Cấu trúc dự án
nearfield-beamforming/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── optimized_nearfield_system.py      # Core simulator
├── demo_script.py                      # Interactive demos
├── verification_script.py             # System verification
├── config/                            # Configuration files
│   ├── system_params.json            # System presets
│   └── simulation_configs.json       # Simulation presets
├── examples/                          # Example scripts
│   ├── basic_usage.py
│   ├── parameter_study.py
│   └── performance_comparison.py
├── docs/                              # Documentation
│   ├── api_reference.md
│   ├── theory_background.md
│   └── troubleshooting.md
├── tests/                             # Unit tests
│   ├── test_simulator.py
│   ├── test_beamforming.py
│   └── test_performance.py
└── results/                           # Simulation results
    ├── plots/
    └── data/
📖 API Reference
Core Classes
OptimizedNearFieldBeamformingSimulator
Main simulator class cho near-field beamforming.
pythonclass OptimizedNearFieldBeamformingSimulator:
    def __init__(self, params: SystemParameters)
    def compute_array_gain_optimized(self, beta, user_pos) -> float
    def grouped_beamforming_optimized(self, positions, group_size) -> np.ndarray
    def run_optimized_simulation(self, config) -> Dict
    def plot_comprehensive_results(self, results, save_dir=None)
SystemParameters
python@dataclass
class SystemParameters:
    M: int = 32                    # LIS rows
    N: int = 32                    # LIS columns  
    lambda_: float = 0.05          # Wavelength (m)
    frequency: float = 6e9         # Frequency (Hz)
SimulationConfig
python@dataclass 
class SimulationConfig:
    num_users_list: List[int] = None      # User counts to test
    z_values: np.ndarray = None           # Distance points
    num_realizations: int = 100           # Monte Carlo runs
    x_range: Tuple = (-10.0, 10.0)       # User x positions
    y_range: Tuple = (-10.0, 10.0)       # User y positions
    n_jobs: int = -1                      # Parallel jobs
Key Methods
Beamforming Methods
python# Far-field beamforming (baseline)
beta = simulator.far_field_beamforming(user_positions)

# Average phase method
beta = simulator.average_phase_beamforming_optimized(user_positions)

# Grouped near-field beamforming (recommended)
beta = simulator.grouped_beamforming_optimized(user_positions, group_size=2)
Performance Evaluation
python# Compute array gains
aag, mag = simulator.compute_aag_mag_batch(beta, user_positions)

# Individual user gains
gains = [simulator.compute_array_gain_optimized(beta, pos) for pos in user_positions]
Utility Functions
python# Create simulator with presets
simulator = create_system_with_presets("standard")  # or "mmwave", "large_array", "small_test"

# Create simulation config with modes
config = create_simulation_config("fast")  # or "standard", "comprehensive"
📊 Benchmarks
Performance Benchmarks (Intel i7-8700K, 16GB RAM)
ConfigurationArray SizeUsersGrouped (2x2) TimeGrouped (4x4) TimeAAG (2x2)AAG (4x4)Small Test16×16512ms8ms245198Standard32×321045ms28ms412348Large Array64×6420180ms95ms892756
Memory Usage
ConfigurationPeak RAM UsageRecommended RAMSmall Test0.5GB2GBStandard1.2GB4GBLarge Array3.5GB8GB
Scaling Performance
python# Example: Performance vs Array Size
array_sizes = [16, 32, 48, 64]
computation_times = [0.012, 0.045, 0.098, 0.180]  # seconds

# Roughly scales as O(N²) due to distance calculations
🤝 Đóng góp
Chúng tôi hoan nghênh mọi đóng góp! Xem CONTRIBUTING.md để biết chi tiết.
Development Setup
bash# Clone và setup development environment
git clone https://github.com/your-repo/nearfield-beamforming.git
cd nearfield-beamforming

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run verification
python verification_script.py
Contribution Guidelines

Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Add tests cho new functionality
Run verification script
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Create Pull Request

🗺 Roadmap
Phase 1 (Q1 2025) - ✅ Completed

 Core simulator implementation
 Parallel processing optimization
 Comprehensive testing framework
 Documentation và examples

Phase 2 (Q2 2025) - 🚧 In Progress

 GPU acceleration với CuPy
 NOMA integration
 Machine learning optimization
 Advanced channel models

Phase 3 (Q3 2025) - 📋 Planned

 RSMA support
 Digital twin integration
 Real-time optimization
 Hardware-in-the-loop testing

Phase 4 (Q4 2025) - 🔮 Future

 6G network integration
 Quantum optimization algorithms
 Edge computing deployment
 Commercial applications

📚 Tài liệu tham khảo
Nghiên cứu gốc
bibtex@article{nearfield_multibeam_2024,
    title={Near-Field Multi-Beamforming for Multi-User LIS-Aided UAV Relay Systems},
    author={Truong Anh Dung and Nguyen Thu Phuong and Pham Thanh Hiep and Le Hai Nam},
    journal={IEEE Conference},
    year={2024},
    pages={1--6}
}
Tài liệu kỹ thuật

Theory Background - Chi tiết lý thuyết toán học
API Reference - Đầy đủ API documentation
Troubleshooting Guide - Giải quyết sự cố
Performance Optimization - Tối ưu hiệu suất

🐛 Báo cáo lỗi
Nếu bạn phát hiện lỗi, vui lòng tạo issue với:

Mô tả chi tiết về lỗi
Steps to reproduce
Expected vs actual behavior
System information (OS, Python version, etc.)
Log files nếu có

❓ FAQ
Q: Tại sao Grouped (2x2) có hiệu suất tốt nhất?
A: Grouped (2x2) cung cấp balance tốt nhất giữa độ phân giải spatial và computational complexity, cho phép fine-grained phase control trong near-field.
Q: Làm thế nào để tối ưu hiệu suất cho arrays lớn?
A: Sử dụng fewer realizations, increase group size, enable GPU acceleration (trong future versions), và ensure sufficient RAM.
Q: Có thể sử dụng cho real-time applications không?
A: Hiện tại optimized cho offline simulations. Real-time optimization đang trong development roadmap.
Q: Làm sao integrate với NOMA/RSMA?
A: Framework đã chuẩn bị sẵn classes NOMAEnhancedSimulator và RSMAEnhancedSimulator. Full implementation trong Phase 2.
📧 Liên hệ

Author: TruongDzung681
Email: dungkhcnmt@gmail.com
Project Link: https://github.com/truongdzung681/NF-BF-LIS-UAV
Research Group: Advanced Wireless Communication Group

📄 License
Distributed under the MIT License. See LICENSE for more information.
🙏 Acknowledgments

Le Quy Don Technical University
Advanced Wireless Communication Group
Open source community contributors
NumPy, SciPy, Matplotlib development teams


⭐ Star this repository if it helped your research!
Made with ❤️ for the wireless communications research community