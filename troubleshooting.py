# troubleshooting.py
"""
Hệ thống troubleshooting và performance monitoring
"""
import psutil
import time
import warnings
import numpy as np

class SystemMonitor:
    """Monitor hiệu suất hệ thống trong quá trình simulation"""
    
    def __init__(self):
        self.start_time = None
        self.memory_snapshots = []
        self.cpu_snapshots = []
    
    def start_monitoring(self):
        """Bắt đầu monitoring"""
        self.start_time = time.time()
        self.memory_snapshots = []
        self.cpu_snapshots = []
        print("Started system monitoring")
    
    def take_snapshot(self, label=""):
        """Chụp snapshot hiệu suất"""
        if self.start_time is None:
            self.start_monitoring()
        
        elapsed = time.time() - self.start_time
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        self.memory_snapshots.append((elapsed, memory_usage, label))
        self.cpu_snapshots.append((elapsed, cpu_usage, label))
        
        if label:
            print(f"[{elapsed:.1f}s] {label}: RAM {memory_usage:.1f}%, CPU {cpu_usage:.1f}%")
    
    def get_summary(self):
        """Lấy summary performance"""
        if not self.memory_snapshots:
            return "No monitoring data available"
        
        max_memory = max(snapshot[1] for snapshot in self.memory_snapshots)
        avg_cpu = np.mean([snapshot[1] for snapshot in self.cpu_snapshots])
        total_time = self.memory_snapshots[-1][0] if self.memory_snapshots else 0
        
        return {
            'total_time': total_time,
            'max_memory_usage': max_memory,
            'avg_cpu_usage': avg_cpu
        }

class TroubleshootingTool:
    """Công cụ chẩn đoán và khắc phục sự cố"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
    
    def diagnose_performance_issues(self, simulator, config):
        """Chẩn đoán vấn đề hiệu suất"""
        issues = []
        recommendations = []
        
        # Kiểm tra memory availability
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 2:
            issues.append(f"Low available memory: {available_memory_gb:.1f} GB")
            recommendations.append("Reduce num_realizations or use smaller arrays")
        
        # Kiểm tra array size vs memory
        total_elements = simulator.params.M * simulator.params.N
        if total_elements > 2048:  # 64x32 hoặc lớn hơn
            estimated_memory_mb = total_elements * config.num_realizations * 8 / (1024**2)  # Rough estimate
            if estimated_memory_mb > 1000:  # > 1GB
                issues.append(f"Large memory requirement estimated: {estimated_memory_mb:.0f} MB")
                recommendations.append("Consider using smaller group sizes or fewer realizations")
        
        # Kiểm tra CPU cores
        cpu_count = psutil.cpu_count()
        if config.n_jobs == -1:
            recommended_jobs = min(cpu_count, 8)  # Cap at 8 for memory reasons
            if cpu_count > recommended_jobs:
                recommendations.append(f"Consider setting n_jobs={recommended_jobs} instead of -1")
        
        # Kiểm tra disk space
        disk_usage = psutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024**3)
        if free_space_gb < 1:
            issues.append(f"Low disk space: {free_space_gb:.1f} GB")
            recommendations.append("Clean up old results or move to larger disk")
        
        return issues, recommendations
    
    def optimize_config_for_system(self, config):
        """Tối ưu config dựa trên capabilities của hệ thống"""
        optimized_config = config
        
        # Memory-based optimization
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 4:
            # Low memory system
            optimized_config.num_realizations = min(config.num_realizations, 20)
            optimized_config.n_jobs = min(config.n_jobs if config.n_jobs > 0 else 4, 2)
            print("Applied low-memory optimizations")
        elif available_memory_gb < 8:
            # Medium memory system
            optimized_config.num_realizations = min(config.num_realizations, 50)
            optimized_config.n_jobs = min(config.n_jobs if config.n_jobs > 0 else psutil.cpu_count(), 4)
            print("Applied medium-memory optimizations")
        
        return optimized_config
    
    def run_performance_test(self, simulator, test_positions=None):
        """Chạy performance test để đo baseline hiệu suất"""
        if test_positions is None:
            test_positions = [(0, 0, 50), (5, 5, 30), (-3, 7, 80)]
        
        print("Running performance benchmark...")
        self.monitor.start_monitoring()
        
        # Test each method
        methods = [
            ("Far-field", lambda: simulator.far_field_beamforming(test_positions)),
            ("Average Phase", lambda: simulator.average_phase_beamforming_optimized(test_positions)),
            ("Grouped (2x2)", lambda: simulator.grouped_beamforming_optimized(test_positions, 2)),
            ("Grouped (4x4)", lambda: simulator.grouped_beamforming_optimized(test_positions, 4)),
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            self.monitor.take_snapshot(f"Starting {method_name}")
            
            # Time the method
            start_time = time.perf_counter()
            beta = method_func()
            elapsed = time.perf_counter() - start_time
            
            # Compute AAG for reference
            aag, mag = simulator.compute_aag_mag_batch(beta, test_positions)
            
            results[method_name] = {
                'time': elapsed,
                'aag': aag,
                'mag': mag
            }
            
            self.monitor.take_snapshot(f"Finished {method_name}")
            print(f"  {method_name}: {elapsed:.4f}s, AAG={aag:.1f}")
        
        system_summary = self.monitor.get_summary()
        print(f"\nSystem summary: {system_summary}")
        
        return results, system_summary

def check_common_issues():
    """Kiểm tra các vấn đề phổ biến"""
    print("=== CHECKING COMMON ISSUES ===")
    
    issues_found = []
    
    # 1. NumPy performance
    print("Checking NumPy performance...")
    start = time.time()
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    numpy_time = time.time() - start
    
    if numpy_time > 2.0:
        issues_found.append(f"Slow NumPy performance: {numpy_time:.2f}s for 1000x1000 matmul")
        print(f"  ❌ NumPy slow: {numpy_time:.2f}s")
        print("  💡 Consider installing OpenBLAS: pip install numpy[openblas]")
    else:
        print(f"  ✓ NumPy performance good: {numpy_time:.2f}s")
    
    # 2. Memory check
    print("\nChecking memory...")
    memory = psutil.virtual_memory()
    if memory.available < 2 * 1024**3:  # Less than 2GB
        issues_found.append(f"Low available memory: {memory.available/(1024**3):.1f} GB")
        print(f"  ❌ Low memory: {memory.available/(1024**3):.1f} GB available")
        print("  💡 Close other applications or reduce simulation parameters")
    else:
        print(f"  ✓ Memory adequate: {memory.available/(1024**3):.1f} GB available")
    
    # 3. CPU check
    print("\nChecking CPU...")
    cpu_count = psutil.cpu_count()
    if cpu_count < 2:
        issues_found.append("Single-core CPU detected")
        print(f"  ⚠ Single-core CPU: parallel processing limited")
    else:
        print(f"  ✓ Multi-core CPU: {cpu_count} cores available")
    
    # 4. Disk space
    print("\nChecking disk space...")
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    if free_gb < 1:
        issues_found.append(f"Low disk space: {free_gb:.1f} GB")
        print(f"  ❌ Low disk space: {free_gb:.1f} GB")
        print("  💡 Clean up old files or change working directory")
    else:
        print(f"  ✓ Disk space adequate: {free_gb:.1f} GB free")
    
    if not issues_found:
        print("\n🎉 No major issues detected!")
    else:
        print(f"\n⚠ Found {len(issues_found)} potential issues:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    return issues_found

if __name__ == "__main__":
    check_common_issues() 
