# test_installation.py
"""
Script kiểm tra cài đặt và chức năng cơ bản
"""
import sys
import time
import numpy as np

def test_basic_imports():
    """Test import các library cần thiết"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy
        import joblib
        from concurrent.futures import ProcessPoolExecutor
        print("✓ Tất cả imports thành công")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_numpy_performance():
    """Test hiệu suất NumPy"""
    print("\nTesting NumPy performance...")
    
    # Matrix multiplication test
    size = 1000
    start_time = time.time()
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    elapsed = time.time() - start_time
    
    print(f"✓ Matrix multiplication ({size}x{size}): {elapsed:.3f}s")
    
    if elapsed > 5.0:
        print("⚠ Hiệu suất NumPy chậm, có thể cần cài đặt BLAS/LAPACK optimized")
    
    return elapsed < 10.0

def test_parallel_processing():
    """Test parallel processing capability"""
    print("\nTesting parallel processing...")
    
    def dummy_task(x):
        return x**2
    
    # Test với ProcessPoolExecutor
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(dummy_task, range(100)))
    elapsed = time.time() - start_time
    
    print(f"✓ Parallel processing test: {elapsed:.3f}s")
    return True

if __name__ == "__main__":
    print("=== KIỂM TRA CÀI ĐẶT NEARFIELD BEAMFORMING ===")
    
    all_passed = True
    all_passed &= test_basic_imports()
    all_passed &= test_numpy_performance()
    all_passed &= test_parallel_processing()
    
    if all_passed:
        print("\n🎉 Tất cả tests passed! Hệ thống sẵn sàng sử dụng.")
    else:
        print("\n❌ Một số tests failed. Kiểm tra lại cài đặt.")
        sys.exit(1)