from optimized_nearfield_system import SystemParameters, OptimizedNearFieldBeamformingSimulator
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