import numpy as np
import matplotlib.pyplot as plt
import os
import time
import gc
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass
from scipy import signal
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')


def _process_z_batch(args):
    """Helper cho xử lý song song theo từng giá trị ``z``.

    Args:
        args: Tuple chứa ``(simulator, z, num_users, config)``

            * ``simulator`` – đối tượng mô phỏng đã khởi tạo
            * ``z`` – khoảng cách từ LIS tới UAV hiện tại (m)
            * ``num_users`` – số người dùng trong lần mô phỏng
            * ``config`` – cấu hình mô phỏng chung

    Returns:
        Tuple ``(z, z_results)`` nơi ``z_results`` là danh sách kết quả của
        từng realization tại khoảng cách ``z``.
    """

    simulator, z, num_users, config = args
    # Different random seed for each process to avoid correlated results
    np.random.seed(int(time.time() * 1000) % 2**32)
    z_results = []
    for _ in range(config.num_realizations):
        result = simulator.process_single_realization_optimized(z, num_users, config)
        z_results.append(result)
    return z, z_results

@dataclass
class SystemParameters:
    """Tham số hệ thống chuẩn hóa.

    Attributes:
        M: Số hàng phần tử của LIS.
        N: Số cột phần tử của LIS.
        lambda_: Bước sóng tín hiệu (m).
        frequency: Tần số sóng mang (Hz).
        d: Khoảng cách giữa các phần tử (m), mặc định ``λ/2``.
    """

    M: int = 32                    # Số hàng LIS
    N: int = 32                    # Số cột LIS
    lambda_: float = 0.05          # Bước sóng (m)
    frequency: float = 6e9         # Tần số (Hz)
    d: float = None                # Khoảng cách phần tử (sẽ tính = λ/2)
    
    def __post_init__(self):
        if self.d is None:
            self.d = self.lambda_ / 2

@dataclass
class SimulationConfig:
    """Cấu hình mô phỏng.

    Attributes:
        num_users_list: Danh sách số lượng người dùng cần mô phỏng.
        z_values: Mảng giá trị khoảng cách ``z`` (m) giữa LIS và người dùng/UAV.
        num_realizations: Số lần lặp lại cho mỗi giá trị ``z``.
        x_range: Khoảng giá trị toạ độ ``x`` của người dùng (m).
        y_range: Khoảng giá trị toạ độ ``y`` của người dùng (m).
        n_jobs: Số tiến trình chạy song song (-1 sử dụng toàn bộ CPU).
    """

    num_users_list: List[int] = None
    z_values: np.ndarray = None
    num_realizations: int = 100
    x_range: Tuple[float, float] = (-10.0, 10.0)
    y_range: Tuple[float, float] = (-10.0, 10.0)
    n_jobs: int = -1

    def __post_init__(self):
        if self.num_users_list is None:
            self.num_users_list = [5, 10]
        if self.z_values is None:
            self.z_values = np.linspace(0.1, 200, 20)

class OptimizedNearFieldBeamformingSimulator:
    """
    Simulator tối ưu cho Near-Field Beamforming với LIS-UAV system
    - Loại bỏ GWO để tăng tốc độ tính toán
    - Tối ưu hóa memory và parallel processing
    - Bổ sung tính năng mở rộng cho nghiên cứu tương lai
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self._initialize_system()
        self._precompute_positions()
        
    def _initialize_system(self):
        """Khởi tạo các thông số hệ thống"""
        # Tính các khoảng cách quan trọng
        self.D_tilde = (self.params.lambda_ / 2) * np.sqrt(
            (self.params.M - 1)**2 + (self.params.N - 1)**2
        )
        self.d_F1 = 2 * self.D_tilde**2 / self.params.lambda_  # ≈ 48.1m
        self.d_F2 = 8 * self.D_tilde**2 / self.params.lambda_  # ≈ 192.2m  
        self.d_0 = 8 * self.D_tilde  # ≈ 8.8m
        
        print(f"=== KHỞI TẠO HỆ THỐNG LIS-UAV ===")
        print(f"LIS size: {self.params.M}×{self.params.N}")
        print(f"Wavelength: {self.params.lambda_}m @ {self.params.frequency/1e9:.1f} GHz")
        print(f"Fresnel boundary: {self.d_0:.1f}m")
        print(f"Fraunhofer boundaries: {self.d_F1:.1f}m, {self.d_F2:.1f}m")
        
    def _precompute_positions(self):
        """Precompute vị trí các phần tử LIS (vectorized)"""
        m_indices = np.arange(self.params.M)
        n_indices = np.arange(self.params.N)
        self.m_grid, self.n_grid = np.meshgrid(m_indices, n_indices, indexing='ij')
        
        # Vị trí các phần tử
        self.x_mn = (self.m_grid - (self.params.M - 1) / 2) * self.params.d
        self.y_mn = (self.n_grid - (self.params.N - 1) / 2) * self.params.d
        
        # Flatten cho tính toán nhanh hơn
        self.x_mn_flat = self.x_mn.flatten()
        self.y_mn_flat = self.y_mn.flatten()
    
    def compute_array_gain_optimized(self, beta: np.ndarray, user_pos: Tuple[float, float, float]) -> float:
        """
        Tính Array Gain tối ưu với vectorization
        """
        x_ue, y_ue, z_ue = user_pos
        
        # Vectorized distance calculation
        if beta.ndim == 2:
            beta_flat = beta.flatten()
        else:
            beta_flat = beta
            
        # Distance từ mỗi element đến user (vectorized)
        dx = x_ue - self.x_mn_flat
        dy = y_ue - self.y_mn_flat
        d_mn = np.sqrt(dx**2 + dy**2 + z_ue**2)
        
        # Phase từ channel
        channel_phase = (2 * np.pi / self.params.lambda_) * d_mn
        
        # Tổng coherent signal (vectorized)
        signal_sum = np.sum(np.exp(1j * (beta_flat - channel_phase)))
        
        return np.abs(signal_sum)
    
    def compute_aag_mag_batch(self, beta: np.ndarray, positions: List[Tuple]) -> Tuple[float, float]:
        """
        Tính AAG và MAG cho batch positions (parallel friendly)
        """
        gains = []
        for pos in positions:
            gain = self.compute_array_gain_optimized(beta, pos)
            gains.append(gain)
            
        gains = np.array(gains)
        aag = np.mean(gains)
        mag = np.min(gains)
        return aag, mag
    
    # ================== BASELINE METHODS (OPTIMIZED) ==================
    
    def far_field_beamforming(self, positions: List[Tuple]) -> np.ndarray:
        """Far-field beamforming: β = 0"""
        return np.zeros((self.params.M, self.params.N))
    
    def average_phase_beamforming_optimized(self, positions: List[Tuple]) -> np.ndarray:
        """Average Phase beamforming tối ưu với vectorization"""
        # Khởi tạo tổng vector phức
        beta_sum = np.zeros(self.params.M * self.params.N, dtype=complex)
        
        # Tính steering vector cho tất cả users
        for x_ue, y_ue, z_ue in positions:
            # Tính góc steering
            r_ue = np.sqrt(x_ue**2 + y_ue**2 + z_ue**2)
            theta = np.arccos(z_ue / r_ue)  # Elevation angle
            phi = np.arctan2(y_ue, x_ue)     # Azimuth angle
            
            # Wave vector components
            kx = (2 * np.pi / self.params.lambda_) * np.sin(theta) * np.cos(phi)
            ky = (2 * np.pi / self.params.lambda_) * np.sin(theta) * np.sin(phi)
            
            # Vectorized steering vector calculation
            beta_user = kx * self.x_mn_flat + ky * self.y_mn_flat
            beta_sum += np.exp(1j * beta_user)
        
        # Phase của tổng vector
        combined_beta = np.angle(beta_sum)
        return combined_beta.reshape(self.params.M, self.params.N)
    
    # ================== GROUPED METHODS (CORE ALGORITHMS) ==================
    
    def grouped_beamforming_optimized(self, positions: List[Tuple], group_size: int) -> np.ndarray:
        """
        Grouped Near-Field Beamforming tối ưu hiệu suất
        """
        num_groups_m = self.params.M // group_size
        num_groups_n = self.params.N // group_size
        beta = np.zeros((self.params.M, self.params.N))
        
        # Precompute group indices
        group_indices = {}
        for i in range(num_groups_m):
            for j in range(num_groups_n):
                group_elements = []
                for m_g in range(group_size):
                    for n_g in range(group_size):
                        m = i * group_size + m_g
                        n = j * group_size + n_g
                        if m < self.params.M and n < self.params.N:
                            group_elements.append((m, n))
                group_indices[(i, j)] = group_elements
        
        # Tính phase cho mỗi group
        for (i, j), elements in group_indices.items():
            group_sum = 0 + 0j
            
            # Vectorized calculation cho group
            element_x = np.array([self.x_mn[m, n] for m, n in elements])
            element_y = np.array([self.y_mn[m, n] for m, n in elements])
            
            for x_ue, y_ue, z_ue in positions:
                # Distance calculation (vectorized)
                dx = x_ue - element_x
                dy = y_ue - element_y
                d_mn = np.sqrt(dx**2 + dy**2 + z_ue**2)
                
                # Contribution của group này đến user này
                user_group_sum = np.sum(np.exp(-1j * 2 * np.pi * d_mn / self.params.lambda_))
                group_sum += user_group_sum
            
            # Phase tối ưu cho group
            optimal_phase = np.angle(group_sum)
            
            # Gán phase cho tất cả elements trong group
            for m, n in elements:
                beta[m, n] = optimal_phase
        
        return beta
    
    # ================== ENHANCED FEATURES FOR FUTURE RESEARCH ==================
    
    def add_channel_noise(self, signal_power: float, snr_db: float) -> float:
        """
        Thêm nhiễu kênh truyền (cho nghiên cứu tương lai)
        """
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn() + 1j * np.random.randn())
        return np.abs(signal_power + noise)
    
    def path_loss_model(self, distance: float, freq_hz: float = None) -> float:
        """
        Mô hình suy hao đường truyền (3GPP Urban Macro)
        """
        if freq_hz is None:
            freq_hz = self.params.frequency
            
        # Path loss in dB
        freq_ghz = freq_hz / 1e9
        pl_db = 28.0 + 22 * np.log10(distance) + 20 * np.log10(freq_ghz)
        
        # Convert to linear scale
        return 10**(-pl_db / 10)
    
    def compute_sinr_with_interference(self, beta: np.ndarray, positions: List[Tuple], 
                                     target_user_idx: int, noise_power: float = 1e-10) -> float:
        """
        Tính SINR với nhiễu đa người dùng (chuẩn bị cho NOMA/RSMA)
        """
        target_pos = positions[target_user_idx]
        target_power = self.compute_array_gain_optimized(beta, target_pos)**2
        
        # Interference từ các user khác (simplified model)
        interference_power = 0
        for i, pos in enumerate(positions):
            if i != target_user_idx:
                interference_power += self.compute_array_gain_optimized(beta, pos)**2 * 0.1  # Cross-talk factor
        
        sinr = target_power / (interference_power + noise_power)
        return sinr
    
    # ================== SIMULATION CORE ==================
    
    def process_single_realization_optimized(self, z: float, num_users: int,
                                           config: SimulationConfig) -> Dict:
        """Xử lý một lần mô phỏng với vị trí người dùng ngẫu nhiên.

        Args:
            z: Khoảng cách từ LIS tới UAV/người dùng (m).
            num_users: Số lượng người dùng trong lần mô phỏng.
            config: Cấu hình mô phỏng đang sử dụng.

        Returns:
            Dictionary chứa AAG và MAG cho từng phương pháp beamforming.
        """

        # Vector hóa việc tạo vị trí người dùng để tăng tốc cho số lượng lớn
        x = np.random.uniform(config.x_range[0], config.x_range[1], size=num_users)
        y = np.random.uniform(config.y_range[0], config.y_range[1], size=num_users)
        positions = list(zip(x, y, np.full(num_users, z)))

        results = {}
        
        # Method 1: Far-field
        beta_ff = self.far_field_beamforming(positions)
        aag_ff, mag_ff = self.compute_aag_mag_batch(beta_ff, positions)
        results['Far-field'] = {'aag': aag_ff, 'mag': mag_ff}
        
        # Method 2: Average Phase  
        beta_ap = self.average_phase_beamforming_optimized(positions)
        aag_ap, mag_ap = self.compute_aag_mag_batch(beta_ap, positions)
        results['Average Phase'] = {'aag': aag_ap, 'mag': mag_ap}
        
        # Method 3-6: Grouped methods (CORE ALGORITHMS)
        for group_size in [2, 4, 8, 16]:
            method_name = f'Grouped ({group_size}x{group_size})'
            try:
                beta_gr = self.grouped_beamforming_optimized(positions, group_size)
                aag_gr, mag_gr = self.compute_aag_mag_batch(beta_gr, positions)
                results[method_name] = {'aag': aag_gr, 'mag': mag_gr}
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                results[method_name] = {'aag': 0, 'mag': 0}
        
        # # GWO methods - DISABLED để tăng tốc độ
        # # Uncomment below để kích hoạt GWO optimization:
        # # 
        # # for group_size in [2, 4, 8, 16]:
        # #     method_name = f'GWO ({group_size}x{group_size})'
        # #     try:
        # #         beta_gwo, aag_gwo = self.gwo_grouped_beamforming(positions, group_size)
        # #         _, mag_gwo = self.compute_aag_mag_batch(beta_gwo, positions)
        # #         results[method_name] = {'aag': aag_gwo, 'mag': mag_gwo}
        # #     except Exception as e:
        # #         print(f"GWO {group_size}x{group_size} failed: {e}")
        # #         # Fallback to grouped method
        # #         beta_fallback = self.grouped_beamforming_optimized(positions, group_size)
        # #         aag_fb, mag_fb = self.compute_aag_mag_batch(beta_fallback, positions)
        # #         results[method_name] = {'aag': aag_fb, 'mag': mag_fb}
        
        return results
    
    def run_optimized_simulation(self, config: SimulationConfig) -> Dict:
        """Chạy mô phỏng chính với khả năng xử lý song song.

        Args:
            config: Cấu hình mô phỏng (số user, dãy ``z``, số lần lặp, ...).

        Returns:
            Kết quả tổng hợp cho tất cả kịch bản người dùng.
        """

        print(f"=== BẮT ĐẦU SIMULATION TỐI ỨU ===")
        print(f"Z values: {len(config.z_values)} điểm")
        print(f"Realizations: {config.num_realizations}/z")
        print(f"User scenarios: {config.num_users_list}")

        start_time = time.time()
        all_results = {}

        for num_users in config.num_users_list:
            print(f"\n--- Simulation cho {num_users} users ---")

            # Số tiến trình song song
            n_jobs = (
                config.n_jobs
                if config.n_jobs != -1
                else min(len(config.z_values), mp.cpu_count())
            )
            n_jobs = max(1, n_jobs)

            args_iterable = (
                (self, z, num_users, config) for z in config.z_values
            )

            user_results = {}
            method_names = None

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for z, z_results in executor.map(_process_z_batch, args_iterable, chunksize=1):
                    if method_names is None:
                        method_names = list(z_results[0].keys())
                        for method in method_names:
                            user_results[method] = {'aag': [], 'mag': []}

                    # Gộp kết quả cho từng z ngay khi nhận được để giảm memory
                    for method in method_names:
                        aag_vals = [r[method]['aag'] for r in z_results]
                        mag_vals = [r[method]['mag'] for r in z_results]
                        user_results[method]['aag'].extend(aag_vals)
                        user_results[method]['mag'].extend(mag_vals)

            # Tính AMAG sau khi đã thu thập hết dữ liệu
            for method in method_names:
                amag = np.mean(user_results[method]['mag'])
                user_results[method]['amag'] = amag
                print(f"  {method}: AAG={np.mean(user_results[method]['aag']):.1f}, AMAG={amag:.1f}")

            all_results[f'{num_users}_users'] = {
                'z_values': config.z_values,
                'num_users': num_users,
                'num_realizations': config.num_realizations,
                'results': user_results,
                'method_names': method_names
            }

        total_time = time.time() - start_time
        print(f"\n=== SIMULATION HOÀN THÀNH: {total_time:.1f}s ===")

        return {
            'system_params': self.params,
            'simulation_config': config,
            'simulation_time': total_time,
            'all_results': all_results
        }
    
    # ================== VISUALIZATION ==================
    
    def plot_comprehensive_results(self, simulation_results: Dict, save_dir: str = None) -> List[plt.Figure]:
        """Vẽ tất cả đồ thị kết quả và trả về danh sách ``Figure``.

        Việc trả về đối tượng ``Figure`` cho phép tích hợp với các giao diện GUI
        mà không phụ thuộc vào backend của Matplotlib. Hàm vẫn giữ nguyên hành vi
        cũ là hiển thị các đồ thị bằng ``plt.show()``.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        all_results = simulation_results['all_results']

        # Style mapping
        style_map = {
            'Far-field': {'color': 'black', 'linestyle': '-', 'marker': 'o'},
            'Average Phase': {'color': 'green', 'linestyle': '--', 'marker': 's'},
            'Grouped (2x2)': {'color': 'red', 'linestyle': '-', 'marker': '^'},
            'Grouped (4x4)': {'color': 'red', 'linestyle': '--', 'marker': 'v'},
            'Grouped (8x8)': {'color': 'red', 'linestyle': ':', 'marker': 'p'},
            'Grouped (16x16)': {'color': 'red', 'linestyle': '-.', 'marker': '*'},
        }

        figures: List[plt.Figure] = []

        for user_key, user_data in all_results.items():
            num_users = user_data['num_users']
            z_values = user_data['z_values']
            results = user_data['results']
            method_names = user_data['method_names']
            num_realizations = user_data['num_realizations']

            # 1. AAG vs Distance
            fig = plt.figure(figsize=(14, 8))
            for method in method_names:
                if method in style_map:
                    style = style_map[method]
                    # Tính AAG trung bình cho mỗi z
                    aag_by_z = []
                    for i, z in enumerate(z_values):
                        start_idx = i * num_realizations
                        end_idx = (i + 1) * num_realizations
                        aag_mean = np.mean(results[method]['aag'][start_idx:end_idx])
                        aag_by_z.append(aag_mean)

                    plt.plot(
                        z_values / self.params.lambda_,
                        aag_by_z,
                        label=method,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markevery=max(1, len(z_values) // 8),
                        markersize=6,
                        linewidth=2,
                    )

            # Vẽ ranh giới
            plt.axvline(
                x=self.d_0 / self.params.lambda_,
                color='gray',
                linestyle=':',
                label=f'Fresnel boundary = {self.d_0:.1f}m',
                alpha=0.7,
            )
            plt.axvline(
                x=self.d_F1 / self.params.lambda_,
                color='gray',
                linestyle='--',
                label=f'Fraunhofer 1 = {self.d_F1:.1f}m',
                alpha=0.7,
            )
            plt.axvline(
                x=self.d_F2 / self.params.lambda_,
                color='gray',
                linestyle='-.',
                label=f'Fraunhofer 2 = {self.d_F2:.1f}m',
                alpha=0.7,
            )

            plt.xlabel('Distance (z/λ)', fontsize=12)
            plt.ylabel('Average Array Gain (AAG)', fontsize=12)
            plt.title(f'AAG vs Distance - {num_users} Users', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(z_values[0] / self.params.lambda_, z_values[-1] / self.params.lambda_)
            plt.ylim(bottom=0)

            # Ensure layout accommodates legend
            fig.tight_layout(rect=[0, 0, 0.85, 1])

            if save_dir:
                fig.savefig(
                    f"{save_dir}/aag_vs_distance_{num_users}users.png",
                    dpi=300,
                    bbox_inches='tight',
                )
            figures.append(fig)
            plt.show()

            # 2. AMAG vs Distance
            fig = plt.figure(figsize=(14, 8))
            for method in method_names:
                if method in style_map:
                    style = style_map[method]
                    amag_by_z = []
                    for i, z in enumerate(z_values):
                        start_idx = i * num_realizations
                        end_idx = (i + 1) * num_realizations
                        mag_mean = np.mean(results[method]['mag'][start_idx:end_idx])
                        amag_by_z.append(mag_mean)

                    plt.plot(
                        z_values / self.params.lambda_,
                        amag_by_z,
                        label=method,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markevery=max(1, len(z_values) // 8),
                        markersize=6,
                        linewidth=2,
                    )

            plt.axvline(
                x=self.d_0 / self.params.lambda_,
                color='gray',
                linestyle=':',
                label=f'Fresnel boundary = {self.d_0:.1f}m',
                alpha=0.7,
            )
            plt.axvline(
                x=self.d_F1 / self.params.lambda_,
                color='gray',
                linestyle='--',
                label=f'Fraunhofer 1 = {self.d_F1:.1f}m',
                alpha=0.7,
            )
            plt.axvline(
                x=self.d_F2 / self.params.lambda_,
                color='gray',
                linestyle='-.',
                label=f'Fraunhofer 2 = {self.d_F2:.1f}m',
                alpha=0.7,
            )

            plt.xlabel('Distance (z/λ)', fontsize=12)
            plt.ylabel('Average Minimum Array Gain (AMAG)', fontsize=12)
            plt.title(f'AMAG vs Distance - {num_users} Users', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(z_values[0] / self.params.lambda_, z_values[-1] / self.params.lambda_)
            plt.ylim(bottom=0)

            fig.tight_layout(rect=[0, 0, 0.85, 1])

            if save_dir:
                fig.savefig(
                    f"{save_dir}/amag_vs_distance_{num_users}users.png",
                    dpi=300,
                    bbox_inches='tight',
                )
            figures.append(fig)
            plt.show()

            # 3. CDF AAG
            fig = plt.figure(figsize=(12, 8))
            for method in method_names:
                if method in style_map and len(results[method]['aag']) > 0:
                    style = style_map[method]
                    data = results[method]['aag']
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                    plt.plot(
                        sorted_data,
                        y,
                        label=method,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=2,
                    )

            plt.xlabel('Average Array Gain', fontsize=12)
            plt.ylabel('CDF', fontsize=12)
            plt.title(f'CDF of AAG - {num_users} Users', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(left=0)
            plt.ylim(0, 1)

            fig.tight_layout(rect=[0, 0, 0.85, 1])

            if save_dir:
                fig.savefig(
                    f"{save_dir}/cdf_aag_{num_users}users.png",
                    dpi=300,
                    bbox_inches='tight',
                )
            figures.append(fig)
            plt.show()

        return figures
    
    def save_results(self, simulation_results: Dict, filepath: str):
        """Lưu kết quả simulation"""
        with open(filepath, 'wb') as f:
            pickle.dump(simulation_results, f)
        print(f"Kết quả đã lưu tại: {filepath}")
    
    def load_results(self, filepath: str) -> Dict:
        """Load kết quả simulation đã lưu"""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"Đã load kết quả từ: {filepath}")
        return results

# ================== UTILITY FUNCTIONS ==================

def create_system_with_presets(preset: str = "standard") -> OptimizedNearFieldBeamformingSimulator:
    """Tạo hệ thống mô phỏng dựa trên preset cấu hình sẵn.

    Args:
        preset: Tên cấu hình hệ thống (``standard``, ``high_freq``, ...).

    Returns:
        Đối tượng ``OptimizedNearFieldBeamformingSimulator`` đã khởi tạo.
    """
    presets = {
        "standard": SystemParameters(M=32, N=32, lambda_=0.05, frequency=6e9),
        "high_freq": SystemParameters(M=32, N=32, lambda_=0.01, frequency=30e9),  # mmWave
        "large_array": SystemParameters(M=64, N=64, lambda_=0.05, frequency=6e9),
        "small_test": SystemParameters(M=16, N=16, lambda_=0.05, frequency=6e9),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return OptimizedNearFieldBeamformingSimulator(presets[preset])

def create_simulation_config(mode: str = "fast") -> SimulationConfig:
    """Tạo cấu hình mô phỏng theo các chế độ định nghĩa sẵn.

    Args:
        mode: Tên chế độ (``fast``, ``standard``, ``comprehensive``...).

    Returns:
        Đối tượng ``SimulationConfig`` tương ứng.
    """
    configs = {
        "fast": SimulationConfig(
            num_users_list=[5],
            z_values=np.linspace(0.1, 200, 10),
            num_realizations=20,
            n_jobs=-1
        ),
        "standard": SimulationConfig(
            num_users_list=[5, 10],
            z_values=np.linspace(0.1, 200, 20),
            num_realizations=50,
            n_jobs=-1
        ),
        "comprehensive": SimulationConfig(
            num_users_list=[5, 10, 20],
            z_values=np.linspace(0.1, 200, 30),
            num_realizations=100,
            n_jobs=-1
        )
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")
    
    return configs[mode]

# ================== FUTURE RESEARCH EXTENSIONS ==================

class NOMAEnhancedSimulator(OptimizedNearFieldBeamformingSimulator):
    """
    Mở rộng cho NOMA integration (prepared for future research)
    """
    
    def __init__(self, params: SystemParameters, noma_enabled: bool = False):
        super().__init__(params)
        self.noma_enabled = noma_enabled
        
    def noma_power_allocation(self, user_positions: List[Tuple], 
                             total_power: float = 1.0) -> np.ndarray:
        """
        NOMA power allocation based on channel conditions
        """
        # Simplified power allocation based on distance
        distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in user_positions]
        # Farther users get more power in NOMA
        power_weights = np.array(distances) / np.sum(distances)
        power_allocation = power_weights * total_power
        return power_allocation
    
    def compute_noma_rate(self, beta: np.ndarray, positions: List[Tuple], 
                         power_allocation: np.ndarray, noise_power: float = 1e-10) -> List[float]:
        """
        Compute achievable rates with NOMA
        """
        rates = []
        for i, pos in enumerate(positions):
            # Simplified NOMA rate calculation
            signal_power = self.compute_array_gain_optimized(beta, pos)**2 * power_allocation[i]
            
            # Interference from other users (simplified)
            interference = sum(self.compute_array_gain_optimized(beta, positions[j])**2 * power_allocation[j] 
                             for j in range(len(positions)) if j != i) * 0.1
            
            sinr = signal_power / (interference + noise_power)
            rate = np.log2(1 + sinr)
            rates.append(rate)
        
        return rates

class RSMAEnhancedSimulator(OptimizedNearFieldBeamformingSimulator):
    """
    Mở rộng cho RSMA integration (prepared for future research)
    """
    
    def __init__(self, params: SystemParameters, rsma_enabled: bool = False):
        super().__init__(params)
        self.rsma_enabled = rsma_enabled
        
    def rsma_precoding_design(self, positions: List[Tuple], 
                             common_stream_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        RSMA precoding design with common and private streams
        """
        # Placeholder for future RSMA implementation
        num_users = len(positions)
        
        # Common stream precoding (simplified)
        common_precoder = self.grouped_beamforming_optimized(positions, group_size=2)
        
        # Private stream precoders (simplified)
        private_precoders = []
        for i, pos in enumerate(positions):
            private_precoder = self.grouped_beamforming_optimized([pos], group_size=4)
            private_precoders.append(private_precoder)
        
        return common_precoder, np.array(private_precoders)

if __name__ == "__main__":
    print("=== OPTIMIZED NEAR-FIELD BEAMFORMING SIMULATOR ===")
    print("Sử dụng các hàm create_system_with_presets() và create_simulation_config()")
    print("để khởi tạo và chạy simulation.")
    print("\nVí dụ:")
    print("simulator = create_system_with_presets('standard')")
    print("config = create_simulation_config('fast')")
    print("results = simulator.run_optimized_simulation(config)")
    print("simulator.plot_comprehensive_results(results)")
