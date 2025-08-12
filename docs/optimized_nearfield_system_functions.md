# Tổng quan các hàm trong `optimized_nearfield_system.py`

Tài liệu này liệt kê và mô tả chi tiết logic quan trọng của từng hàm và phương thức trong module `optimized_nearfield_system.py`.

## Hàm trợ giúp cấp module

### `_process_z_batch(args)`
- **Mục đích:** Xử lý một tập giá trị `z` trong quá trình mô phỏng song song.
- **Logic chính:**
  - Nhận bộ tham số gồm simulator, khoảng cách `z`, số người dùng và cấu hình mô phỏng.
  - Thiết lập seed ngẫu nhiên khác nhau cho từng tiến trình.
  - Lặp qua số lần mô phỏng (`num_realizations`) và gọi `process_single_realization_optimized` cho mỗi lần.
  - Trả về khoảng cách `z` và danh sách kết quả tương ứng.

## Các dataclass cấu hình

### `SystemParameters`
- **Mục đích:** Lưu trữ kích thước LIS, bước sóng và các tham số liên quan.
- **Logic chính:**
  - Thuộc tính `d` (khoảng cách phần tử) được tính mặc định bằng `lambda_/2` thông qua `__post_init__`.

### `SimulationConfig`
- **Mục đích:** Định nghĩa danh sách số người dùng, tập giá trị khoảng cách `z` và số lần mô phỏng.
- **Logic chính:**
  - Thiết lập giá trị mặc định cho `num_users_list` và `z_values` nếu không được cung cấp.

## Lớp `OptimizedNearFieldBeamformingSimulator`

### `__init__(self, params)`
- Khởi tạo simulator và gọi `_initialize_system` cùng `_precompute_positions` để chuẩn bị các biến toàn cục.

### `_initialize_system(self)`
- **Logic chính:**
  - Tính các ranh giới Fresnel/Fraunhofer dựa trên kích thước LIS.
  - In thông tin cấu hình hệ thống.

### `_precompute_positions(self)`
- **Logic chính:**
  - Tạo lưới chỉ số của các phần tử LIS và tính toạ độ `(x_mn, y_mn)`.
  - Làm phẳng các ma trận toạ độ để phục vụ các phép tính vector hóa.

### `compute_array_gain_optimized(self, beta, user_pos)`
- **Logic chính:**
  - Làm phẳng ma trận pha `beta` nếu cần.
  - Tính khoảng cách từ từng phần tử đến người dùng bằng vector hóa.
  - Tính pha kênh và tổng tín hiệu coherent để trả về độ lợi mảng.

### `compute_aag_mag_batch(self, beta, positions)`
- **Logic chính:**
  - Tính độ lợi mảng cho từng vị trí trong `positions`.
  - Trả về AAG (trung bình) và MAG (nhỏ nhất) của tập vị trí.

### `far_field_beamforming(self, positions)`
- Trả về ma trận pha bằng 0, tương ứng với beamforming miền xa.

### `average_phase_beamforming_optimized(self, positions)`
- **Logic chính:**
  - Tính vector steering cho từng người dùng và cộng các vector phức.
  - Pha tổng hợp được sử dụng làm cấu hình beamforming trung bình.

### `grouped_beamforming_optimized(self, positions, group_size)`
- **Logic chính:**
  - Chia ma trận LIS thành các nhóm con kích thước `group_size`.
  - Với mỗi nhóm, tính tổng đóng góp phức của tất cả người dùng và đặt pha tối ưu cho toàn nhóm.

### `add_channel_noise(self, signal_power, snr_db)`
- **Logic chính:**
  - Chuyển đổi SNR dB sang hệ số tuyến tính.
  - Sinh nhiễu Gaussian phức và cộng vào công suất tín hiệu.

### `path_loss_model(self, distance, freq_hz=None)`
- **Logic chính:**
  - Tính suy hao đường truyền theo mô hình 3GPP Urban Macro và trả về hệ số tuyến tính.

### `compute_sinr_with_interference(self, beta, positions, target_user_idx, noise_power)`
- **Logic chính:**
  - Tính công suất tín hiệu mục tiêu và công suất nhiễu từ các người dùng khác.
  - Trả về tỉ số SINR.

### `process_single_realization_optimized(self, z, num_users, config)`
- **Logic chính:**
  - Sinh ngẫu nhiên vị trí người dùng trong mặt phẳng `xy` với khoảng cách `z` cố định.
  - Tính AAG và MAG cho các phương pháp beamforming khác nhau (far-field, average-phase, grouped).
  - Trả về từ điển kết quả cho mỗi phương pháp.

### `run_optimized_simulation(self, config)`
- **Logic chính:**
  - Lặp qua từng kịch bản số người dùng và sử dụng `ProcessPoolExecutor` để xử lý song song các giá trị `z`.
  - Gom và tính toán trung bình kết quả cho từng phương pháp beamforming.

### `plot_comprehensive_results(self, simulation_results, save_dir=None)`
- **Logic chính:**
  - Vẽ các đồ thị AAG/AMAG theo khoảng cách và biểu đồ CDF.
  - Cho phép lưu hình ảnh nếu cung cấp `save_dir`.

### `save_results(self, simulation_results, filepath)`
- Lưu kết quả mô phỏng vào file pickle.

### `load_results(self, filepath)`
- Đọc kết quả mô phỏng đã lưu từ file pickle.

## Hàm tiện ích

### `create_system_with_presets(preset="standard")`
- **Logic chính:**
  - Định nghĩa nhiều bộ tham số hệ thống mẫu và trả về simulator tương ứng với preset được chọn.

### `create_simulation_config(mode="fast")`
- **Logic chính:**
  - Cung cấp các cấu hình mô phỏng tiêu chuẩn (nhanh, chuẩn, toàn diện) với số người dùng, dải khoảng cách và số lần mô phỏng khác nhau.

## Mở rộng cho nghiên cứu tương lai

### `NOMAEnhancedSimulator`
- **`__init__(self, params, noma_enabled=False)`**: kế thừa simulator gốc và bật/tắt chế độ NOMA.
- **`noma_power_allocation(self, user_positions, total_power)`**: phân bổ công suất dựa trên khoảng cách người dùng.
- **`compute_noma_rate(self, beta, positions, power_allocation, noise_power)`**: tính tốc độ đạt được của từng người dùng với mô hình nhiễu đơn giản.

### `RSMAEnhancedSimulator`
- **`__init__(self, params, rsma_enabled=False)`**: kế thừa simulator gốc và bật/tắt chế độ RSMA.
- **`rsma_precoding_design(self, positions, common_stream_ratio)`**: thiết kế precoder chung và riêng cho RSMA bằng phương pháp nhóm đơn giản.

