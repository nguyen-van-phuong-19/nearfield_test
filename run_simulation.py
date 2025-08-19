"""Script chạy mô phỏng Near-Field Beamforming.

Cho phép lựa chọn preset hệ thống và chế độ mô phỏng, phù hợp khi cần
mô phỏng với số lượng người dùng lớn."""

import argparse
from optimized_nearfield_system import create_system_with_presets, create_simulation_config


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số đầu vào từ dòng lệnh.

    Returns:
        Namespace chứa ``preset``, ``mode`` và ``users``.
    """
    parser = argparse.ArgumentParser(description="Chạy mô phỏng LIS-UAV")
    parser.add_argument("--preset", default="standard", help="Tên preset hệ thống")
    parser.add_argument("--mode", default="fast", help="Chế độ cấu hình mô phỏng")
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="Số người dùng mô phỏng (ghi đè cấu hình)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Khởi tạo simulator từ preset
    simulator = create_system_with_presets(args.preset)

    # Tạo cấu hình mô phỏng
    config = create_simulation_config(args.mode)
    if args.users is not None:
        # Ghi chú: num_users_list là danh sách, nên chuyển thành list một phần tử
        config.num_users_list = [args.users]

    # Chạy mô phỏng và lưu kết quả
    results = simulator.run_optimized_simulation(config)
    simulator.plot_comprehensive_results(results, save_dir="my_results")


if __name__ == "__main__":
    main()
