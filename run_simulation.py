"""Script chạy mô phỏng Near-Field Beamforming với tùy chỉnh tham số."""

import json
import argparse
from pathlib import Path

from optimized_nearfield_system import create_system_with_presets, create_simulation_config
from random_params import random_basic_params

PARAM_FILE = Path("config/user_params.json")


def load_saved_params() -> dict:
    """Đọc tham số mô phỏng đã lưu hoặc trả về giá trị mặc định."""
    if PARAM_FILE.exists():
        with open(PARAM_FILE, "r") as f:
            return json.load(f)
    return {"preset": "standard", "mode": "fast", "users": None}


def save_params(params: dict) -> None:
    """Lưu tham số mô phỏng cho các lần chạy sau."""
    PARAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f)


def customize_params(current: dict) -> dict:
    """Tương tác để người dùng điều chỉnh tham số mô phỏng."""
    print("Nhập tên preset hệ thống (standard, high_freq, large_array, small_test).")
    preset = input(f"Preset [{current['preset']}]: ").strip() or current['preset']

    print("Nhập chế độ mô phỏng (fast, standard, comprehensive).")
    mode = input(f"Mode [{current['mode']}]: ").strip() or current['mode']

    print("Nhập số người dùng (ví dụ 5). Để trống để giữ nguyên giá trị hiện tại.")
    users_str = input(f"Số người dùng [{current['users']}]: ").strip()
    users = int(users_str) if users_str else current['users']

    return {"preset": preset, "mode": mode, "users": users}


def choose_parameters() -> dict:
    """Hiển thị menu lựa chọn và trả về tham số mô phỏng cuối cùng."""
    saved = load_saved_params()
    print("=== Thiết lập tham số mô phỏng ===")
    print(f"1. Dùng tham số đã lưu (preset={saved['preset']}, mode={saved['mode']}, users={saved['users']})")
    print("2. Dùng tham số mặc định (preset=standard, mode=fast)")
    print("3. Tùy chỉnh tham số")
    choice = input("Lựa chọn [1/2/3]: ").strip()

    if choice == "2":
        params = {"preset": "standard", "mode": "fast", "users": None}
    elif choice == "3":
        params = customize_params(saved)
    else:
        params = saved

    save_params(params)
    return params


def choose_parameters_random_enabled() -> dict:
    """Like choose_parameters but with an extra randomize option."""
    saved = load_saved_params()
    print("=== Thiết lập tham số mô phỏng ===")
    print(f"1. Dùng tham số đã lưu (preset={saved['preset']}, mode={saved['mode']}, users={saved['users']})")
    print("2. Dùng tham số mặc định (preset=standard, mode=fast)")
    print("3. Tuỳ chỉnh tham số")
    print("4. Random tham số hợp lệ (preset/mode/users)")
    choice = input("Lựa chọn [1/2/3/4]: ").strip()

    if choice == "2":
        params = {"preset": "standard", "mode": "fast", "users": None}
    elif choice == "3":
        params = customize_params(saved)
    elif choice == "4":
        params = random_basic_params()
        print(f"Chọn ngẫu nhiên: preset={params['preset']}, mode={params['mode']}, users={params['users']}")
    else:
        params = saved

    save_params(params)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Near-Field simulation (tuỳ chỉnh hoặc random tham số)")
    parser.add_argument("--randomize", "-r", action="store_true", help="Randomize preset/mode/users with valid values")
    args = parser.parse_args()

    if args.randomize:
        params = random_basic_params()
        print(f"[Random] preset={params['preset']}, mode={params['mode']}, users={params['users']}")
        save_params(params)
    else:
        params = choose_parameters_random_enabled()

    simulator = create_system_with_presets(params["preset"])

    config = create_simulation_config(params["mode"])
    if params["users"] is not None:
        config.num_users_list = [params["users"]]

    results = simulator.run_optimized_simulation(config)
    simulator.plot_comprehensive_results(results, save_dir="my_results")


if __name__ == "__main__":
    main()
