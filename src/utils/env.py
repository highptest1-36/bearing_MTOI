# -*- coding: utf-8 -*-
"""
env.py — KIỂM TRA MÔI TRƯỜNG & THIẾT LẬP CHUNG (GPU A100, seed, device).

Mục tiêu:
  - In rõ thông tin máy: GPU gì, RAM bao nhiêu, đĩa còn trống bao nhiêu, phiên bản thư viện.
  - Cảnh báo nếu đang chạy CPU (nhắc bạn đổi runtime sang GPU A100).
  - Cố định seed để kết quả tái lập được (reproducible) — rất quan trọng cho paper.
  - Trả về 'device' (cuda hoặc cpu) để các module khác dùng.
"""

import os                       # đọc biến môi trường, thao tác hệ thống
import random                   # seed cho random thuần Python
import shutil                   # lấy dung lượng đĩa
import platform                 # lấy thông tin hệ điều hành/python


def set_seed(seed: int = 42):
    """Cố định mọi nguồn ngẫu nhiên để thí nghiệm tái lập được."""
    random.seed(seed)                          # seed cho module random của Python
    os.environ["PYTHONHASHSEED"] = str(seed)   # seed cho hash (ổn định thứ tự set/dict)
    try:
        import numpy as np                     # import numpy (nếu có)
        np.random.seed(seed)                   # seed cho numpy
    except Exception:
        pass                                   # nếu chưa cài numpy thì bỏ qua
    try:
        import torch                           # import torch (nếu có)
        torch.manual_seed(seed)                # seed cho CPU
        torch.cuda.manual_seed_all(seed)       # seed cho tất cả GPU
        # 2 dòng dưới giúp kết quả ổn định hơn (đánh đổi 1 chút tốc độ) — bật khi cần reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass                                   # nếu chưa cài torch thì bỏ qua
    return seed


def get_device(verbose: bool = True):
    """Trả về 'cuda' nếu có GPU, ngược lại 'cpu'. In cảnh báo nếu đang CPU."""
    import torch                               # cần torch để hỏi GPU
    if torch.cuda.is_available():              # nếu có GPU
        device = "cuda"                        # dùng GPU
        if verbose:
            # In tên GPU (ví dụ 'NVIDIA A100-SXM4-40GB') và tổng VRAM.
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU] Đang dùng GPU: {name} | VRAM ~ {vram_gb:.1f} GB")
            # Cảnh báo nhẹ nếu KHÔNG phải A100 (vẫn chạy được nhưng có thể chậm hơn).
            if "A100" not in name:
                print("[GPU] LƯU Ý: GPU hiện tại không phải A100. Vẫn chạy được nhưng có thể chậm hơn.")
    else:
        device = "cpu"                         # không có GPU -> dùng CPU
        if verbose:
            print("[GPU] !!! CẢNH BÁO: KHÔNG phát hiện GPU — đang chạy CPU rất chậm.")
            print("[GPU] >>> Hãy đổi runtime: Colab menu 'Runtime' -> 'Change runtime type' -> 'A100 GPU'.")
    return device


def print_env_report(seed: int = 42):
    """In một báo cáo môi trường đầy đủ để bạn theo dõi đầu mỗi phiên chạy."""
    print("=" * 70)
    print(" BÁO CÁO MÔI TRƯỜNG (MTOI-Bearing) ".center(70, "="))
    print("=" * 70)

    # --- Thông tin hệ điều hành & Python ---
    print(f"[OS]     {platform.platform()}")                 # tên/phiên bản hệ điều hành
    print(f"[Python] {platform.python_version()}")           # phiên bản Python

    # --- Thông tin RAM (đọc từ /proc/meminfo trên Linux/Colab) ---
    try:
        with open("/proc/meminfo") as f:                     # mở file thông tin bộ nhớ
            meminfo = f.read()                               # đọc toàn bộ nội dung
        # Dòng "MemTotal: 12345 kB" -> lấy số kB rồi đổi sang GB.
        total_kb = int([l for l in meminfo.splitlines() if l.startswith("MemTotal")][0].split()[1])
        print(f"[RAM]    Tổng RAM ~ {total_kb/1024/1024:.1f} GB")
    except Exception:
        print("[RAM]    Không đọc được thông tin RAM.")

    # --- Thông tin đĩa của /content (nơi giải nén raw) ---
    try:
        total, used, free = shutil.disk_usage("/content")   # lấy dung lượng đĩa local
        print(f"[DISK]   /content: tổng {total/1e9:.0f}GB | đã dùng {used/1e9:.0f}GB | trống {free/1e9:.0f}GB")
    except Exception:
        print("[DISK]   Không đọc được dung lượng đĩa.")

    # --- Phiên bản các thư viện khoa học chính ---
    for libname in ["numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"]:
        try:
            mod = __import__(libname)                        # thử import thư viện
            ver = getattr(mod, "__version__", "?")           # lấy version nếu có
            print(f"[LIB]    {libname:12s} {ver}")
        except Exception:
            print(f"[LIB]    {libname:12s} CHƯA CÀI")        # báo nếu chưa cài

    # --- Kiểm tra Google Drive đã mount chưa ---
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    print(f"[DRIVE]  Google Drive mounted: {'CÓ' if drive_mounted else 'CHƯA (cần chạy mount)'}")

    # --- Thiết lập seed & device ---
    set_seed(seed)                                           # cố định seed
    print(f"[SEED]   Đã đặt seed = {seed}")
    device = get_device(verbose=True)                        # in thông tin GPU/CPU
    print("=" * 70)
    return device


def mount_drive():
    """Mount Google Drive (chỉ dùng khi chạy trong Colab). An toàn nếu đã mount rồi."""
    # Nếu đã mount rồi thì không làm gì.
    if os.path.exists("/content/drive/MyDrive"):
        print("[DRIVE] Drive đã được mount sẵn.")
        return True
    try:
        from google.colab import drive          # chỉ có trong môi trường Colab
        drive.mount("/content/drive")           # mở popup yêu cầu cấp quyền 1 lần
        print("[DRIVE] Mount Drive thành công.")
        return True
    except Exception as e:
        print(f"[DRIVE] Không mount được Drive (có thể không chạy trên Colab): {e}")
        return False


def setup_environment(seed: int = 42, do_mount: bool = True):
    """
    Hàm gọi 1 lần ở đầu MỖI phase:
      - (tuỳ chọn) mount Drive
      - tạo sẵn thư mục
      - in báo cáo môi trường
      - trả về device để dùng tiếp.
    """
    if do_mount:
        mount_drive()                            # đảm bảo Drive sẵn sàng
    # Import muộn để tránh lỗi vòng (paths cũng import nhẹ).
    from src.utils.paths import ensure_dirs
    ensure_dirs()                                # tạo sẵn mọi thư mục
    device = print_env_report(seed=seed)         # in báo cáo + lấy device
    return device
