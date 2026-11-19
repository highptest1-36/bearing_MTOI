# -*- coding: utf-8 -*-
"""
phase0_setup.py — BƯỚC 0: KIỂM TRA MÔI TRƯỜNG & CHUẨN BỊ.

Chạy đầu tiên mỗi phiên Colab để:
  - Mount Google Drive.
  - In báo cáo môi trường (GPU A100? RAM? đĩa? thư viện?).
  - Tạo sẵn mọi thư mục.
  - In trạng thái tiến trình hiện tại (đã làm tới phase nào).
Lệnh chạy:  !python scripts/phase0_setup.py
"""

import sys
from pathlib import Path
# Thêm thư mục gốc dự án vào sys.path để 'import src...' chạy được dù đứng ở đâu.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker


def main():
    # Thiết lập môi trường: mount Drive + in báo cáo + lấy device.
    device = setup_environment(seed=42, do_mount=True)
    # In trạng thái các phase đã hoàn thành (đọc state.json trên Drive).
    PhaseTracker().summary()
    print(f"\n>>> Thiết bị tính toán: {device}")
    print(">>> Nếu device='cpu', hãy đổi runtime sang A100 GPU rồi chạy lại phase0.")
    return device


if __name__ == "__main__":
    main()
