# -*- coding: utf-8 -*-
"""
segmentation.py — PHASE 2: CẮT TÍN HIỆU RUNG THÀNH CỬA SỔ (sub-window).

Hai cấp thời gian (rất quan trọng, theo plan mục 5):
  - Cấp 1 (sub-window): cắt mỗi file/giờ thành nhiều cửa sổ ngắn W_{h,j} ∈ R^{2×L} -> để TRAIN mạng.
  - Cấp 2 (hour/snapshot): MTOI, RUL, stage, warning tính ở cấp GIỜ (mỗi giờ 1 giá trị).

QUYẾT ĐỊNH THỰC TẾ (để chạy được trên Colab, RAM/đĩa có hạn):
  - Mỗi file Mendeley có ~2 triệu mẫu -> ~1000 cửa sổ L=4096/giờ. 129 giờ -> >100k cửa sổ.
  - Ta KHÔNG giữ hết (tốn RAM/đĩa); thay vào đó LẤY MẪU tối đa `max_windows_per_hour` cửa sổ/giờ
    (mặc định 64) trải đều khắp file -> đủ đại diện cho trạng thái giờ đó, mà file npy chỉ ~270MB.
  - Việc lấy mẫu này được GHI LOG rõ ràng (không "âm thầm cắt bớt").

Đầu ra (ghi lên Drive để sống sót disconnect):
  - X_vib_windows.npy : mảng [N, 2, L] float32 (toàn bộ cửa sổ rung đã lấy mẫu).
  - window_hour_ids.npy : mảng [N] int — mỗi cửa sổ thuộc giờ nào.
  - temp_by_hour.csv : mỗi giờ 1 dòng (T_bearing_mean, T_atm_mean, DeltaT_mean).
"""

import numpy as np                 # mảng số
import pandas as pd                # bảng nhiệt độ
from pathlib import Path

from src.utils.paths import PROC_MENDELEY
from src.utils.logger import get_logger, section, progress_iter, Timer
from src.data import mendeley_loader


def make_windows_from_signal(vib, L=4096, overlap=0.5, max_windows=None):
    """
    Cắt 1 tín hiệu rung (2, T) thành các cửa sổ (2, L).
      - L: độ dài cửa sổ (số mẫu).
      - overlap: tỉ lệ chồng lấp (0.5 = chồng 50%) -> bước nhảy hop = L*(1-overlap).
      - max_windows: nếu đặt -> chỉ giữ tối đa bấy nhiêu cửa sổ, chọn ĐỀU (deterministic) trên trục
        thời gian để vẫn phủ toàn file (không dùng ngẫu nhiên -> kết quả tái lập 100%).
    Trả về mảng [n, 2, L].
    """
    T = vib.shape[1]                                   # tổng số mẫu của tín hiệu
    hop = max(1, int(L * (1.0 - overlap)))             # bước nhảy giữa 2 cửa sổ liên tiếp
    # Các vị trí bắt đầu cửa sổ: 0, hop, 2*hop, ... sao cho cửa sổ không vượt quá T.
    starts = list(range(0, T - L + 1, hop))
    if len(starts) == 0:                               # tín hiệu ngắn hơn L -> không cắt được
        return np.empty((0, 2, L), dtype=np.float32)

    # Nếu cần giới hạn số cửa sổ -> chọn ĐỀU (evenly) trên trục thời gian để vẫn phủ cả file.
    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)  # chỉ số đều
        starts = [starts[i] for i in np.unique(idx)]   # lấy các vị trí đã chọn (bỏ trùng)

    # Trích từng cửa sổ và xếp chồng thành mảng [n, 2, L].
    wins = np.stack([vib[:, s:s + L] for s in starts], axis=0).astype(np.float32)
    return wins


def build_mendeley_windows(L=4096, overlap=0.5, max_windows_per_hour=64,
                           max_files=None, max_rows=None, seed=42, out_dir=None):
    """
    PHASE 2 cho Mendeley: duyệt 129 giờ, cắt cửa sổ, ghi X_vib_windows.npy + window_hour_ids.npy
    + temp_by_hour.csv.
    Tham số test nhanh: max_files (chỉ vài giờ đầu), max_rows (đọc 1 phần mỗi file).
    """
    logger = get_logger("segmentation")
    section(f"PHASE 2 — CẮT CỬA SỔ (L={L}, overlap={overlap}, "
            f"max_windows/giờ={max_windows_per_hour})", logger)
    out_dir = Path(out_dir or PROC_MENDELEY)               # thư mục lưu (mặc định trên Drive)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)                                   # cố định seed (lấy mẫu cửa sổ là đều/deterministic)
    all_windows = []                                       # gom cửa sổ của mọi giờ
    all_hour_ids = []                                      # gom hour_id tương ứng
    temp_rows = []                                         # gom nhiệt độ theo giờ

    # Duyệt từng giờ (generator nạp từng file -> tiết kiệm RAM).
    files = mendeley_loader.list_mendeley_files()
    n_hours = len(files) if max_files is None else min(max_files, len(files))
    gen = mendeley_loader.iter_mendeley(max_rows=max_rows, max_files=max_files)

    with Timer("Cắt cửa sổ toàn bộ Mendeley", logger):
        for hour_id, data in progress_iter(gen, total=n_hours, desc="Cắt cửa sổ theo giờ"):
            vib = data["vib"]                              # (2, T) tín hiệu rung giờ này
            wins = make_windows_from_signal(vib, L=L, overlap=overlap,
                                            max_windows=max_windows_per_hour)
            all_windows.append(wins)                       # thêm cửa sổ giờ này
            all_hour_ids.append(np.full(len(wins), hour_id, dtype=np.int32))  # gắn hour_id
            # Ghi nhận nhiệt độ trung bình giờ này (dùng cho G_h và temp encoder).
            tb = data["temp_bearing_mean"]                 # nhiệt độ ổ bi
            ta = data["temp_atm_mean"]                     # nhiệt độ môi trường
            temp_rows.append({"hour_id": hour_id, "T_bearing_mean": tb,
                              "T_atm_mean": ta, "DeltaT_mean": tb - ta,
                              "timestamp": data.get("timestamp")})
            logger.info(f"  giờ {hour_id:3d}: {wins.shape[0]} cửa sổ | "
                        f"T_bearing={tb:.2f}°C ΔT={tb-ta:.2f}°C")

    # Gộp tất cả cửa sổ thành 1 mảng lớn [N, 2, L].
    X = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, 2, L), np.float32)
    hour_ids = np.concatenate(all_hour_ids, axis=0) if all_hour_ids else np.empty((0,), np.int32)

    # Ghi ra đĩa (Drive).
    np.save(out_dir / "X_vib_windows.npy", X)              # mảng cửa sổ
    np.save(out_dir / "window_hour_ids.npy", hour_ids)     # nhãn giờ cho từng cửa sổ
    pd.DataFrame(temp_rows).to_csv(out_dir / "temp_by_hour.csv", index=False)  # nhiệt độ theo giờ

    logger.info(f"HOÀN TẤT PHASE 2: X={X.shape} ({X.nbytes/1e6:.1f} MB) | "
                f"{len(temp_rows)} giờ -> lưu tại {out_dir}")
    return {
        "X_path": out_dir / "X_vib_windows.npy",
        "hour_ids_path": out_dir / "window_hour_ids.npy",
        "temp_path": out_dir / "temp_by_hour.csv",
        "n_windows": int(X.shape[0]),
        "n_hours": len(temp_rows),
    }
