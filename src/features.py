# -*- coding: utf-8 -*-
"""
features.py — PHASE 3: TRÍCH ĐẶC TRƯNG TÍN HIỆU (theo plan mục 7-8).

Với mỗi cửa sổ rung (2, L) ta tính 5 đặc trưng cho CẢ trục x và y:
  - RMS            : năng lượng rung      = sqrt(mean(x^2))
  - Kurtosis       : mức xung/va đập      = mean((x-μ)^4) / (σ^4 + eps)   (KHÔNG trừ 3)
  - Crest factor   : độ nhọn tín hiệu     = max|x| / (RMS + eps)
  - Spectral energy: năng lượng phổ       = sum |FFT(x)|^2 trên dải B
  - Envelope spec. : bắt impulse hỏng     = sum |FFT(envelope_Hilbert(x))|^2 trên dải B

Sau đó GỘP về cấp GIỜ bằng MEDIAN (trung vị) để giảm nhiễu/outlier (plan mục 8):
  f_h = median_j ( f_{h,j} )

Đầu ra:
  - window_features.csv : mỗi cửa sổ 1 dòng (10 cột đặc trưng + hour_id).
  - hour_features.csv   : mỗi giờ 1 dòng (10 đặc trưng rung + 3 cột nhiệt độ).
"""

import numpy as np                 # tính toán
import pandas as pd                # bảng
from pathlib import Path
from scipy.signal import hilbert   # biến đổi Hilbert -> đường bao (envelope)

from src.utils.paths import PROC_MENDELEY
from src.utils.logger import get_logger, section, progress_iter, Timer

EPS = 1e-12                        # số rất nhỏ tránh chia cho 0

# Tên 10 cột đặc trưng cấp cửa sổ (x và y cho mỗi loại).
FEATURE_NAMES = [
    "RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y",
    "SE_x", "SE_y", "ESE_x", "ESE_y",
]


def _rms(x):
    """RMS = căn bậc hai của trung bình bình phương (năng lượng rung)."""
    return np.sqrt(np.mean(x * x) + EPS)


def _kurtosis(x):
    """Kurtosis (Pearson, không trừ 3) = mean((x-μ)^4) / (σ^4 + eps)."""
    mu = np.mean(x)                                    # trung bình
    sigma2 = np.mean((x - mu) ** 2)                    # phương sai (σ^2)
    num = np.mean((x - mu) ** 4)                       # moment bậc 4
    return num / (sigma2 ** 2 + EPS)                   # chia cho σ^4


def _crest_factor(x):
    """Crest factor = biên độ đỉnh / RMS = max|x| / (RMS + eps)."""
    return np.max(np.abs(x)) / (_rms(x) + EPS)


def _spectral_energy(x, band=None):
    """
    Năng lượng phổ = tổng |FFT(x)|^2 trên dải tần B.
    band=None -> dùng toàn phổ (bỏ thành phần 1 chiều DC ở index 0).
    band=(lo, hi) -> chỉ lấy các bin tần số trong khoảng [lo, hi] (theo CHỈ SỐ bin).
    """
    X = np.fft.rfft(x)                                 # FFT 1 phía (tín hiệu thực)
    p = np.abs(X) ** 2                                 # phổ công suất
    p[0] = 0.0                                         # bỏ DC (thành phần tần số 0)
    if band is not None:                              # nếu giới hạn dải tần
        lo, hi = band
        mask = np.zeros_like(p, dtype=bool)
        mask[lo:hi] = True
        p = p * mask
    return float(np.sum(p))                            # tổng năng lượng


def _envelope_spectral_energy(x, band=None):
    """
    Envelope-spectrum energy: lấy đường bao Hilbert rồi tính năng lượng phổ của đường bao.
    Rất nhạy với impulse/điều biên do hỏng ổ bi (bearing modulation).
    """
    analytic = hilbert(x)                              # tín hiệu giải tích (phức)
    env = np.abs(analytic)                             # đường bao = |tín hiệu giải tích|
    env = env - np.mean(env)                           # bỏ thành phần 1 chiều của đường bao
    return _spectral_energy(env, band=band)            # năng lượng phổ của đường bao


def window_feature_vector(win, band=None):
    """
    Tính 10 đặc trưng cho 1 cửa sổ (2, L).
    win[0] = trục x, win[1] = trục y. Trả về vector 10 phần tử theo thứ tự FEATURE_NAMES.
    """
    x, y = win[0], win[1]                              # tách 2 trục
    return np.array([
        _rms(x), _rms(y),                              # RMS_x, RMS_y
        _kurtosis(x), _kurtosis(y),                    # Kurt_x, Kurt_y
        _crest_factor(x), _crest_factor(y),            # CF_x, CF_y
        _spectral_energy(x, band), _spectral_energy(y, band),    # SE_x, SE_y
        _envelope_spectral_energy(x, band), _envelope_spectral_energy(y, band),  # ESE_x, ESE_y
    ], dtype=np.float64)


def build_features(proc_dir=None, band=None):
    """
    PHASE 3: đọc X_vib_windows.npy + window_hour_ids.npy + temp_by_hour.csv (từ Phase 2),
    tính đặc trưng từng cửa sổ, gộp median theo giờ, ghi window_features.csv & hour_features.csv.
    """
    logger = get_logger("features")
    section("PHASE 3 — TRÍCH ĐẶC TRƯNG (window & hour level)", logger)
    proc_dir = Path(proc_dir or PROC_MENDELEY)

    # --- Nạp đầu vào từ Phase 2 ---
    X = np.load(proc_dir / "X_vib_windows.npy")        # [N, 2, L]
    hour_ids = np.load(proc_dir / "window_hour_ids.npy")  # [N]
    temp_df = pd.read_csv(proc_dir / "temp_by_hour.csv")   # nhiệt độ theo giờ
    logger.info(f"Nạp {X.shape[0]} cửa sổ, {temp_df.shape[0]} giờ.")

    # --- Tính đặc trưng từng cửa sổ ---
    feats = np.zeros((X.shape[0], len(FEATURE_NAMES)), dtype=np.float64)
    with Timer("Tính đặc trưng từng cửa sổ", logger):
        for i in progress_iter(range(X.shape[0]), total=X.shape[0], desc="Đặc trưng cửa sổ"):
            feats[i] = window_feature_vector(X[i], band=band)

    # Tạo bảng đặc trưng cấp cửa sổ.
    win_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    win_df.insert(0, "hour_id", hour_ids)              # thêm cột hour_id ở đầu
    win_df.to_csv(proc_dir / "window_features.csv", index=False)
    logger.info(f"Đã ghi window_features.csv ({win_df.shape}).")

    # --- Gộp về cấp GIỜ bằng MEDIAN ---
    hour_feat = win_df.groupby("hour_id")[FEATURE_NAMES].median().reset_index()
    # Ghép thêm 3 cột nhiệt độ theo hour_id.
    hour_feat = hour_feat.merge(
        temp_df[["hour_id", "T_bearing_mean", "T_atm_mean", "DeltaT_mean"]],
        on="hour_id", how="left")
    hour_feat = hour_feat.sort_values("hour_id").reset_index(drop=True)
    hour_feat.to_csv(proc_dir / "hour_features.csv", index=False)
    logger.info(f"Đã ghi hour_features.csv ({hour_feat.shape}).")
    print(hour_feat.head().to_string(index=False))     # xem nhanh 5 giờ đầu

    return {
        "window_features": proc_dir / "window_features.csv",
        "hour_features": proc_dir / "hour_features.csv",
    }
