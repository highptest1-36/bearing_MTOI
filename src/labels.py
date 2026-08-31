# -*- coding: utf-8 -*-
"""
labels.py — PHASE 5: TẠO NHÃN RUL / Degradation / Stage / Early-warning (plan mục 11).

Quy ước thời điểm hỏng: H_fail = giờ cuối cùng (H-1) vì đây là dữ liệu run-to-failure.
  RUL_h = (H_fail - h) / H_fail   -> 1 ở đầu đời, 0 lúc hỏng.
  Deg_h = 1 - RUL_h = h / H_fail  -> 0 ở đầu đời, 1 lúc hỏng.

Stage (4 mức) — bản khởi đầu theo tỉ lệ vòng đời p_h = h/H_fail:
  p<0.60 -> 0 Normal | 0.60<=p<0.80 -> 1 Early | 0.80<=p<0.95 -> 2 Degraded | p>=0.95 -> 3 Critical

Stage — bản mạnh hơn (tuỳ chọn): dựa vào ngưỡng MTOI thay cho tỉ lệ thời gian.

Early-warning: Warning_h = 1 nếu MTOI_h > τ liên tục q giờ (mặc định τ=0.6, q=3).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import PROC_MENDELEY
from src.utils.logger import get_logger, section
from src import metrics


# Cadence lấy mẫu mỗi snapshot theo dataset (GIÂY/snapshot) — để quy RUL về GIỜ tuyệt đối.
#   PRONOSTIA: 1 file acc mỗi 10 s | XJTU-SY: 1 file mỗi 1 phút | Mendeley: 1 file mỗi 1 giờ.
#   IMS: 1 snapshot mỗi 10 PHÚT (readme NASA; 43 file đầu của test 1 là 5 phút — bỏ qua sai lệch nhỏ này).
SAMPLE_INTERVAL_S = {"pronostia": 10.0, "xjtu_sy": 60.0, "mendeley": 3600.0, "ims": 600.0}


def make_rul_deg(H):
    """Tạo RUL và Deg cho H giờ. Trả về (rul, deg) — mỗi cái mảng độ dài H."""
    H_fail = max(H - 1, 1)                                # thời điểm hỏng = giờ cuối
    h = np.arange(H)                                      # 0,1,...,H-1
    rul = (H_fail - h) / H_fail                           # giảm dần 1 -> 0
    deg = h / H_fail                                      # tăng dần 0 -> 1
    return rul, deg


def make_stage_by_time(H):
    """Stage 0..3 theo tỉ lệ vòng đời (bản khởi đầu, đơn giản & ổn định)."""
    H_fail = max(H - 1, 1)
    p = np.arange(H) / H_fail                             # tỉ lệ vòng đời
    stage = np.zeros(H, dtype=int)                        # mặc định 0 (Normal)
    stage[(p >= 0.60) & (p < 0.80)] = 1                  # Early degradation
    stage[(p >= 0.80) & (p < 0.95)] = 2                  # Degraded
    stage[p >= 0.95] = 3                                  # Critical
    return stage


def make_stage_by_mtoi(mtoi, thresholds=(0.2, 0.5, 0.8)):
    """
    Stage 0..3 theo ngưỡng MTOI (bản mạnh hơn): cắt theo 3 ngưỡng tăng dần.
    mtoi<t0 -> 0 | t0<=mtoi<t1 -> 1 | t1<=mtoi<t2 -> 2 | mtoi>=t2 -> 3.
    """
    mtoi = np.asarray(mtoi, float)
    t0, t1, t2 = thresholds
    stage = np.zeros(len(mtoi), dtype=int)
    stage[(mtoi >= t0) & (mtoi < t1)] = 1
    stage[(mtoi >= t1) & (mtoi < t2)] = 2
    stage[mtoi >= t2] = 3
    return stage


def make_warning(mtoi, tau=0.6, q=3):
    """Early-warning: 1 nếu MTOI vượt τ liên tục q giờ (dùng luật trong metrics)."""
    _, flags = metrics.warning_from_threshold(mtoi, tau=tau, q=q)
    return flags


def build_labels(proc_dir=None, stage_mode="time", tau=0.6, q=3,
                 mtoi_col="MTOI_learnable", mtoi_thresholds=(0.2, 0.5, 0.8),
                 sample_interval_s=None, cap_onset_frac=0.4):
    """
    PHASE 5: đọc mtoi_by_hour.csv -> tạo labels_by_hour.csv.
    stage_mode: 'time' (theo tỉ lệ vòng đời) hoặc 'mtoi' (theo ngưỡng MTOI).

    RUL-CENTRIC (đơn vị thời gian tuyệt đối — thêm cột, KHÔNG phá reader cũ):
      sample_interval_s : giây/snapshot (PRONOSTIA 10, XJTU 60, Mendeley 3600). None -> đơn vị = snapshot.
      cap_onset_frac    : mốc tỉ-lệ-vòng-đời CỐ ĐỊNH để cắt RUL (piecewise/capped); KHÔNG lấy từ MTOI
                          (tránh circularity). RUL_capped = min(1, (H_fail-h)/((1-cap_onset_frac)*H_fail)).
    Cột thêm: H_fail, RUL_hours, RUL_capped, life_hours, sample_interval_s.
    """
    logger = get_logger("labels")
    section("PHASE 5 — TẠO NHÃN (RUL / Deg / Stage / Warning)", logger)
    proc_dir = Path(proc_dir or PROC_MENDELEY)

    mtoi_df = pd.read_csv(proc_dir / "mtoi_by_hour.csv").sort_values("hour_id").reset_index(drop=True)
    H = len(mtoi_df)
    mtoi = mtoi_df[mtoi_col].to_numpy(float)             # đường MTOI dùng để sinh warning/stage

    rul, deg = make_rul_deg(H)                           # RUL & Deg (chuẩn hoá [0,1], target chính)
    if stage_mode == "mtoi":                            # chọn cách gán stage
        stage = make_stage_by_mtoi(mtoi, thresholds=mtoi_thresholds)
    else:
        stage = make_stage_by_time(H)
    # Lấy CẢ giờ cảnh báo đầu tiên (first) lẫn mảng cờ -> dùng 'first' để log cho chính xác.
    first_warning, warning = metrics.warning_from_threshold(mtoi, tau=tau, q=q)

    # --- Đơn vị thời gian tuyệt đối (cho RUL-centric) ---
    H_fail = max(H - 1, 1)                                # thời điểm hỏng (snapshot cuối)
    h = np.arange(H)
    interval = float(sample_interval_s) if sample_interval_s else 1.0   # giây/snapshot (1 nếu None)
    life_hours = H_fail * interval / 3600.0              # tổng vòng đời (giờ; nếu None -> đơn vị snapshot)
    rul_hours = (H_fail - h) * interval / 3600.0         # giờ còn lại tới hỏng (true)
    R_early = max((1.0 - cap_onset_frac) * H_fail, 1.0)  # độ dài vùng suy giảm (onset CỐ ĐỊNH)
    rul_capped = np.minimum(1.0, (H_fail - h) / R_early) # piecewise: phẳng 1.0 ở vùng khoẻ, dốc về 0

    # --- Quỹ đạo MTOI: vận tốc (ΔMTOI) + gia tốc (Δ²MTOI) -> "suy giảm NHANH hay CHẬM" cho RUL ---
    def _traj(col):
        if col not in mtoi_df.columns or H < 2:
            return np.zeros(H), np.zeros(H)
        x = mtoi_df[col].to_numpy(float)
        vel = np.gradient(x)            # đạo hàm bậc 1 (mượt)
        acc = np.gradient(vel)          # đạo hàm bậc 2
        return vel, acc
    learn_vel, learn_acc = _traj("MTOI_learnable")   # MTOI đầy đủ (E+C+G)
    woG_vel, woG_acc = _traj("MTOI_woG")             # MTOI vib-only (E+C)

    out = pd.DataFrame({
        "hour_id": mtoi_df["hour_id"].to_numpy(),
        "RUL": rul, "Deg": deg,
        "stage_label": stage, "warning_label": warning,
        # --- RUL-centric (additive) ---
        "RUL_hours": rul_hours, "RUL_capped": rul_capped,
        "H_fail": int(H_fail), "life_hours": life_hours,
        "sample_interval_s": interval,
        # --- Quỹ đạo MTOI (cho HI-input dạng trajectory) ---
        "MTOI_learnable_vel": learn_vel, "MTOI_learnable_acc": learn_acc,
        "MTOI_woG_vel": woG_vel, "MTOI_woG_acc": woG_acc,
    })

    # --- QUỸ ĐẠO MTOI: vận tốc ΔMTOI + gia tốc Δ²MTOI (per-bearing, file này = 1 bearing) ---
    #   Cho biết "đang suy giảm NHANH/CHẬM" -> tín hiệu mạnh cho RUL (đưa vào RUL head qua hi).
    def _traj(x):
        x = np.asarray(x, float)
        vel = np.zeros_like(x); acc = np.zeros_like(x)
        if len(x) >= 2: vel[1:] = x[1:] - x[:-1]
        if len(x) >= 3: acc[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
        return vel, acc
    for col in ("MTOI_learnable", "MTOI_woG", "MTOI_fixed"):
        if col in mtoi_df.columns:
            v, a = _traj(mtoi_df[col].to_numpy(float))
            out[f"{col}_vel"] = v; out[f"{col}_acc"] = a

    out_path = proc_dir / "labels_by_hour.csv"
    out.to_csv(out_path, index=False)

    # In phân bố nhãn + đơn vị thời gian để bạn kiểm tra.
    unit = f"{life_hours:.2f} giờ" if sample_interval_s else f"{H_fail} snapshot (interval chưa đặt)"
    logger.info(f"Vòng đời: H_fail={H_fail} snapshot | tổng = {unit} | RUL_hours[0]={rul_hours[0]:.3f}h")
    logger.info(f"Phân bố stage: {np.bincount(stage, minlength=4).tolist()} (Normal/Early/Degraded/Critical)")
    logger.info(f"Số giờ cảnh báo: {int(warning.sum())}/{H} | giờ cảnh báo đầu tiên: "
                f"{first_warning if first_warning is not None else 'không có'}")
    logger.info(f"Đã ghi {out_path}")
    return {"labels_path": out_path}
