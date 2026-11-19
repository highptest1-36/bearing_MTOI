# -*- coding: utf-8 -*-
"""
cross_dataset.py — PHASE 9: KIỂM CHỨNG CHÉO trên PRONOSTIA & XJTU-SY (plan mục 16, Exp 6).

Mục tiêu: chứng minh MTOI GENERALIZE sang dataset khác và VẪN CHẠY khi THIẾU nhiệt độ
(modality-adaptive). Vì vậy ở đây ta dùng phiên bản MTOI RÚT GỌN (vib-only): MTOI = a·Ẽ + b·C̃.

Cách làm cho MỖI bearing run:
  1) Duyệt từng snapshot (hour/minute) -> cắt cửa sổ -> tính 10 đặc trưng rung -> median theo snapshot
     => tạo hour_features.csv (chỉ phần rung).
  2) Gọi lại src.mtoi.build_mtoi(has_temp=False) -> mtoi_by_hour.csv (E,C -> MTOI).
  3) Tính chất lượng HI: monotonicity, |Spearman với độ suy giảm|, lead time.

Lưu kết quả tổng: results/tables/table6_cross_dataset_results.csv.
GHI CHÚ: dùng vib-only cho cả PRONOSTIA & XJTU để pipeline đồng nhất & đơn giản; muốn dùng
         nhiệt độ của PRONOSTIA (fusion) là phần mở rộng — xem ghi chú trong HUONG_DAN.md.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import PROC_PRONOSTIA, PROC_XJTU, TABLES_DIR
from src.utils.logger import get_logger, section, progress_iter, Timer
from src.segmentation import make_windows_from_signal
from src.features import window_feature_vector, FEATURE_NAMES
from src.mtoi import build_mtoi
from src import metrics


def build_hour_features_generic(out_dir, hour_iter, L=4096, overlap=0.5,
                                max_windows_per_hour=32, band=None, logger=None):
    """
    Tạo hour_features.csv (chỉ đặc trưng rung) cho 1 bearing từ generator (hour_id, vib(2,N)).
    Ghi vào out_dir/hour_features.csv và trả về số snapshot.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hour_id, vib in hour_iter:                         # duyệt từng snapshot
        wins = make_windows_from_signal(vib, L=L, overlap=overlap,
                                        max_windows=max_windows_per_hour)
        if len(wins) == 0:                                 # snapshot quá ngắn -> bỏ
            continue
        # Tính đặc trưng từng cửa sổ rồi lấy median -> 1 vector/snapshot.
        feats = np.stack([window_feature_vector(w, band=band) for w in wins], axis=0)
        med = np.median(feats, axis=0)
        row = {"hour_id": hour_id}
        row.update({name: med[i] for i, name in enumerate(FEATURE_NAMES)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("hour_id").reset_index(drop=True)
    df.to_csv(out_dir / "hour_features.csv", index=False)
    if logger:
        logger.info(f"  -> {len(df)} snapshot, ghi {out_dir/'hour_features.csv'}")
    return len(df)


def run_one_bearing(name, hour_iter, out_dir, L=4096, overlap=0.5,
                    max_windows_per_hour=32, temp_array=None, logger=None):
    """
    Chạy pipeline MTOI cho 1 bearing -> trả về 1 dòng metric.
    temp_array: nếu truyền (PRONOSTIA có nhiệt độ) -> xây MTOI ĐẦY ĐỦ (E+C+G, fusion nhiệt độ);
                nếu None (XJTU hoặc PRONOSTIA không có nhiệt độ) -> MTOI vib-only (E+C) — modality-adaptive.
    """
    logger = logger or get_logger("cross")
    n = build_hour_features_generic(out_dir, hour_iter, L=L, overlap=overlap,
                                    max_windows_per_hour=max_windows_per_hour, logger=logger)
    if n < 5:                                              # quá ít snapshot -> bỏ qua
        logger.info(f"  Bỏ qua {name} (chỉ {n} snapshot).")
        return None

    has_temp = temp_array is not None                     # có dùng nhiệt độ không
    if has_temp:
        # GẮN nhiệt độ vào hour_features để MTOI dùng được (PRONOSTIA chỉ có nhiệt độ Ổ BI, không có
        # ambient -> dùng nhiệt độ ổ bi cho CẢ abnormality (qua cột nhiệt độ) và xu hướng G).
        df = pd.read_csv(out_dir / "hour_features.csv")
        # Nội suy nhiệt độ về đúng số hàng của hour_features (đảm bảo khớp 1-1 theo thứ tự thời gian).
        t = np.asarray(temp_array, dtype=float)
        xq = np.linspace(0.0, 1.0, len(df)); xp = np.linspace(0.0, 1.0, len(t))
        t_aligned = np.interp(xq, xp, t)
        df["T_bearing_mean"] = t_aligned                  # nhiệt độ ổ bi mỗi snapshot
        df["T_atm_mean"] = 0.0                            # không có ambient -> đặt 0 (hằng số)
        df["DeltaT_mean"] = t_aligned                     # G dùng độ dốc nhiệt độ ổ bi
        df.to_csv(out_dir / "hour_features.csv", index=False)
        logger.info(f"  -> gắn nhiệt độ ({len(t_aligned)} snapshot) -> MTOI ĐẦY ĐỦ (vib+temp).")

    # Xây MTOI: has_temp=True (E+C+G, fusion) cho PRONOSTIA có nhiệt độ; False (E+C) cho còn lại.
    build_mtoi(proc_dir=out_dir, baseline_frac=0.2, train_frac=0.6,
               has_temp=has_temp, dataset_name=name)
    mtoi = pd.read_csv(out_dir / "mtoi_by_hour.csv")
    time_ref = mtoi["deg_ref"].to_numpy()                  # thời gian chuẩn hoá 0->1 (độ suy giảm theo thời gian)

    # QUAN TRỌNG (đã sửa theo review): trọng số LEARNABLE được fit để TỐI ĐA hoá tương quan với
    # chính time_ref -> báo cáo spearman(MTOI_learnable, time_ref) sẽ TỰ TUẦN HOÀN (thổi phồng).
    # Vì vậy chỉ số GENERALIZATION dùng MTOI_fixed (trọng số CỐ ĐỊNH, KHÔNG fit theo time_ref) ->
    # tương quan với thời gian là thước đo TRUNG THỰC. Monotonicity của learnable vẫn báo cáo tham khảo.
    hi_fixed = mtoi["MTOI_fixed"].to_numpy()
    hi_learn = mtoi["MTOI_learnable"].to_numpy()
    failure_hour = len(hi_fixed) - 1
    hi_n = (hi_fixed - hi_fixed.min()) / (hi_fixed.max() - hi_fixed.min() + 1e-9)
    onset = max(1, int(0.6 * len(hi_n)))
    ew = metrics.early_warning_metrics(hi_n, tau=0.6, failure_hour=failure_hour, onset_hour=onset, q=3)
    return {
        "bearing": name, "n_snapshots": int(n),
        "mtoi_type": ("vib+temp (full, fixed-weight)" if has_temp else "vib-only (fixed-weight)"),
        "monotonicity_fixed": round(metrics.monotonicity(hi_fixed), 4),
        "monotonicity_learnable": round(metrics.monotonicity(hi_learn), 4),
        "spearman_with_time": round(abs(metrics.spearman_corr(hi_fixed, time_ref)), 4),
        "lead_time": ew["lead_time"],
        "false_alarm_rate": round(ew["false_alarm_rate"], 4),
    }


def run_xjtu(max_bearings=6, L=4096, overlap=0.5, max_windows_per_hour=32):
    """Chạy cross-validation trên vài bearing XJTU-SY (vib-only)."""
    from src.data import xjtu_loader
    logger = get_logger("cross")
    section("PHASE 9 — CROSS-DATASET: XJTU-SY (vib-only MTOI)", logger)
    bearings = xjtu_loader.find_bearing_dirs()
    rows = []
    for i, (name, bdir) in enumerate(list(bearings.items())[:max_bearings]):
        logger.info(f"[XJTU] ({i+1}) bearing {name}")
        out_dir = PROC_XJTU / name
        with Timer(f"XJTU {name}", logger):
            row = run_one_bearing(f"xjtu_{name}", xjtu_loader.iter_xjtu_bearing(bdir),
                                  out_dir, L=L, overlap=overlap,
                                  max_windows_per_hour=max_windows_per_hour, logger=logger)
        if row:
            row["dataset"] = "XJTU-SY"; rows.append(row)
    return rows


def run_pronostia(max_bearings=4, L=2560, overlap=0.0, max_windows_per_hour=1):
    """Chạy cross-validation trên vài bearing PRONOSTIA (vib-only)."""
    from src.data import pronostia_loader
    logger = get_logger("cross")
    section("PHASE 9 — CROSS-DATASET: PRONOSTIA/FEMTO (vib-only MTOI)", logger)
    bearings = pronostia_loader.find_bearing_dirs()
    rows = []
    for i, (name, bdir) in enumerate(list(bearings.items())[:max_bearings]):
        logger.info(f"[PRONOSTIA] ({i+1}) bearing {name}")
        out_dir = PROC_PRONOSTIA / name
        # NẠP NHIỆT ĐỘ (nếu bearing có) -> MTOI ĐẦY ĐỦ vib+temp; nếu không có -> tự dùng vib-only.
        n_acc = pronostia_loader.count_acc_files(bdir)
        temp_arr = pronostia_loader.load_pronostia_temperature_aligned(bdir, n_acc)
        if temp_arr is not None:
            logger.info(f"  bearing {name} CÓ nhiệt độ -> dùng fusion (vib+temp).")
        else:
            logger.info(f"  bearing {name} KHÔNG có nhiệt độ -> dùng vib-only (adaptive).")
        with Timer(f"PRONOSTIA {name}", logger):
            row = run_one_bearing(f"pronostia_{name}", pronostia_loader.iter_pronostia_bearing(bdir),
                                  out_dir, L=L, overlap=overlap,
                                  max_windows_per_hour=max_windows_per_hour,
                                  temp_array=temp_arr, logger=logger)
        if row:
            row["dataset"] = "PRONOSTIA"; rows.append(row)
    return rows


def run_cross_dataset(do_xjtu=True, do_pronostia=True, max_bearings=4):
    """Chạy toàn bộ Phase 9 và ghi bảng tổng hợp Table 6."""
    logger = get_logger("cross")
    rows = []
    if do_pronostia:
        rows += run_pronostia(max_bearings=max_bearings)
    if do_xjtu:
        rows += run_xjtu(max_bearings=max_bearings)
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table6_cross_dataset_results.csv"
    table.to_csv(out, index=False)
    logger.info(f"Đã ghi {out}")
    if len(table):
        print(table.to_string(index=False))
    return table
