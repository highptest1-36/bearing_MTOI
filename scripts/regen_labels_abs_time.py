# -*- coding: utf-8 -*-
"""
regen_labels_abs_time.py — TÁI TẠO labels_by_hour.csv (THÊM cột thời gian tuyệt đối) cho MỌI
artifact đã xử lý, KHÔNG cần giải nén/extract lại vibration.

Vì sao cần: RUL-centric báo cáo MAE/RMSE theo GIỜ. labels_by_hour.csv cũ chỉ có RUL chuẩn hoá.
build_labels() bản mới thêm H_fail/RUL_hours/RUL_capped/life_hours/sample_interval_s từ
mtoi_by_hour.csv (đã có sẵn) -> chỉ đọc/ghi CSV nhỏ, vài giây.

  - PRONOSTIA & XJTU: lặp mọi bearing trong _manifest.csv, build_labels với interval theo dataset.
  - Mendeley: build_labels với interval 3600 s, giữ stage_mode/tau/q theo config (không đổi nhãn cũ).

Nhãn RUL/Deg/stage/warning KHÔNG đổi (cùng công thức) — chỉ THÊM cột. Idempotent.

Chạy:  python scripts/regen_labels_abs_time.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.paths import proc_dir_for, PROC_MENDELEY
from src.utils.logger import get_logger, section
from src.labels import build_labels, SAMPLE_INTERVAL_S


def regen_perbearing(dataset, logger):
    proc_root = proc_dir_for(dataset)
    man_path = proc_root / "_manifest.csv"
    if not man_path.exists():
        logger.info(f"[{dataset}] chưa có manifest -> bỏ qua."); return 0
    man = pd.read_csv(man_path)
    interval = SAMPLE_INTERVAL_S.get(dataset)
    n = 0
    for bearing in man["bearing"].tolist():
        bdir = proc_root / bearing
        if not (bdir / "mtoi_by_hour.csv").exists():
            logger.info(f"  [skip] {bearing}: thiếu mtoi_by_hour.csv"); continue
        # Khớp tham số mà lobo.build_bearing_artifacts đã dùng (stage_mode='time', mtoi_col learnable).
        build_labels(proc_dir=bdir, stage_mode="time", tau=0.6, q=3,
                     mtoi_col="MTOI_learnable", sample_interval_s=interval)
        n += 1
    logger.info(f"[{dataset}] regen {n}/{len(man)} bearing (interval={interval}s).")
    return n


def regen_mendeley(logger):
    if not (PROC_MENDELEY / "mtoi_by_hour.csv").exists():
        logger.info("[mendeley] thiếu mtoi_by_hour.csv -> bỏ qua."); return 0
    # Giữ stage_mode/tau/q theo config để KHÔNG đổi nhãn cũ của Mendeley.
    try:
        from src.utils.config import load_config
        cfg = load_config(ROOT / "configs/main_mendeley.yaml")
        sm, tau, q = cfg.labels.stage_mode, cfg.labels.tau, cfg.labels.q
    except Exception:
        sm, tau, q = "time", 0.6, 3
    build_labels(proc_dir=PROC_MENDELEY, stage_mode=sm, tau=tau, q=q,
                 sample_interval_s=SAMPLE_INTERVAL_S["mendeley"])
    logger.info("[mendeley] regen labels (interval=3600s).")
    return 1


def main():
    logger = get_logger("regen_labels")
    section("REGEN LABELS — thêm đơn vị thời gian tuyệt đối", logger)
    total = 0
    total += regen_mendeley(logger)
    for ds in ("pronostia", "xjtu_sy"):
        total += regen_perbearing(ds, logger)
    logger.info(f"HOÀN TẤT. Đã regen {total} labels_by_hour.csv (thêm H_fail/RUL_hours/life_hours...).")


if __name__ == "__main__":
    main()
