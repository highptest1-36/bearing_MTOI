# -*- coding: utf-8 -*-
"""
phase2_preprocess.py — PHASE 2: CẮT CỬA SỔ TÍN HIỆU RUNG.

Đọc 129 CSV -> cắt cửa sổ [2, L] -> ghi:
  data/processed/mendeley/X_vib_windows.npy
  data/processed/mendeley/window_hour_ids.npy
  data/processed/mendeley/temp_by_hour.csv
Lệnh chạy:  !python scripts/phase2_preprocess.py
Mẹo TEST nhanh: sửa configs/main_mendeley.yaml -> segmentation.max_files: 5 (chỉ 5 giờ).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import PROC_MENDELEY
from src.segmentation import build_mendeley_windows

PHASE = "phase2_preprocess"


def main(config="configs/main_mendeley.yaml", force=False):
    setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)                      # đọc cấu hình

    outputs = [PROC_MENDELEY / f for f in
               ("X_vib_windows.npy", "window_hour_ids.npy", "temp_by_hour.csv")]
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return

    tracker.mark_running(PHASE)
    seg = cfg.segmentation
    res = build_mendeley_windows(
        L=seg.window_length, overlap=seg.overlap,
        max_windows_per_hour=seg.max_windows_per_hour,
        max_files=seg.get("max_files"), max_rows=seg.get("max_rows"),
        seed=cfg.get("seed", 42),
    )
    tracker.mark_done(PHASE, outputs=outputs, meta={"n_windows": res["n_windows"],
                                                    "n_hours": res["n_hours"]})
    print(f"\n[{PHASE}] HOÀN TẤT. {res['n_windows']} cửa sổ / {res['n_hours']} giờ.")


if __name__ == "__main__":
    main()
