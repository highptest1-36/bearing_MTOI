# -*- coding: utf-8 -*-
"""
phase3_features.py — PHASE 3: TRÍCH ĐẶC TRƯNG (RMS, Kurtosis, CF, SE, ESE).

Đọc X_vib_windows.npy -> tính đặc trưng từng cửa sổ -> gộp median theo giờ. Ghi:
  data/processed/mendeley/window_features.csv
  data/processed/mendeley/hour_features.csv
Lệnh chạy:  !python scripts/phase3_features.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.paths import PROC_MENDELEY
from src.features import build_features

PHASE = "phase3_features"


def main(force=False):
    setup_environment(do_mount=True)
    tracker = PhaseTracker()
    outputs = [PROC_MENDELEY / "window_features.csv", PROC_MENDELEY / "hour_features.csv"]
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)
    build_features(proc_dir=PROC_MENDELEY)
    tracker.mark_done(PHASE, outputs=outputs)
    print(f"\n[{PHASE}] HOÀN TẤT.")


if __name__ == "__main__":
    main()
