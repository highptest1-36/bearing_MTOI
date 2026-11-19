# -*- coding: utf-8 -*-
"""
phase9_cross_dataset.py — PHASE 9: KIỂM CHỨNG CHÉO PRONOSTIA & XJTU-SY.

Việc làm:
  1) Giải nén PRONOSTIA (zip-trong-zip) và XJTU-SY (zip-chứa-RAR-nhiều-phần).
  2) Chạy MTOI vib-only cho vài bearing mỗi dataset -> đo monotonicity/Spearman/lead time.
  3) Ghi results/tables/table6_cross_dataset_results.csv.
Lệnh chạy:  !python scripts/phase9_cross_dataset.py
LƯU Ý: bước giải nén XJTU cần 'unrar' (Colab thường có; nếu thiếu: !apt-get -qq install -y unrar).
       Bước này NẶNG & TỐN ĐĨA — chỉ chạy sau khi pipeline chính (Phase 1-8) đã ổn.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.paths import TABLES_DIR
from src.data.extract import extract_pronostia, extract_xjtu
from src.cross_dataset import run_cross_dataset

PHASE = "phase9_cross_dataset"


def main(do_pronostia=True, do_xjtu=True, max_bearings=4, force=False):
    setup_environment(do_mount=True)
    tracker = PhaseTracker()
    out_path = TABLES_DIR / "table6_cross_dataset_results.csv"
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    # 1) Giải nén (idempotent).
    if do_pronostia:
        extract_pronostia()
    if do_xjtu:
        extract_xjtu()

    # 2) Chạy cross-dataset.
    run_cross_dataset(do_xjtu=do_xjtu, do_pronostia=do_pronostia, max_bearings=max_bearings)

    tracker.mark_done(PHASE, outputs=[out_path])
    print(f"\n[{PHASE}] HOÀN TẤT. Bảng cross-dataset: {out_path}")


if __name__ == "__main__":
    main()
