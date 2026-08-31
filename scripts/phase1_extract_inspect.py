# -*- coding: utf-8 -*-
"""
phase1_extract_inspect.py — PHASE 1: GIẢI NÉN MENDELEY + KHẢO SÁT DỮ LIỆU.

Việc làm:
  1) Giải nén dataset chính Mendeley (zip -> 129 CSV) ra ổ local /content/MTOI_raw/mendeley.
  2) Khảo sát -> tạo data/processed/dataset_summary.csv.
Có checkpoint: nếu đã làm xong (file tồn tại) thì TỰ BỎ QUA.
Lệnh chạy:  !python scripts/phase1_extract_inspect.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.paths import PROCESSED_DIR
from src.data.extract import extract_mendeley
from src.data.inspect import build_dataset_summary

PHASE = "phase1_extract_inspect"


def main(force=False):
    setup_environment(do_mount=True)                      # môi trường + Drive
    tracker = PhaseTracker()                              # quản lý checkpoint

    summary_path = PROCESSED_DIR / "dataset_summary.csv"
    # Nếu đã xong & file còn -> bỏ qua (resume).
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành trước đó -> BỎ QUA. (xoá để chạy lại: force=True)")
        return

    tracker.mark_running(PHASE)                           # đánh dấu đang chạy
    # 1) Giải nén Mendeley (idempotent — tự bỏ qua nếu đã có 129 CSV).
    extract_mendeley()
    # 2) Khảo sát & ghi summary.
    build_dataset_summary(which=("mendeley",), out_path=summary_path)

    tracker.mark_done(PHASE, outputs=[summary_path])      # đánh dấu xong
    print(f"\n[{PHASE}] HOÀN TẤT. Xem {summary_path}")


if __name__ == "__main__":
    main()
