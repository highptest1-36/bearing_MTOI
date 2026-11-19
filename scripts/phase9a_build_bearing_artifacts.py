# -*- coding: utf-8 -*-
"""
phase9a_build_bearing_artifacts.py — XỬ LÝ ĐA-BEARING -> GHI ARTIFACT LÊN DRIVE.

Mục tiêu (đáp ứng yêu cầu "extract lưu đúng chỗ thư mục"):
  - Với MỖI bearing của PRONOSTIA & XJTU-SY, tạo ĐỦ 6 file (giống Mendeley) và GHI LÊN DRIVE:
      data/processed/pronostia/<bearing>/{X_vib_windows.npy, window_hour_ids.npy,
                                          temp_by_hour.csv, hour_features.csv,
                                          mtoi_by_hour.csv, labels_by_hour.csv}
      data/processed/xjtu_sy/<bearing>/{...}
  - Ghi manifest: data/processed/<dataset>/_manifest.csv

Idempotent: bearing đã có labels_by_hour.csv sẽ được BỎ QUA (chạy lại an toàn sau disconnect).
RAW tự giải nén lại nếu thiếu (từ zip trên Drive) — KHÔNG mất công.

Chạy:  python scripts/phase9a_build_bearing_artifacts.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.logger import get_logger, section
from src.utils.checkpoint import PhaseTracker
from src.utils.paths import raw_dir_for
from src.data.extract import extract_pronostia, extract_xjtu
from src.lobo import process_dataset


def _ensure_raw(dataset, logger):
    """Giải nén RAW nếu thiếu (raw ở /content local, mất khi disconnect -> tự bù)."""
    raw = raw_dir_for(dataset)
    has = any(raw.rglob("*.csv")) if raw.exists() else False
    if has:
        logger.info(f"[{dataset}] RAW đã có ({raw}).")
        return
    logger.info(f"[{dataset}] RAW thiếu -> giải nén lại từ zip trên Drive ...")
    if dataset == "pronostia":
        extract_pronostia()
    else:
        extract_xjtu()


def main(do_pronostia=True, do_xjtu=True, max_bearings=None, force=False):
    device = setup_environment(seed=42, do_mount=False)
    logger = get_logger("phase9a")
    tracker = PhaseTracker()
    section("PHASE 9A — BUILD PER-BEARING ARTIFACTS (PRONOSTIA + XJTU) -> DRIVE", logger)
    logger.info(f"Device: {device}")

    phase = "phase9a_build_bearing_artifacts"
    if tracker.is_done(phase) and not force:
        logger.info("Phase 9A đã DONE (theo state.json). Dùng force=True để chạy lại.")
        return
    tracker.mark_running(phase)

    outputs = []
    if do_pronostia:
        _ensure_raw("pronostia", logger)
        man = process_dataset("pronostia", max_bearings=max_bearings, force=force, logger=logger)
        outputs.append(str((raw_dir_for("pronostia"))))  # placeholder
    if do_xjtu:
        _ensure_raw("xjtu_sy", logger)
        man = process_dataset("xjtu_sy", max_bearings=max_bearings, force=force, logger=logger)

    from src.utils.paths import PROC_PRONOSTIA, PROC_XJTU
    outs = [str(PROC_PRONOSTIA / "_manifest.csv"), str(PROC_XJTU / "_manifest.csv")]
    tracker.mark_done(phase, outputs=outs)
    logger.info("HOÀN TẤT Phase 9A. Artifact per-bearing đã ghi lên Drive (persistent).")


if __name__ == "__main__":
    main()
