# -*- coding: utf-8 -*-
"""
phase9a_ims.py — SINH ARTIFACT PER-BEARING CHO DATASET IMS/NASA (external validation, R3 #2).

Tương đương phase9a_build_bearing_artifacts.py nhưng cho IMS. Tách riêng vì:
  - IMS không có zip trên Drive để tự giải nén (tải từ NASA Prognostics Data Repository),
  - IMS chỉ có 4 ổ bi chạy đến hỏng thật -> LOBO 4 fold, dùng làm BẰNG CHỨNG BỔ TRỢ
    trong phụ lục, KHÔNG phải benchmark chính (xem EXPERIMENT_RUNBOOK_v2.md, Tier 4).

Điều kiện tiên quyết: RAW đã giải nén tại $MTOI_RAW_DIR/ims (mặc định /content/MTOI_raw/ims)
với 3 thư mục 1st_test/, 2nd_test/, 4th_test/txt/.

Idempotent: ổ bi đã có labels_by_hour.csv sẽ được BỎ QUA -> chạy lại an toàn sau disconnect.

Chạy:
    python3 scripts/phase9a_ims.py                 # cả 4 ổ bi
    python3 scripts/phase9a_ims.py --max-bearings 1  # smoke test
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment              # noqa: E402
from src.utils.logger import get_logger, section         # noqa: E402
from src.utils.paths import raw_dir_for, proc_dir_for    # noqa: E402
from src.lobo import process_dataset                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bearings", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    setup_environment(seed=42, do_mount=False)
    logger = get_logger("phase9a_ims")
    section("PHASE 9A-IMS — BUILD PER-BEARING ARTIFACTS (IMS/NASA) -> DRIVE", logger)

    raw = raw_dir_for("ims")
    if not Path(raw).exists():
        logger.error(f"Chưa có RAW IMS tại {raw}.")
        logger.error("Tải: https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip")
        logger.error("  unzip -> IMS.7z -> 7z x -> 3 file .rar -> unrar x vào thư mục trên.")
        return 2

    from src.data import ims_loader as ld
    desc = ld.describe_bearings(raw)
    logger.info("Ổ bi IMS dùng được (chỉ các ổ chạy đến hỏng thật):")
    for line in desc.to_string(index=False).splitlines():
        logger.info("  " + line)

    man = process_dataset("ims", max_bearings=a.max_bearings, force=a.force, logger=logger)
    logger.info(f"HOÀN TẤT. Manifest: {proc_dir_for('ims') / '_manifest.csv'} "
                f"({len(man)} ổ bi).")

    # Ghi kèm bảng mô tả để dựng Table I của bài (cadence, số kênh vật lý, dạng hỏng).
    out = proc_dir_for("ims") / "_bearing_description.csv"
    desc.to_csv(out, index=False)
    logger.info(f"Ghi mô tả dataset cho Table I: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
