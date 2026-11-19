# -*- coding: utf-8 -*-
"""
make_splits.py — TẠO FILE CHIA TẬP CHUYÊN NGHIỆP (leakage-free, chronological).

Ghi data/splits/mendeley_split.csv với mỗi giờ 1 dòng:
  hour_id | timestamp | split (train/val/test) | RUL | stage_label | warning_label
Cách chia ĐÚNG như khi train (chia theo THỜI GIAN, không random theo cửa sổ -> chống leakage).
Lệnh chạy:  python scripts/make_splits.py
ĐIỀU KIỆN: đã chạy Phase 4 (có labels_by_hour.csv).
"""

import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.config import load_config
from src.utils.paths import PROC_MENDELEY, SPLITS_DIR
from src.utils.logger import get_logger, section
from src.datasets import chronological_split


def main(config="configs/main_mendeley.yaml"):
    setup_environment(do_mount=True)
    logger = get_logger("make_splits")
    cfg = load_config(ROOT / config)
    section("TẠO FILE CHIA TẬP (data/splits/mendeley_split.csv)", logger)

    labels = pd.read_csv(PROC_MENDELEY / "labels_by_hour.csv")
    temp = pd.read_csv(PROC_MENDELEY / "temp_by_hour.csv")[["hour_id", "timestamp"]]
    df = labels.merge(temp, on="hour_id", how="left").sort_values("hour_id").reset_index(drop=True)

    # Chia theo thời gian (cùng tham số với lúc train).
    train_h, val_h, test_h = chronological_split(df["hour_id"].to_numpy(),
                                                 train=cfg.split.train, val=cfg.split.val)
    def which(h):
        return "train" if h in train_h else ("val" if h in val_h else "test")
    df["split"] = df["hour_id"].apply(which)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLITS_DIR / "mendeley_split.csv"
    cols = ["hour_id", "timestamp", "split", "RUL", "Deg", "stage_label", "warning_label"]
    df[cols].to_csv(out, index=False)

    # In tóm tắt số giờ mỗi tập.
    counts = df["split"].value_counts().to_dict()
    logger.info(f"Đã ghi {out}")
    logger.info(f"Số giờ mỗi tập: train={counts.get('train',0)} | "
                f"val={counts.get('val',0)} | test={counts.get('test',0)}")
    print(df[cols].groupby("split").agg(n=("hour_id", "count"),
                                        hour_min=("hour_id", "min"),
                                        hour_max=("hour_id", "max")).to_string())
    return out


if __name__ == "__main__":
    main()
