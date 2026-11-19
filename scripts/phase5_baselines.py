# -*- coding: utf-8 -*-
"""
phase5_baselines.py — PHASE 5: BASELINES (HI comparison + ML truyền thống).

Tạo:
  results/tables/table3_health_indicator_comparison.csv  (so MTOI với RMS/Kurt/Temp/PCA/AE...)
  results/tables/baseline_ml_results.csv                 (SVM/RF/XGBoost cho stage & RUL)
Lệnh chạy:  !python scripts/phase5_baselines.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import TABLES_DIR
from src.baselines import build_hi_comparison, run_ml_baselines

PHASE = "phase5_baselines"


def main(config="configs/main_mendeley.yaml", force=False):
    setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    outputs = [TABLES_DIR / "table3_health_indicator_comparison.csv",
               TABLES_DIR / "baseline_ml_results.csv"]
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    # (A) So sánh Health Indicator.
    build_hi_comparison(baseline_frac=cfg.mtoi.baseline_frac)
    # (B) ML baselines.
    run_ml_baselines(train_frac=cfg.split.train, val_frac=cfg.split.val)

    tracker.mark_done(PHASE, outputs=outputs)
    print(f"\n[{PHASE}] HOÀN TẤT.")


if __name__ == "__main__":
    main()
