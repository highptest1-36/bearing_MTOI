# -*- coding: utf-8 -*-
"""
phase6b_rolling_eval.py — PHASE 6B: ROLLING EVALUATION (plan mục 14.2).

Đánh giá kiểu "cửa sổ trượt mở rộng" (expanding-window) để chứng minh model ổn định theo thời gian:
  R1: Train 0-50%  -> Test 50-70%
  R2: Train 0-70%  -> Test 70-85%
  R3: Train 0-85%  -> Test 85-100%
Mỗi vòng huấn luyện 1 model RIÊNG (proposed_learnable) trên phần train tương ứng rồi test trên cửa sổ test.
Ghi: results/tables/rolling_eval_results.csv
Lệnh chạy:  !python scripts/phase6b_rolling_eval.py
ĐIỀU KIỆN: đã chạy xong Phase 2-4 (có X_vib_windows + mtoi + labels).
"""

import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import PROC_MENDELEY, TABLES_DIR
from src.utils.logger import get_logger, section
from src.datasets import make_datasets
from src.train import train_model
from src.evaluate import evaluate_model

PHASE = "phase6b_rolling_eval"

# Mỗi vòng: (tên, train_frac, val_frac, test_end). train+val = mốc bắt đầu test.
ROUNDS = [
    ("R1_train50_test50-70", 0.40, 0.10, 0.70),
    ("R2_train70_test70-85", 0.60, 0.10, 0.85),
    ("R3_train85_test85-100", 0.75, 0.10, 1.00),
]


def main(config="configs/main_mendeley.yaml", force=False):
    device = setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    logger = get_logger("phase6b")
    out_path = TABLES_DIR / "rolling_eval_results.csv"
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    section("PHASE 6B — ROLLING EVALUATION (expanding window)", logger)
    results = []
    for name, tr, va, t_end in ROUNDS:
        logger.info(f"\n===== {name} (train={tr+va:.0%}, test tới {t_end:.0%}) =====")
        # Dataset với cửa sổ test giới hạn (test_end).
        data = make_datasets(proc_dir=PROC_MENDELEY, train=tr, val=va,
                             mtoi_target="MTOI_learnable", test_end=t_end)
        out = train_model(
            data, run_name=f"roll_{name}", device=device, use_temp=True, fusion="gated",
            loss_weights=dict(cfg.loss), epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            num_workers=cfg.train.num_workers, model_cfg=dict(cfg.model), log=logger,
        )
        m = evaluate_model(out["model"], data["test"], device=device, run_name=f"roll_{name}")["metrics"]
        m["round"] = name
        results.append(m)
        pd.DataFrame(results).to_csv(out_path, index=False)   # lưu tạm sau mỗi vòng

    tracker.mark_done(PHASE, outputs=[out_path])
    print(f"\n[{PHASE}] HOÀN TẤT. Bảng rolling: {out_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
