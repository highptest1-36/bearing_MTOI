# -*- coding: utf-8 -*-
"""
phase6_train_proposed.py — PHASE 6: HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT + BASELINE DEEP.

Huấn luyện & đánh giá lần lượt:
  - vib_only_cnn        : 1D-CNN chỉ dùng rung (baseline deep).
  - fusion_concat       : đa mô thức, fusion = concat (baseline fusion).
  - proposed_wo_mtoi    : đề xuất nhưng λ1=0 (KHÔNG giám sát MTOI).
  - proposed_fixed_mtoi : đề xuất, nhãn MTOI = MTOI_fixed.
  - proposed_learnable  : ĐỀ XUẤT ĐẦY ĐỦ (nhãn MTOI = MTOI_learnable).
Ghi: results/tables/proposed_results.csv + predictions_*.csv + checkpoints/*.pt.

CHECKPOINT: mỗi run tự lưu '<run>_last.pt' mỗi epoch -> rớt mạng vẫn train tiếp được.
Lệnh chạy:  !python scripts/phase6_train_proposed.py
A100: có thể tăng batch_size lên 256/512 trong configs/main_mendeley.yaml.
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

PHASE = "phase6_train_proposed"


def run_one(run_name, cfg, device, logger, use_temp=True, fusion="gated",
            mtoi_target="MTOI_learnable", loss_override=None, vib_encoder="cnn",
            use_uncertainty=False):
    """Huấn luyện + đánh giá 1 cấu hình; trả về dict metric (1 dòng kết quả).
    vib_encoder: 'cnn'|'lstm'|'gru'|'transformer' -> dùng cho các baseline deep khác nhau."""
    # Tạo dataset với đúng cột MTOI mục tiêu (fixed/learnable).
    data = make_datasets(proc_dir=PROC_MENDELEY, train=cfg.split.train,
                         val=cfg.split.val, mtoi_target=mtoi_target)
    # Gộp trọng số loss mặc định + override (vd λ1=0 cho biến thể w/o MTOI).
    loss_weights = dict(cfg.loss)
    if loss_override:
        loss_weights.update(loss_override)
    # model_cfg KHÔNG được ghi đè use_uncertainty bằng giá trị cũ trong yaml -> tách riêng để truyền cờ.
    mcfg = dict(cfg.model); mcfg["use_uncertainty"] = use_uncertainty
    # Huấn luyện (tự resume nếu có checkpoint).
    out = train_model(
        data, run_name=run_name, device=device,
        use_temp=use_temp, fusion=fusion, vib_encoder=vib_encoder,
        loss_weights=loss_weights, use_uncertainty=use_uncertainty,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
        num_workers=cfg.train.num_workers, model_cfg=mcfg, log=logger,
    )
    # Đánh giá trên test.
    ev = evaluate_model(out["model"], data["test"], device=device, run_name=run_name)
    return ev["metrics"]


def main(config="configs/main_mendeley.yaml", force=False):
    device = setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    logger = get_logger("phase6")
    out_path = TABLES_DIR / "proposed_results.csv"
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    section("PHASE 6 — HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT & BASELINE DEEP", logger)
    results = []
    # Danh sách cấu hình (tên, tham số). Chạy lần lượt — mỗi cái có checkpoint riêng.
    runs = [
        # --- Baseline deep vibration-only (plan 15.2): CNN, LSTM, GRU, Transformer (đều λ1=0, không MTOI) ---
        dict(run_name="vib_only_cnn",         use_temp=False, fusion="gated",  vib_encoder="cnn",         mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        dict(run_name="vib_only_lstm",        use_temp=False, fusion="gated",  vib_encoder="lstm",        mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        dict(run_name="vib_only_gru",         use_temp=False, fusion="gated",  vib_encoder="gru",         mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        dict(run_name="vib_only_transformer", use_temp=False, fusion="gated",  vib_encoder="transformer", mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        # --- Baseline fusion (concat) ---
        dict(run_name="fusion_concat",        use_temp=True,  fusion="concat", vib_encoder="cnn",         mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        # --- Các biến thể đề xuất ---
        dict(run_name="proposed_wo_mtoi",     use_temp=True,  fusion="gated",  vib_encoder="cnn",         mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
        dict(run_name="proposed_fixed_mtoi",  use_temp=True,  fusion="gated",  vib_encoder="cnn",         mtoi_target="MTOI_fixed"),
        # ĐỀ XUẤT ĐẦY ĐỦ + UNCERTAINTY (q10/q50/q90) -> headline, có PICP/MPIW.
        dict(run_name="proposed_learnable",   use_temp=True,  fusion="gated",  vib_encoder="cnn",         mtoi_target="MTOI_learnable", use_uncertainty=True),
    ]
    for r in runs:
        logger.info(f"\n========== RUN: {r['run_name']} ==========")
        m = run_one(cfg=cfg, device=device, logger=logger, **r)
        results.append(m)
        # Lưu kết quả TẠM sau MỖI run (nếu rớt mạng vẫn còn các run trước).
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info(f"Đã cập nhật {out_path} ({len(results)} run).")

    tracker.mark_done(PHASE, outputs=[out_path])
    print(f"\n[{PHASE}] HOÀN TẤT. Bảng kết quả: {out_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
