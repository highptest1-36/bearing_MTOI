# -*- coding: utf-8 -*-
"""
phase7_ablation.py — PHASE 7: ABLATION STUDY (plan mục 16, Experiment 3).

Huấn luyện & đánh giá các biến thể khai báo trong configs/ablation.yaml, ghi:
  results/tables/ablation_table.csv
Mỗi biến thể có checkpoint riêng -> rớt mạng vẫn tiếp tục.
Lệnh chạy:  !python scripts/phase7_ablation.py

Đã TỰ ĐỘNG HOÁ ĐẦY ĐỦ các biến thể: full, w/o temperature, w/o L_mtoi, w/o L_smooth,
w/o L_mono, fixed_mtoi, concat_fusion, VÀ w/o E / w/o C / w/o G (dùng các cột MTOI_woE/woC/woG
do Phase 4 tạo sẵn). Không cần chỉnh tay nữa.
"""

import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import TABLES_DIR
from src.utils.logger import get_logger, section
from scripts.phase6_train_proposed import run_one   # tái sử dụng hàm train+eval

PHASE = "phase7_ablation"

# Ánh xạ tên biến thể -> tham số cho run_one (mtoi_target/use_temp/fusion/loss_override).
VARIANTS = {
    "full_model":      dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_learnable"),
    # SỬA: chỉ bỏ NHIỆT ĐỘ, GIỮ giám sát MTOI (lambda1 mặc định). Trước đây vô tình tắt cả lambda1
    # -> ablation bị nhập nhằng 2 yếu tố (bỏ nhiệt độ + bỏ MTOI) cùng lúc.
    "wo_temperature":  dict(use_temp=False, fusion="gated",  mtoi_target="MTOI_learnable"),
    "wo_L_mtoi":       dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
    "wo_L_smooth":     dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_learnable", loss_override={"lambda3": 0.0}),
    "wo_L_mono":       dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_learnable", loss_override={"lambda4": 0.0}),
    "fixed_mtoi":      dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_fixed"),
    "concat_fusion":   dict(use_temp=True,  fusion="concat", mtoi_target="MTOI_learnable"),
    # Bỏ TỪNG thành phần MTOI (E/C/G) — dùng cột MTOI tương ứng đã tạo ở Phase 4 (tự động).
    "wo_E":            dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_woE"),
    "wo_C":            dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_woC"),
    "wo_G":            dict(use_temp=True,  fusion="gated",  mtoi_target="MTOI_woG"),
}


def main(config="configs/main_mendeley.yaml", force=False):
    device = setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    logger = get_logger("phase7")
    out_path = TABLES_DIR / "ablation_table.csv"
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    section("PHASE 7 — ABLATION STUDY", logger)
    results = []
    for name, params in VARIANTS.items():
        logger.info(f"\n===== Ablation: {name} =====")
        m = run_one(run_name=f"abl_{name}", cfg=cfg, device=device, logger=logger, **params)
        m["variant"] = name
        results.append(m)
        pd.DataFrame(results).to_csv(out_path, index=False)   # lưu tạm sau mỗi biến thể

    tracker.mark_done(PHASE, outputs=[out_path])
    print(f"\n[{PHASE}] HOÀN TẤT. Bảng ablation: {out_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
