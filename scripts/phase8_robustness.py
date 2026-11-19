# -*- coding: utf-8 -*-
"""
phase8_robustness.py — PHASE 8: ROBUSTNESS (plan mục 16, Experiment 5).

Nạp lại 2 model đã huấn luyện (proposed_learnable & fusion_concat) từ checkpoint best,
chạy các kịch bản nhiễu/thiếu sensor trên tập test, ghi bảng + vẽ hình.
  results/tables/table5_robustness_*.csv
  results/figures/fig5_robustness.png
Lệnh chạy:  !python scripts/phase8_robustness.py
ĐIỀU KIỆN: phải chạy xong Phase 6 (đã có checkpoint *_best.pt).
"""

import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import PROC_MENDELEY, CHECKPOINTS_DIR, TABLES_DIR
from src.utils.logger import get_logger
from src.datasets import make_datasets
from src.models.mtoi_model import MTOIModel
from src.robustness import run_robustness
from src import visualization as viz

PHASE = "phase8_robustness"


def load_trained(run_name, cfg, device, use_temp, fusion, use_uncertainty=False):
    """Dựng lại kiến trúc khớp lúc train rồi nạp trọng số best.
    use_uncertainty PHẢI khớp lúc train (proposed_learnable train với uncertainty=True ->
    có unc_head); nếu không sẽ lỗi 'Unexpected key unc_head.*' khi load_state_dict."""
    model = MTOIModel(emb_dim=cfg.model.emb_dim, n_stages=cfg.model.n_stages,
                      use_temp=use_temp, temp_dim=cfg.model.temp_dim, fusion=fusion,
                      use_uncertainty=use_uncertainty).to(device)
    ckpt = CHECKPOINTS_DIR / f"{run_name}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Thiếu checkpoint {ckpt}. Hãy chạy Phase 6 trước.")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    return model


def main(config="configs/main_mendeley.yaml", force=False):
    device = setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    logger = get_logger("phase8")
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    # Dựng dataset test (cùng cách chia như khi train) + lấy thống kê chuẩn hoá nhiệt độ.
    data = make_datasets(proc_dir=PROC_MENDELEY, train=cfg.split.train, val=cfg.split.val)
    temp_mean = data["norm"]["temp_mean"]; temp_std = data["norm"]["temp_std"]

    # Nạp 2 model & chạy robustness (truyền temp_mean/temp_std để can thiệp nhiệt độ đúng đơn vị °C).
    m_prop = load_trained("proposed_learnable", cfg, device, use_temp=True, fusion="gated",
                          use_uncertainty=True)   # proposed_learnable train CÓ uncertainty head
    t_prop = run_robustness(m_prop, data["test"], device=device, run_name="proposed_learnable",
                            temp_mean=temp_mean, temp_std=temp_std)

    m_fuse = load_trained("fusion_concat", cfg, device, use_temp=True, fusion="concat")
    run_robustness(m_fuse, data["test"], device=device, run_name="fusion_concat",
                   temp_mean=temp_mean, temp_std=temp_std)

    # Vẽ đường robustness cho model đề xuất.
    viz.plot_robustness(TABLES_DIR / "table5_robustness_proposed_learnable.csv")

    tracker.mark_done(PHASE, outputs=[TABLES_DIR / "table5_robustness_proposed_learnable.csv"])
    print(f"\n[{PHASE}] HOÀN TẤT.")


if __name__ == "__main__":
    main()
