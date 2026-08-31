# -*- coding: utf-8 -*-
"""
phase4_mtoi_labels.py — PHASE 4: XÂY MTOI++ và TẠO NHÃN.

Đọc hour_features.csv -> tính E,C,G -> MTOI_fixed & MTOI_learnable -> tạo RUL/stage/warning. Ghi:
  data/processed/mendeley/mtoi_by_hour.csv
  data/processed/mendeley/labels_by_hour.csv
  results/tables/learned_mtoi_weights_mendeley.csv
Lệnh chạy:  !python scripts/phase4_mtoi_labels.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.checkpoint import PhaseTracker
from src.utils.config import load_config
from src.utils.paths import PROC_MENDELEY
from src.mtoi import build_mtoi
from src.labels import build_labels, SAMPLE_INTERVAL_S

PHASE = "phase4_mtoi_labels"


def main(config="configs/main_mendeley.yaml", force=False):
    setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    outputs = [PROC_MENDELEY / "mtoi_by_hour.csv", PROC_MENDELEY / "labels_by_hour.csv"]
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    # 1) Xây MTOI (E,C,G -> fixed & learnable).
    build_mtoi(
        proc_dir=PROC_MENDELEY,
        baseline_frac=cfg.mtoi.baseline_frac, train_frac=cfg.mtoi.train_frac,
        k_temp=cfg.mtoi.k_temp, has_temp=cfg.dataset.has_temperature,
        use_sigmoid=cfg.mtoi.use_sigmoid, dataset_name=cfg.dataset.name,
    )
    # 2) Tạo nhãn (RUL/Deg/stage/warning) + đơn vị thời gian tuyệt đối (Mendeley = 3600 s/giờ).
    build_labels(proc_dir=PROC_MENDELEY, stage_mode=cfg.labels.stage_mode,
                 tau=cfg.labels.tau, q=cfg.labels.q,
                 sample_interval_s=SAMPLE_INTERVAL_S["mendeley"])

    # 3) Vẽ ngay vài hình để bạn QUAN SÁT quỹ đạo MTOI (Figure 3, 4, 7).
    from src import visualization as viz
    viz.plot_mtoi_curve(proc_dir=PROC_MENDELEY)
    viz.plot_early_warning(proc_dir=PROC_MENDELEY, tau=cfg.labels.tau, q=cfg.labels.q)
    viz.plot_learned_weights(dataset_name=cfg.dataset.name)
    viz.plot_hi_comparison(proc_dir=PROC_MENDELEY)

    tracker.mark_done(PHASE, outputs=outputs)
    print(f"\n[{PHASE}] HOÀN TẤT. Hãy mở mtoi_by_hour.csv để xem quỹ đạo MTOI.")


if __name__ == "__main__":
    main()
