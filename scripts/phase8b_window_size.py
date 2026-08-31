# -*- coding: utf-8 -*-
"""
phase8b_window_size.py — PHASE 8B: QUÉT KÍCH THƯỚC CỬA SỔ (plan mục 16, Exp5).

Thử các độ dài cửa sổ L = 2048 / 4096 / 8192 và so model đề xuất với fusion baseline.
Với MỖI L: chạy lại mini-pipeline (cắt cửa sổ -> đặc trưng -> MTOI -> nhãn) vào thư mục RIÊNG
(data/processed/mendeley/winL/) rồi train + đánh giá.
Ghi: results/tables/window_size_results.csv
Lệnh chạy:  !python scripts/phase8b_window_size.py

CẢNH BÁO: ĐÂY LÀ PHASE NẶNG NHẤT (đọc lại 129 CSV cho mỗi L + train 2 model/L).
          Có thể mất 1-2 giờ. Có checkpoint nên rớt mạng chạy lại được. Chỉ chạy khi cần bảng này.
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
from src.segmentation import build_mendeley_windows
from src.features import build_features
from src.mtoi import build_mtoi
from src.labels import build_labels
from src.datasets import make_datasets
from src.train import train_model
from src.evaluate import evaluate_model

PHASE = "phase8b_window_size"
WINDOW_SIZES = [2048, 4096, 8192]          # các độ dài cửa sổ cần thử


def run_for_L(L, cfg, device, logger):
    """Chạy mini-pipeline + train proposed & fusion cho 1 giá trị L; trả 2 dòng kết quả."""
    wdir = PROC_MENDELEY / f"win{L}"        # thư mục processed riêng cho L này
    wdir.mkdir(parents=True, exist_ok=True)

    # 1) Cắt cửa sổ với L này (bỏ qua nếu đã có file).
    if not (wdir / "X_vib_windows.npy").exists():
        build_mendeley_windows(L=L, overlap=cfg.segmentation.overlap,
                               max_windows_per_hour=cfg.segmentation.max_windows_per_hour,
                               max_files=cfg.segmentation.get("max_files"),
                               max_rows=cfg.segmentation.get("max_rows"), out_dir=wdir)
    # 2) Đặc trưng -> MTOI -> nhãn (idempotent theo file).
    if not (wdir / "hour_features.csv").exists():
        build_features(proc_dir=wdir)
    if not (wdir / "mtoi_by_hour.csv").exists():
        build_mtoi(proc_dir=wdir, baseline_frac=cfg.mtoi.baseline_frac,
                   train_frac=cfg.mtoi.train_frac, k_temp=cfg.mtoi.k_temp,
                   has_temp=cfg.dataset.has_temperature, use_sigmoid=cfg.mtoi.use_sigmoid,
                   dataset_name=f"mendeley_win{L}")
    if not (wdir / "labels_by_hour.csv").exists():
        build_labels(proc_dir=wdir, stage_mode=cfg.labels.stage_mode, tau=cfg.labels.tau, q=cfg.labels.q)

    rows = []
    # 3) Train + đánh giá proposed_learnable và fusion_concat cho L này.
    for run, use_temp, fusion, l1 in [("proposed", True, "gated", None), ("fusion", True, "concat", 0.0)]:
        data = make_datasets(proc_dir=wdir, train=cfg.split.train, val=cfg.split.val)
        lw = dict(cfg.loss)
        if l1 is not None:
            lw["lambda1"] = l1
        out = train_model(data, run_name=f"winL{L}_{run}", device=device, use_temp=use_temp,
                          fusion=fusion, loss_weights=lw, epochs=cfg.train.epochs,
                          batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                          weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
                          num_workers=cfg.train.num_workers, model_cfg=dict(cfg.model), log=logger)
        m = evaluate_model(out["model"], data["test"], device=device, run_name=f"winL{L}_{run}")["metrics"]
        rows.append({"window_size": L, "model": run, "stage_macro_f1": round(m["stage_macro_f1"], 4),
                     "rul_rmse": round(m["rul_rmse"], 4), "mtoi_spearman": round(m["mtoi_spearman"], 4)})
    return rows


def main(config="configs/main_mendeley.yaml", force=False):
    device = setup_environment(do_mount=True)
    tracker = PhaseTracker()
    cfg = load_config(ROOT / config)
    logger = get_logger("phase8b")
    out_path = TABLES_DIR / "window_size_results.csv"
    if tracker.is_done(PHASE) and not force:
        print(f"[{PHASE}] Đã hoàn thành -> BỎ QUA.")
        return
    tracker.mark_running(PHASE)

    section("PHASE 8B — QUÉT KÍCH THƯỚC CỬA SỔ (2048/4096/8192)", logger)
    results = []
    for L in WINDOW_SIZES:
        logger.info(f"\n========== WINDOW SIZE L = {L} ==========")
        results += run_for_L(L, cfg, device, logger)
        pd.DataFrame(results).to_csv(out_path, index=False)   # lưu tạm sau mỗi L

    tracker.mark_done(PHASE, outputs=[out_path])
    print(f"\n[{PHASE}] HOÀN TẤT. Bảng window-size: {out_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
