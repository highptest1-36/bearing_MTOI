# -*- coding: utf-8 -*-
"""
phase9b_lobo_train.py — HUẤN LUYỆN LEAVE-ONE-BEARING-OUT (LOBO) trên PRONOSTIA & XJTU-SY.

Với MỖI dataset đa-bearing:
  - Lặp qua từng bearing làm TEST (held-out, chưa từng thấy), 1 bearing làm VAL, còn lại TRAIN.
  - Huấn luyện mô hình đề xuất (+ vài baseline) trên TRAIN, đánh giá trên bearing held-out.
  - Tổng hợp metric trung bình ± lệch chuẩn QUA CÁC FOLD -> bảng main-performance + stage cân bằng.

Bảng xuất ra (results/tables/):
  - lobo_<dataset>_<config>_perfold.csv : 1 dòng / fold (metric của bearing held-out).
  - lobo_<dataset>_summary.csv          : mean ± std qua các fold, mỗi config 1 dòng.

Chạy (sau khi phase9a đã build artifact):
  python scripts/phase9b_lobo_train.py
Tham số gọi từ Python: main(datasets=('pronostia','xjtu_sy'), configs=None, max_folds=None,
                             epochs=30, train_subsample=1, force=False)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.env import setup_environment
from src.utils.logger import get_logger, section, Timer
from src.utils.checkpoint import PhaseTracker
from src.utils.paths import TABLES_DIR
from src.train import train_model
from src.evaluate import evaluate_model
from src.lobo import make_folds, prepare_fold, datasets_from_pack


# Các cấu hình mô hình để so sánh trong bảng LOBO main-performance.
#   use_temp/fusion/vib_encoder/mtoi_target/loss_override khớp tham số train_model.
# Cột HI (đầu vào RUL head). vib = MTOI vib-only (E+C, bỏ G nhiễu) + quỹ đạo; full = có thêm G.
HI_VIB_STATIC = ["E_norm", "C_norm", "MTOI_woG"]
HI_VIB_TRAJ   = ["E_norm", "C_norm", "MTOI_woG", "MTOI_woG_vel", "MTOI_woG_acc"]
HI_FULL_TRAJ  = ["E_norm", "C_norm", "G_norm", "MTOI_learnable", "MTOI_learnable_vel", "MTOI_learnable_acc"]
NO_MTOI  = dict(lambda1=0.0)                       # tắt nhánh MTOI loss (baseline)
LOW_AUX  = dict(lambda1=0.1, w_stage=0.0)          # MTOI loss nhẹ + BỎ stage (giảm nhiễu đa-nhiệm)


def configs_for(dataset, has_temp_ds):
    """
    KẾ HOẠCH MODALITY-ADAPTIVE (backbone Transformer). MTOI vib-only (E+C) + QUỸ ĐẠO làm đầu vào RUL.
      A transformer_vib   : baseline mạnh nhất (vib-only, KHÔNG MTOI) — mốc phải vượt.
      B mtoi_vib_static   : + MTOI_vib tĩnh làm đầu vào RUL.
      C mtoi_vib_traj  ⭐ : + QUỸ ĐẠO MTOI (level+vận tốc+gia tốc) — mục tiêu vượt A.
      D mtoi_full_temp    : ÉP đa mô thức (temp + full MTOI có G) — test temp có hại không.
      E mtoi_adaptive     : C + nhánh temp qua gated fusion + masking (model TỰ chọn dùng temp).
    """
    base = [
        ("transformer_vib", dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=False,
                                 mtoi_target="MTOI_woG", loss_override=NO_MTOI)),
        ("mtoi_vib_static", dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=True,
                                 hi_cols=HI_VIB_STATIC, mtoi_target="MTOI_woG", loss_override=LOW_AUX)),
        ("mtoi_vib_traj",   dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=True,
                                 hi_cols=HI_VIB_TRAJ, mtoi_target="MTOI_woG", loss_override=LOW_AUX,
                                 use_uncertainty=True)),
        # --- Baseline hiện đại vib-only (reviewer Q1, KHÔNG MTOI) — so sánh công bằng cùng protocol ---
        ("tcn_vib",             dict(use_temp=False, fusion="gated", vib_encoder="tcn",
                                     use_hi=False, mtoi_target="MTOI_woG", loss_override=NO_MTOI)),
        ("tcn_transformer_vib", dict(use_temp=False, fusion="gated", vib_encoder="tcn_transformer",
                                     use_hi=False, mtoi_target="MTOI_woG", loss_override=NO_MTOI)),
        ("cnn_bilstm_attn_vib", dict(use_temp=False, fusion="gated", vib_encoder="cnn_bilstm_attn",
                                     use_hi=False, mtoi_target="MTOI_woG", loss_override=NO_MTOI)),
        # --- Ablation TRÊN LOBO (reviewer Q1 v2): tách đóng góp từng phần (cùng encoder transformer) ---
        # E-only: index chỉ là Mahalanobis abnormality -> kiểm tra C/G có thật sự cần.
        ("abl_E_only",     dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=True,
                                hi_cols=["E_norm"], mtoi_target="E_norm", loss_override=LOW_AUX)),
        # MTOI loss BẬT nhưng KHÔNG feed HI vào RUL head -> tách đóng góp của "index-guided head".
        ("abl_no_idxhead", dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=False,
                                mtoi_target="MTOI_woG", loss_override=LOW_AUX)),
        # Bỏ smoothness penalty / bỏ monotonicity penalty (trên nền E+C = mtoi_vib_static).
        ("abl_no_smooth",  dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=True,
                                hi_cols=HI_VIB_STATIC, mtoi_target="MTOI_woG",
                                loss_override=dict(lambda1=0.1, w_stage=0.0, w_rul_smooth=0.0, w_mtoi_smooth=0.0))),
        ("abl_no_mono",    dict(use_temp=False, fusion="gated", vib_encoder="transformer", use_hi=True,
                                hi_cols=HI_VIB_STATIC, mtoi_target="MTOI_woG",
                                loss_override=dict(lambda1=0.1, w_stage=0.0, w_rul_mono=0.0, w_mtoi_mono=0.0))),
    ]
    if has_temp_ds:                                # 2 config đa mô thức (chỉ dataset có nhiệt độ)
        base += [
            ("mtoi_full_temp", dict(use_temp=True, fusion="gated", vib_encoder="transformer", use_hi=True,
                                    hi_cols=HI_FULL_TRAJ, mtoi_target="MTOI_learnable", loss_override=LOW_AUX)),
            ("mtoi_adaptive",  dict(use_temp=True, fusion="gated", vib_encoder="transformer", use_hi=True,
                                    hi_cols=HI_VIB_TRAJ, mtoi_target="MTOI_woG", loss_override=LOW_AUX,
                                    use_uncertainty=True)),
        ]
    return base


# RUL-FIRST: metric chính (RUL theo giờ + score + PI) trước, rồi cảnh báo, MTOI, stage.
METRIC_KEYS = ["rul_mae", "rul_rmse", "rul_r2",
               "rul_mae_hours", "rul_rmse_hours", "rul_phm_score", "rul_asym_score",
               "rul_picp", "rul_mpiw",
               "first_warning", "lead_time", "false_alarm_rate",
               "mtoi_rmse", "mtoi_spearman", "mtoi_monotonicity",
               "stage_acc", "stage_macro_f1", "stage_bal_acc"]


def train_eval_config(dataset, pack, cfg_name, cfg_kwargs, device,
                      epochs, train_subsample, logger, rul_col="RUL_capped"):
    """Train 1 config trên 1 fold ĐÃ NẠP (pack), đánh giá bearing held-out -> dict metric (1 dòng)."""
    ho = pack["holdout"]
    hi_cols = cfg_kwargs.get("hi_cols")                # danh sách cột HI (E/C/[G]/MTOI/vel/acc) hoặc None
    ds = datasets_from_pack(pack, mtoi_target=cfg_kwargs["mtoi_target"],
                            train_subsample=train_subsample, rul_col=rul_col, hi_cols=hi_cols)
    run_name = f"lobo_{dataset}_{cfg_name}_{ho}"
    out = train_model(
        {"train": ds["train"], "val": ds["val"], "test": ds["test"]},
        run_name=run_name, device=device,
        use_temp=cfg_kwargs["use_temp"], fusion=cfg_kwargs["fusion"],
        vib_encoder=cfg_kwargs["vib_encoder"], loss_weights=cfg_kwargs["loss_override"],
        use_uncertainty=cfg_kwargs.get("use_uncertainty", False),
        use_hi=cfg_kwargs.get("use_hi", False),
        hi_dim=(len(hi_cols) if hi_cols else 4),       # khớp số chiều HI với hi_cols
        epochs=epochs, batch_size=256, patience=5, num_workers=4, resume=True, log=logger,
    )
    res = evaluate_model(out["model"], ds["test"], device=device,
                         run_name=run_name, save=False)
    # Lưu đường dự đoán trên bearing held-out (config đề xuất có HI) -> phục vụ VẼ HÌNH ở phase9c.
    if cfg_name in ("mtoi_vib_traj", "mtoi_adaptive") and res.get("hour_df") is not None:
        from src.utils.paths import PREDICTIONS_DIR
        outp = PREDICTIONS_DIR / f"lobo_{dataset}_{ho}_proposed.csv"
        try:
            res["hour_df"].to_csv(outp, index=False)
        except Exception:
            pass
    row = {"dataset": dataset, "config": cfg_name, "holdout": ho,
           "n_train_bearings": ds["n_train_bearings"],
           "n_test_windows": len(ds["test"])}
    row.update({k: res["metrics"].get(k) for k in METRIC_KEYS})
    return row


def summarize(perfold_df):
    """Tổng hợp mean ± std qua các fold cho mỗi (dataset, config)."""
    rows = []
    for (dsn, cfg), g in perfold_df.groupby(["dataset", "config"]):
        row = {"dataset": dsn, "config": cfg, "n_folds": len(g)}
        for k in METRIC_KEYS:
            vals = pd.to_numeric(g[k], errors="coerce").dropna()
            row[f"{k}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            row[f"{k}_std"] = round(float(vals.std(ddof=0)), 4) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(datasets=("pronostia", "xjtu_sy"), configs=None, max_folds=None,
         epochs=20, train_subsample=None, force=False, local_ckpt=True,
         rul_target="RUL_capped"):
    device = setup_environment(seed=42, do_mount=False)
    logger = get_logger("phase9b")
    tracker = PhaseTracker()
    section("PHASE 9B — LOBO TRAINING (PRONOSTIA + XJTU)", logger)

    # Mặc định: XJTU lấy thưa cửa sổ train x4 (nhiều cửa sổ/snapshot -> giảm tải, fold nhanh hơn);
    # PRONOSTIA giữ nguyên (1 cửa sổ/snapshot). Truyền dict để ghi đè khi cần.
    if train_subsample is None:
        train_subsample = {"xjtu_sy": 4}

    # Checkpoint per-fold KHÔNG cần bền trên Drive (resume mức fold đã có qua perfold.csv) ->
    # chuyển sang ổ LOCAL nhanh để TRÁNH ghi hàng chục GB lên Drive mỗi epoch (nhanh hơn nhiều).
    if local_ckpt:
        import src.train as _t
        local_dir = Path("/content/lobo_ckpts"); local_dir.mkdir(parents=True, exist_ok=True)
        _t.CHECKPOINTS_DIR = local_dir
        logger.info(f"Checkpoint LOBO -> LOCAL {local_dir} (không ghi Drive mỗi epoch).")

    for dataset in datasets:
        folds, has_temp_ds = make_folds(dataset)
        if max_folds:
            folds = folds[:max_folds]
        cfg_list = configs_for(dataset, has_temp_ds)
        if configs:
            cfg_list = [c for c in cfg_list if c[0] in configs]
        logger.info(f"[{dataset}] {len(folds)} fold × {len(cfg_list)} config "
                    f"(has_temp={has_temp_ds}, epochs={epochs}, subsample={train_subsample})")

        perfold_path = TABLES_DIR / f"lobo_{dataset}_perfold.csv"
        done = set()
        if perfold_path.exists() and not force:
            prev = pd.read_csv(perfold_path)
            done = set(zip(prev["config"], prev["holdout"]))
            rows = prev.to_dict("records")
        else:
            rows = []

        sub = train_subsample.get(dataset, 1) if isinstance(train_subsample, dict) else train_subsample
        for fi, fold in enumerate(folds):
            ho = fold["holdout"]
            pending = [(n, k) for n, k in cfg_list if (n, ho) not in done or force]
            if not pending:
                continue
            # NẠP fold 1 LẦN -> dùng cho mọi config (tiết kiệm I/O Drive).
            try:
                pack = prepare_fold(dataset, ho, fold["val"], fold["train"], logger=logger)
            except Exception as e:
                logger.info(f"  LỖI nạp fold {ho}: {e}")
                continue
            for cfg_name, cfg_kwargs in pending:
                with Timer(f"[{dataset}] fold {fi+1}/{len(folds)} {ho} :: {cfg_name}", logger):
                    try:
                        row = train_eval_config(dataset, pack, cfg_name, cfg_kwargs,
                                                device, epochs, sub, logger, rul_col=rul_target)
                        rows.append(row)
                    except Exception as e:
                        logger.info(f"  LỖI fold {ho} {cfg_name}: {e}")
                        continue
                # Ghi tăng tiến sau MỖI lần train (sống sót disconnect).
                pd.DataFrame(rows).to_csv(perfold_path, index=False)
            del pack  # giải phóng RAM fold trước khi sang fold sau

        perfold_df = pd.DataFrame(rows)
        summ = summarize(perfold_df)
        summ_path = TABLES_DIR / f"lobo_{dataset}_summary.csv"
        summ.to_csv(summ_path, index=False)
        logger.info(f"[{dataset}] Ghi {summ_path}")
        if len(summ):
            print(f"\n===== LOBO SUMMARY: {dataset} =====")
            print(summ.to_string(index=False))

    tracker.mark_done("phase9b_lobo_train",
                      outputs=[str(TABLES_DIR / f"lobo_{d}_summary.csv") for d in datasets])
    logger.info("HOÀN TẤT Phase 9B.")


if __name__ == "__main__":
    main()
