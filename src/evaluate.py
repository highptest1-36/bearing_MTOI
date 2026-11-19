# -*- coding: utf-8 -*-
"""
evaluate.py — ĐÁNH GIÁ MÔ HÌNH (RUL-PRIMARY): dự đoán cấp cửa sổ -> gộp cấp GIỜ -> tính metric.

Quy trình:
  1) Chạy model trên test, thu (stage_pred, mtoi_pred, rul_pred, [q10/q50/q90]) cho từng cửa sổ.
  2) GỘP về cấp GIỜ: stage = vote đa số; mtoi/rul/quantile = trung bình theo giờ.
  3) Tính metric — RUL LÀ CHÍNH:
        RUL   : MAE/RMSE/R2 (chuẩn hoá) + MAE/RMSE theo GIỜ + PHM-Score + Asymmetric-Score.
        Cảnh báo sớm (HEADLINE): suy ra từ RUL dự đoán (RUL tụt dưới ngưỡng) -> first/lead/FAR.
        Uncertainty (nếu có quantile): PICP + MPIW.
        MTOI  : RMSE, |Spearman| với RUL, Monotonicity (phụ — chỉ số giải thích).
        Stage : Accuracy/Macro-F1/Balanced-Acc (phụ trợ).
  4) Lưu predictions.csv (cấp giờ) để vẽ hình & lập bảng.

GHI CHÚ đơn vị: nếu labels có 'life_hours' (đã regen với cadence) -> báo MAE/RMSE theo GIỜ THẬT.
Nếu không -> chỉ có metric chuẩn hoá (log rõ 'normalized'). first_warning/lead_time tính theo VỊ TRÍ
trong test (snapshot); để lấy giờ tuyệt đối cần đường dự đoán TOÀN vòng đời (xem phase4/5).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.paths import PREDICTIONS_DIR
from src.utils.logger import get_logger, section
from src import metrics


@torch.no_grad()
def predict_windows(model, dataset, device="cuda", batch_size=256):
    """Chạy model trên dataset, trả về DataFrame dự đoán cấp CỬA SỔ (kèm H_fail/life_hours/quantile nếu có)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    for batch in loader:
        vib = batch["vib"].to(device)
        temp = batch["temp"].to(device) if "temp" in batch else None
        hi = batch["hi"].to(device) if "hi" in batch else None
        has_temp = batch["has_temp"].to(device) if "has_temp" in batch else None
        out = model(vib, temp, hi, has_temp)
        d = {
            "hour_id": batch["hour_id"].numpy(),
            "stage_pred": out["stage_logits"].argmax(dim=1).cpu().numpy(),
            "mtoi_pred": out["mtoi"].cpu().numpy(),
            "rul_pred": out["rul"].cpu().numpy(),
            "stage_true": batch["stage"].numpy(),
            "mtoi_true": batch["mtoi"].numpy(),
            "rul_true": batch["rul"].numpy(),
        }
        if "H_fail" in batch:          d["H_fail"] = batch["H_fail"].numpy()
        if "rul_scale_hours" in batch: d["rul_scale_hours"] = batch["rul_scale_hours"].numpy()
        if "rul_quantiles" in out:                       # khoảng dự đoán (nếu bật uncertainty)
            q = out["rul_quantiles"].cpu().numpy()
            d["rul_q10"] = q[:, 0]; d["rul_q50"] = q[:, 1]; d["rul_q90"] = q[:, 2]
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def aggregate_to_hour(win_df):
    """Gộp dự đoán cấp cửa sổ -> cấp GIỜ (stage: vote đa số; còn lại: trung bình)."""
    def majority(s):
        return s.value_counts().idxmax()
    agg_map = dict(
        stage_pred=("stage_pred", majority), stage_true=("stage_true", "first"),
        mtoi_pred=("mtoi_pred", "mean"), mtoi_true=("mtoi_true", "first"),
        rul_pred=("rul_pred", "mean"), rul_true=("rul_true", "first"),
    )
    for c in ("H_fail", "rul_scale_hours"):              # giữ scalar/bearing (lấy 'first')
        if c in win_df.columns:
            agg_map[c] = (c, "first")
    for c in ("rul_q10", "rul_q50", "rul_q90"):          # quantile -> trung bình theo giờ
        if c in win_df.columns:
            agg_map[c] = (c, "mean")
    agg = (win_df.groupby("hour_id").agg(**agg_map)
           .reset_index().sort_values("hour_id").reset_index(drop=True))
    return agg


def evaluate_model(model, dataset, device="cuda", run_name="proposed", save=True,
                   tau=0.6, q=3):
    """
    Đánh giá đầy đủ trên 1 dataset (thường là test). Trả về dict metric + DataFrame cấp giờ.
    RUL là metric CHÍNH; cảnh báo sớm headline suy ra từ RUL dự đoán.
    """
    logger = get_logger("evaluate")
    section(f"ĐÁNH GIÁ: {run_name}", logger)

    win_df = predict_windows(model, dataset, device=device)
    hour_df = aggregate_to_hour(win_df)
    n = len(hour_df)

    rul_true = hour_df["rul_true"].to_numpy()
    rul_pred = hour_df["rul_pred"].to_numpy()

    # --- RUL chuẩn hoá ---
    rul_mae = metrics.mae(rul_true, rul_pred)
    rul_rmse = metrics.rmse(rul_true, rul_pred)
    rul_r2 = metrics.r2_score(rul_true, rul_pred)

    # --- RUL theo GIỜ tuyệt đối + score (dùng rul_scale_hours: đúng cho cả linear lẫn capped) ---
    if "rul_scale_hours" in hour_df.columns:
        life = hour_df["rul_scale_hours"].to_numpy()
        pred_h = rul_pred * life                          # giờ còn lại dự đoán
        true_h = rul_true * life                          # giờ còn lại thật
        rul_mae_hours = metrics.mae(true_h, pred_h)
        rul_rmse_hours = metrics.rmse(true_h, pred_h)
        rul_phm = metrics.phm_score(true_h, pred_h)       # phạt over-predict (kiểu PHM08)
        rul_asym = metrics.asymmetric_score(true_h, pred_h)
        eval_unit = "hours"
    else:
        rul_mae_hours = rul_rmse_hours = rul_phm = rul_asym = None
        eval_unit = "normalized"

    # --- Cảnh báo sớm HEADLINE: suy ra từ RUL dự đoán (degradation = 1 - RUL tăng dần) ---
    deg_pred = 1.0 - rul_pred
    failure_pos = n - 1
    onset_pos = max(1, int(0.6 * n))
    ew = metrics.early_warning_metrics(deg_pred, tau=tau, failure_hour=failure_pos,
                                       onset_hour=onset_pos, q=q)
    # --- Cảnh báo sớm PHỤ: từ MTOI dự đoán (giữ để đối chiếu) ---
    ew_mtoi = metrics.early_warning_metrics(hour_df["mtoi_pred"].to_numpy(), tau=tau,
                                            failure_hour=failure_pos, onset_hour=onset_pos, q=q)

    # --- MTOI/VTOI (phụ) ---
    # R2 #23: |Spearman| che giấu DẤU. Một chỉ số suy thoái phải TĂNG khi RUL GIẢM -> kỳ vọng ÂM.
    # Ta báo cáo CẢ HAI: signed (dùng cho paper) và abs (giữ để tương thích bảng/script cũ).
    mtoi_rmse = metrics.rmse(hour_df["mtoi_true"], hour_df["mtoi_pred"])
    mtoi_spear_signed = metrics.spearman_corr(hour_df["mtoi_pred"], rul_true)
    mtoi_spear = abs(mtoi_spear_signed)
    mtoi_spear_deg = metrics.spearman_corr(hour_df["mtoi_pred"], 1.0 - rul_true)  # kỳ vọng DƯƠNG
    mtoi_mono = metrics.monotonicity(hour_df["mtoi_pred"])

    # --- Stage (phụ trợ) ---
    stage_m = metrics.classification_metrics(hour_df["stage_true"], hour_df["stage_pred"])

    # --- Uncertainty / khoảng dự đoán (nếu có quantile) ---
    rul_picp = rul_mpiw = None
    if "rul_q10" in hour_df.columns and "rul_q90" in hour_df.columns:
        rul_picp = metrics.picp(rul_true, hour_df["rul_q10"], hour_df["rul_q90"])
        rul_mpiw = metrics.mpiw(hour_df["rul_q10"], hour_df["rul_q90"])

    results = {
        "run": run_name, "eval_unit": eval_unit,
        # ----- RUL: CHÍNH -----
        "rul_mae": rul_mae, "rul_rmse": rul_rmse, "rul_r2": rul_r2,
        "rul_mae_hours": rul_mae_hours, "rul_rmse_hours": rul_rmse_hours,
        "rul_phm_score": rul_phm, "rul_asym_score": rul_asym,
        "rul_picp": rul_picp, "rul_mpiw": rul_mpiw,
        # ----- Cảnh báo sớm headline (từ RUL) -----
        "first_warning": ew["first_warning"], "lead_time": ew["lead_time"],
        "false_alarm_rate": ew["false_alarm_rate"],
        # ----- MTOI/VTOI (phụ) -----
        "mtoi_rmse": mtoi_rmse, "mtoi_spearman": mtoi_spear, "mtoi_monotonicity": mtoi_mono,
        "mtoi_spearman_signed": mtoi_spear_signed,      # vs RUL   -> kỳ vọng ÂM  (R2 #23)
        "mtoi_spearman_vs_deg": mtoi_spear_deg,         # vs Deg   -> kỳ vọng DƯƠNG
        "mtoi_first_warning": ew_mtoi["first_warning"], "mtoi_lead_time": ew_mtoi["lead_time"],
        "mtoi_false_alarm_rate": ew_mtoi["false_alarm_rate"],
        # ----- Stage (phụ trợ) -----
        "stage_acc": stage_m["accuracy"], "stage_macro_f1": stage_m["macro_f1"],
        "stage_bal_acc": stage_m["balanced_accuracy"],
    }

    # In RUL-FIRST.
    hh = (f"MAE_h={rul_mae_hours:.3f} RMSE_h={rul_rmse_hours:.3f} "
          f"PHM={rul_phm:.1f} Asym={rul_asym:.3f}") if eval_unit == "hours" else "(chưa có giờ tuyệt đối)"
    logger.info(f"RUL[{eval_unit}]: MAE={rul_mae:.4f} RMSE={rul_rmse:.4f} R2={rul_r2:.3f} | {hh}")
    logger.info(f"Cảnh báo (từ RUL): first={ew['first_warning']} lead={ew['lead_time']} FAR={ew['false_alarm_rate']:.3f}")
    if rul_picp is not None:
        logger.info(f"Khoảng dự đoán: PICP={rul_picp:.3f} MPIW={rul_mpiw:.3f}")
    logger.info(f"MTOI(phụ): RMSE={mtoi_rmse:.4f} |Spearman|={mtoi_spear:.3f} Mono={mtoi_mono:.3f}")
    logger.info(f"Stage(phụ): Acc={stage_m['accuracy']:.3f} MacroF1={stage_m['macro_f1']:.3f}")

    if save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PREDICTIONS_DIR / f"predictions_{run_name}.csv"
        hour_df.to_csv(out_path, index=False)
        logger.info(f"Đã lưu dự đoán cấp giờ: {out_path}")

    return {"metrics": results, "hour_df": hour_df}
