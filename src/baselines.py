# -*- coding: utf-8 -*-
"""
baselines.py — CÁC BASELINE (plan mục 15): so sánh để chứng minh MTOI & model tốt hơn.

Gồm 2 nhóm:
  (A) HEALTH INDICATOR baselines (Table 3): so MTOI với RMS, Kurtosis, Crest, Temperature,
      Temp-slope, Mahalanobis E, State-change C, PCA-HI, Autoencoder-HI, Fixed/Learnable MTOI.
      Đánh giá mỗi HI bằng: Monotonicity, |Spearman với RUL|, Smoothness, Lead time, False alarm.
  (B) MODEL baselines truyền thống (Table 2 phần ML): SVM, Random Forest, XGBoost
      cho phân loại stage + hồi quy RUL, dùng đặc trưng cấp cửa sổ, chia theo thời gian.

(Các baseline deep: 1D-CNN, LSTM, Transformer, fusion-concat dùng lại src/train.py với cờ khác nhau.)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import PROC_MENDELEY, TABLES_DIR
from src.utils.logger import get_logger, section
from src import metrics


# ============================ (A) HEALTH INDICATOR COMPARISON ============================

def _pca_hi(F, baseline_idx):
    """PCA-HI: chiếu đặc trưng lên trục chính số 1 học từ baseline khoẻ; HI = |điểm PC1|."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)                              # 1 thành phần chính
    pca.fit(F[baseline_idx])                               # học từ baseline (leakage-free)
    score = pca.transform(F)[:, 0]                         # điểm PC1 cho mọi giờ
    return np.abs(score - np.median(score[baseline_idx]))  # lệch so với baseline -> HI dương


def _autoencoder_hi(F, baseline_idx, epochs=200):
    """Autoencoder-HI: AE nhỏ học tái tạo trạng thái khoẻ; HI = sai số tái tạo."""
    import torch, torch.nn as nn
    X = torch.tensor(F, dtype=torch.float32)
    d = X.shape[1]
    ae = nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, d))  # AE rất nhỏ
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    base = X[list(baseline_idx)]                            # chỉ train trên baseline khoẻ
    for _ in range(epochs):                                # vòng lặp huấn luyện AE
        opt.zero_grad()
        loss = ((ae(base) - base) ** 2).mean()             # MSE tái tạo
        loss.backward(); opt.step()
    with torch.no_grad():
        rec = ae(X)                                        # tái tạo mọi giờ
        err = ((rec - X) ** 2).mean(dim=1).numpy()         # sai số -> HI
    return err


def build_hi_comparison(proc_dir=None, baseline_frac=0.2):
    """Tạo Table 3: so sánh chất lượng các HI khác nhau (gồm MTOI)."""
    logger = get_logger("baselines")
    section("BASELINE (A) — SO SÁNH HEALTH INDICATOR (Table 3)", logger)
    proc_dir = Path(proc_dir or PROC_MENDELEY)

    hf = pd.read_csv(proc_dir / "hour_features.csv").sort_values("hour_id").reset_index(drop=True)
    mtoi = pd.read_csv(proc_dir / "mtoi_by_hour.csv").sort_values("hour_id").reset_index(drop=True)
    labels = pd.read_csv(proc_dir / "labels_by_hour.csv").sort_values("hour_id").reset_index(drop=True)
    H = len(hf)
    rul = labels["RUL"].to_numpy()
    failure_hour = H - 1
    n_base = max(2, int(baseline_frac * H))
    baseline_idx = np.arange(n_base)

    # Ma trận đặc trưng cho PCA/AE.
    feat_cols = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y",
                 "SE_x", "SE_y", "ESE_x", "ESE_y", "T_bearing_mean", "DeltaT_mean"]
    F = hf[feat_cols].to_numpy(float)
    # Chuẩn hoá đơn giản (z-score toàn cục) cho PCA/AE ổn định.
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)

    # Định nghĩa các HI cần so sánh (mỗi HI là 1 chuỗi theo giờ).
    his = {
        "RMS": (hf["RMS_x"] + hf["RMS_y"]).to_numpy() / 2,
        "Kurtosis": (hf["Kurt_x"] + hf["Kurt_y"]).to_numpy() / 2,
        "CrestFactor": (hf["CF_x"] + hf["CF_y"]).to_numpy() / 2,
        "Temperature": hf["T_bearing_mean"].to_numpy(),
        "TempSlope": np.gradient(hf["DeltaT_mean"].to_numpy()),
        "Mahalanobis_E": mtoi["E_h"].to_numpy(),
        "StateChange_C": mtoi["C_h"].to_numpy(),
        "PCA_HI": _pca_hi(F, baseline_idx),
        "AE_HI": _autoencoder_hi(F, baseline_idx),
        "MTOI_fixed": mtoi["MTOI_fixed"].to_numpy(),
        "MTOI_learnable": mtoi["MTOI_learnable"].to_numpy(),
    }

    # Đánh giá mỗi HI.
    rows = []
    for name, hi in his.items():
        hi = np.asarray(hi, float)
        # Chuẩn hoá min-max để so sánh lead time/false alarm công bằng (cùng thang [0,1]).
        hi_n = (hi - hi.min()) / (hi.max() - hi.min() + 1e-9)
        ew = metrics.early_warning_metrics(hi_n, tau=0.6, failure_hour=failure_hour, q=3)
        rows.append({
            "indicator": name,
            "monotonicity": round(metrics.monotonicity(hi), 4),
            "spearman_with_RUL": round(abs(metrics.spearman_corr(hi, rul)), 4),
            "smoothness": round(metrics.smoothness(hi_n), 2),
            "lead_time": ew["lead_time"],
            "false_alarm_rate": round(ew["false_alarm_rate"], 4),
        })
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table3_health_indicator_comparison.csv"
    table.to_csv(out, index=False)
    logger.info(f"Đã ghi {out}")
    print(table.to_string(index=False))
    return table


# ============================ (B) TRADITIONAL ML BASELINES ============================

def run_ml_baselines(proc_dir=None, train_frac=0.6, val_frac=0.2):
    """SVM / Random Forest / XGBoost cho stage (phân loại) & RUL (hồi quy)."""
    logger = get_logger("baselines")
    section("BASELINE (B) — ML TRUYỀN THỐNG (SVM/RF/XGBoost)", logger)
    proc_dir = Path(proc_dir or PROC_MENDELEY)

    win = pd.read_csv(proc_dir / "window_features.csv")
    labels = pd.read_csv(proc_dir / "labels_by_hour.csv")
    win = win.merge(labels, on="hour_id", how="left")     # gắn nhãn theo giờ vào từng cửa sổ

    feat_cols = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y",
                 "SE_x", "SE_y", "ESE_x", "ESE_y"]
    # Chia theo thời gian (theo hour_id) để chống leakage.
    hrs = np.sort(win["hour_id"].unique())
    H = len(hrs)
    test_hours = set(hrs[int((train_frac + val_frac) * H):])
    train_hours = set(hrs[:int(train_frac * H)])
    tr = win[win["hour_id"].isin(train_hours)]
    te = win[win["hour_id"].isin(test_hours)]
    Xtr, Xte = tr[feat_cols].to_numpy(), te[feat_cols].to_numpy()
    # Chuẩn hoá theo train.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    models = {
        "SVM": (SVC(), SVR()),
        "RandomForest": (RandomForestClassifier(n_estimators=200, n_jobs=-1),
                         RandomForestRegressor(n_estimators=200, n_jobs=-1)),
    }
    # XGBoost nếu có cài.
    try:
        from xgboost import XGBClassifier, XGBRegressor
        models["XGBoost"] = (XGBClassifier(n_estimators=300, max_depth=5, n_jobs=-1,
                                           use_label_encoder=False, eval_metric="mlogloss"),
                             XGBRegressor(n_estimators=300, max_depth=5, n_jobs=-1))
    except Exception:
        logger.info("XGBoost chưa cài -> bỏ qua (cài bằng: !pip install xgboost).")

    rows = []
    # Chia theo thời gian TRONG 1 bearing: các stage xuất hiện tuần tự, không chồng lấn ->
    # train (giờ đầu) thường chỉ có stage healthy. Phân loại stage khi đó ill-posed
    # (test toàn stage chưa từng thấy). -> Bỏ phân loại stage, CHỈ báo RUL (luôn hợp lệ).
    # Phân loại stage hợp lệ nằm ở LOBO cross-bearing (phase9b), nơi train thấy đủ stage.
    n_stage_classes = tr["stage_label"].nunique()
    stage_ok = n_stage_classes >= 2
    if not stage_ok:
        logger.info(f"  ⚠️ TRAIN chỉ có {n_stage_classes} lớp stage (split theo thời gian, 1 bearing) "
                    f"-> BỎ phân loại stage cho ML baseline (ill-posed); chỉ xuất RUL. "
                    f"Stage classification hợp lệ ở LOBO cross-bearing.")
    for name, (clf, reg) in models.items():
        reg.fit(Xtr, tr["RUL"])                                      # RUL regression (luôn hợp lệ)
        te = te.copy()
        te["rul_pred"] = reg.predict(Xte)
        if stage_ok:
            clf.fit(Xtr, tr["stage_label"])
            te["stage_pred"] = clf.predict(Xte)
            agg = te.groupby("hour_id").agg(
                stage_pred=("stage_pred", lambda s: s.value_counts().idxmax()),
                stage_true=("stage_label", "first"),
                rul_pred=("rul_pred", "mean"), rul_true=("RUL", "first")).reset_index()
            cm = metrics.classification_metrics(agg["stage_true"], agg["stage_pred"])
            stage_acc, stage_f1 = round(cm["accuracy"], 4), round(cm["macro_f1"], 4)
        else:
            agg = te.groupby("hour_id").agg(
                rul_pred=("rul_pred", "mean"), rul_true=("RUL", "first")).reset_index()
            stage_acc, stage_f1 = float("nan"), float("nan")
        rows.append({
            "method": name,
            "stage_acc": stage_acc, "stage_macro_f1": stage_f1,
            "rul_mae": round(metrics.mae(agg["rul_true"], agg["rul_pred"]), 4),
            "rul_rmse": round(metrics.rmse(agg["rul_true"], agg["rul_pred"]), 4),
        })
        logger.info(f"  {name}: stageF1={rows[-1]['stage_macro_f1']} rulRMSE={rows[-1]['rul_rmse']}")

    table = pd.DataFrame(rows)
    out = TABLES_DIR / "baseline_ml_results.csv"
    table.to_csv(out, index=False)
    logger.info(f"Đã ghi {out}")
    return table
