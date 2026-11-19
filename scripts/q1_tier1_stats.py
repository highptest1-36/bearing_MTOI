# -*- coding: utf-8 -*-
"""
q1_tier1_stats.py — TIER 1 (phản hồi reviewer Q1): thống kê nghiêm túc TỪ DỮ LIỆU CÓ SẴN.
KHÔNG train lại. Sinh:
  results/tables/q1_stats_summary.csv     — mean +/- std +/- CI95 mỗi config (LOBO per-fold).
  results/tables/q1_wilcoxon.csv          — Wilcoxon signed-rank ghép cặp proposed vs baseline.
  results/tables/q1_early_warning.csv      — precision/recall/FAR/lead-time early-warning (per-bearing + tổng).
Chạy:  python scripts/q1_tier1_stats.py
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PRED = ROOT / "results" / "predictions"

PROPOSED = "mtoi_vib_static"     # cấu hình đề xuất tốt nhất (theo lobo_main_performance)
BASELINE = "transformer_vib"     # baseline mạnh nhất hiện có
METRICS = ["rul_mae_hours", "rul_rmse_hours", "mtoi_spearman", "mtoi_monotonicity",
           "stage_macro_f1", "lead_time", "false_alarm_rate"]
LOWER_BETTER = {"rul_mae_hours", "rul_rmse_hours", "false_alarm_rate"}


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2: return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return (x.mean() - h, x.mean() + h)


def summarize(perfold_csv, dataset):
    d = pd.read_csv(perfold_csv)
    rows = []
    for cfg, g in d.groupby("config"):
        row = {"dataset": dataset, "config": cfg, "n_folds": len(g)}
        for m in METRICS:
            if m in g:
                v = g[m].to_numpy(float)
                lo, hi = ci95(v)
                row[f"{m}_mean"] = round(np.nanmean(v), 4)
                row[f"{m}_std"] = round(np.nanstd(v, ddof=1), 4)
                row[f"{m}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(perfold_csv, dataset):
    d = pd.read_csv(perfold_csv)
    out = []
    a = d[d.config == PROPOSED].set_index("holdout")
    b = d[d.config == BASELINE].set_index("holdout")
    common = sorted(set(a.index) & set(b.index))
    for m in ["rul_mae_hours", "rul_rmse_hours", "mtoi_spearman", "mtoi_monotonicity", "stage_macro_f1"]:
        if m not in d: continue
        xa = a.loc[common, m].to_numpy(float)
        xb = b.loc[common, m].to_numpy(float)
        mask = ~(np.isnan(xa) | np.isnan(xb))
        xa, xb = xa[mask], xb[mask]
        if len(xa) < 3 or np.allclose(xa, xb):
            out.append({"dataset": dataset, "metric": m, "n": len(xa), "note": "không đủ/đồng nhất"})
            continue
        try:
            w, p = stats.wilcoxon(xa, xb)
        except Exception as e:
            w, p = np.nan, np.nan
        better = "proposed" if ((np.median(xa) < np.median(xb)) == (m in LOWER_BETTER)) else "baseline"
        out.append({
            "dataset": dataset, "metric": m, "n": len(xa),
            f"{PROPOSED}_median": round(float(np.median(xa)), 4),
            f"{BASELINE}_median": round(float(np.median(xb)), 4),
            "median_diff": round(float(np.median(xa - xb)), 4),
            "wilcoxon_W": round(float(w), 3) if not np.isnan(w) else np.nan,
            "p_value": round(float(p), 4) if not np.isnan(p) else np.nan,
            "significant_0.05": (p < 0.05) if not np.isnan(p) else False,
            "favors": better,
        })
    return pd.DataFrame(out)


def early_warning(tau=0.6, q=3):
    """Early-warning từ predictions của model đề xuất.
    Quy tắc cảnh báo: mtoi_pred > tau trong q giờ liên tiếp.
    'Suy giảm thật' = stage_true >= 1. Healthy = stage_true == 0.
    """
    files = sorted(glob.glob(str(PRED / "*_proposed.csv")))
    rows = []
    for f in files:
        name = os.path.basename(f).replace("_proposed.csv", "")
        ds = "PRONOSTIA" if "pronostia" in name else ("XJTU-SY" if "xjtu" in name else "?")
        d = pd.read_csv(f).sort_values("hour_id").reset_index(drop=True)
        if "mtoi_pred" not in d or "stage_true" not in d:
            continue
        alarm_raw = (d["mtoi_pred"].to_numpy(float) > tau).astype(int)
        # yêu cầu q giờ liên tiếp
        alarm = np.zeros_like(alarm_raw)
        run = 0
        for i, a in enumerate(alarm_raw):
            run = run + 1 if a else 0
            if run >= q:
                alarm[i] = 1
        deg = (d["stage_true"].to_numpy(float) >= 1).astype(int)
        healthy = (d["stage_true"].to_numpy(float) == 0)
        TP = int(((alarm == 1) & (deg == 1)).sum())
        FP = int(((alarm == 1) & (deg == 0)).sum())
        FN = int(((alarm == 0) & (deg == 1)).sum())
        prec = TP / (TP + FP) if (TP + FP) else np.nan
        rec = TP / (TP + FN) if (TP + FN) else np.nan
        far = FP / healthy.sum() if healthy.sum() else np.nan
        # lead time (giờ): từ cảnh báo bền đầu tiên đến hỏng
        H_fail = float(d["hour_id"].iloc[-1]) if len(d) else np.nan
        scale = float(d["rul_scale_hours"].iloc[0]) if "rul_scale_hours" in d and len(d) else np.nan
        first_alarm = int(np.argmax(alarm == 1)) if alarm.any() else -1
        lead_h = (H_fail - first_alarm) * scale / H_fail if (first_alarm >= 0 and H_fail > 0) else 0.0
        # onset thật (giờ đầu tiên suy giảm)
        onset = int(np.argmax(deg == 1)) if deg.any() else -1
        warn_before_onset = (first_alarm >= 0 and onset >= 0 and first_alarm <= onset)
        rows.append({
            "dataset": ds, "bearing": name, "n_hours": len(d),
            "precision": round(prec, 3) if not np.isnan(prec) else np.nan,
            "recall": round(rec, 3) if not np.isnan(rec) else np.nan,
            "far": round(far, 4) if not np.isnan(far) else np.nan,
            "lead_time_h": round(lead_h, 3),
            "warned_before_onset": warn_before_onset,
        })
    df = pd.DataFrame(rows)
    # tổng hợp theo dataset
    agg = df.groupby("dataset").agg(
        n_bearings=("bearing", "size"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        far=("far", "mean"), lead_time_h=("lead_time_h", "mean"),
        pct_warned_before_onset=("warned_before_onset", "mean"),
    ).round(3).reset_index()
    return df, agg


def main():
    print("=== TIER 1: thống kê từ dữ liệu có sẵn (không train lại) ===\n")
    summ = pd.concat([summarize(TAB / "lobo_pronostia_perfold.csv", "PRONOSTIA"),
                      summarize(TAB / "lobo_xjtu_sy_perfold.csv", "XJTU-SY")], ignore_index=True)
    summ.to_csv(TAB / "q1_stats_summary.csv", index=False)
    print(f"[1/3] q1_stats_summary.csv ({len(summ)} dòng)")

    wil = pd.concat([paired_tests(TAB / "lobo_pronostia_perfold.csv", "PRONOSTIA"),
                     paired_tests(TAB / "lobo_xjtu_sy_perfold.csv", "XJTU-SY")], ignore_index=True)
    wil.to_csv(TAB / "q1_wilcoxon.csv", index=False)
    print(f"[2/3] q1_wilcoxon.csv ({len(wil)} dòng) — {PROPOSED} vs {BASELINE}")
    print(wil.to_string(index=False))

    ew, ew_agg = early_warning()
    ew.to_csv(TAB / "q1_early_warning.csv", index=False)
    ew_agg.to_csv(TAB / "q1_early_warning_summary.csv", index=False)
    print(f"\n[3/3] q1_early_warning.csv ({len(ew)} bearing) + summary:")
    print(ew_agg.to_string(index=False))

    print("\n=== TÓM TẮT cho paper (RUL MAE giờ, |rho|, monotonicity) ===")
    key = summ[summ.config.isin([PROPOSED, BASELINE, "mtoi_full_temp"])][
        ["dataset", "config", "rul_mae_hours_mean", "rul_mae_hours_ci95",
         "mtoi_spearman_mean", "mtoi_monotonicity_mean", "stage_macro_f1_mean"]]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
