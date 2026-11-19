# -*- coding: utf-8 -*-
"""
q1_tier_c_analysis.py — TIER C (reviewer Q1 v2): phân tích sâu từ DỮ LIỆU CÓ SẴN (không train lại).
  C1: partial/detrended Spearman(MTOI, defect-energy | lifetime) + so RMS/Kurtosis -> chống confound thời gian.
  C2: early-warning so MTOI vs RMS-3sigma vs Kurtosis-3sigma (onset ĐỘC LẬP theo stage) + PR curve.
  C3: weight stability (a,b,c) across bearings.
Sinh: results/tables/q1_c1_partial_corr.csv, q1_c2_ew_comparison.csv, q1_c2_pr_curve.csv, q1_c3_weight_stability.csv
Chạy: python scripts/q1_tier_c_analysis.py
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.q1_tier3_physics import per_hour_defect_energy, shaft_freq, XJTU
TAB = ROOT / "results" / "tables"
PROC = ROOT.parent / "data" / "processed"


def _pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def partial_spearman(x, y, z):
    """Spearman riêng phần corr(x,y | z): rank-transform rồi partial-Pearson."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = _pear(rx, ry), _pear(rx, rz), _pear(ry, rz)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)) + 1e-12
    return (rxy - rxz * ryz) / denom


def c1_partial_correlation():
    """MTOI vs RMS vs Kurtosis: tương quan với defect-energy, THÔ và RIÊNG PHẦN (kiểm soát lifetime)."""
    rows = []
    bearings = sorted([d for d in XJTU.iterdir() if d.is_dir() and "Bearing" in d.name])
    for bd in bearings:
        fr = shaft_freq(bd.name)
        try:
            de, _ = per_hour_defect_energy(bd, fr)
            mt = pd.read_csv(bd / "mtoi_by_hour.csv")[["hour_id", "MTOI_learnable"]]
            hf = pd.read_csv(bd / "hour_features.csv")
            hf["RMS"] = hf["RMS_x"] + hf["RMS_y"]; hf["KURT"] = hf["Kurt_x"] + hf["Kurt_y"]
            df = de.merge(mt, on="hour_id").merge(hf[["hour_id", "RMS", "KURT"]], on="hour_id").sort_values("hour_id")
        except Exception as ex:
            continue
        if len(df) < 10:
            continue
        life = df["hour_id"].to_numpy(float)
        defe = df["ALL"].to_numpy(float)
        rec = {"bearing": bd.name, "n_hours": len(df)}
        for hi_name, col in [("MTOI", "MTOI_learnable"), ("RMS", "RMS"), ("Kurtosis", "KURT")]:
            hi = df[col].to_numpy(float)
            rec[f"{hi_name}_raw"] = round(spearmanr(hi, defe)[0], 3)
            rec[f"{hi_name}_partial"] = round(partial_spearman(hi, defe, life), 3)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_c1_partial_corr.csv", index=False)
    # tóm tắt: median raw vs partial
    summ = {hi: (round(out[f"{hi}_raw"].median(), 3), round(out[f"{hi}_partial"].median(), 3))
            for hi in ["MTOI", "RMS", "Kurtosis"]}
    print("[C1] median (raw, partial) Spearman với defect-energy:")
    for hi, (r, p) in summ.items():
        print(f"   {hi:9s}: raw={r}  partial(|lifetime)={p}")
    print(f"   -> {len(out)} bearing. File: q1_c1_partial_corr.csv")
    return out


def _warn_first(signal, healthy_frac=0.2, k=3.0, q=3, fixed_tau=None):
    """Trả index cảnh báo bền đầu tiên. Nếu fixed_tau: signal>tau; else: 3-sigma trên baseline healthy."""
    n = len(signal)
    if fixed_tau is not None:
        alarm_raw = signal > fixed_tau
    else:
        nh = max(5, int(healthy_frac * n))
        mu, sd = signal[:nh].mean(), signal[:nh].std() + 1e-9
        alarm_raw = signal > mu + k * sd
    run = 0
    for i in range(n):
        run = run + 1 if alarm_raw[i] else 0
        if run >= q:
            return i - q + 1
    return -1


def c2_early_warning_comparison():
    """So MTOI(tau=0.6) vs RMS-3sigma vs Kurtosis-3sigma. Onset ĐỘC LẬP = giờ đầu stage_true>=1."""
    rows = []
    for ds_name, ds_dir in [("PRONOSTIA", PROC / "pronostia"), ("XJTU-SY", PROC / "xjtu_sy")]:
        for bd in sorted([d for d in ds_dir.iterdir() if d.is_dir()]):
            try:
                mt = pd.read_csv(bd / "mtoi_by_hour.csv")
                hf = pd.read_csv(bd / "hour_features.csv")
                lab = pd.read_csv(bd / "labels_by_hour.csv")
                df = mt.merge(hf, on="hour_id").merge(lab[["hour_id", "stage_label"]], on="hour_id").sort_values("hour_id")
            except Exception:
                continue
            if len(df) < 12 or (df["stage_label"] >= 1).sum() == 0:
                continue
            H = len(df)
            onset = int(np.argmax((df["stage_label"] >= 1).to_numpy()))
            deg = (df["stage_label"].to_numpy() >= 1)
            detectors = {
                "MTOI": _warn_first(df["MTOI_learnable"].to_numpy(float), fixed_tau=0.6),
                "RMS_3sigma": _warn_first((df["RMS_x"] + df["RMS_y"]).to_numpy(float)),
                "Kurtosis_3sigma": _warn_first((df["Kurt_x"] + df["Kurt_y"]).to_numpy(float)),
            }
            for name, fa in detectors.items():
                # precision/recall ở mức giờ: cảnh báo từ fa trở đi
                pred = np.zeros(H, bool)
                if fa >= 0:
                    pred[fa:] = True
                TP = int((pred & deg).sum()); FP = int((pred & ~deg).sum()); FN = int((~pred & deg).sum())
                prec = TP / (TP + FP) if (TP + FP) else np.nan
                rec = TP / (TP + FN) if (TP + FN) else np.nan
                lead = (onset - fa) if (fa >= 0 and fa <= onset) else (0 if fa < 0 else -(fa - onset))
                rows.append({"dataset": ds_name, "bearing": bd.name, "detector": name,
                             "first_alarm": fa, "onset": onset, "lead_snapshots": lead,
                             "precision": round(prec, 3) if not np.isnan(prec) else np.nan,
                             "recall": round(rec, 3) if not np.isnan(rec) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_c2_ew_comparison.csv", index=False)
    agg = out.groupby(["dataset", "detector"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        lead=("lead_snapshots", "mean")).round(3).reset_index()
    print("\n[C2] Early-warning so sánh (onset độc lập theo stage):")
    print(agg.to_string(index=False))
    print("   File: q1_c2_ew_comparison.csv")
    return out


def c2_pr_curve():
    """PR curve cho MTOI: quét ngưỡng tau, gộp toàn bộ bearing 2 dataset (mức giờ)."""
    Y, S = [], []
    for ds_dir in [PROC / "pronostia", PROC / "xjtu_sy"]:
        for bd in sorted([d for d in ds_dir.iterdir() if d.is_dir()]):
            try:
                mt = pd.read_csv(bd / "mtoi_by_hour.csv"); lab = pd.read_csv(bd / "labels_by_hour.csv")
                df = mt.merge(lab[["hour_id", "stage_label"]], on="hour_id")
            except Exception:
                continue
            Y.append((df["stage_label"].to_numpy() >= 1).astype(int)); S.append(df["MTOI_learnable"].to_numpy(float))
    if not Y:
        return
    y = np.concatenate(Y); s = np.concatenate(S)
    rows = []
    for tau in np.linspace(0.05, 0.95, 19):
        pred = s > tau
        TP = int((pred & (y == 1)).sum()); FP = int((pred & (y == 0)).sum()); FN = int(((~pred) & (y == 1)).sum())
        prec = TP / (TP + FP) if (TP + FP) else 1.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        rows.append({"tau": round(float(tau), 3), "precision": round(prec, 3), "recall": round(rec, 3)})
    pd.DataFrame(rows).to_csv(TAB / "q1_c2_pr_curve.csv", index=False)
    print("   PR curve (MTOI) -> q1_c2_pr_curve.csv")


def c3_weight_stability():
    """Mean±std của a,b,c qua các file trọng số có sẵn (per-dataset learnable)."""
    files = glob.glob(str(TAB / "learned_mtoi_weights_*.csv"))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        ln = d[d["type"] == "learnable"]
        for _, r in ln.iterrows():
            rows.append({"source": os.path.basename(f).replace("learned_mtoi_weights_", "").replace(".csv", ""),
                         "a_E": r["a_E"], "b_C": r["b_C"], "c_G": r["c_G"], "fitness": r.get("fitness", np.nan)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_c3_weight_stability.csv", index=False)
    print("\n[C3] Learned weights (per source):")
    print(out.to_string(index=False))
    return out


def main():
    print("=== TIER C: phân tích sâu (không train lại) ===")
    c1_partial_correlation()
    c2_early_warning_comparison()
    c2_pr_curve()
    c3_weight_stability()


if __name__ == "__main__":
    main()
