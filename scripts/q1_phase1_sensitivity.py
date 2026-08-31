# -*- coding: utf-8 -*-
"""
q1_phase1_sensitivity.py — Pha 1: sensitivity của chất lượng INDEX theo healthy-baseline fraction H0.
Trả lời "tại sao 20%?". CPU thuần, không train lại. Đo độ ổn định của thành phần abnormality (Mahalanobis E)
khi đổi H0 ∈ {10,15,20,25,30}%: |Spearman(E, RUL)| (median qua bearing) per dataset.
Baseline trong-bearing (first H0%); đây là sensitivity của INDEX CONSTRUCTION, không phải RUL (không retrain).
Sinh: results/tables/q1_phase1_sensitivity.csv. Chạy: python scripts/q1_phase1_sensitivity.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT.parent / "data" / "processed"
PRED = ROOT / "results" / "predictions"
TAB = ROOT / "results" / "tables"
HC = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y", "SE_x", "SE_y", "ESE_x", "ESE_y"]
H0S = [0.10, 0.15, 0.20, 0.25, 0.30]


def lobo_names(ds_key):
    return [f.stem.replace(f"lobo_{ds_key}_", "").replace("_proposed", "")
            for f in sorted(PRED.glob(f"lobo_{ds_key}_*_proposed.csv"))]


def mahalanobis(X, h0frac):
    n = len(X); nh = max(5, int(h0frac * n))
    base = X[:nh]
    mu = base.mean(0); cov = np.cov(base, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    inv = np.linalg.pinv(cov)
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def main():
    rows = []
    for ds, key in [("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy")]:
        names = lobo_names(key)
        for h0 in H0S:
            rhos = []
            for nm in names:
                bd = PROC / key / nm
                try:
                    hf = pd.read_csv(bd / "hour_features.csv")
                    lab = pd.read_csv(bd / "labels_by_hour.csv")[["hour_id", "RUL_capped"]]
                    df = hf.merge(lab, on="hour_id").sort_values("hour_id")
                except Exception:
                    continue
                if len(df) < 12:
                    continue
                E = mahalanobis(df[HC].to_numpy(float), h0)
                rho, _ = spearmanr(E, df["RUL_capped"].to_numpy(float))
                if rho == rho:
                    rhos.append(abs(rho))
            rows.append({"dataset": ds, "healthy_baseline_pct": int(h0 * 100),
                         "median_abs_spearman_E_RUL": round(float(np.median(rhos)), 3),
                         "n_bearings": len(rhos)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_phase1_sensitivity.csv", index=False)
    print("=== Index abnormality quality vs healthy-baseline fraction (median |Spearman(E,RUL)|) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
