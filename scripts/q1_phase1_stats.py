# -*- coding: utf-8 -*-
"""
q1_phase1_stats.py — Pha 1: thống kê BỔ SUNG cho paper (CPU, không train lại).
  - Wilcoxon + rank-biserial effect size + win/loss, proposed (mtoi_vib_static) vs từng baseline.
  - Holm-adjusted p (qua 4 baseline / dataset) — baseline nào sống ở 0.05.
  - PHM score + asymmetric score (trung bình per-fold) của proposed.
  - Partial-corr: median / IQR / MIN của MTOI vs RMS (từ q1_c1_partial_corr.csv).
Sinh: results/tables/q1_phase1_stats.csv (+ in ra màn hình). Chạy: python scripts/q1_phase1_stats.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PROP = "mtoi_vib_static"
BASES = ["transformer_vib", "tcn_vib", "tcn_transformer_vib", "cnn_bilstm_attn_vib"]
NAME = {"transformer_vib": "Transformer", "tcn_vib": "TCN",
        "tcn_transformer_vib": "TCN-Transformer", "cnn_bilstm_attn_vib": "CNN-BiLSTM-Attn"}


def rank_biserial(a, b):
    """Matched-pairs rank-biserial cho Wilcoxon signed-rank: (W+ - W-)/(W+ + W-).
    a,b = error proposed/baseline; d = b - a (>0 nghĩa proposed tốt hơn)."""
    d = np.asarray(b) - np.asarray(a)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.abs(d))) + 1
    wp = r[d > 0].sum(); wn = r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def holm(pvals):
    """Holm step-down adjusted p-values, trả dict {idx: p_adj}."""
    idx = np.argsort(pvals)
    m = len(pvals); adj = [0.0] * m; run = 0.0
    for k, i in enumerate(idx):
        val = (m - k) * pvals[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


def main():
    rows = []
    for ds, pf in [("PRONOSTIA", "lobo_pronostia_perfold.csv"),
                   ("XJTU-SY", "lobo_xjtu_sy_perfold.csv")]:
        d = pd.read_csv(TAB / pf)
        prop = d[d.config == PROP].set_index("holdout")["rul_mae_hours"]
        pvals, recs = [], []
        for b in BASES:
            base = d[d.config == b].set_index("holdout")["rul_mae_hours"]
            common = sorted(set(prop.index) & set(base.index))
            a = prop.loc[common].to_numpy(); bb = base.loc[common].to_numpy()
            try:
                _, p = wilcoxon(a, bb)
            except Exception:
                p = np.nan
            rb = rank_biserial(a, bb)
            wins = int((a < bb).sum())
            pvals.append(p)
            recs.append({"dataset": ds, "baseline": NAME[b], "n": len(common),
                         "wilcoxon_p": round(p, 4), "rank_biserial": round(rb, 3),
                         "prop_wins": f"{wins}/{len(common)}"})
        adj = holm(pvals)
        for r, pa in zip(recs, adj):
            r["holm_p"] = round(pa, 4); r["holm_sig_0.05"] = pa < 0.05
            rows.append(r)
        # PHM + asym của proposed
        sub = d[d.config == PROP]
        phm = sub["rul_phm_score"].dropna(); asym = sub["rul_asym_score"].dropna()
        print(f"[{ds}] PHM score (proposed) mean={phm.mean():.3f}  asym mean={asym.mean():.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_phase1_stats.csv", index=False)
    print("\n=== Wilcoxon + effect size + Holm (proposed vs baseline) ===")
    print(out.to_string(index=False))

    # Partial-corr median/IQR/min
    pc = pd.read_csv(TAB / "q1_c1_partial_corr.csv")
    print("\n=== Partial-corr (MTOI vs defect-energy | lifetime) ===")
    for col in ["MTOI_partial", "RMS_partial", "Kurtosis_partial"]:
        if col in pc:
            v = pc[col].dropna()
            print(f"  {col:18s} median={v.median():.3f}  IQR=[{v.quantile(.25):.3f},{v.quantile(.75):.3f}]  "
                  f"min={v.min():.3f}  n_neg={(v<0).sum()}/{len(v)}")
    print(f"  raw MTOI median={pc['MTOI_raw'].median():.3f}" if "MTOI_raw" in pc else "")


if __name__ == "__main__":
    main()
