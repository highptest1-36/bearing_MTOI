# -*- coding: utf-8 -*-
"""
q1_per_condition_xjtu.py — REVIEWER Q1: phân tách XJTU-SY theo 3 ĐIỀU KIỆN vận hành
(35Hz12kN, 37.5Hz11kN, 40Hz10kN) để xem gain ổn định hay do vài bearing/điều kiện.

Đọc results/tables/lobo_xjtu_sy_perfold.csv (đã có rul_mae_hours per bearing).
Sinh results/tables/q1_per_condition_xjtu.csv. CPU thuần, không train lại.
Chạy: python scripts/q1_per_condition_xjtu.py
"""
from pathlib import Path
import re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PROP = "mtoi_vib_static"
BEST_BASE = "cnn_bilstm_attn_vib"     # baseline mạnh nhất trên XJTU (Bảng 2)


def condition_of(holdout):
    m = re.match(r"(.+?)_Bearing", holdout)
    return m.group(1) if m else holdout


def main():
    d = pd.read_csv(TAB / "lobo_xjtu_sy_perfold.csv")
    d["cond"] = d["holdout"].map(condition_of)
    conds = sorted(d["cond"].unique())
    rows = []
    for cond in conds + ["ALL"]:
        sub = d if cond == "ALL" else d[d["cond"] == cond]
        prop = sub[sub.config == PROP].set_index("holdout")["rul_mae_hours"]
        base = sub[sub.config == BEST_BASE].set_index("holdout")["rul_mae_hours"]
        common = sorted(set(prop.index) & set(base.index))
        a = prop.loc[common].to_numpy(); b = base.loc[common].to_numpy()
        try:
            _, p = wilcoxon(a, b)
        except Exception:
            p = np.nan
        won = int((a < b).sum())
        rows.append({
            "condition": cond, "n_bearings": len(common),
            "proposed_mae_h": round(float(a.mean()), 3),
            "proposed_std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3),
            "best_baseline_mae_h": round(float(b.mean()), 3),
            "best_baseline_std": round(float(b.std(ddof=1)) if len(b) > 1 else 0.0, 3),
            "proposed_wins": f"{won}/{len(common)}",
            "wilcoxon_p": round(float(p), 4) if p == p else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "q1_per_condition_xjtu.csv", index=False)
    print("=== XJTU-SY per-condition (proposed vs best baseline =", BEST_BASE, ") ===")
    print(out.to_string(index=False))
    print("\n[bảng] q1_per_condition_xjtu.csv")


if __name__ == "__main__":
    main()
