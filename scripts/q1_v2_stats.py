# -*- coding: utf-8 -*-
"""
q1_v2_stats.py — TỔNG HỢP KẾT QUẢ CHÍNH + THỐNG KÊ cho bản nộp lại (CPU, không train lại).

Sinh ra (results/tables/v2/):
  A. main_results.csv        — Bảng 2: median/IQR/worst/mean, gộp QUA SEED (R2 #7, #28)
  B. wilcoxon.csv            — Bảng 3: Wilcoxon + rank-biserial + Holm, trên lỗi TRUNG BÌNH QUA SEED (R2 #27)
  C. factorial_2x2.csv       — Bảng 5: ablation giai thừa conditioning × aux-loss (R2 #2)
  D. controls.csv            — Bảng 6: control battery + so sánh cặp với đề xuất (R2 #25)
  E. classical.csv           — Bảng 7: baseline cổ điển, ĐÚNG cấu hình đề xuất (sửa lỗi A6, R2 #10)
  F. seed_variability.csv    — biến thiên qua seed (R2 #9, #27)
  G. per_bearing_all.csv     — phụ lục: mọi bearing × config × seed (R2 #28)

Chạy: python scripts/q1_v2_stats.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAB = ROOT / "results" / "tables" / "v2"
PRED = ROOT / "results" / "predictions" / "v2"
PROC = ROOT.parent / "data" / "processed"
TAB.mkdir(parents=True, exist_ok=True)

PROPOSED = "vtoi_static"
BASELINES = ["transformer_vib", "tcn_vib", "tcn_transformer_vib", "cnn_bilstm_attn_vib"]
PRIMARY = "rul_mae"            # R2 #7: metric CHÍNH là lifetime-fraction chuẩn hoá
SECONDARY = "rul_mae_hours"    # thang giờ = RETROSPECTIVE, chỉ để so sánh giữa các bearing
CAP_ONSET = 0.4
HC = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y", "SE_x", "SE_y", "ESE_x", "ESE_y"]


# ============================================================ tiện ích
def load_perfold():
    """Gộp mọi lobo_v2_<ds>_seed*_perfold.csv."""
    fs = sorted(TAB.glob("lobo_v2_*_seed*_perfold.csv"))
    if not fs:
        raise SystemExit(f"Không tìm thấy perfold nào trong {TAB}. Chạy phase9b_v2.py trước.")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    print(f"[nạp] {len(fs)} file | {len(df)} dòng | seeds={sorted(df.seed.unique())} "
          f"| configs={df.config.nunique()}")
    return df


def seed_avg(df, metric):
    """Lỗi TRUNG BÌNH QUA SEED cho từng (dataset, config, bearing) -> đơn vị ghép cặp hợp lệ."""
    return (df.groupby(["dataset", "config", "holdout"])[metric]
              .mean().reset_index().rename(columns={metric: "err"}))


def rank_biserial(a, b):
    """Matched-pairs rank-biserial: tỉ lệ hạng ủng hộ b (proposed) trừ ủng hộ a."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    """Holm step-down adjusted p-values."""
    idx = np.argsort(pvals)
    n, out, run = len(pvals), np.empty(len(pvals)), 0.0
    for k, i in enumerate(idx):
        run = max(run, (n - k) * pvals[i])
        out[i] = min(run, 1.0)
    return out


def paired_test(sa, sb, label_a, label_b):
    """Wilcoxon ghép cặp theo bearing giữa 2 config (sa=baseline, sb=proposed)."""
    m = pd.merge(sa, sb, on=["dataset", "holdout"], suffixes=("_a", "_b"))
    if len(m) < 3:
        return None
    a, b = m["err_a"].to_numpy(), m["err_b"].to_numpy()
    try:
        _, p = wilcoxon(a, b)
    except Exception:
        p = np.nan
    return {"comparison": f"{label_b} vs {label_a}", "n": len(m),
            "mean_a": round(a.mean(), 4), "mean_b": round(b.mean(), 4),
            "median_diff": round(float(np.median(a - b)), 4),
            "p_value": round(float(p), 4) if p == p else np.nan,
            "rank_biserial": round(rank_biserial(a, b), 3),
            "prop_wins": f"{int((b < a).sum())}/{len(m)}"}


# ============================================================ A. main results
def a_main_results(df):
    rows = []
    for (ds, cfg), g in df.groupby(["dataset", "config"]):
        r = {"dataset": ds, "config": cfg, "n_seeds": g.seed.nunique(),
             "n_bearings": g.holdout.nunique()}
        for met, tag in [(PRIMARY, "norm"), (SECONDARY, "hours")]:
            sa = seed_avg(g, met)["err"]
            r[f"{tag}_median"] = round(float(sa.median()), 4)
            r[f"{tag}_q25"] = round(float(sa.quantile(.25)), 4)
            r[f"{tag}_q75"] = round(float(sa.quantile(.75)), 4)
            r[f"{tag}_worst"] = round(float(sa.max()), 4)
            r[f"{tag}_mean"] = round(float(sa.mean()), 4)
            r[f"{tag}_std"] = round(float(sa.std(ddof=1)), 4) if len(sa) > 1 else 0.0
        for k in ["mtoi_spearman_signed", "mtoi_spearman_vs_deg", "mtoi_monotonicity",
                  "rul_rmse_hours", "rul_phm_score", "rul_asym_score"]:
            if k in g.columns:
                v = pd.to_numeric(g[k], errors="coerce").dropna()
                if len(v):
                    r[f"{k}_mean"] = round(float(v.mean()), 4)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["dataset", "norm_mean"])
    out.to_csv(TAB / "main_results.csv", index=False)
    print("\n=== A. MAIN RESULTS (metric chính = normalized lifetime fraction) ===")
    print(out[["dataset", "config", "n_seeds", "norm_median", "norm_q25", "norm_q75",
               "norm_worst", "norm_mean", "hours_mean"]].to_string(index=False))
    return out


# ============================================================ B. wilcoxon vs baselines
def b_wilcoxon(df):
    rows = []
    for ds, g in df.groupby("dataset"):
        sb = seed_avg(g[g.config == PROPOSED], PRIMARY)
        block = []
        for bl in BASELINES:
            sa = seed_avg(g[g.config == bl], PRIMARY)
            r = paired_test(sa, sb, bl, PROPOSED)
            if r:
                r["dataset"] = ds
                block.append(r)
        if block:
            ps = np.array([b["p_value"] for b in block], float)
            for b, ph in zip(block, holm(np.nan_to_num(ps, nan=1.0))):
                b["holm_p"] = round(float(ph), 4)
                b["holm_sig_0.05"] = bool(ph < 0.05)
            rows += block
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "wilcoxon.csv", index=False)
    print("\n=== B. WILCOXON (lỗi trung bình qua seed, ghép cặp theo bearing) ===")
    if len(out):
        print(out[["dataset", "comparison", "n", "mean_a", "mean_b", "p_value",
                   "holm_p", "rank_biserial", "prop_wins"]].to_string(index=False))
    return out


# ============================================================ C. factorial 2x2
def c_factorial(df):
    """R2 #2: conditioning (use_hi) × auxiliary loss (lambda1) — thiết kế giai thừa đầy đủ."""
    cells = {("off", "off"): "transformer_vib", ("off", "on"): "abl_no_idxhead",
             ("on", "off"): "abl_cond_noaux",   ("on", "on"): PROPOSED}
    rows = []
    for ds, g in df.groupby("dataset"):
        avail = {k: v for k, v in cells.items() if v in set(g.config)}
        for (cond, aux), cfg in avail.items():
            sa = seed_avg(g[g.config == cfg], PRIMARY)["err"]
            rows.append({"dataset": ds, "conditioning": cond, "aux_loss": aux, "config": cfg,
                         "n": len(sa), "mean": round(float(sa.mean()), 4),
                         "median": round(float(sa.median()), 4)})
        if len(avail) == 4:                       # hiệu ứng chính + tương tác
            m = {k: seed_avg(g[g.config == v], PRIMARY)["err"].mean() for k, v in avail.items()}
            rows.append({"dataset": ds, "conditioning": "MAIN EFFECT", "aux_loss": "-",
                         "config": "cond(on)-cond(off)", "n": 0,
                         "mean": round(float((m[("on", "off")] + m[("on", "on")]) / 2
                                             - (m[("off", "off")] + m[("off", "on")]) / 2), 4)})
            rows.append({"dataset": ds, "conditioning": "-", "aux_loss": "MAIN EFFECT",
                         "config": "aux(on)-aux(off)", "n": 0,
                         "mean": round(float((m[("off", "on")] + m[("on", "on")]) / 2
                                             - (m[("off", "off")] + m[("on", "off")]) / 2), 4)})
            rows.append({"dataset": ds, "conditioning": "INTERACTION", "aux_loss": "INTERACTION",
                         "config": "cond x aux", "n": 0,
                         "mean": round(float(m[("on", "on")] - m[("on", "off")]
                                             - m[("off", "on")] + m[("off", "off")]), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "factorial_2x2.csv", index=False)
    print("\n=== C. ABLATION GIAI THỪA 2x2 (conditioning × aux loss) ===")
    print(out.to_string(index=False))
    return out


# ============================================================ D. control battery
def d_controls(df):
    """R2 #25. In kèm CẢNH BÁO nếu control âm (random/shuffled/elapsed) ngang đề xuất."""
    ctls = sorted([c for c in df.config.unique() if c.startswith("ctl_")])
    rows = []
    for ds, g in df.groupby("dataset"):
        sb = seed_avg(g[g.config == PROPOSED], PRIMARY)
        for cfg in ctls + ["abl_no_idxhead", PROPOSED]:
            if cfg not in set(g.config):
                continue
            sa = seed_avg(g[g.config == cfg], PRIMARY)
            r = {"dataset": ds, "config": cfg, "n": len(sa),
                 "mean": round(float(sa["err"].mean()), 4),
                 "median": round(float(sa["err"].median()), 4),
                 "hi_cols": g[g.config == cfg]["hi_cols"].iloc[0]}
            if cfg != PROPOSED:
                t = paired_test(sa, sb, cfg, PROPOSED)
                r.update({"p_vs_proposed": t["p_value"], "proposed_wins": t["prop_wins"]})
            rows.append(r)
    out = pd.DataFrame(rows).sort_values(["dataset", "mean"])
    out.to_csv(TAB / "controls.csv", index=False)
    print("\n=== D. CONTROL BATTERY (R2 #25) ===")
    print(out.to_string(index=False))

    print("\n--- GO / NO-GO ---")
    for ds, g in out.groupby("dataset"):
        prop = g[g.config == PROPOSED]["mean"]
        if not len(prop):
            continue
        prop = float(prop.iloc[0])
        for neg, name in [("ctl_random", "trọng số ngẫu nhiên"),
                          ("ctl_shuffled", "xáo trộn thời gian"),
                          ("ctl_elapsed", "đồng hồ thuần")]:
            v = g[g.config == neg]["mean"]
            if not len(v):
                continue
            v = float(v.iloc[0])
            flag = "🔴 KHÔNG ĐẠT" if v <= prop * 1.02 else "✅ đạt"
            print(f"  [{ds}] {name:22s}: {v:.4f} vs đề xuất {prop:.4f}  -> {flag}")
    return out


# ============================================================ E. classical baselines
def e_classical(df):
    """
    Baseline cổ điển trên CÙNG feature, CÙNG fold, CÙNG mức snapshot.
    SỬA LỖI A6: lấy dự đoán deep từ ĐÚNG config đề xuất (`vtoi_static`), không phải config khác.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        import xgboost as xgb
        HAS_XGB = True
    except Exception:
        HAS_XGB = False
        print("[cảnh báo] thiếu xgboost -> bỏ dòng XGBoost")

    seeds = sorted(df.seed.unique())
    rows = []
    for ds_name, key in [("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy")]:
        ddir = PROC / key
        # tập bearing = ĐÚNG các holdout của config đề xuất
        bearings = sorted(df[(df.dataset == key) & (df.config == PROPOSED)].holdout.unique())
        if not bearings:
            continue

        # deep proposed, snapshot-level, trung bình qua seed
        deep = {}
        for b in bearings:
            vals = []
            for s in seeds:
                f = PRED / f"lobo_v2_{key}_s{s}_{b}_{PROPOSED}.csv"
                if f.exists():
                    p = pd.read_csv(f)
                    vals.append(float(np.mean(np.abs(p.rul_pred - p.rul_true))))
            if vals:
                deep[b] = float(np.mean(vals))
        if not deep:
            print(f"[bỏ qua] {ds_name}: không có file dự đoán v2 của {PROPOSED}")
            continue

        data = {}
        for b in bearings:
            bd = ddir / b
            if not bd.is_dir():
                continue
            hf = pd.read_csv(bd / "hour_features.csv")
            lb = pd.read_csv(bd / "labels_by_hour.csv")[["hour_id", "RUL_capped", "life_hours"]]
            data[b] = hf.merge(lb, on="hour_id").sort_values("hour_id").reset_index(drop=True)

        methods = [("Naive mean", None, "mean"),
                   ("RMS-only, Ridge", ["RMS_x", "RMS_y"], "ridge"),
                   ("Handcrafted, Ridge", HC, "ridge"),
                   ("Handcrafted, RF", HC, "rf")]
        if HAS_XGB:
            methods.append(("Handcrafted, XGBoost", HC, "xgb"))

        for mname, cols, kind in methods:
            per = {}
            for held in data:
                te = data[held]
                tr = pd.concat([data[b] for b in data if b != held], ignore_index=True)
                ytr, yte = tr.RUL_capped.to_numpy(float), te.RUL_capped.to_numpy(float)
                if cols is None:
                    pred = np.full(len(te), float(ytr.mean()))
                else:
                    Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
                    if kind == "ridge":
                        sc = StandardScaler().fit(Xtr)
                        pred = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
                    elif kind == "rf":
                        pred = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                     random_state=42).fit(Xtr, ytr).predict(Xte)
                    else:
                        pred = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                                random_state=42, verbosity=0
                                                ).fit(Xtr, ytr).predict(Xte)
                per[held] = float(np.mean(np.abs(np.clip(pred, 0, 1) - yte)))

            common = [b for b in per if b in deep]
            a = np.array([per[b] for b in common]); p = np.array([deep[b] for b in common])
            try:
                _, pv = wilcoxon(a, p)
            except Exception:
                pv = np.nan
            rows.append({"dataset": ds_name, "predictor": mname,
                         "mae_norm_mean": round(float(a.mean()), 4),
                         "mae_norm_std": round(float(a.std(ddof=1)), 4), "n": len(common),
                         "proposed_wins": f"{int((p < a).sum())}/{len(common)}",
                         "p_vs_proposed": round(float(pv), 4) if pv == pv else np.nan})

        d = np.array([deep[b] for b in deep])
        rows.append({"dataset": ds_name, "predictor": f"VTOI-conditioned ({PROPOSED}, deep)",
                     "mae_norm_mean": round(float(d.mean()), 4),
                     "mae_norm_std": round(float(d.std(ddof=1)), 4), "n": len(d),
                     "proposed_wins": "-", "p_vs_proposed": np.nan})

    out = pd.DataFrame(rows)
    if len(out):
        ps = out.p_vs_proposed.to_numpy(float)
        mask = ~np.isnan(ps)
        adj = np.full(len(ps), np.nan)
        if mask.sum():
            adj[mask] = holm(ps[mask])
        out["holm_p"] = np.round(adj, 4)
    out.to_csv(TAB / "classical.csv", index=False)
    print("\n=== E. CLASSICAL BASELINES (đúng cấu hình đề xuất, mức snapshot, normalized) ===")
    print(out.to_string(index=False))
    return out


# ============================================================ F/G
def f_seed_variability(df):
    rows = []
    for (ds, cfg), g in df.groupby(["dataset", "config"]):
        per_seed = g.groupby("seed")[PRIMARY].mean()
        rows.append({"dataset": ds, "config": cfg, "n_seeds": len(per_seed),
                     "mean_over_seeds": round(float(per_seed.mean()), 4),
                     "std_over_seeds": round(float(per_seed.std(ddof=1)), 4) if len(per_seed) > 1 else 0.0,
                     "min_seed": round(float(per_seed.min()), 4),
                     "max_seed": round(float(per_seed.max()), 4)})
    out = pd.DataFrame(rows).sort_values(["dataset", "mean_over_seeds"])
    out.to_csv(TAB / "seed_variability.csv", index=False)
    print("\n=== F. BIẾN THIÊN QUA SEED (R2 #9, #27) ===")
    print(out.to_string(index=False))
    return out


def g_per_bearing(df):
    cols = ["dataset", "config", "seed", "holdout", PRIMARY, SECONDARY,
            "mtoi_spearman_signed", "vtoi_a", "vtoi_b", "vtoi_sat_frac_test"]
    out = df[[c for c in cols if c in df.columns]].sort_values(["dataset", "config", "holdout", "seed"])
    out.to_csv(TAB / "per_bearing_all.csv", index=False)
    print(f"\n=== G. PHỤ LỤC per-bearing: {len(out)} dòng -> per_bearing_all.csv ===")
    return out


def main():
    df = load_perfold()
    a_main_results(df); b_wilcoxon(df); c_factorial(df)
    d_controls(df); e_classical(df); f_seed_variability(df); g_per_bearing(df)
    print(f"\n[XONG] Mọi bảng đã ghi vào {TAB}")


if __name__ == "__main__":
    main()
