# -*- coding: utf-8 -*-
"""
q1_classical_baselines.py — REVIEWER Q1: baseline CỔ ĐIỂN / NGÂY THƠ trên feature handcrafted,
dưới CÙNG giao thức leave-one-bearing-out, RUL MAE quy về GIỜ y hệt pipeline deep.

Mục đích (bảo vệ novelty): chứng minh "deep index-guided head" THẮNG cách dùng tầm thường
cùng feature (Mahalanobis-only, RMS-only, MTOI-scalar-only, Ridge/RF/XGBoost handcrafted, naive-mean)
=> gain KHÔNG chỉ là feature-engineering.

Khớp pipeline (CÔNG BẰNG TUYỆT ĐỐI):
  - Target = RUL_capped (piecewise, cap_onset_frac=0.4) — đúng nhãn deep model dùng.
  - Quy giờ: mae_hours = mae_norm * (life_hours * 0.6).   [đã kiểm chứng = tỉ lệ trong perfold]
  - CÙNG TẬP BEARING: chỉ dùng các bearing có file dự đoán LOBO của đề xuất
    (results/predictions/lobo_<ds>_<bearing>_proposed.csv) -> ĐÚNG 17 PRONOSTIA + 15 XJTU,
    tránh leakage do thư mục processed chứa cả Learning/Test/Full trùng bearing vật lý.
  - CÙNG GRANULARITY: deep model lấy MAE SNAPSHOT-level từ file dự đoán
    (mean|rul_pred-rul_true|*rul_scale_hours), classical cũng snapshot-level -> so apples-to-apples.
  - Wilcoxon ghép cặp theo bearing giữa classical và deep-proposed (cùng snapshot-level).

CPU thuần, KHÔNG GPU, KHÔNG train lại mạng. Chạy: python scripts/q1_classical_baselines.py
Sinh: results/tables/q1_classical_baselines.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT.parent / "data" / "processed"
TAB = ROOT / "results" / "tables"
PRED = ROOT / "results" / "predictions"
CAP_ONSET = 0.4                      # khớp src/labels.py / datasets.py (RUL_capped)
HC = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y", "SE_x", "SE_y", "ESE_x", "ESE_y"]


def lobo_bearings(ds_key):
    """Tập bearing LOBO = các holdout có file dự đoán proposed (đúng tập deep model dùng)."""
    out = {}
    for f in sorted(PRED.glob(f"lobo_{ds_key}_*_proposed.csv")):
        name = f.stem.replace(f"lobo_{ds_key}_", "").replace("_proposed", "")
        out[name] = f
    return out


def deep_proposed_hours(pred_csv):
    """MAE SNAPSHOT-level (giờ) của deep proposed từ file dự đoán."""
    p = pd.read_csv(pred_csv)
    err = np.abs(p["rul_pred"].to_numpy(float) - p["rul_true"].to_numpy(float))
    return float(np.mean(err * p["rul_scale_hours"].to_numpy(float)))


def load_bearing(bdir):
    """Trả DataFrame per-snapshot: 10 handcrafted + E_norm/C_norm + MTOI_learnable + y(RUL_capped) + life_hours."""
    hf = pd.read_csv(bdir / "hour_features.csv")
    lab = pd.read_csv(bdir / "labels_by_hour.csv")[["hour_id", "RUL_capped", "life_hours"]]
    mt = pd.read_csv(bdir / "mtoi_by_hour.csv")[["hour_id", "E_norm", "C_norm", "MTOI_learnable"]]
    df = hf.merge(lab, on="hour_id").merge(mt, on="hour_id").sort_values("hour_id").reset_index(drop=True)
    return df


# (tên method, danh sách cột feature hoặc None nếu naive, loại model)
METHODS = [
    ("Naive mean predictor", None, "mean"),
    ("RMS-only (Ridge)", ["RMS_x", "RMS_y"], "ridge"),
    ("Mahalanobis-only (Ridge)", ["E_norm"], "ridge"),
    ("MTOI scalar-only (Ridge)", ["MTOI_learnable"], "ridge"),
    ("MTOI components E+C (Ridge)", ["E_norm", "C_norm"], "ridge"),
    ("Handcrafted (Ridge)", HC, "ridge"),
    ("Handcrafted (Random Forest)", HC, "rf"),
    ("Handcrafted (XGBoost)", HC, "xgb"),
]


def fit_predict(kind, cols, Xtr, ytr, Xte):
    if kind == "mean":
        return np.full(len(Xte), float(ytr.mean()))
    if kind == "ridge":
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    if kind == "rf":
        m = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42).fit(Xtr, ytr)
        return m.predict(Xte)
    if kind == "xgb":
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             random_state=42, verbosity=0).fit(Xtr, ytr)
        return m.predict(Xte)
    raise ValueError(kind)


def run_dataset(ds_name, ds_dir, ds_key):
    universe = lobo_bearings(ds_key)                       # {bearing_name: pred_csv} — ĐÚNG tập LOBO
    data = {name: load_bearing(ds_dir / name) for name in universe if (ds_dir / name).is_dir()}
    prop = {name: deep_proposed_hours(universe[name]) for name in data}   # deep snapshot-level (giờ)

    rows = []
    per_bearing_hours = {m[0]: {} for m in METHODS}
    for mname, cols, kind in METHODS:
        for held in data:
            te = data[held]
            tr = pd.concat([data[b] for b in data if b != held], ignore_index=True)
            ytr, yte = tr["RUL_capped"].to_numpy(float), te["RUL_capped"].to_numpy(float)
            if cols is None:
                Xtr = np.zeros((len(tr), 1)); Xte = np.zeros((len(te), 1))
            else:
                Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
            pred = np.clip(fit_predict(kind, cols, Xtr, ytr, Xte), 0.0, 1.0)
            mae_norm = float(np.mean(np.abs(pred - yte)))
            scale = float(te["life_hours"].iloc[0]) * (1.0 - CAP_ONSET)  # = life_hours * 0.6
            per_bearing_hours[mname][held] = mae_norm * scale

        hrs = np.array([per_bearing_hours[mname][b] for b in data])
        a = np.array([per_bearing_hours[mname][b] for b in data])
        p = np.array([prop[b] for b in data])
        try:
            _, pval = wilcoxon(a, p)
        except Exception:
            pval = np.nan
        won = int((p < a).sum())   # số bearing đề xuất TỐT HƠN (thấp hơn) classical
        rows.append({"dataset": ds_name, "method": mname,
                     "mae_h_mean": round(float(hrs.mean()), 3),
                     "mae_h_std": round(float(hrs.std(ddof=1)), 3),
                     "n_bearings": len(hrs),
                     "proposed_wins": f"{won}/{len(hrs)}",
                     "wilcoxon_p_vs_proposed": round(float(pval), 4) if pval == pval else np.nan})
    # dòng tham chiếu: deep proposed snapshot-level (cùng cách tính)
    dh = np.array([prop[b] for b in data])
    rows.append({"dataset": ds_name, "method": "MTOI vib-static (proposed, deep)",
                 "mae_h_mean": round(float(dh.mean()), 3),
                 "mae_h_std": round(float(dh.std(ddof=1)), 3),
                 "n_bearings": len(dh), "proposed_wins": "-",
                 "wilcoxon_p_vs_proposed": np.nan})
    return rows


def main():
    all_rows = []
    for ds, sub, key in [("PRONOSTIA", PROC / "pronostia", "pronostia"),
                         ("XJTU-SY", PROC / "xjtu_sy", "xjtu_sy")]:
        print(f"=== {ds} ===")
        rows = run_dataset(ds, sub, key)
        all_rows += rows
        for r in rows:
            tag = ""
            if r["wilcoxon_p_vs_proposed"] == r["wilcoxon_p_vs_proposed"]:
                tag = f"  (Wilcoxon vs proposed p={r['wilcoxon_p_vs_proposed']})"
            print(f"  {r['method']:38s} MAE={r['mae_h_mean']:.3f}±{r['mae_h_std']:.3f} h{tag}")
    out = pd.DataFrame(all_rows)
    out.to_csv(TAB / "q1_classical_baselines.csv", index=False)
    print(f"\n[bảng] q1_classical_baselines.csv ({len(out)} dòng)")


if __name__ == "__main__":
    main()
