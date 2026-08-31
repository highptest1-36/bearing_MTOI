# -*- coding: utf-8 -*-
"""Xuất quỹ đạo chỉ số của bearing held-out (VTOI vs các HI học sâu) để vẽ Figure."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import vtoi as V
from src.lobo import make_folds
from src.lobo_v2 import fit_fold_vtoi
from src.utils.paths import proc_dir_for, TABLES_DIR
from scripts.q1_v2_gapfill import _fit_deep_hi, _fit_ssl_hi, BASE_FRAC

PICK = {"pronostia": ["Full_Test_Set_Bearing1_3", "Learning_set_Bearing1_1"],
        "xjtu_sy":   ["35Hz12kN_Bearing1_3", "37.5Hz11kN_Bearing2_3"]}

rows = []
for ds, names in PICK.items():
    folds, _ = make_folds(ds); proc = proc_dir_for(ds)
    for f in folds:
        ho = f["holdout"]
        if ho not in names: continue
        cond, _, _, _ = fit_fold_vtoi(ds, ho, f["val"], f["train"], seed=42)
        hf = pd.read_csv(proc / ho / "hour_features.csv").sort_values("hour_id")
        Xte = hf[V.VIB_FEATURES].to_numpy(float)
        Xtr = np.concatenate([(lambda a: a[:max(int(BASE_FRAC*len(a)), 5)])(
            pd.read_csv(proc / n / "hour_features.csv")[V.VIB_FEATURES].to_numpy(float))
            for n in f["train"]], axis=0)
        cand = {"VTOI": cond[ho].VTOI.to_numpy(float),
                "RMS": (hf.RMS_x + hf.RMS_y).to_numpy(float),
                "Autoencoder HI": _fit_deep_hi(Xtr, Xte, "ae"),
                "Self-supervised HI": _fit_ssl_hi(Xtr, Xte),
                "Variational HI": _fit_deep_hi(Xtr, Xte, "vae")}
        H = len(hf); t = np.arange(H) / max(H - 1, 1)
        for k, v in cand.items():
            v = np.asarray(v, float)
            v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)   # min-max để so sánh
            for i in range(H):
                rows.append({"dataset": ds, "bearing": ho, "indicator": k,
                             "life_frac": round(float(t[i]), 5), "value": round(float(v[i]), 5)})
        print(f"  [{ds}] {ho}: {H} snapshot")
out = TABLES_DIR / "v2" / "hi_trajectories.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("Ghi", out, len(rows), "dòng")
