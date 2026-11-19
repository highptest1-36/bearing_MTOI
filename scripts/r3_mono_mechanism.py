# -*- coding: utf-8 -*-
"""
r3_mono_mechanism.py - Round-3 revision, Reviewer 4 item (e): why does adding about twelve per
cent of the state-change magnitude raise sign-based monotonicity on PRONOSTIA?

For every held-out bearing (index fitted on the training bearings of its fold) it records:
  ties          fraction of zero first differences (the clipping-bound hypothesis)
  signed        (#rises - #falls)/(H-1), raw and after the Fig. 3 smoothing
  flips         steps whose sign in the index differs from the sign in the abnormality
                component, split into rise-to-fall and fall-to-rise

Output: results/tables/v3/mono_mechanism_per_bearing.csv and mono_mechanism.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import r3_hi_analyses as H                                   # noqa: E402
from src.lobo import make_folds                              # noqa: E402

OUT = ROOT / "results" / "tables" / "v3"


def signed(x):
    d = np.diff(np.asarray(x, float))
    return float(((d > 0).sum() - (d < 0).sum()) / len(d))


def main():
    rows = []
    for ds, key in H.DATASETS:
        folds, _ = make_folds(key)
        names = sorted({n for f in folds for n in f["train"] + f["val"] + [f["holdout"]]})
        hfs = {n: H.load_hf(key, n) for n in names}
        comps = {n: H.components(hfs[n]) for n in names}
        for f in folds:
            p, ap = H.fit_apply(comps, f["train"], f["holdout"])
            E, C, V = ap["E_norm"], ap["C_norm"], ap["VTOI"]
            dE, dV = np.diff(E), np.diff(V)
            flip = np.sign(dV) != np.sign(dE)
            r = {"dataset": ds, "bearing": f["holdout"], "a": p["a"], "steps": len(dV),
                 "flips": int(flip.sum()), "rise_to_fall": int((flip & (dV < 0)).sum()),
                 "fall_to_rise": int((flip & (dV > 0)).sum())}
            for nm, v in (("E", E), ("C", C), ("V", V)):
                d = np.diff(v)
                r[f"ties_{nm}"] = float(np.mean(d == 0))
                r[f"signed_{nm}"] = signed(v)
                r[f"signed_smooth_{nm}"] = signed(H.smooth(v))
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "mono_mechanism_per_bearing.csv", index=False)
    g = d.groupby("dataset").agg(
        n=("bearing", "count"), weight=("a", "median"),
        ties_E=("ties_E", "median"), ties_V=("ties_V", "median"),
        signed_E=("signed_E", "median"), signed_V=("signed_V", "median"),
        signed_smooth_E=("signed_smooth_E", "median"), signed_smooth_V=("signed_smooth_V", "median"),
        n_fall_excess_V=("signed_V", lambda s: int((s < 0).sum())),
        steps=("steps", "sum"), flips=("flips", "sum"), rise_to_fall=("rise_to_fall", "sum"))
    g["flip_share"] = g.flips / g.steps
    g["rise_to_fall_share"] = g.rise_to_fall / g.flips
    g.round(4).to_csv(OUT / "mono_mechanism.csv")
    print(g.round(4).T.to_string())


if __name__ == "__main__":
    main()
