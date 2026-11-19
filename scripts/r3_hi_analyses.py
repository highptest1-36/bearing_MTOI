# -*- coding: utf-8 -*-
"""
r3_hi_analyses.py - Round-3 revision (Access-2026-44252). Index-level analyses, no retraining.

  --part quality   R4(e) raw and smoothed criteria for every indicator, plus the state-change
                   component alone (R3.5); mechanism diagnostics for the monotonicity gain.
  --part sens      R4(g) baseline fraction 0.1/0.2/0.3 with the diagonal-fallback count, and
                   R3.6 percentile bounds of the scaling step (0/100, 1/99, 5/95).
  --part robust    R3.4 additive noise (SNR 20/10/5 dB), constant gain error (installation),
                   linear sensitivity drift; features recomputed from the stored raw windows.
  --part ew        R4(f) precision-recall curves per detector family, precision at matched
                   recall, and the per-snapshot definition of precision and recall.

Outputs go to results/tables/v3/. Every fit uses the training bearings of the fold only.
Smoothing (R4 e) is the centred rolling median over five per cent of the record used in Fig. 3.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import vtoi as V                                              # noqa: E402
from src.lobo import make_folds                                        # noqa: E402
from src.utils.paths import proc_dir_for                               # noqa: E402

OUT = ROOT / "results" / "tables" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("/content/r3_feature_cache")
DATASETS = (("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy"))
BASE_FRAC = 0.2


# ------------------------------------------------------------------ criteria
def mono(x):
    d = np.diff(np.asarray(x, float))
    return float(abs((d > 0).sum() - (d < 0).sum()) / len(d)) if len(d) else 0.0


def trend(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x, np.arange(len(x)))[0, 1]))


def progn(series):
    xf = np.array([np.asarray(s, float)[-1] for s in series])
    rng = np.array([abs(np.asarray(s, float)[0] - np.asarray(s, float)[-1]) for s in series])
    return float(np.exp(-np.std(xf) / (np.mean(rng) + 1e-12)))


def smooth(x, frac=0.05):
    """Centred rolling median over five per cent of the record (the Fig. 3 smoothing)."""
    x = pd.Series(np.asarray(x, float))
    w = max(3, int(round(frac * len(x))))
    return x.rolling(w, center=True, min_periods=1).median().to_numpy()


def summarise(series_by_ind, ds_name, extra=None):
    rows = []
    for k, ss in series_by_ind.items():
        if not ss:
            continue
        sm = [smooth(s) for s in ss]
        r = {"dataset": ds_name, "indicator": k, "n": len(ss),
             "mono": np.median([mono(s) for s in ss]),
             "trend": np.median([trend(s) for s in ss]),
             "trend_min": np.min([trend(s) for s in ss]),
             "progn": progn(ss),
             "mono_smooth": np.median([mono(s) for s in sm]),
             "trend_smooth": np.median([trend(s) for s in sm]),
             "progn_smooth": progn(sm)}
        if extra:
            r.update(extra)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ index on given features
def components(hf, beta=BASE_FRAC):
    return V.compute_raw_components(hf, baseline_frac=beta)


def fit_apply(comp_by_name, train, holdout, lo=1.0, hi=99.0, objective="mono+corr"):
    trc = [comp_by_name[n] for n in train]
    degs = [np.arange(len(c["E_raw"])) / max(len(c["E_raw"]) - 1, 1) for c in trc]
    p = V.fit_vtoi_params(trc, degs, lo_pct=lo, hi_pct=hi, objective=objective)
    return p, V.apply_vtoi(comp_by_name[holdout], p)


def load_hf(key, name):
    return pd.read_csv(proc_dir_for(key) / name / "hour_features.csv").sort_values("hour_id") \
             .reset_index(drop=True)


# ================================================================== part: quality (R4 e, R3.5)
def part_quality():
    sys.path.insert(0, str(ROOT / "scripts"))
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from q1_v2_gapfill import _fit_deep_hi, _fit_ssl_hi
    from q1_v2_tier3 import fit_coble_parameter

    rows, mech = [], []
    for ds_name, key in DATASETS:
        folds, _ = make_folds(key)
        names = sorted({n for f in folds for n in f["train"] + f["val"] + [f["holdout"]]})
        hfs = {n: load_hf(key, n) for n in names}
        comps = {n: components(hfs[n]) for n in names}
        S = {k: [] for k in ["RMS", "Kurtosis", "Principal-component HI", "Autoencoder HI",
                             "Self-supervised HI", "Variational HI", "Mahalanobis HI",
                             "State-change component", "Coble optimal parameter",
                             "VTOI, label-free fit", "VTOI (proposed)"]}
        for f in folds:
            ho = f["holdout"]
            p, ap = fit_apply(comps, f["train"], ho)
            _, apm = fit_apply(comps, f["train"], ho, objective="mono")
            hf = hfs[ho]
            Xtr_list = [hfs[n][V.VIB_FEATURES].to_numpy(float) for n in f["train"]]
            sc = StandardScaler().fit(np.concatenate(Xtr_list))
            Xho = sc.transform(hf[V.VIB_FEATURES].to_numpy(float))
            deg = np.arange(len(hf)) / max(len(hf) - 1, 1)
            pca = PCA(n_components=1).fit(sc.transform(np.concatenate(Xtr_list)))
            pc1 = pca.transform(Xho)[:, 0]
            if spearmanr(pc1, deg)[0] < 0:
                pc1 = -pc1
            healthy = np.concatenate([a[:max(int(BASE_FRAC * len(a)), 5)] for a in Xtr_list])
            Xte = hf[V.VIB_FEATURES].to_numpy(float)
            w_c = fit_coble_parameter([sc.transform(X) for X in Xtr_list], seed=42)
            S["RMS"].append((hf.RMS_x + hf.RMS_y).to_numpy(float))
            S["Kurtosis"].append((hf.Kurt_x + hf.Kurt_y).to_numpy(float))
            S["Principal-component HI"].append(pc1)
            S["Autoencoder HI"].append(_fit_deep_hi(healthy, Xte, "ae"))
            S["Self-supervised HI"].append(_fit_ssl_hi(healthy, Xte))
            S["Variational HI"].append(_fit_deep_hi(healthy, Xte, "vae"))
            S["Mahalanobis HI"].append(ap["E_norm"])
            S["State-change component"].append(ap["C_norm"])
            S["Coble optimal parameter"].append(Xho @ w_c)
            S["VTOI, label-free fit"].append(apm["VTOI"])
            S["VTOI (proposed)"].append(ap["VTOI"])
            # mechanism diagnostics (R4 e): ties, clipping, sign counts
            for nm, v in (("Abnormality", ap["E_norm"]), ("State-change", ap["C_norm"]),
                          ("VTOI", ap["VTOI"])):
                d = np.diff(v)
                mech.append({"dataset": ds_name, "bearing": ho, "series": nm, "a": p["a"],
                             "H": len(v), "frac_tie": float(np.mean(d == 0)),
                             "frac_at_0": float(np.mean(v <= 0)),
                             "frac_at_1": float(np.mean(v >= 1)),
                             "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                             "mono": mono(v)})
            print(f"  [{ds_name}] {ho} done", flush=True)
        rows += summarise(S, ds_name)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(OUT / "hi_quality_smoothed.csv", index=False)
    pd.DataFrame(mech).to_csv(OUT / "mono_ties_per_bearing.csv", index=False)
    m = (pd.DataFrame(mech).groupby(["dataset", "series"])
           [["frac_tie", "frac_at_0", "frac_at_1", "n_pos", "n_neg", "mono"]].median().round(4))
    m.to_csv(OUT / "mono_ties.csv")
    print(df.to_string(index=False)); print(m.to_string())


# ================================================================== part: sens (R4 g, R3.6)
def part_sens():
    rows = []
    settings = [("beta", 0.1, (1, 99)), ("beta", 0.2, (1, 99)), ("beta", 0.3, (1, 99)),
                ("bounds", 0.2, (0, 100)), ("bounds", 0.2, (5, 95))]
    for ds_name, key in DATASETS:
        folds, _ = make_folds(key)
        names = sorted({n for f in folds for n in f["train"] + f["val"] + [f["holdout"]]})
        hfs = {n: load_hf(key, n) for n in names}
        for kind, beta, (lo, hi) in settings:
            comps = {n: components(hfs[n], beta=beta) for n in names}
            n_diag = sum(c["info"]["estimator"] == "diagonal" for c in comps.values())
            n_enlarged = sum(c["info"]["baseline_frac_effective"] > beta + 0.005
                             for c in comps.values())
            S = {"VTOI": [], "Mahalanobis HI": []}; a_s = []
            for f in folds:
                p, ap = fit_apply(comps, f["train"], f["holdout"], lo=lo, hi=hi)
                S["VTOI"].append(ap["VTOI"]); S["Mahalanobis HI"].append(ap["E_norm"])
                a_s.append(p["a"])
            extra = {"setting": kind, "beta": beta, "bounds": f"{lo}/{hi}",
                     "a_median": float(np.median(a_s)), "n_diagonal": n_diag,
                     "n_baseline_enlarged": n_enlarged, "n_bearings": len(names)}
            rows += summarise(S, ds_name, extra)
            print(f"  [{ds_name}] {kind} beta={beta} bounds={lo}/{hi} done", flush=True)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    print(df.to_string(index=False))


# ================================================================== part: robust (R3.4)
def _feat_from_windows(X, hid):
    from src.features import window_feature_vector
    F = np.array([window_feature_vector(w) for w in X])
    df = pd.DataFrame(F, columns=V.VIB_FEATURES); df.insert(0, "hour_id", hid)
    return df.groupby("hour_id")[V.VIB_FEATURES].median().reset_index()


def _perturb(X, hid, kind, rng):
    X = np.array(X, dtype=np.float64)
    H = hid.max() + 1
    if kind == "clean":
        return X
    if kind.startswith("snr"):
        snr = float(kind[3:])
        nb = max(int(BASE_FRAC * H), 1)
        base = X[hid < nb]
        rms = np.sqrt((base ** 2).mean(axis=(0, 2)))            # per channel, healthy level
        sig = rms / (10 ** (snr / 20))
        return X + rng.standard_normal(X.shape) * sig[None, :, None]
    if kind == "gain":                                          # installation / sensitivity
        return X * np.array([0.7, 1.3])[None, :, None]
    if kind == "drift":                                         # sensitivity drift +20 % over life
        g = 1.0 + 0.2 * (hid / max(H - 1, 1))
        return X * g[:, None, None]
    raise ValueError(kind)


def robust_features(key, name, kind):
    """Features of one bearing under one perturbation, computed block by block (float32) so the
    longest XJTU-SY record (about 38,000 windows) fits in memory."""
    f = CACHE / key / kind / f"{name}.csv"
    if f.exists():
        return pd.read_csv(f)
    from src.features import window_feature_vector
    import zlib
    d = proc_dir_for(key) / name
    X = np.load(d / "X_vib_windows.npy", mmap_mode="r")
    hid = np.load(d / "window_hour_ids.npy")
    H = int(hid.max()) + 1
    rng = np.random.default_rng(zlib.crc32(f"{name}|{kind}".encode()))
    sig = None
    if kind.startswith("snr"):
        nb = max(int(BASE_FRAC * H), 1)
        idx = np.where(hid < nb)[0]
        acc, cnt = np.zeros(X.shape[1]), 0
        for s0 in range(0, len(idx), 2000):
            xb = np.asarray(X[idx[s0:s0 + 2000]], dtype=np.float64)
            acc += (xb ** 2).sum(axis=(0, 2)); cnt += xb.shape[0] * xb.shape[2]
        rms = np.sqrt(acc / cnt)
        sig = rms / (10 ** (float(kind[3:]) / 20))
    feats = []
    for s0 in range(0, len(X), 2000):
        x = np.asarray(X[s0:s0 + 2000], dtype=np.float32)
        h = hid[s0:s0 + 2000]
        if kind.startswith("snr"):
            x = x + (rng.standard_normal(x.shape).astype(np.float32) * sig[None, :, None].astype(np.float32))
        elif kind == "gain":
            x = x * np.array([0.7, 1.3], dtype=np.float32)[None, :, None]
        elif kind == "drift":
            x = x * (1.0 + 0.2 * (h / max(H - 1, 1))).astype(np.float32)[:, None, None]
        feats.append(np.array([window_feature_vector(w) for w in x]))
    F = np.concatenate(feats)
    df = pd.DataFrame(F, columns=V.VIB_FEATURES); df.insert(0, "hour_id", hid)
    df = df.groupby("hour_id")[V.VIB_FEATURES].median().reset_index()
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def _first_persistent(sig, thr, q=3):
    run = 0
    for i, v in enumerate(sig):
        run = run + 1 if v > thr else 0
        if run >= q:
            return i - q + 1
    return -1


def _pr(first, onset, n):
    if first < 0:
        return 0.0, 0.0
    pred = np.zeros(n, bool); pred[first:] = True
    true = np.zeros(n, bool); true[onset:] = True
    tp = (pred & true).sum()
    return tp / max(pred.sum(), 1), tp / max(true.sum(), 1)


def part_robust(kinds=("clean", "gain", "drift", "snr20", "snr10", "snr5")):
    rows = []
    for ds_name, key in DATASETS:
        folds, _ = make_folds(key)
        names = sorted({n for f in folds for n in f["train"] + f["val"] + [f["holdout"]]})
        for kind in kinds:
            hfs = {n: robust_features(key, n, kind) for n in names}
            comps = {n: components(hfs[n]) for n in names}
            S = {"VTOI (proposed)": [], "Mahalanobis HI": [], "RMS": [], "Kurtosis": []}
            prec, rec = [], []
            for f in folds:
                ho = f["holdout"]
                p, ap = fit_apply(comps, f["train"], ho)
                S["VTOI (proposed)"].append(ap["VTOI"]); S["Mahalanobis HI"].append(ap["E_norm"])
                S["RMS"].append((hfs[ho].RMS_x + hfs[ho].RMS_y).to_numpy(float))
                S["Kurtosis"].append((hfs[ho].Kurt_x + hfs[ho].Kurt_y).to_numpy(float))
                # train-selected threshold (same max-F1 rule as the paper), then held-out P/R
                best, bf = 0.6, -1
                for tau in np.arange(0.3, 0.95, 0.05):
                    f1 = []
                    for n in f["train"]:
                        v = V.apply_vtoi(comps[n], p)["VTOI"]; m = len(v)
                        pp, rr = _pr(_first_persistent(v, tau), max(1, int(0.6 * m)), m)
                        f1.append(0 if pp + rr == 0 else 2 * pp * rr / (pp + rr))
                    if np.mean(f1) > bf:
                        bf, best = np.mean(f1), tau
                v = ap["VTOI"]; m = len(v)
                pp, rr = _pr(_first_persistent(v, best), max(1, int(0.6 * m)), m)
                prec.append(pp); rec.append(rr)
            rows += summarise(S, ds_name, {"perturbation": kind})
            rows.append({"dataset": ds_name, "indicator": "VTOI early warning",
                         "perturbation": kind, "ew_precision": float(np.mean(prec)),
                         "ew_recall": float(np.mean(rec))})
            print(f"  [{ds_name}] {kind} done", flush=True)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(OUT / "robustness.csv", index=False)
    print(df.to_string(index=False))


# ================================================================== part: ew (R4 f)
def _sigma(sig, k, frac=0.2):
    n = len(sig); nh = max(5, int(frac * n))
    return float(sig[:nh].mean() + k * sig[:nh].std() + 1e-9)


def part_ew(onset_frac=0.6):
    """Threshold sweeps per detector family. Precision and recall are per-snapshot quantities
    computed within each held-out bearing (an alarm, once raised, persists to the end of the
    record; snapshots at or after the onset are positives) and then averaged over bearings, a
    bearing on which a detector never fires contributing zero precision and zero recall."""
    from src.lobo_v2 import fit_fold_vtoi
    curves = []
    cached = OUT / "pr_curves_per_bearing.csv"
    for ds_name, key in ([] if cached.exists() else DATASETS):
        for f in make_folds(key)[0]:
            ho = f["holdout"]
            cond, _, _, _ = fit_fold_vtoi(key, ho, f["val"], f["train"], seed=42)
            hf = load_hf(key, ho)
            sigs = {"VTOI": (cond[ho].VTOI.to_numpy(float), np.arange(0.05, 1.0, 0.025), "tau"),
                    "RMS k-sigma": ((hf.RMS_x + hf.RMS_y).to_numpy(float),
                                    np.arange(0.25, 40.01, 0.25), "k"),
                    "Kurtosis k-sigma": ((hf.Kurt_x + hf.Kurt_y).to_numpy(float),
                                         np.arange(0.25, 40.01, 0.25), "k")}
            for fam, (s, grid, kind) in sigs.items():
                n = len(s); onset = max(1, int(onset_frac * n))
                for g in grid:
                    thr = g if kind == "tau" else _sigma(s, g)
                    p, r = _pr(_first_persistent(s, thr), onset, n)
                    curves.append({"dataset": ds_name, "bearing": ho, "family": fam,
                                   "param": round(float(g), 4), "precision": p, "recall": r})
    c = pd.read_csv(cached) if cached.exists() else pd.DataFrame(curves)
    c.to_csv(cached, index=False)
    agg = (c.groupby(["dataset", "family", "param"])[["precision", "recall"]].mean()
             .reset_index())
    agg.to_csv(OUT / "pr_curves.csv", index=False)
    rows = []
    for (ds, fam), g in agg.groupby(["dataset", "family"]):
        g = g.sort_values("recall")
        # upper envelope: best precision achievable at recall >= target
        for target in (0.4, 0.6, 0.8):
            ok = g[g.recall >= target]
            rows.append({"dataset": ds, "family": fam, "recall_target": target,
                         "precision": float(ok.precision.max()) if len(ok) else np.nan})
        # interpolated average precision: envelope env(r) = best precision at recall >= r,
        # integrated over a fine grid on [0, 1] (zero where no operating point reaches r)
        R, P = g.recall.to_numpy(), g.precision.to_numpy()
        grid = np.linspace(0, 1, 1001)
        env = np.array([P[R >= r].max() if (R >= r).any() else 0.0 for r in grid])
        rows.append({"dataset": ds, "family": fam, "recall_target": "AUPRC",
                     "precision": float(np.trapezoid(env, grid))})
    out = pd.DataFrame(rows)
    out["precision"] = out["precision"].round(3)
    out.to_csv(OUT / "ew_matched_recall.csv", index=False)
    print(out.pivot_table(index=["dataset", "family"], columns="recall_target",
                          values="precision", aggfunc="first").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["quality", "sens", "robust", "ew"])
    a = ap.parse_args()
    {"quality": part_quality, "sens": part_sens, "robust": part_robust, "ew": part_ew}[a.part]()
