# -*- coding: utf-8 -*-
"""
r3_rul_stats.py - Round-3 revision (Access-2026-44252), Reviewer 4 items (a) and (c).

No retraining. Reads the saved per-snapshot predictions of every configuration and writes
results/tables/v3/:

  constant_baselines.csv   (a) MAE and RMSE of constant predictors (0.5 and the pooled mean of the
                           training-bearing targets of each fold) next to the trained models.
  prognostic_metrics.csv   (a) alpha-lambda accuracy and prognostic horizon (Saxena et al., 2008).
  equivalence.csv          (c) paired differences against the proposed configuration with 90% and
                           95% confidence intervals and two one-sided tests (Lakens, 2017).

EQUIVALENCE MARGIN - fixed before any equivalence test was computed:
  DELTA = 0.01 in normalised-RUL units, i.e. one per cent of the degradation phase. It is about
  twice the seed-to-seed standard deviation of the proposed configuration on PRONOSTIA (0.0042)
  and therefore the smallest difference three seeds could plausibly resolve.

Error metric (the quantity the paper reports, written out for R4(a)):
  target   y_h = min(1, (H_fail - h) / ((1 - c) H_fail)), c = 0.4   (capped, normalised RUL)
  bearing  MAE_b = mean_h |yhat_h - y_h|  over all snapshots h of held-out bearing b,
           where yhat_h is the mean of the window-level predictions in snapshot h
  report   MAE_b averaged over the three seeds, then summarised over bearings.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lobo import make_folds                      # noqa: E402
from src.utils.paths import proc_dir_for             # noqa: E402

PRED = ROOT / "results" / "predictions" / "v2"
OUT = ROOT / "results" / "tables" / "v3"
OUT.mkdir(parents=True, exist_ok=True)

DELTA = 0.01
SEEDS = (42, 43, 44)
DATASETS = (("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy"))
PROPOSED = "vtoi_static"
NAMES = {
    "vtoi_static": "VTOI-conditioned (proposed)", "vtoi_traj": "VTOI-conditioned, trajectory",
    "transformer_vib": "Transformer", "tcn_vib": "TCN", "tcn_transformer_vib": "TCN-Transformer",
    "cnn_bilstm_attn_vib": "CNN-BiLSTM-Attention", "abl_no_idxhead": "No conditioning channel",
    "abl_cond_noaux": "Conditioning only", "abl_E_only": "Abnormality component only",
    "abl_no_smooth": "No smoothness penalty", "abl_no_mono": "No monotonicity penalty",
    "ctl_scalar": "Fused scalar only", "ctl_EC": "Raw components only",
    "ctl_hc10": "Ten handcrafted features", "ctl_random": "Randomly weighted index",
    "ctl_shuffled": "Time-shuffled index", "ctl_elapsed": "Elapsed time only",
    "ctl_lifefrac": "Oracle lifetime fraction", "ctl_nodeg": "Label-free index",
}
CONTROL_ARMS = ["ctl_lifefrac", "ctl_elapsed", "ctl_hc10", "ctl_scalar", "ctl_random",
                "ctl_nodeg", "ctl_EC", "ctl_shuffled", "abl_no_idxhead", "transformer_vib"]


def capped_target(H, c=0.4):
    H_fail = max(H - 1, 1)
    h = np.arange(H)
    return np.minimum(1.0, (H_fail - h) / max((1 - c) * H_fail, 1.0))


def load_pred(key, seed, ho, cfg):
    f = PRED / f"lobo_v2_{key}_s{seed}_{ho}_{cfg}.csv"
    return pd.read_csv(f) if f.exists() else None


# ------------------------------------------------------------------ (a) constants + Saxena
def alpha_lambda(y, yhat, alpha=0.2):
    """Fraction of degradation-phase snapshots whose prediction lies in the alpha cone
    [(1-alpha) y, (1+alpha) y] around the true remaining life (Saxena et al., 2008)."""
    m = y < 1.0                                       # degradation phase only
    if m.sum() == 0:
        return np.nan
    yy, pp = y[m], yhat[m]
    return float(np.mean((pp >= (1 - alpha) * yy) & (pp <= (1 + alpha) * yy)))


def prognostic_horizon(y, yhat, alpha=0.1):
    """Fraction of the degradation phase, counted back from failure, over which every
    prediction stays within +-alpha of the true remaining life (Saxena et al., 2008)."""
    m = np.where(y < 1.0)[0]
    if len(m) == 0:
        return np.nan
    ok = np.abs(yhat[m] - y[m]) <= alpha
    run = 0
    for v in ok[::-1]:
        if not v:
            break
        run += 1
    return float(run / len(m))


def part_a():
    rows, prog = [], []
    for ds, key in DATASETS:
        folds, _ = make_folds(key)
        proc = proc_dir_for(key)
        for f in folds:
            ho = f["holdout"]
            tr_targets = []
            for n in f["train"]:
                H = sum(1 for _ in open(proc / n / "hour_features.csv")) - 1
                tr_targets.append(capped_target(H))
            const_mean = float(np.mean(np.concatenate(tr_targets)))
            ref = load_pred(key, 42, ho, PROPOSED)
            y = ref.rul_true.to_numpy(float)
            for cname, c in (("Constant 0.5", 0.5), ("Constant training mean", const_mean)):
                rows.append({"dataset": ds, "bearing": ho, "predictor": cname,
                             "mae": float(np.mean(np.abs(y - c))),
                             "rmse": float(np.sqrt(np.mean((y - c) ** 2))),
                             "const": round(c, 4)})
            for cfg in NAMES:
                maes, rmses, al, ph = [], [], [], []
                for s in SEEDS:
                    p = load_pred(key, s, ho, cfg)
                    if p is None:
                        continue
                    yt, yp = p.rul_true.to_numpy(float), p.rul_pred.to_numpy(float)
                    maes.append(np.mean(np.abs(yp - yt)))
                    rmses.append(np.sqrt(np.mean((yp - yt) ** 2)))
                    al.append(alpha_lambda(yt, yp)); ph.append(prognostic_horizon(yt, yp))
                if not maes:
                    continue
                rows.append({"dataset": ds, "bearing": ho, "predictor": NAMES[cfg],
                             "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
                             "const": np.nan})
                prog.append({"dataset": ds, "bearing": ho, "predictor": NAMES[cfg],
                             "alpha_lambda": float(np.nanmean(al)),
                             "prog_horizon": float(np.nanmean(ph))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "constant_baselines_per_bearing.csv", index=False)
    summ = (df.groupby(["dataset", "predictor"])
              .agg(n=("bearing", "count"), mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                   rmse_mean=("rmse", "mean"), const=("const", "mean"))
              .round(4).reset_index().sort_values(["dataset", "mae_mean"]))
    summ.to_csv(OUT / "constant_baselines.csv", index=False)
    pg = pd.DataFrame(prog)
    pg.to_csv(OUT / "prognostic_metrics_per_bearing.csv", index=False)
    ps = (pg.groupby(["dataset", "predictor"])
            .agg(n=("bearing", "count"), alpha_lambda=("alpha_lambda", "mean"),
                 prog_horizon=("prog_horizon", "mean"))
            .round(3).reset_index().sort_values(["dataset", "alpha_lambda"], ascending=[True, False]))
    ps.to_csv(OUT / "prognostic_metrics.csv", index=False)

    # paired test of each trained model against the training-mean constant
    tests = []
    for ds, _ in DATASETS:
        d = df[df.dataset == ds].pivot(index="bearing", columns="predictor", values="mae")
        for cname in ("Constant 0.5", "Constant training mean"):
            for m in ("VTOI-conditioned (proposed)", "Transformer", "Elapsed time only"):
                diff = d[m] - d[cname]
                p = stats.wilcoxon(d[m], d[cname]).pvalue
                tests.append({"dataset": ds, "model": m, "constant": cname,
                              "mean_diff": round(float(diff.mean()), 4), "p": round(float(p), 4),
                              "model_better": f"{int((diff < 0).sum())}/{len(diff)}"})
    pd.DataFrame(tests).to_csv(OUT / "constant_tests.csv", index=False)
    print(summ.to_string(index=False)); print(ps.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))


# ------------------------------------------------------------------ (c) equivalence
def tost_paired(d, delta):
    """Two one-sided paired t-tests (Lakens, 2017). H0: |mean diff| >= delta."""
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (m + delta) / se                           # H0: mean <= -delta
    t_hi = (m - delta) / se                           # H0: mean >= +delta
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return float(max(p_lo, p_hi))


def tost_wilcoxon(d, delta):
    p_lo = stats.wilcoxon(d + delta, alternative="greater").pvalue
    p_hi = stats.wilcoxon(d - delta, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def boot_ci(d, level=0.95, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def part_c():
    pb = pd.read_csv(ROOT / "results" / "tables" / "v2" / "per_bearing_all.csv")
    sa = pb.groupby(["dataset", "config", "holdout"]).rul_mae.mean().reset_index()
    rows = []
    for ds, key in DATASETS:
        w = sa[sa.dataset == key].pivot(index="holdout", columns="config", values="rul_mae")
        block = []
        for arm in CONTROL_ARMS:
            d = (w[arm] - w[PROPOSED]).to_numpy(float)
            n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
            t90 = stats.t.ppf(0.95, n - 1); t95 = stats.t.ppf(0.975, n - 1)
            b_lo, b_hi = boot_ci(d)
            block.append({"dataset": ds, "arm": NAMES[arm], "n": n,
                          "mean_diff": m, "ci90_lo": m - t90 * se, "ci90_hi": m + t90 * se,
                          "ci95_lo": m - t95 * se, "ci95_hi": m + t95 * se,
                          "boot95_lo": b_lo, "boot95_hi": b_hi,
                          "p_tost_t": tost_paired(d, DELTA),
                          "p_tost_wilcoxon": tost_wilcoxon(d, DELTA),
                          "p_difference": float(stats.wilcoxon(d).pvalue)})
        # Holm family = the nine arms of the control battery; the Transformer comparison
        # (not part of the battery) is a single test and is left uncorrected.
        fam = [r for r in block if r["arm"] != "Transformer"]
        adj = holm([r["p_tost_t"] for r in fam])
        for r, a in zip(fam, adj):
            r["p_tost_holm"] = float(a)
        for r in block:
            r.setdefault("p_tost_holm", r["p_tost_t"])
        for r in block:
            a = r["p_tost_holm"]
            r["equivalent"] = bool(a < 0.05)
            r["delta"] = DELTA
        rows += block
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(OUT / "equivalence.csv", index=False)
    print(df[["dataset", "arm", "mean_diff", "ci90_lo", "ci90_hi", "boot95_lo", "boot95_hi",
              "p_tost_t", "p_tost_holm", "equivalent"]].to_string(index=False))


# ------------------------------------------------------------------ table completion (round 3)
# Cells of Tables 8 and 10 that were left as '-' although the saved predictions allow them to be
# computed. No retraining. Holm families are kept separate and within one dataset each:
#   constant references (2 per dataset), deep baselines (4, results/tables/v2/wilcoxon.csv),
#   classical regressors on the handcrafted features (4), control arms (9, part_c).
VARIANTS = ["vtoi_traj", "abl_no_idxhead", "abl_cond_noaux", "abl_E_only", "abl_no_smooth",
            "abl_no_mono"]
CLASSICAL = ["Handcrafted, XGBoost", "Handcrafted, RF", "Handcrafted, Ridge", "RMS-only, Ridge"]


def part_tables():
    from scipy.stats import wilcoxon
    rows = []
    # (1) constants: spread over bearings, alpha-lambda, Wins and Wilcoxon p against the proposed
    cb = pd.read_csv(OUT / "constant_baselines_per_bearing.csv")
    for ds, key in DATASETS:
        folds, _ = make_folds(key)
        proc = proc_dir_for(key)
        d = cb[cb.dataset == ds].pivot(index="bearing", columns="predictor", values="mae")
        prop = d[NAMES[PROPOSED]]
        fam = []
        for cname in ("Constant 0.5", "Constant training mean"):
            al = []
            for f in folds:
                ho = f["holdout"]
                if cname == "Constant 0.5":
                    c = 0.5
                else:
                    tr = []
                    for n in f["train"]:
                        H = sum(1 for _ in open(proc / n / "hour_features.csv")) - 1
                        tr.append(capped_target(H))
                    c = float(np.mean(np.concatenate(tr)))
                y = load_pred(key, 42, ho, PROPOSED).rul_true.to_numpy(float)
                al.append(alpha_lambda(y, np.full_like(y, c)))
            fam.append({"dataset": ds, "predictor": cname, "family": "constant",
                        "q25": d[cname].quantile(0.25), "q75": d[cname].quantile(0.75),
                        "alpha_lambda": float(np.mean(al)),
                        "wins": f"{int((prop < d[cname]).sum())}/{len(d)}",
                        "p": float(wilcoxon(prop, d[cname]).pvalue)})
        for r, a in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = float(a)
        rows += fam
    # (2) variants of the proposed configuration: Wins against the proposed configuration
    pb = pd.read_csv(ROOT / "results" / "tables" / "v2" / "per_bearing_all.csv")
    sa = pb.groupby(["dataset", "config", "holdout"]).rul_mae.mean().reset_index()
    for ds, key in DATASETS:
        w = sa[sa.dataset == key].pivot(index="holdout", columns="config", values="rul_mae")
        for v in VARIANTS:
            rows.append({"dataset": ds, "predictor": NAMES[v], "config": v, "family": "variant",
                         "wins": f"{int((w[PROPOSED] < w[v]).sum())}/{len(w)}"})
    ext = pd.DataFrame(rows)
    ext.to_csv(OUT / "rul_table_extras.csv", index=False)
    print(ext.to_string(index=False))

    # (3) classical regressors: Holm within each dataset. classical.csv (v2) corrected over both
    # datasets at once (ten tests), contrary to the stated protocol. Its raw p values carry four
    # decimals; the exact two-sided Wilcoxon p (no ties, n <= 50) lies on a discrete lattice, so
    # the exact value is recovered as the unique lattice point that rounds to the stored value.
    def lattice(n):
        N = n * (n + 1) // 2
        cnt = np.zeros(N + 1); cnt[0] = 1
        for k in range(1, n + 1):
            cnt[k:] = cnt[k:] + cnt[:-k].copy()
        cdf = np.cumsum(cnt) / 2 ** n
        return np.unique(np.minimum(1.0, 2 * cdf[: N // 2 + 1]))

    def exact(p, n):
        cand = [v for v in lattice(n) if round(v, 4) == round(p, 4)]
        assert len(cand) == 1, (p, n, cand)
        return float(cand[0])

    c = pd.read_csv(ROOT / "results" / "tables" / "v2" / "classical.csv")
    out = []
    for ds, _ in DATASETS:
        g = c[(c.dataset == ds) & c.predictor.isin(CLASSICAL)].copy()
        p = np.array([exact(v, int(n)) for v, n in zip(g.p_vs_proposed, g.n)])
        for (_, r), pe, a in zip(g.iterrows(), p, holm(p)):
            out.append({"dataset": ds, "predictor": r.predictor, "n": int(r.n),
                        "p_vs_proposed": r.p_vs_proposed, "p_exact": pe,
                        "holm_p_within_dataset": float(a),
                        "holm_p_v2_across_datasets": r.holm_p})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "classical_holm.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    import sys as _s
    if "--tables" in _s.argv:
        part_tables()
    else:
        if "--c-only" not in _s.argv:
            part_a()
        part_c()
