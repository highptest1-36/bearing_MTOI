# -*- coding: utf-8 -*-
"""
r3_attribution_ig.py - Round-3 revision, Reviewer 4 item (d).

The RUL head of the proposed model reads the concatenation of a 128-dimensional vibration
embedding e and the 3-dimensional conditioning vector c, through a two-layer MLP. Raw gradient
norms of the two groups are not comparable (different dimension and scale). This script reports
scale-aware attributions on the 96 saved checkpoints (32 folds x 3 seeds):

  grad x input   sum_j |(x_j - xbar_j) * dRUL/dx_j| per group, xbar = per-bearing mean input
  IG             integrated gradients (Sundararajan et al., 2017) from the per-bearing mean
                 input, 32 steps; completeness: sum of attributions = f(x) - f(xbar)
  share_c        mean |group attribution of c| / (mean |of c| + mean |of e|)

and three interventions on c that are larger than freezing at the healthy value:
  failure value  c set to its value at the last snapshot of the bearing
  reversed       c reversed in time (same values, opposite direction)
  healthy value  c frozen at its early-life mean (the original intervention)

Window-level MAE, as in the original attribution table. Output: results/tables/v3/attribution_ig*.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.mtoi_model import MTOIModel                    # noqa: E402
from src.lobo import make_folds, datasets_from_pack            # noqa: E402
from src.lobo_v2 import prepare_fold_v2
import src.lobo_v2 as _lv2
_lv2.save_fold_params = lambda *a, **k: None   # do not rewrite the released per-fold parameter records                        # noqa: E402
from src.utils.logger import get_logger                        # noqa: E402

OUT = ROOT / "results" / "tables" / "v3"
CKPT = ROOT / "results" / "checkpoints" / "v2_keep"
HI_STATIC = ["E_norm", "C_norm", "VTOI"]
DEV = "cpu"
LIGHT = True


def light_pack(dsn, ho, f, logger):
    """Memory-light equivalent of prepare_fold_v2 for inference on the held-out bearing only.

    The input normalisation statistics are the per-axis mean and standard deviation over all
    training windows, accumulated block by block instead of concatenating every window."""
    from src.lobo_v2 import fit_fold_vtoi, COND_COLS
    from src.lobo import load_bearing_pack, HID_STRIDE
    from src.utils.paths import proc_dir_for
    cond, _, _, meta = fit_fold_vtoi(dsn, ho, f["val"], f["train"], seed=42, logger=logger)
    proc = proc_dir_for(dsn)
    s1 = np.zeros(2); s2 = np.zeros(2); n = 0
    for nm in f["train"]:
        X = np.load(proc / nm / "X_vib_windows.npy", mmap_mode="r")
        for i in range(0, len(X), 1000):
            x = np.asarray(X[i:i + 1000], dtype=np.float64)
            s1 += x.sum((0, 2)); s2 += (x ** 2).sum((0, 2)); n += x.shape[0] * x.shape[2]
    mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0)) + 1e-8
    all_names = list(f["train"]) + list(f["val"]) + [ho]
    name2idx = {k: i for i, k in enumerate(sorted(set(all_names)))}
    X, gh, tg = load_bearing_pack(proc / ho, name2idx[ho], has_temp=0)
    c = cond[ho].copy()
    c["hour_id"] = c["hour_id"].astype(np.int64) + name2idx[ho] * HID_STRIDE
    tg = tg.drop(columns=[x for x in COND_COLS if x in tg.columns], errors="ignore")
    tg = tg.merge(c, on="hour_id", how="left")
    from src.lobo_v2 import TEMP_INPUT_COLS
    norm = dict(vib_mean=mean.reshape(2, 1).astype(np.float32),
                vib_std=std.reshape(2, 1).astype(np.float32),
                temp_mean=np.zeros(len(TEMP_INPUT_COLS), np.float32),
                temp_std=np.ones(len(TEMP_INPUT_COLS), np.float32))
    return {"tr": None, "va": None, "te": (X, gh, tg), "norm": norm,
            "n_train_bearings": len(f["train"]), "n_val_bearings": len(f["val"]),
            "holdout": ho, "vtoi_meta": meta}


@torch.no_grad()
def embed_all(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    E, C, Y = [], [], []
    for b in loader:
        E.append(model.vib_encoder(b["vib"]).cpu()); C.append(b["hi"].cpu()); Y.append(b["rul"].cpu())
    return torch.cat(E), torch.cat(C), torch.cat(Y)


def head(model, e, c):
    return model.sigmoid(model.rul_head(torch.cat([e, c], dim=1))).squeeze(-1)


def ig(model, e, c, e0, c0, steps=32):
    ae, ac = torch.zeros_like(e), torch.zeros_like(c)
    for k in range(1, steps + 1):
        t = k / steps
        ee = (e0 + t * (e - e0)).requires_grad_(True)
        cc = (c0 + t * (c - c0)).requires_grad_(True)
        ge, gc = torch.autograd.grad(head(model, ee, cc).sum(), [ee, cc])
        ae += ge; ac += gc
    return (e - e0) * ae / steps, (c - c0) * ac / steps


def run(seeds=(42, 43, 44), datasets=("pronostia", "xjtu_sy")):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    logger = get_logger("r3_attr")
    rows = []
    out_f = OUT / "attribution_ig_per_bearing.csv"
    done = set()
    if out_f.exists():
        prev = pd.read_csv(out_f); rows = prev.to_dict("records")
        done = set(zip(prev.dataset, prev.bearing, prev.seed))
    for dsn in datasets:
        for f in make_folds(dsn)[0]:
            ho = f["holdout"]
            todo = [s for s in seeds if (dsn, ho, s) not in done
                    and (CKPT / f"v2_{dsn}_s{s}_vtoi_static_{ho}_best.pt").exists()]
            if not todo:
                continue
            pack = (light_pack(dsn, ho, f, logger) if LIGHT
                    else prepare_fold_v2(dsn, ho, f["val"], f["train"], seed=42, logger=logger))
            d = datasets_from_pack(pack, mtoi_target="VTOI", train_subsample=1,
                                   rul_col="RUL_capped", hi_cols=HI_STATIC)
            for s in todo:
                model = MTOIModel(use_temp=False, vib_encoder="transformer", use_hi=True,
                                  hi_dim=len(HI_STATIC)).to(DEV)
                model.load_state_dict(torch.load(CKPT / f"v2_{dsn}_s{s}_vtoi_static_{ho}_best.pt",
                                                 map_location=DEV)["model"])
                model.eval()
                e, c, y = embed_all(model, d["test"])
                with torch.no_grad():
                    base = head(model, e, c)
                    mae = lambda p: float((p - y).abs().mean())
                    n = len(c); nh = max(3, n // 20)
                    c_healthy = c[:nh].mean(0, keepdim=True).expand_as(c)
                    c_fail = c[-1:].expand_as(c)
                    r = {"dataset": dsn, "bearing": ho, "seed": s, "mae_base": mae(base),
                         "mae_c_healthy": mae(head(model, e, c_healthy)),
                         "mae_c_failure": mae(head(model, e, c_fail)),
                         "mae_c_reversed": mae(head(model, e, torch.flip(c, [0]))),
                         "mae_e_mean": mae(head(model, e.mean(0, keepdim=True).expand_as(e), c)),
                         "c_iqr": float(np.subtract(*np.percentile(c[:, 2].numpy(), [75, 25])))}
                e0 = e.mean(0, keepdim=True).expand_as(e)
                c0 = c.mean(0, keepdim=True).expand_as(c)
                # gradient x input (centred)
                ee = e.clone().requires_grad_(True); cc = c.clone().requires_grad_(True)
                ge, gc = torch.autograd.grad(head(model, ee, cc).sum(), [ee, cc])
                gxi_e = ((e - e0) * ge).sum(1).abs(); gxi_c = ((c - c0) * gc).sum(1).abs()
                ie, ic = ig(model, e, c, e0, c0)
                ie_s, ic_s = ie.sum(1), ic.sum(1)
                with torch.no_grad():
                    gap = head(model, e, c) - head(model, e0, c0)
                r.update({
                    "raw_grad_ratio": float(gc.abs().sum(1).mean() / (ge.norm(dim=1).mean() + 1e-12)),
                    "gxi_share_c": float(gxi_c.mean() / (gxi_c.mean() + gxi_e.mean() + 1e-12)),
                    "ig_share_c": float(ic_s.abs().mean() / (ic_s.abs().mean() + ie_s.abs().mean() + 1e-12)),
                    "ig_completeness_err": float((ie_s + ic_s - gap).abs().mean()),
                })
                rows.append(r)
                logger.info(f"[{dsn}/{ho}/s{s}] base={r['mae_base']:.4f} fail={r['mae_c_failure']:.4f} "
                            f"rev={r['mae_c_reversed']:.4f} ig_share_c={r['ig_share_c']:.3f} "
                            f"gxi_share_c={r['gxi_share_c']:.3f}")
                pd.DataFrame(rows).to_csv(out_f, index=False)
            del pack
    df = pd.DataFrame(rows)
    df["d_healthy"] = df.mae_c_healthy - df.mae_base
    df["d_failure"] = df.mae_c_failure - df.mae_base
    df["d_reversed"] = df.mae_c_reversed - df.mae_base
    df["d_e_mean"] = df.mae_e_mean - df.mae_base
    sa = df.groupby(["dataset", "bearing"]).mean(numeric_only=True).reset_index()
    summ = sa.groupby("dataset")[["mae_base", "d_healthy", "d_failure", "d_reversed", "d_e_mean",
                                  "raw_grad_ratio", "gxi_share_c", "ig_share_c",
                                  "ig_completeness_err", "c_iqr"]].mean().round(4)
    summ.to_csv(OUT / "attribution_ig.csv")
    print(summ.T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pronostia,xjtu_sy")
    a = ap.parse_args()
    run(datasets=tuple(a.datasets.split(",")))
