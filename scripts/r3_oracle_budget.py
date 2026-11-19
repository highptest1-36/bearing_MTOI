# -*- coding: utf-8 -*-
"""
r3_oracle_budget.py - Round-3 revision, Reviewer 4 item (b).

Question: the oracle arm receives the lifetime fraction, from which the capped RUL target is a
deterministic function, yet its error is 0.118 / 0.142. Is the regressor under-trained?

Evidence already in hand (logs of the original runs): the oracle arm reaches the 20-epoch cap in
most runs without triggering early stopping, while the other arms stop near epoch 10.

This script retrains ONLY the oracle arm (and, optionally, the proposed arm) with a longer budget,
on the same folds, seeds, loss and sampler, and writes results to a separate table so the original
results are never overwritten:
    results/tables/v3/oracle_budget.csv
    results/predictions/v3/oracle_budget_<dataset>_s<seed>_<holdout>_<cfg>_e<epochs>.csv

Usage:
    python scripts/r3_oracle_budget.py --dataset pronostia --stages 20,40,60 --folds 0,1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils.env import setup_environment            # noqa: E402
from src.utils.logger import get_logger                 # noqa: E402
from src.train import train_model                       # noqa: E402
from src.evaluate import evaluate_model                 # noqa: E402
from src.lobo import make_folds, datasets_from_pack     # noqa: E402
from src.lobo_v2 import prepare_fold_v2
import src.lobo_v2 as _lv2
_lv2.save_fold_params = lambda *a, **k: None   # do not rewrite the released per-fold parameter records                 # noqa: E402
import phase9b_v2 as P                                  # noqa: E402

OUT_T = ROOT / "results" / "tables" / "v3"
OUT_P = ROOT / "results" / "predictions" / "v3"
CKPT = Path("/content/r3_oracle_ckpts")
CFGS = {"ctl_lifefrac": dict(P.TIER1)["ctl_lifefrac"], "vtoi_static": dict(P.TIER0)["vtoi_static"]}


def run(dataset, stages, fold_ids, seed, cfg_names, lr):
    OUT_T.mkdir(parents=True, exist_ok=True); OUT_P.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    import src.train as _t
    _t.CHECKPOINTS_DIR = CKPT
    device = setup_environment(seed=seed, do_mount=False)
    logger = get_logger(f"r3_oracle_{dataset}")
    folds, _ = make_folds(dataset)
    sub = 4 if dataset == "xjtu_sy" else 1
    out_csv = OUT_T / "oracle_budget.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    def _done(ho, cfg_name, budget):
        return any((r["dataset"], r["config"], r["holdout"], int(r["seed"]), int(r["epochs"])) ==
                   (dataset, cfg_name, ho, seed, budget) for r in rows)
    for fi in fold_ids:
        f = folds[fi]; ho = f["holdout"]
        if all(_done(ho, c, b) for c in cfg_names for b in stages):
            logger.info(f"SKIP {ho}: every budget already in {out_csv.name}")
            continue                                     # resume after a disconnect without re-preparing data
        pack = prepare_fold_v2(dataset, ho, f["val"], f["train"], seed=seed, logger=logger)
        for cfg_name in cfg_names:
            cfg = CFGS[cfg_name]
            ds = datasets_from_pack(pack, mtoi_target=cfg["mtoi_target"], train_subsample=sub,
                                    rul_col="RUL_capped", hi_cols=cfg["hi_cols"])
            # one model trained in stages; evaluated (best-by-validation weights) at each budget
            run_name = f"r3_{dataset}_s{seed}_{cfg_name}_staged_{ho}"
            for budget in stages:
                if _done(ho, cfg_name, budget):
                    continue
                t0 = time.time()
                out = train_model({"train": ds["train"], "val": ds["val"], "test": ds["test"]},
                                  run_name=run_name, device=device, use_temp=False,
                                  fusion="gated", vib_encoder=cfg["vib_encoder"],
                                  loss_weights=cfg["loss_override"], use_uncertainty=False,
                                  use_hi=cfg["use_hi"], hi_dim=len(cfg["hi_cols"]),
                                  epochs=budget, batch_size=256, lr=lr, patience=10 ** 6,
                                  num_workers=P._NUM_WORKERS, resume=True, log=logger,
                                  sampler_mode="bearing", seed=seed)
                res = evaluate_model(out["model"], ds["test"], device=device,
                                     run_name=run_name, save=False)
                hd = res.get("hour_df")
                slope = np.nan
                if hd is not None:
                    hd.to_csv(OUT_P / f"oracle_budget_{dataset}_s{seed}_{ho}_{cfg_name}_e{budget}.csv",
                              index=False)
                    slope = float(np.polyfit(hd.rul_true.to_numpy(float),
                                             hd.rul_pred.to_numpy(float), 1)[0])
                h = out.get("history") or {}
                vl = h.get("val_loss", []) if isinstance(h, dict) else []
                rows.append({"dataset": dataset, "config": cfg_name, "holdout": ho, "seed": seed,
                             "epochs": budget, "lr": lr, "best_epoch": int(np.argmin(vl)) + 1 if vl else np.nan,
                             "best_val": out.get("best_val"),
                             "rul_mae": res["metrics"].get("rul_mae"),
                             "rul_rmse": res["metrics"].get("rul_rmse"), "slope": slope,
                             "minutes": round((time.time() - t0) / 60, 1)})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                logger.info(f"DONE {cfg_name} {ho} budget={budget}: mae={rows[-1]['rul_mae']:.4f} "
                            f"slope={slope:.3f} best_epoch={rows[-1]['best_epoch']}")
        del pack


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pronostia")
    ap.add_argument("--stages", default="20,40,60")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="ctl_lifefrac")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    if a.folds == "all":
        fold_ids = list(range(len(make_folds(a.dataset)[0])))
    else:
        fold_ids = [int(x) for x in a.folds.split(",")]
    run(a.dataset, [int(x) for x in a.stages.split(",")], fold_ids, a.seed, a.configs.split(","),
        a.lr)
