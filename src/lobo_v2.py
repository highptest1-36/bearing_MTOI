# -*- coding: utf-8 -*-
"""
lobo_v2.py — ASSEMBLER LOBO **LEAKAGE-FREE** (bản v2 cho vòng nộp lại IEEE Access).

Khác bản cũ (`src/lobo.py:prepare_fold`) ở ĐÚNG MỘT ĐIỂM, nhưng là điểm sống còn:
  CŨ : chỉ số VTOI được tính SẴN cho từng bearing (trọng số + biên min-max fit trên chính
       bearing đó, dùng cả H_fail của nó) -> bearing held-out bị leakage nhãn.
  MỚI: mỗi fold TỰ fit tham số VTOI **chỉ trên các bearing TRAIN**, rồi ÁP ĐÓNG BĂNG cho
       val + held-out test. Có assert fail-loud chống leakage.

Đồng thời sinh luôn TOÀN BỘ cột control của R2 #25 (random / shuffled / elapsed / oracle /
handcrafted / mono-only) để Tier 1 chỉ việc đổi `hi_cols`, không phải chạy lại pipeline.

Tái sử dụng nguyên vẹn: make_folds(), physical_base(), dedup_bearings(), load_bearing_pack(),
datasets_from_pack() của `src/lobo.py`. KHÔNG cần chạy lại phase9a — vì E_raw/C_raw chỉ cần
`hour_features.csv` vốn đã có sẵn trên Drive.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import proc_dir_for, TABLES_DIR
from src.utils.logger import get_logger
from src.lobo import (HID_STRIDE, load_bearing_pack, make_folds, physical_base,
                      group_key, datasets_from_pack)
from src.datasets import TEMP_INPUT_COLS
from src import vtoi as V


# Các cột conditioning được sinh cho mọi bearing của fold (dùng cho đề xuất + control).
COND_COLS = ["E_norm", "C_norm", "VTOI", "VTOI_vel", "VTOI_acc",
             "VTOI_mono", "VTOI_rand", "VTOI_shuf",
             "elapsed_norm", "Deg_oracle", "vtoi_sat_frac"] + V.HC_COLS


def _load_bearing_frames(proc_root, name):
    """Đọc hour_features.csv + labels_by_hour.csv của 1 bearing (không đụng tín hiệu thô)."""
    bdir = Path(proc_root) / name
    hf = pd.read_csv(bdir / "hour_features.csv").sort_values("hour_id").reset_index(drop=True)
    lb = pd.read_csv(bdir / "labels_by_hour.csv").sort_values("hour_id").reset_index(drop=True)
    return hf, lb


def fit_fold_vtoi(dataset, holdout_name, val_names, train_names, seed=42, logger=None):
    """
    B1: fit tham số VTOI CHỈ trên bearing TRAIN của fold.
    B2: áp đóng băng cho train + val + test, sinh mọi cột conditioning/control.

    Trả về (cond_by_bearing: dict[name -> DataFrame], params, params_mono, meta).
    """
    logger = logger or get_logger("lobo_v2")
    proc_root = proc_dir_for(dataset)
    all_names = list(train_names) + list(val_names) + [holdout_name]

    # ---------- Đọc & tính thành phần thô (KHÔNG dùng nhãn) cho MỌI bearing của fold ----------
    comps, degs, hfs, infos = {}, {}, {}, {}
    for n in all_names:
        hf, lb = _load_bearing_frames(proc_root, n)
        c = V.compute_raw_components(hf)
        comps[n] = c
        hfs[n] = hf
        infos[n] = c["info"]
        degs[n] = (lb["Deg"].to_numpy(float) if "Deg" in lb.columns
                   else np.arange(len(hf)) / max(len(hf) - 1, 1))

    # ---------- FIT: CHỈ bearing TRAIN ----------
    tr_comps = [comps[n] for n in train_names]
    tr_degs = [degs[n] for n in train_names]
    params = V.fit_vtoi_params(tr_comps, tr_degs, objective="mono+corr")
    params_mono = V.fit_vtoi_params(tr_comps, tr_degs, objective="mono")   # control ctl_nodeg
    V.assert_no_leakage(params, holdout_name, train_names, val_names)

    # Thống kê chuẩn hoá handcrafted — cũng CHỈ từ TRAIN (control ctl_hc10).
    hc_tr = np.concatenate([hfs[n][V.VIB_FEATURES].to_numpy(float) for n in train_names], axis=0)
    hc_stats = (hc_tr.mean(axis=0), hc_tr.std(axis=0))

    # H_max của TRAIN cho `elapsed_norm` (KHÔNG dùng H_fail của test — nếu dùng là oracle).
    H_max_train = max(len(hfs[n]) for n in train_names)

    # ---------- APPLY: đóng băng tham số, sinh cột cho MỌI bearing ----------
    fold_name = f"{dataset}|{holdout_name}"
    cond = {}
    for n in all_names:
        cond[n] = V.build_conditioning_columns(
            comps[n], params, params_mono, bearing_name=n, fold_name=fold_name,
            H_max_train=H_max_train, deg=degs[n],
            hc_features=hfs[n], hc_stats=hc_stats, seed=seed)

    meta = {
        "dataset": dataset, "holdout": holdout_name, "seed": seed,
        "n_train_bearings": len(train_names), "H_max_train": int(H_max_train),
        "a": params["a"], "b": params["b"], "fitness": params["fitness"],
        "a_mono": params_mono["a"], "b_mono": params_mono["b"],
        "E_lo": params["E_lo"], "E_hi": params["E_hi"],
        "C_lo": params["C_lo"], "C_hi": params["C_hi"],
        "holdout_sat_frac": float(cond[holdout_name]["vtoi_sat_frac"].iloc[0]),
        "holdout_n_baseline": infos[holdout_name]["n_baseline"],
        "holdout_cov_estimator": infos[holdout_name]["estimator"],
        "holdout_shrinkage": infos[holdout_name]["shrinkage"],
    }
    logger.info(f"  [VTOI fold {holdout_name}] a={params['a']:.3f} b={params['b']:.3f} "
                f"fit={params['fitness']:.3f} (train {len(train_names)}b) | "
                f"a_mono={params_mono['a']:.3f} | sat(test)={meta['holdout_sat_frac']:.3f}")
    return cond, params, params_mono, meta


def save_fold_params(dataset, holdout_name, params, params_mono, meta, seed=42):
    """
    Ghi tham số RIÊNG cho từng fold — KHÔNG ghi đè (sửa lỗi A5/R2 #14 của bản cũ, vốn để
    mọi bearing ghi đè lên cùng một file rồi báo cáo giá trị của bearing cuối cùng).
    Cũng ghi luôn đường quét trọng số để dựng Fig. nhạy cảm (R3 #3).
    """
    d = TABLES_DIR / "v2_vtoi_params"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_seed{seed}_{holdout_name}"
    pd.DataFrame([meta]).to_csv(d / f"params_{tag}.csv", index=False)
    pd.DataFrame(params["sweep"], columns=["a", "fitness"]).to_csv(
        d / f"sweep_{tag}.csv", index=False)


def prepare_fold_v2(dataset, holdout_name, val_names, train_names, seed=42, logger=None):
    """
    Tương đương `lobo.prepare_fold` nhưng chỉ số VTOI được fit LEAKAGE-FREE theo fold.
    Trả về 'pack' dùng được ngay với `lobo.datasets_from_pack`.
    """
    logger = logger or get_logger("lobo_v2")
    proc_root = proc_dir_for(dataset)

    # GUARD chống leakage theo ổ bi VẬT LÝ (PRONOSTIA lặp cùng ổ bi giữa Test_set/Full_Test_Set).
    b_ho = group_key(holdout_name)
    for nm in list(train_names) + list(val_names):
        assert group_key(nm) != b_ho, (
            f"LEAKAGE: '{nm}' cùng nhóm độc lập với holdout '{holdout_name}' "
            f"(nhóm='{b_ho}'). PRONOSTIA/XJTU: cùng ổ bi vật lý; IMS: cùng lần chạy.")

    # ---------- (A) Fit + apply VTOI theo fold ----------
    cond, params, params_mono, meta = fit_fold_vtoi(
        dataset, holdout_name, val_names, train_names, seed=seed, logger=logger)
    save_fold_params(dataset, holdout_name, params, params_mono, meta, seed=seed)

    # ---------- (B) Nạp cửa sổ + đích cấp snapshot (như bản cũ) ----------
    all_names = list(train_names) + list(val_names) + [holdout_name]
    name2idx = {n: i for i, n in enumerate(sorted(set(all_names)))}
    man = pd.read_csv(proc_root / "_manifest.csv")
    has_temp_map = dict(zip(man["bearing"], man["has_temp"].astype(int)))

    def pack_group(names):
        Xs, ghs, tgs = [], [], []
        for n in names:
            X, gh, tg = load_bearing_pack(proc_root / n, name2idx[n],
                                          has_temp=has_temp_map.get(n, 0))
            # ---- TIÊM cột conditioning của fold, ghi đè mọi cột VTOI cũ (bị leak) ----
            c = cond[n].copy()
            c["hour_id"] = c["hour_id"].astype(np.int64) + name2idx[n] * HID_STRIDE
            tg = tg.drop(columns=[x for x in COND_COLS if x in tg.columns], errors="ignore")
            tg = tg.merge(c, on="hour_id", how="left")
            assert tg["VTOI"].notna().all(), f"Thiếu cột VTOI sau merge cho bearing {n}"
            Xs.append(X); ghs.append(gh); tgs.append(tg)
        if not Xs:
            return None
        return (np.concatenate(Xs, 0), np.concatenate(ghs, 0),
                pd.concat(tgs, ignore_index=True))

    tr = pack_group(list(train_names))
    va = pack_group(list(val_names))
    te = pack_group([holdout_name])
    if tr is None or te is None:
        raise RuntimeError(f"Fold {holdout_name}: thiếu dữ liệu train/test.")

    X_tr, gh_tr, tg_tr = tr
    vib_mean = X_tr.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
    vib_std = (X_tr.std(axis=(0, 2), keepdims=True)[0] + 1e-8).astype(np.float32)
    # Nhánh nhiệt độ đã bị loại khỏi paper VTOI -> giữ thống kê trung tính cho tương thích API.
    temp_mean = np.zeros(len(TEMP_INPUT_COLS), np.float32)
    temp_std = np.ones(len(TEMP_INPUT_COLS), np.float32)

    logger.info(f"  Fold[{holdout_name}]: train {len(train_names)}b/{len(gh_tr)}w | "
                f"val {len(val_names)}b | test 1b/{len(te[1])}w")
    return {
        "tr": tr, "va": va, "te": te,
        "norm": dict(vib_mean=vib_mean, vib_std=vib_std,
                     temp_mean=temp_mean, temp_std=temp_std),
        "n_train_bearings": len(train_names), "n_val_bearings": len(val_names),
        "holdout": holdout_name, "vtoi_meta": meta,
    }


__all__ = ["prepare_fold_v2", "fit_fold_vtoi", "make_folds", "datasets_from_pack", "COND_COLS"]
