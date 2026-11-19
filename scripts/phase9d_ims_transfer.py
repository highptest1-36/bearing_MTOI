# -*- coding: utf-8 -*-
r"""
phase9d_ims_transfer.py — EXTERNAL VALIDATION TRÊN IMS BẰNG **TRANSFER ĐÓNG BĂNG** (R3 #2).

------------------------------------------------------------------------------------------
VÌ SAO KHÔNG LÀM LOBO NGAY TRÊN IMS
------------------------------------------------------------------------------------------
IMS chỉ có **3 lần chạy đến hỏng độc lập** (4 ổ bi, nhưng Test 1 đóng góp 2 ổ bi nằm trên
CÙNG một trục, cùng tải, đo ĐỒNG THỜI). Nếu chia fold theo ổ bi thì fold giữ lại
`Test1_Bearing3` sẽ có `Test1_Bearing4` trong train/val — mô hình vẫn thấy rung động của
CHÍNH giàn thử đó tại CHÍNH khoảng thời gian đó. Đó đúng là loại leakage mà bài này phê phán,
nên dùng nó ở đây sẽ **tự mâu thuẫn với luận điểm trung tâm**.

Nếu gom nhóm theo lần chạy (leave-one-run-out, xem `lobo.group_key`) thì leakage-free, nhưng
3/4 fold chỉ còn **1 ổ bi train** — dưới ngưỡng tối thiểu để fit tham số chỉ số VTOI.

------------------------------------------------------------------------------------------
THIẾT KẾ ĐƯỢC CHỌN — mạnh hơn cả hai
------------------------------------------------------------------------------------------
Huấn luyện **hoàn toàn trên XJTU-SY**, rồi áp **NGUYÊN VẸN** lên cả 4 ổ bi IMS:

  * tham số chỉ số VTOI  : fit trên 12 ổ bi TRAIN của XJTU, ĐÓNG BĂNG
  * trọng số mạng        : train trên 12 ổ bi XJTU, val trên 3 ổ bi XJTU (mỗi điều kiện 1)
  * IMS                  : KHÔNG đóng góp một gradient nào, không một nhãn nào

Đây là bài kiểm tra tổng quát hoá **thật**: máy khác, cảm biến khác, tốc độ lấy mẫu khác.
Leakage-free theo định nghĩa (tập test đến từ một dataset chưa từng được chạm tới).

------------------------------------------------------------------------------------------
BỐN KHÁC BIỆT MIỀN — PHẢI CÔNG BỐ ĐẦY ĐỦ TRONG BÀI
------------------------------------------------------------------------------------------
  1. Tốc độ lấy mẫu : XJTU 25,6 kHz  vs  IMS 20 kHz. Cùng độ dài cửa sổ L=4096 mẫu, nên
                      cửa sổ IMS phủ 204,8 ms còn XJTU phủ 160 ms. KHÔNG hiệu chỉnh lại.
  2. Nhịp snapshot  : XJTU 1 phút     vs  IMS 10 phút.
  3. Số kênh vật lý : IMS test 2 và 3 chỉ có 1 gia tốc kế -> nhân đôi thành [x, x]
                      (xem `ims_loader`). Test 1 có đủ 2 kênh.
  4. Thang đo cảm biến: gia tốc kế khác nhau, đơn vị khác nhau. Cửa sổ IMS được chuẩn hoá
                      bằng thống kê của **20 % đầu đời của chính ổ bi đó** — ĐÚNG quy ước
                      baseline khoẻ mà VTOI vẫn dùng (`vtoi.compute_raw_components`,
                      `baseline_frac=0.2`), KHÔNG dùng nhãn hỏng. Đây chính là việc mà một
                      lần triển khai thật sẽ làm: ghi máy lúc còn khoẻ rồi hiệu chuẩn.
                      Ghi rõ trong bài để không ai hiểu nhầm là hiệu chỉnh có giám sát.

So sánh chính: `vtoi_static` (có conditioning) vs `transformer_vib` (không) — CÙNG encoder,
CÙNG loss, CÙNG dữ liệu train, khác DUY NHẤT ở vector conditioning.

Chạy:
    python3 scripts/phase9d_ims_transfer.py --epochs 20 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment                              # noqa: E402
from src.utils.logger import get_logger, section                         # noqa: E402
from src.utils.paths import proc_dir_for, TABLES_DIR, PREDICTIONS_DIR    # noqa: E402
from src.datasets import MTOIWindowDataset, TEMP_INPUT_COLS              # noqa: E402
from src.lobo import (HID_STRIDE, load_bearing_pack, dedup_bearings,     # noqa: E402
                      datasets_from_pack)
from src.lobo_v2 import COND_COLS, _load_bearing_frames                  # noqa: E402
from src.train import train_model                                        # noqa: E402
from src.evaluate import evaluate_model                                  # noqa: E402
from src import vtoi as V                                                # noqa: E402

# Cùng 2 config với phương án dự phòng của runbook Tier 4.
HI_STATIC = ["E_norm", "C_norm", "VTOI"]
CONFIGS = [
    ("vtoi_static",     dict(vib_encoder="transformer", use_hi=True,  hi_cols=HI_STATIC,
                             mtoi_target="VTOI", loss_override=dict(lambda1=0.1, w_stage=0.0))),
    ("transformer_vib", dict(vib_encoder="transformer", use_hi=False, hi_cols=None,
                             mtoi_target="VTOI", loss_override=dict(lambda1=0.0, w_stage=0.0))),
    # --- HAI CONTROL ÂM (R2 #25) — bắt buộc phải có ở đây ---
    # Nếu không chạy, phản biện hiển nhiên sẽ là: "quỹ đạo đúng dấu chỉ vì bạn đút VTOI vào
    # làm đầu vào thôi". Hai control này dùng ĐÚNG cùng một kênh đầu vào nhưng nội dung vô
    # nghĩa (trọng số ngẫu nhiên / xáo trộn thời gian). Nếu chúng cũng cho ρ đúng dấu thì
    # kết quả KHÔNG đến từ VTOI và phải báo cáo như vậy.
    ("ctl_random",      dict(vib_encoder="transformer", use_hi=True, hi_cols=["VTOI_rand"],
                             mtoi_target="VTOI", loss_override=dict(lambda1=0.1, w_stage=0.0))),
    ("ctl_shuffled",    dict(vib_encoder="transformer", use_hi=True, hi_cols=["VTOI_shuf"],
                             mtoi_target="VTOI", loss_override=dict(lambda1=0.1, w_stage=0.0))),
]
BASELINE_FRAC = 0.2          # khớp vtoi.compute_raw_components — baseline khoẻ, không dùng nhãn
METRIC_KEYS = ["rul_mae", "rul_rmse", "rul_r2", "rul_mae_hours", "rul_rmse_hours",
               "rul_phm_score", "rul_asym_score", "first_warning", "lead_time",
               "false_alarm_rate", "mtoi_rmse", "mtoi_spearman", "mtoi_spearman_signed",
               "mtoi_spearman_vs_deg", "mtoi_monotonicity", "stage_acc", "stage_macro_f1",
               "stage_bal_acc"]


def _xjtu_condition(name):
    """'35Hz12kN_Bearing1_1' -> '35Hz12kN' (dùng để chọn val trải đủ 3 điều kiện)."""
    return name.split("_Bearing")[0]


def build_source(seed, logger):
    """
    Dựng nguồn XJTU: fit VTOI trên 12 ổ bi TRAIN, val 3 ổ bi (mỗi điều kiện 1), đóng gói pack.
    Trả về dict có tr/va/norm + tham số VTOI đã đóng băng để áp cho IMS.
    """
    proc = proc_dir_for("xjtu_sy")
    man = dedup_bearings(pd.read_csv(proc / "_manifest.csv"), logger=logger)
    names = sorted(man["bearing"].tolist())

    # val = ổ bi ĐẦU TIÊN (theo thứ tự tên) của MỖI điều kiện vận hành -> val trải đủ 3 điều kiện,
    # và cách chọn hoàn toàn tất định (không phụ thuộc seed) để tái lập được.
    val_names, seen = [], set()
    for n in names:
        c = _xjtu_condition(n)
        if c not in seen:
            seen.add(c); val_names.append(n)
    train_names = [n for n in names if n not in val_names]
    logger.info(f"[nguồn XJTU] train {len(train_names)} ổ bi | val {len(val_names)} ổ bi "
                f"({', '.join(val_names)})")

    # ---- thành phần thô + fit tham số VTOI CHỈ trên train ----
    comps, degs, hfs = {}, {}, {}
    for n in train_names + val_names:
        hf, lb = _load_bearing_frames(proc, n)
        comps[n] = V.compute_raw_components(hf)
        hfs[n] = hf
        degs[n] = (lb["Deg"].to_numpy(float) if "Deg" in lb.columns
                   else np.arange(len(hf)) / max(len(hf) - 1, 1))
    params = V.fit_vtoi_params([comps[n] for n in train_names],
                               [degs[n] for n in train_names], objective="mono+corr")
    params_mono = V.fit_vtoi_params([comps[n] for n in train_names],
                                    [degs[n] for n in train_names], objective="mono")
    hc_tr = np.concatenate([hfs[n][V.VIB_FEATURES].to_numpy(float) for n in train_names], axis=0)
    hc_stats = (hc_tr.mean(axis=0), hc_tr.std(axis=0))
    H_max_train = max(len(hfs[n]) for n in train_names)
    logger.info(f"[nguồn XJTU] tham số VTOI ĐÓNG BĂNG: a={params['a']:.3f} b={params['b']:.3f} "
                f"fitness={params['fitness']:.3f} | H_max_train={H_max_train}")

    # ---- đóng gói cửa sổ XJTU ----
    name2idx = {n: i for i, n in enumerate(sorted(train_names + val_names))}
    has_temp_map = dict(zip(man["bearing"], man["has_temp"].astype(int)))

    def pack_group(group):
        Xs, ghs, tgs = [], [], []
        for n in group:
            X, gh, tg = load_bearing_pack(proc / n, name2idx[n],
                                          has_temp=has_temp_map.get(n, 0))
            c = V.build_conditioning_columns(
                comps[n], params, params_mono, bearing_name=n, fold_name="xjtu_source",
                H_max_train=H_max_train, deg=degs[n],
                hc_features=hfs[n], hc_stats=hc_stats, seed=seed).copy()
            c["hour_id"] = c["hour_id"].astype(np.int64) + name2idx[n] * HID_STRIDE
            tg = tg.drop(columns=[x for x in COND_COLS if x in tg.columns], errors="ignore")
            tg = tg.merge(c, on="hour_id", how="left")
            assert tg["VTOI"].notna().all(), f"thiếu VTOI sau merge: {n}"
            Xs.append(X); ghs.append(gh); tgs.append(tg)
        return (np.concatenate(Xs, 0), np.concatenate(ghs, 0),
                pd.concat(tgs, ignore_index=True))

    tr = pack_group(train_names)
    va = pack_group(val_names)
    vib_mean = tr[0].mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
    vib_std = (tr[0].std(axis=(0, 2), keepdims=True)[0] + 1e-8).astype(np.float32)
    norm = dict(vib_mean=vib_mean, vib_std=vib_std,
                temp_mean=np.zeros(len(TEMP_INPUT_COLS), np.float32),
                temp_std=np.ones(len(TEMP_INPUT_COLS), np.float32))
    logger.info(f"[nguồn XJTU] cửa sổ: train={len(tr[1])} | val={len(va[1])}")
    return dict(tr=tr, va=va, norm=norm, params=params, params_mono=params_mono,
                hc_stats=hc_stats, H_max_train=H_max_train,
                train_names=train_names, val_names=val_names)


def build_ims_test(ims_name, src, seed, logger):
    """
    Dựng tập test cho 1 ổ bi IMS, áp tham số VTOI ĐÓNG BĂNG của XJTU.
    Chuẩn hoá cửa sổ bằng 20 % ĐẦU ĐỜI của chính ổ bi đó (baseline khoẻ, KHÔNG dùng nhãn).
    """
    proc = proc_dir_for("ims")
    hf, lb = _load_bearing_frames(proc, ims_name)
    comps = V.compute_raw_components(hf)
    deg = (lb["Deg"].to_numpy(float) if "Deg" in lb.columns
           else np.arange(len(hf)) / max(len(hf) - 1, 1))
    cond = V.build_conditioning_columns(
        comps, src["params"], src["params_mono"], bearing_name=ims_name,
        fold_name="xjtu->ims", H_max_train=src["H_max_train"], deg=deg,
        hc_features=hf, hc_stats=src["hc_stats"], seed=seed).copy()

    idx = 0                                   # test đứng riêng -> chỉ số bearing bắt đầu từ 0
    X, gh, tg = load_bearing_pack(proc / ims_name, idx, has_temp=0)
    cond["hour_id"] = cond["hour_id"].astype(np.int64) + idx * HID_STRIDE
    tg = tg.drop(columns=[x for x in COND_COLS if x in tg.columns], errors="ignore")
    tg = tg.merge(cond, on="hour_id", how="left")
    assert tg["VTOI"].notna().all(), f"thiếu VTOI sau merge: {ims_name}"

    # --- chuẩn hoá đầu vào theo baseline KHOẺ của chính ổ bi (label-free) ---
    H = len(hf)
    n_base = max(int(BASELINE_FRAC * H), 1)
    base_hids = set((np.arange(n_base) + idx * HID_STRIDE).tolist())
    m = np.isin(gh, list(base_hids))
    Xb = X[m] if m.any() else X
    ims_norm = dict(
        vib_mean=Xb.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32),
        vib_std=(Xb.std(axis=(0, 2), keepdims=True)[0] + 1e-8).astype(np.float32),
        temp_mean=np.zeros(len(TEMP_INPUT_COLS), np.float32),
        temp_std=np.ones(len(TEMP_INPUT_COLS), np.float32))
    logger.info(f"  [{ims_name}] {H} snapshot | {len(gh)} cửa sổ | "
                f"baseline chuẩn hoá = {n_base} snapshot đầu ({m.sum()} cửa sổ) | "
                f"VTOI[min,med,max]=[{cond['VTOI'].min():.3f},"
                f"{cond['VTOI'].median():.3f},{cond['VTOI'].max():.3f}]")
    return (X, gh, tg), ims_norm, comps["info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=4,
                    help="lấy thưa cửa sổ TRAIN của XJTU (khớp train_subsample của tier0)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    device = setup_environment(seed=a.seed, do_mount=False)
    logger = get_logger("ims_transfer")
    section("PHASE 9D — EXTERNAL VALIDATION IMS BẰNG TRANSFER ĐÓNG BĂNG XJTU->IMS (R3 #2)", logger)

    out_csv = TABLES_DIR / "v2" / f"ims_transfer_seed{a.seed}.csv"
    done = set()
    if out_csv.exists() and not a.force:
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["config"], prev["holdout"]))
        logger.info(f"Đã có {len(done)} kết quả -> bỏ qua (resume).")
    else:
        prev = pd.DataFrame()

    src = build_source(a.seed, logger)
    ims_names = sorted(pd.read_csv(proc_dir_for("ims") / "_manifest.csv")["bearing"].tolist())
    logger.info(f"Ổ bi IMS để đánh giá: {ims_names}")

    # Dựng sẵn tập test IMS 1 lần (dùng lại cho mọi config).
    ims_tests = {}
    for n in ims_names:
        ims_tests[n] = build_ims_test(n, src, a.seed, logger)

    rows = []
    for cfg_name, cfg in CONFIGS:
        pending = [n for n in ims_names if (cfg_name, n) not in done]
        if not pending:
            logger.info(f"[{cfg_name}] đã xong toàn bộ -> bỏ qua.")
            continue

        # ---- HUẤN LUYỆN MỘT LẦN trên XJTU (không phụ thuộc ổ bi IMS nào) ----
        pack = dict(tr=src["tr"], va=src["va"], te=src["va"], norm=src["norm"],
                    n_train_bearings=len(src["train_names"]), holdout="xjtu_source")
        ds = datasets_from_pack(pack, mtoi_target=cfg["mtoi_target"],
                                train_subsample=a.subsample, rul_col="RUL_capped",
                                hi_cols=cfg["hi_cols"])
        run_name = f"v2_imstransfer_s{a.seed}_{cfg_name}"
        logger.info(f"\n=== [{cfg_name}] huấn luyện trên XJTU (1 lần, dùng chung cho 4 ổ bi IMS) ===")
        t0 = time.time()
        out = train_model(
            {"train": ds["train"], "val": ds["val"], "test": ds["test"]},
            run_name=run_name, device=device, use_temp=False, fusion="gated",
            vib_encoder=cfg["vib_encoder"], loss_weights=cfg["loss_override"],
            use_uncertainty=False, use_hi=cfg["use_hi"],
            hi_dim=(len(cfg["hi_cols"]) if cfg["hi_cols"] else 4),
            epochs=a.epochs, batch_size=256, patience=5, num_workers=4, resume=True,
            log=logger, sampler_mode="bearing", seed=a.seed)
        logger.info(f"[{cfg_name}] huấn luyện xong trong {time.time()-t0:.0f}s")

        # ---- ĐÁNH GIÁ trên từng ổ bi IMS ----
        for n in pending:
            te, ims_norm, info = ims_tests[n]
            X, gh, tg = te
            ds_te = MTOIWindowDataset(X, gh, tg, set(np.unique(gh).tolist()),
                                      **ims_norm, mtoi_col=cfg["mtoi_target"],
                                      rul_col="RUL_capped", hi_cols=cfg["hi_cols"])
            res = evaluate_model(out["model"], ds_te, device=device,
                                 run_name=f"{run_name}_{n}", save=False)
            if res.get("hour_df") is not None:
                d = PREDICTIONS_DIR / "v2"; d.mkdir(parents=True, exist_ok=True)
                res["hour_df"].to_csv(d / f"ims_transfer_s{a.seed}_{n}_{cfg_name}.csv",
                                      index=False)
            row = {"dataset": "ims", "protocol": "frozen_transfer_xjtu_to_ims",
                   "config": cfg_name, "holdout": n, "seed": a.seed,
                   "n_train_bearings": len(src["train_names"]),
                   "n_test_windows": len(ds_te),
                   "hi_cols": "|".join(cfg["hi_cols"]) if cfg["hi_cols"] else "",
                   "vtoi_a": src["params"]["a"], "vtoi_b": src["params"]["b"],
                   "vtoi_fitness": src["params"]["fitness"],
                   "n_baseline_snapshots": info.get("n_baseline"),
                   "cov_estimator": info.get("estimator")}
            row.update({k: res["metrics"].get(k) for k in METRIC_KEYS})
            rows.append(row)
            logger.info(f"  [{cfg_name} | {n}] MAE={row['rul_mae']:.4f} "
                        f"MAE_h={row['rul_mae_hours']:.3f} "
                        f"rho_signed={row['mtoi_spearman_signed']}")

            df = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)          # ghi sau MỖI ổ bi -> an toàn disconnect

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    if len(df):
        section("KẾT QUẢ TRANSFER XJTU -> IMS", logger)
        piv = df.pivot_table(index="holdout", columns="config",
                             values=["rul_mae", "mtoi_spearman_signed"])
        print(piv.to_string())
        print("\nTrung bình qua 4 ổ bi IMS:")
        print(df.groupby("config")[["rul_mae", "rul_mae_hours",
                                    "mtoi_spearman_signed", "mtoi_monotonicity"]]
              .mean().round(4).to_string())
    logger.info(f"Ghi {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
