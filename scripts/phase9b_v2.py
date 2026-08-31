# -*- coding: utf-8 -*-
"""
phase9b_v2.py — HUẤN LUYỆN LOBO **v2 LEAKAGE-FREE** cho vòng nộp lại IEEE Access.

KHÁC BẢN CŨ (`phase9b_lobo_train.py`) Ở 6 ĐIỂM — mỗi điểm sửa một lỗi reviewer đã/sẽ bắt:

  1. [B1 / R2 #6,#7,#8] Dùng `lobo_v2.prepare_fold_v2` -> tham số VTOI fit CHỈ trên bearing
     TRAIN của fold rồi đóng băng. Bản cũ fit trên chính bearing held-out (target leakage).
  2. [B5 / R2 #24]      MỌI config dùng CÙNG một loss: `w_stage = 0.0`. Bản cũ để baseline
     giữ w_stage=0.2 mặc định còn proposed đặt 0.0 -> confound toàn bộ Bảng 2.
  3. [A6 / R2 #10,#12]  LƯU FILE DỰ ĐOÁN CHO MỌI CONFIG, tên file có `cfg_name`. Bản cũ chỉ
     lưu cho 2 config và bị GHI ĐÈ -> Bảng 6 báo cáo nhầm cấu hình.
  4. [A5 / R2 #14]      Tham số VTOI ghi RIÊNG từng fold, không ghi đè.
  5. [R2 #9,#27]        Hỗ trợ `seed` -> chạy nhiều seed, perfold/checkpoint tách theo seed.
  6. [R2 #17]           `sampler_mode='bearing'` -> batch gom theo ổ bi, snapshot liên tiếp.

  + Bỏ hẳn 2 config dùng nhiệt độ (paper VTOI là vibration-only).
  + Bỏ `use_uncertainty` khỏi mọi config -> loss ĐỒNG NHẤT tuyệt đối. Khoảng dự đoán sẽ được
    tạo POST-HOC bằng split-conformal (`q1_v2_conformal.py`), mạnh hơn và không đụng loss.

CHẠY (an toàn với disconnect — tự bỏ qua phần đã xong nhờ perfold.csv):
    python scripts/phase9b_v2.py --tier 0 --seed 42
    python scripts/phase9b_v2.py --tier 1 --seed 42
    python scripts/phase9b_v2.py --tier 2 --seed 43        # (tier 2 = lặp tier 0 với seed khác)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Số worker của DataLoader: bám theo số CPU thực tế (Colab A100 thường 8-12 core; CPU-only 2 core).
_NUM_WORKERS = max(0, min(4, (os.cpu_count() or 2) - 1))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.env import setup_environment
from src.utils.logger import get_logger, section, Timer
from src.utils.paths import TABLES_DIR, PREDICTIONS_DIR
from src.train import train_model
from src.evaluate import evaluate_model
from src.lobo import make_folds, datasets_from_pack
from src.lobo_v2 import prepare_fold_v2
from src.vtoi import HC_COLS

# =========================================================================================
# TRỌNG SỐ LOSS — ĐỒNG NHẤT CHO MỌI CONFIG (sửa confound B5)
#   w_stage=0.0 ở CẢ HAI -> khác biệt duy nhất giữa proposed và baseline là ĐẦU VÀO conditioning.
# =========================================================================================
NO_AUX = dict(lambda1=0.0, w_stage=0.0)                       # không có loss phụ VTOI
LOW_AUX = dict(lambda1=0.1, w_stage=0.0)                      # có loss phụ VTOI (như proposed)
NO_SMOOTH = dict(lambda1=0.1, w_stage=0.0, w_rul_smooth=0.0, w_mtoi_smooth=0.0)
NO_MONO = dict(lambda1=0.1, w_stage=0.0, w_rul_mono=0.0, w_mtoi_mono=0.0)

HI_STATIC = ["E_norm", "C_norm", "VTOI"]                      # đề xuất
HI_TRAJ = ["E_norm", "C_norm", "VTOI", "VTOI_vel", "VTOI_acc"]


def _cfg(hi_cols=None, enc="transformer", loss=LOW_AUX, target="VTOI"):
    """Rút gọn khai báo config. use_hi tự suy ra từ hi_cols."""
    return dict(vib_encoder=enc, use_hi=bool(hi_cols), hi_cols=hi_cols,
                mtoi_target=target, loss_override=loss, use_temp=False, fusion="gated")


# =========================================================================================
# TIER 0 — SO SÁNH CHÍNH + ABLATION GIAI THỪA 2x2
# =========================================================================================
TIER0 = [
    # --- Đề xuất ---
    ("vtoi_static",         _cfg(HI_STATIC)),                       # conditioning ON,  aux ON
    ("vtoi_traj",           _cfg(HI_TRAJ)),
    # --- Baseline hiện đại, vib-only, KHÔNG conditioning, CÙNG loss ---
    ("transformer_vib",     _cfg(None, "transformer",      NO_AUX)),  # cond OFF, aux OFF
    ("tcn_vib",             _cfg(None, "tcn",              NO_AUX)),
    ("tcn_transformer_vib", _cfg(None, "tcn_transformer",  NO_AUX)),
    ("cnn_bilstm_attn_vib", _cfg(None, "cnn_bilstm_attn",  NO_AUX)),
    # --- Ablation: ô còn thiếu của bảng 2x2 (R2 #2) ---
    ("abl_no_idxhead",      _cfg(None,      loss=LOW_AUX)),          # cond OFF, aux ON
    ("abl_cond_noaux",      _cfg(HI_STATIC, loss=NO_AUX)),           # cond ON,  aux OFF  << MỚI
    # --- Ablation thành phần & prior ---
    ("abl_E_only",          _cfg(["E_norm"], target="E_norm")),
    ("abl_no_smooth",       _cfg(HI_STATIC, loss=NO_SMOOTH)),
    ("abl_no_mono",         _cfg(HI_STATIC, loss=NO_MONO)),
]

# =========================================================================================
# TIER 1 — CONTROL BATTERY (Reviewer 2, ý #25) — mục quan trọng nhất về khoa học
#   Mọi dòng dùng CÙNG encoder / CÙNG loss / CÙNG fold; khác DUY NHẤT vector conditioning.
# =========================================================================================
TIER1 = [
    ("ctl_scalar",   _cfg(["VTOI"])),                 # #1,#26,R1.5 — chỉ scalar VTOI
    ("ctl_EC",       _cfg(["E_norm", "C_norm"])),     # #25 — nối thẳng thành phần, không fusion
    ("ctl_hc10",     _cfg(HC_COLS)),                  # #25 — 10 feature handcrafted thô
    ("ctl_random",   _cfg(["VTOI_rand"])),            # #25 — trọng số NGẪU NHIÊN  << control âm
    ("ctl_shuffled", _cfg(["VTOI_shuf"])),            # #25 — xáo trộn thời gian   << control âm
    ("ctl_elapsed",  _cfg(["elapsed_norm"])),         # #25 — chỉ là "đồng hồ"?
    ("ctl_lifefrac", _cfg(["Deg_oracle"])),           # #25 — ORACLE, trần trên
    ("ctl_nodeg",    _cfg(["VTOI_mono"])),            # #6  — index KHÔNG dùng nhãn deg
]

TIERS = {0: TIER0, 1: TIER1, 2: TIER0}                # tier 2 = lặp tier 0 với seed khác

METRIC_KEYS = ["rul_mae", "rul_rmse", "rul_r2",
               "rul_mae_hours", "rul_rmse_hours", "rul_phm_score", "rul_asym_score",
               "first_warning", "lead_time", "false_alarm_rate",
               "mtoi_rmse", "mtoi_spearman", "mtoi_spearman_signed", "mtoi_spearman_vs_deg",
               "mtoi_monotonicity", "stage_acc", "stage_macro_f1", "stage_bal_acc"]

LOCAL_CKPT = Path("/content/lobo_v2_ckpts")               # checkpoint tạm (nhanh, mất khi reset)
CKPT_KEEP_DIR = Path(__file__).resolve().parents[1] / "results" / "checkpoints" / "v2_keep"
PERSIST_CKPT_CONFIGS = {"vtoi_static", "transformer_vib"}  # chỉ giữ 2 config -> ~130 MB tổng


def perfold_path(dataset, seed):
    return TABLES_DIR / "v2" / f"lobo_v2_{dataset}_seed{seed}_perfold.csv"


# =========================================================================================
# NHẬT KÝ CHẠY (append-only, trên Drive) — ghi nhận MỌI lần chạy kể cả LỖI.
#   * journal.jsonl : 1 dòng JSON / lần train (thời lượng, metric chính, lỗi nếu có)
#   * heartbeat.json: vị trí HIỆN TẠI + ETA -> status_v2.py đọc để báo tiến độ
# Nhờ file này, sau khi disconnect ta biết CHÍNH XÁC đã chạy gì, mất bao lâu, hỏng ở đâu.
# =========================================================================================
JOURNAL_DIR = Path(__file__).resolve().parents[1] / "results" / "logs" / "v2"


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def journal(rec):
    """Ghi 1 bản ghi vào journal.jsonl (append, flush ngay -> sống sót disconnect đột ngột)."""
    try:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _now(), **rec}
        with open(JOURNAL_DIR / "journal.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass                                              # nhật ký hỏng KHÔNG được làm hỏng run


def heartbeat(**kw):
    """Ghi trạng thái hiện tại (ghi đè) — status_v2.py đọc để biết đang chạy tới đâu."""
    try:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        (JOURNAL_DIR / "heartbeat.json").write_text(
            json.dumps({"ts": _now(), **kw}, ensure_ascii=False, default=str, indent=2))
    except Exception:
        pass


def train_eval_one(dataset, pack, cfg_name, cfg, device, epochs, subsample,
                   seed, sampler_mode, logger, rul_col="RUL_capped", force=False):
    """Train + eval 1 config trên 1 fold ĐÃ NẠP. Trả về 1 dòng metric."""
    ho = pack["holdout"]
    hi_cols = cfg.get("hi_cols")
    ds = datasets_from_pack(pack, mtoi_target=cfg["mtoi_target"],
                            train_subsample=subsample, rul_col=rul_col, hi_cols=hi_cols)
    run_name = f"v2_{dataset}_s{seed}_{cfg_name}_{ho}"

    if force:                                          # xoá checkpoint cũ để train LẠI thật sự
        for suf in ("_last.pt", "_best.pt"):
            p = LOCAL_CKPT / f"{run_name}{suf}"
            if p.exists():
                p.unlink()

    out = train_model(
        {"train": ds["train"], "val": ds["val"], "test": ds["test"]},
        run_name=run_name, device=device,
        use_temp=False, fusion="gated",
        vib_encoder=cfg["vib_encoder"], loss_weights=cfg["loss_override"],
        use_uncertainty=False,                         # loss ĐỒNG NHẤT — PI làm post-hoc
        use_hi=cfg["use_hi"], hi_dim=(len(hi_cols) if hi_cols else 4),
        epochs=epochs, batch_size=256, patience=5, num_workers=_NUM_WORKERS, resume=True,
        log=logger, sampler_mode=sampler_mode, seed=seed,
    )
    res = evaluate_model(out["model"], ds["test"], device=device, run_name=run_name, save=False)

    # --- LƯU DỰ ĐOÁN CHO MỌI CONFIG, tên file CÓ cfg_name (sửa lỗi A6) ---
    if res.get("hour_df") is not None:
        d = PREDICTIONS_DIR / "v2"
        d.mkdir(parents=True, exist_ok=True)
        res["hour_df"].to_csv(d / f"lobo_v2_{dataset}_s{seed}_{ho}_{cfg_name}.csv", index=False)

    # --- Giữ CHECKPOINT của config đề xuất lên Drive (~2 MB/fold) để chạy attribution (R2 #19) ---
    if cfg_name in PERSIST_CKPT_CONFIGS:
        try:
            import shutil
            dst = CKPT_KEEP_DIR
            dst.mkdir(parents=True, exist_ok=True)
            src = LOCAL_CKPT / f"{run_name}_best.pt"
            if src.exists():
                shutil.copy2(src, dst / f"{run_name}_best.pt")
        except Exception as e:
            logger.info(f"  (không lưu được checkpoint lên Drive: {e})")

    m = pack.get("vtoi_meta", {})
    row = {"dataset": dataset, "config": cfg_name, "holdout": ho, "seed": seed,
           "n_train_bearings": ds["n_train_bearings"], "n_test_windows": len(ds["test"]),
           "hi_cols": "|".join(hi_cols) if hi_cols else "",
           "vtoi_a": m.get("a"), "vtoi_b": m.get("b"), "vtoi_fitness": m.get("fitness"),
           "vtoi_sat_frac_test": m.get("holdout_sat_frac")}
    row.update({k: res["metrics"].get(k) for k in METRIC_KEYS})
    return row


def summarize(df):
    """mean ± sample std QUA CÁC FOLD cho mỗi (dataset, config, seed) + median/IQR/worst (R2 #28)."""
    # Nếu MỌI fold đều lỗi thì df rỗng và không có cột nào -> groupby sẽ ném KeyError('dataset')
    # che mất nguyên nhân thật (đã ghi trong journal). Trả về bảng rỗng để lỗi gốc hiện ra.
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for (dsn, cfg, sd), g in df.groupby(["dataset", "config", "seed"]):
        r = {"dataset": dsn, "config": cfg, "seed": sd, "n_folds": len(g)}
        for k in METRIC_KEYS:
            v = pd.to_numeric(g[k], errors="coerce").dropna()
            if not len(v):
                continue
            r[f"{k}_mean"] = round(float(v.mean()), 4)
            r[f"{k}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0
            if k in ("rul_mae", "rul_mae_hours"):      # R2 #28: median/IQR/worst là headline
                r[f"{k}_median"] = round(float(v.median()), 4)
                r[f"{k}_q25"] = round(float(v.quantile(0.25)), 4)
                r[f"{k}_q75"] = round(float(v.quantile(0.75)), 4)
                r[f"{k}_worst"] = round(float(v.max()), 4)
        rows.append(r)
    return pd.DataFrame(rows)


def main(tier=0, seed=42, datasets=("pronostia", "xjtu_sy"), configs=None,
         epochs=20, max_folds=None, force=False, sampler_mode="bearing",
         train_subsample=None, rul_target="RUL_capped"):
    device = setup_environment(seed=seed, do_mount=False)
    logger = get_logger(f"phase9b_v2_t{tier}_s{seed}")
    section(f"PHASE 9B-v2 — LOBO LEAKAGE-FREE | tier={tier} seed={seed} "
            f"sampler={sampler_mode}", logger)

    if train_subsample is None:
        train_subsample = {"xjtu_sy": 4}               # PHẢI công bố trong paper (hiện chưa nhắc)
    logger.info(f"train_subsample={train_subsample} (lấy thưa cửa sổ TRAIN — công bố trong paper)")

    LOCAL_CKPT.mkdir(parents=True, exist_ok=True)
    import src.train as _t
    _t.CHECKPOINTS_DIR = LOCAL_CKPT                     # không ghi hàng chục GB lên Drive
    (TABLES_DIR / "v2").mkdir(parents=True, exist_ok=True)

    cfg_list = TIERS[tier]
    if configs:
        cfg_list = [c for c in cfg_list if c[0] in configs]
    logger.info(f"Sẽ chạy {len(cfg_list)} config: {[c[0] for c in cfg_list]}")

    for dataset in datasets:
        folds, _ = make_folds(dataset)
        if max_folds:
            folds = folds[:max_folds]

        pf = perfold_path(dataset, seed)
        if pf.exists() and not force:
            prev = pd.read_csv(pf)
            done = set(zip(prev["config"], prev["holdout"]))
            rows = prev.to_dict("records")
        else:
            done, rows = set(), []
        logger.info(f"[{dataset}] {len(folds)} fold × {len(cfg_list)} config | "
                    f"đã xong {len(done)} | file: {pf.name}")

        sub = train_subsample.get(dataset, 1) if isinstance(train_subsample, dict) else train_subsample

        # --- Đếm tổng việc còn lại của dataset này -> phục vụ ETA ---
        remaining = sum(1 for fold in folds for n, _ in cfg_list
                        if (n, fold["holdout"]) not in done or force)
        completed_now, durations = 0, []
        t_ds0 = time.time()
        journal({"event": "dataset_start", "tier": tier, "seed": seed, "dataset": dataset,
                 "n_folds": len(folds), "n_configs": len(cfg_list), "remaining": remaining})

        for fi, fold in enumerate(folds):
            ho = fold["holdout"]
            pending = [(n, k) for n, k in cfg_list if (n, ho) not in done or force]
            if not pending:
                continue
            # Nạp fold 1 LẦN + fit VTOI leakage-free 1 LẦN -> dùng chung cho mọi config.
            t_fold = time.time()
            try:
                pack = prepare_fold_v2(dataset, ho, fold["val"], fold["train"],
                                       seed=seed, logger=logger)
            except Exception as e:
                logger.info(f"  LỖI nạp fold {ho}: {type(e).__name__}: {e}")
                journal({"event": "fold_load_error", "tier": tier, "seed": seed,
                         "dataset": dataset, "holdout": ho, "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-1500:]})
                continue
            journal({"event": "fold_ready", "tier": tier, "seed": seed, "dataset": dataset,
                     "holdout": ho, "fold": f"{fi+1}/{len(folds)}",
                     "load_s": round(time.time() - t_fold, 1),
                     "vtoi_a": pack["vtoi_meta"]["a"], "vtoi_b": pack["vtoi_meta"]["b"],
                     "vtoi_fitness": pack["vtoi_meta"]["fitness"],
                     "sat_frac_test": pack["vtoi_meta"]["holdout_sat_frac"]})

            for cfg_name, cfg in pending:
                eta = (np.median(durations) * (remaining - completed_now)) if durations else None
                heartbeat(stage=f"tier{tier}", seed=seed, dataset=dataset,
                          fold=f"{fi+1}/{len(folds)}", holdout=ho, config=cfg_name,
                          done_this_run=completed_now, remaining=remaining - completed_now,
                          eta_hours=round(eta / 3600, 2) if eta else None,
                          elapsed_hours=round((time.time() - t_ds0) / 3600, 2))
                t0 = time.time()
                with Timer(f"[{dataset}] fold {fi+1}/{len(folds)} {ho} :: {cfg_name}", logger):
                    try:
                        rows = [r for r in rows
                                if not (r.get("config") == cfg_name and r.get("holdout") == ho)]
                        row = train_eval_one(dataset, pack, cfg_name, cfg, device,
                                             epochs, sub, seed, sampler_mode, logger,
                                             rul_col=rul_target, force=force)
                        rows.append(row)
                        dt = time.time() - t0
                        durations.append(dt); completed_now += 1
                        journal({"event": "run_ok", "tier": tier, "seed": seed, "dataset": dataset,
                                 "holdout": ho, "config": cfg_name, "duration_s": round(dt, 1),
                                 "rul_mae": row.get("rul_mae"),
                                 "rul_mae_hours": row.get("rul_mae_hours"),
                                 "rho_signed": row.get("mtoi_spearman_signed")})
                    except Exception as e:
                        logger.info(f"  LỖI fold {ho} {cfg_name}: {type(e).__name__}: {e}")
                        journal({"event": "run_error", "tier": tier, "seed": seed,
                                 "dataset": dataset, "holdout": ho, "config": cfg_name,
                                 "duration_s": round(time.time() - t0, 1),
                                 "error": f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()[-1500:]})
                        continue
                pd.DataFrame(rows).to_csv(pf, index=False)   # ghi tăng tiến -> sống sót disconnect
                if durations:
                    left = remaining - completed_now
                    logger.info(f"    tiến độ {completed_now}/{remaining} | "
                                f"trung vị {np.median(durations)/60:.1f} ph/run | "
                                f"còn ~{np.median(durations)*left/3600:.1f} h")
            del pack

        df = pd.DataFrame(rows)
        summ = summarize(df)
        sp = TABLES_DIR / "v2" / f"lobo_v2_{dataset}_seed{seed}_summary.csv"
        summ.to_csv(sp, index=False)
        logger.info(f"[{dataset}] Ghi {sp}")
        if len(summ):
            cols = [c for c in ["config", "n_folds", "rul_mae_hours_mean", "rul_mae_hours_median",
                                "rul_mae_hours_worst", "rul_mae_mean", "mtoi_spearman_signed_mean"]
                    if c in summ.columns]
            print(f"\n===== LOBO v2 SUMMARY: {dataset} (seed {seed}, tier {tier}) =====")
            print(summ.sort_values("rul_mae_hours_mean")[cols].to_string(index=False))

    heartbeat(stage=f"tier{tier}", seed=seed, dataset="-", config="-", status="HOÀN TẤT")
    journal({"event": "tier_done", "tier": tier, "seed": seed, "datasets": list(datasets)})
    logger.info(f"HOÀN TẤT tier {tier} seed {seed}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--datasets", type=str, default="pronostia,xjtu_sy")
    ap.add_argument("--configs", type=str, default="")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sampler", type=str, default="bearing", choices=["bearing", "shuffle"])
    a = ap.parse_args()
    main(tier=a.tier, seed=a.seed, epochs=a.epochs,
         datasets=tuple(x for x in a.datasets.split(",") if x),
         configs=[x for x in a.configs.split(",") if x] or None,
         max_folds=a.max_folds or None, force=a.force, sampler_mode=a.sampler)
