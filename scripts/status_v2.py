# -*- coding: utf-8 -*-
"""
status_v2.py — BÁO CÁO TIẾN ĐỘ v2. Chạy ĐẦU TIÊN sau mỗi lần reconnect Colab.

Trả lời đúng 3 câu hỏi: đã xong bao nhiêu? đang thiếu gì? lệnh tiếp theo là gì?
Chạy: python scripts/status_v2.py     (hoặc: bash run_v2.sh status)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DRIVE = ROOT.parent

TAB = ROOT / "results" / "tables" / "v2"
PAR = ROOT / "results" / "tables" / "v2_vtoi_params"
PRED = ROOT / "results" / "predictions" / "v2"
CKPT = ROOT / "results" / "checkpoints" / "v2_keep"
LOGS = ROOT / "results" / "logs" / "v2"
STATE = DRIVE / ".v2_state"

EXPECTED_FOLDS = {"pronostia": 17, "xjtu_sy": 15}
TIER0_CFGS = ["vtoi_static", "vtoi_traj", "transformer_vib", "tcn_vib", "tcn_transformer_vib",
              "cnn_bilstm_attn_vib", "abl_no_idxhead", "abl_cond_noaux", "abl_E_only",
              "abl_no_smooth", "abl_no_mono"]
TIER1_CFGS = ["ctl_scalar", "ctl_EC", "ctl_hc10", "ctl_random", "ctl_shuffled",
              "ctl_elapsed", "ctl_lifefrac", "ctl_nodeg"]
# Tier 2 chỉ lặp config CỐT LÕI (khớp TIER2_CONFIGS trong run_v2.sh)
TIER2_CFGS = ["vtoi_static", "transformer_vib", "tcn_vib", "tcn_transformer_vib",
              "cnn_bilstm_attn_vib", "abl_no_idxhead", "abl_cond_noaux", "abl_E_only"]
TIER3_TABLES = ["weights_distribution.csv", "weight_sweep.csv", "hi_quality.csv",
                "vtoi_range.csv", "early_warning_v2.csv", "onset_sensitivity.csv",
                "conformal.csv", "deployable_hours.csv"]
FINAL_TABLES = ["main_results.csv", "wilcoxon.csv", "factorial_2x2.csv",
                "controls.csv", "classical.csv", "seed_variability.csv", "per_bearing_all.csv"]


def bar(done, total, width=28):
    if total <= 0:
        return "[" + "?" * width + "]"
    n = int(width * done / total)
    return "[" + "#" * n + "." * (width - n) + f"] {done}/{total}"


def load_all():
    fs = sorted(TAB.glob("lobo_v2_*_seed*_perfold.csv"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def main():
    print("=" * 78)
    print(" TIẾN ĐỘ THÍ NGHIỆM v2 — Access-2026-31251 ".center(78, "="))
    print("=" * 78)

    # ---- môi trường ----
    try:
        import torch
        gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "KHÔNG CÓ GPU  <-- Runtime > Change runtime type > A100")
    except Exception:
        gpu = "chưa cài torch"
    print(f"\n[MÔI TRƯỜNG] {gpu}")

    df = load_all()
    if df.empty:
        print("\n[TRẠNG THÁI] Chưa có kết quả v2 nào.")
        print("\n>>> LỆNH TIẾP THEO:  bash run_v2.sh tier0")
        return

    seeds = sorted(df.seed.unique())
    print(f"[DỮ LIỆU]    {len(df)} dòng | seed = {seeds}")

    # ---- tiến độ từng tier ----
    print("\n--- TIẾN ĐỘ ---")
    todo = []

    for tier, cfgs, seed_list in [("Tier 0 (chính+ablation)", TIER0_CFGS, [42]),
                                  ("Tier 1 (control)", TIER1_CFGS, [42])]:
        tot = done = 0
        for s in seed_list:
            for ds, nf in EXPECTED_FOLDS.items():
                tot += nf * len(cfgs)
                g = df[(df.seed == s) & (df.dataset == ds) & (df.config.isin(cfgs))]
                done += len(g.drop_duplicates(["config", "holdout"]))
        print(f"  {tier:28s} {bar(done, tot)}")
        if done < tot:
            todo.append("tier0" if "Tier 0" in tier else "tier1")

    tot = done = 0
    extra_seeds = [x for x in seeds if x != 42] or [43, 44]
    for s in extra_seeds:
        for ds, nf in EXPECTED_FOLDS.items():
            tot += nf * len(TIER2_CFGS)
            g = df[(df.seed == s) & (df.dataset == ds) & (df.config.isin(TIER2_CFGS))]
            done += len(g.drop_duplicates(["config", "holdout"]))
    print(f"  {f'Tier 2 ({len(extra_seeds)} seed bổ sung)':28s} {bar(done, tot)}")
    if done < tot:
        todo.append("tier2")

    n3 = sum((TAB / t).exists() for t in TIER3_TABLES)
    print(f"  {'Tier 3 (phân tích CPU)':28s} {bar(n3, len(TIER3_TABLES))}")
    if n3 < len(TIER3_TABLES):
        todo.append("tier3")

    nck = len(list(CKPT.glob("*vtoi_static*_best.pt"))) if CKPT.exists() else 0
    natt = 1 if (TAB / "attribution.csv").exists() else 0
    print(f"  {'Attribution (R2 #19)':28s} {bar(natt, 1)}  (checkpoint đã lưu: {nck})")
    if not natt:
        todo.append("attrib")

    nf = sum((TAB / t).exists() for t in FINAL_TABLES)
    print(f"  {'Bảng cuối (analyze)':28s} {bar(nf, len(FINAL_TABLES))}")
    if nf < len(FINAL_TABLES):
        todo.append("analyze")

    # ---- kết quả sơ bộ ----
    d42 = df[(df.seed == 42) & (df.config.isin(["vtoi_static", "transformer_vib",
                                                "abl_no_idxhead", "abl_cond_noaux",
                                                "ctl_random", "ctl_shuffled", "ctl_elapsed"]))]
    if len(d42):
        print("\n--- KẾT QUẢ SƠ BỘ (seed 42, RUL MAE chuẩn hoá, thấp = tốt) ---")
        p = (d42.groupby(["dataset", "config"])
                .agg(n=("holdout", "count"), mae=("rul_mae", "mean"))
                .round(4).reset_index().sort_values(["dataset", "mae"]))
        print(p.to_string(index=False))

    # ---- tham số VTOI ----
    pfs = sorted(PAR.glob("params_*seed42*.csv"))
    if pfs:
        w = pd.concat([pd.read_csv(f) for f in pfs], ignore_index=True)
        print("\n--- TRỌNG SỐ VTOI (leakage-free, theo fold) ---")
        for ds, g in w.groupby("dataset"):
            print(f"  {ds:12s} a: median {g.a.median():.3f} "
                  f"[{g.a.quantile(.25):.3f}, {g.a.quantile(.75):.3f}] "
                  f"range [{g.a.min():.3f}, {g.a.max():.3f}] | n={len(g)} fold | "
                  f"bão hoà test {g.holdout_sat_frac.mean():.3f}")

    # ---- nhật ký chạy: tốc độ, ETA, lỗi ----
    jl = LOGS / "journal.jsonl"
    if jl.exists():
        recs = []
        for line in jl.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
        ok = [r for r in recs if r.get("event") == "run_ok" and r.get("duration_s")]
        err = [r for r in recs if r.get("event") in ("run_error", "fold_load_error")]
        if ok:
            d = np.array([r["duration_s"] for r in ok], float)
            print(f"\n--- NHẬT KÝ CHẠY ({len(ok)} lần train thành công) ---")
            print(f"  thời lượng/run: trung vị {np.median(d)/60:.1f} ph | "
                  f"p90 {np.percentile(d, 90)/60:.1f} ph | tổng GPU đã dùng {d.sum()/3600:.1f} h")
            # ETA cho phần còn lại (dùng trung vị đã đo được, chính xác hơn ước lượng lý thuyết)
            med = float(np.median(d))
            est = {"tier0": 11 * 32, "tier1": 8 * 32, "tier2": len(extra_seeds) * 8 * 32}
            left = []
            for t in ("tier0", "tier1", "tier2"):
                if t in todo:
                    cfgs = {"tier0": TIER0_CFGS, "tier1": TIER1_CFGS, "tier2": TIER2_CFGS}[t]
                    sds = [42] if t != "tier2" else extra_seeds
                    n_tot = sum(len(cfgs) * nf for nf in EXPECTED_FOLDS.values()) * len(sds)
                    n_done = sum(len(df[(df.seed == s) & (df.dataset == ds) &
                                        (df.config.isin(cfgs))].drop_duplicates(["config", "holdout"]))
                                 for s in sds for ds in EXPECTED_FOLDS)
                    left.append((t, max(n_tot - n_done, 0)))
            if left:
                print("  ước tính còn lại (dựa trên tốc độ THỰC ĐO):")
                for t, n in left:
                    print(f"    {t:6s}: {n:4d} run  ≈ {n*med/3600:5.1f} h GPU")
                print(f"    {'TỔNG':6s}: {sum(n for _, n in left):4d} run  "
                      f"≈ {sum(n for _, n in left)*med/3600:5.1f} h GPU")
        if err:
            print(f"\n  ⚠️  {len(err)} lần chạy LỖI — 3 lỗi gần nhất:")
            for r in err[-3:]:
                print(f"    [{r.get('ts')}] {r.get('dataset')}/{r.get('holdout')}/"
                      f"{r.get('config')}: {str(r.get('error'))[:90]}")

    hb = LOGS / "heartbeat.json"
    if hb.exists():
        try:
            h = json.loads(hb.read_text())
            print(f"\n--- VỊ TRÍ LẦN CHẠY GẦN NHẤT ({h.get('ts')}) ---")
            print(f"  {h.get('stage')} seed{h.get('seed')} | {h.get('dataset')} "
                  f"fold {h.get('fold')} {h.get('holdout')} :: {h.get('config')} "
                  f"| {h.get('status', 'đang chạy')}")
            if h.get("eta_hours"):
                print(f"  ETA lúc đó: ~{h['eta_hours']} h (đã chạy {h.get('elapsed_hours')} h)")
        except Exception:
            pass

    npred = len(list(PRED.glob("*.csv"))) if PRED.exists() else 0
    print(f"\n[FILE] dự đoán v2: {npred} | tham số fold: {len(list(PAR.glob('params_*.csv')))} "
          f"| bảng v2: {len(list(TAB.glob('*.csv')))}")

    # ---- lệnh tiếp theo ----
    order = ["tier0", "tier1", "tier3", "attrib", "tier2", "analyze"]
    nxt = next((t for t in order if t in todo), None)
    print("\n" + "=" * 78)
    if nxt is None:
        print(">>> TẤT CẢ ĐÃ XONG. Bước tiếp: điền số vào RESPONSE_TO_REVIEWERS_v2.md và sửa .tex")
    else:
        print(f">>> LỆNH TIẾP THEO:   bash run_v2.sh {nxt}")
        if nxt == "tier2":
            print("    (chạy `bash run_v2.sh gate` TRƯỚC — đừng đầu tư tier2 nếu cổng KHÔNG ĐẠT)")
    print("=" * 78)


if __name__ == "__main__":
    main()
