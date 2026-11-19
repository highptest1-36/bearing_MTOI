# -*- coding: utf-8 -*-
"""
smoke_test_rul.py — CỔNG SMOKE-TEST trước khi chạy ma trận RUL-centric nhiều giờ GPU.

Kiểm 4 bất biến (mỗi cái 1 dòng PASS/FAIL):
  [1] Alias loss: loss_override={'lambda1':0} vẫn TẮT nhánh MTOI (w_mtoi==0) — giữ ablation w/o-MTOI.
  [2] Quantile ĐƠN ĐIỆU: model(use_uncertainty=True) cho q10<=q50<=q90 trên đầu vào ngẫu nhiên.
  [3] Dedup LOBO: make_folds() -> PRONOSTIA 17 fold, XJTU 15 fold, KHÔNG fold nào leak (base trùng).
  [4] End-to-end 1 fold XJTU (proposed + uncertainty, 2 epoch, 2 bearing nhỏ): evaluate trả về
      rul_mae_hours / rul_phm_score / rul_picp (đơn vị 'hours'), và quantile không chéo.

Chạy:  python scripts/smoke_test_rul.py
Thoát !=0 nếu BẤT KỲ kiểm tra nào fail (để dùng làm cổng trước ma trận).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.env import setup_environment
from src.utils.logger import get_logger, section
from src.utils.paths import proc_dir_for

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def test_alias():
    from src.losses import resolve_weights
    w0 = resolve_weights({"lambda1": 0.0})
    w1 = resolve_weights(None)
    check("1. alias lambda1->w_mtoi (ablation w/o-MTOI)",
          w0["w_mtoi"] == 0.0 and w1["w_mtoi"] > 0.0,
          f"w_mtoi(override)={w0['w_mtoi']} vs default={w1['w_mtoi']}")


def test_quantile_monotone(device):
    from src.models.mtoi_model import MTOIModel
    m = MTOIModel(use_temp=False, use_uncertainty=True, vib_encoder="cnn").to(device).eval()
    with torch.no_grad():
        vib = torch.randn(64, 2, 4096, device=device)
        out = m(vib)
    q = out["rul_quantiles"].cpu().numpy()
    ok = bool(np.all(q[:, 0] <= q[:, 1] + 1e-6) and np.all(q[:, 1] <= q[:, 2] + 1e-6))
    inrange = bool(np.all(q >= 0) and np.all(q <= 1))
    check("2. quantile đơn điệu q10<=q50<=q90 in (0,1)", ok and inrange,
          f"min={q.min():.3f} max={q.max():.3f}")


def test_new_inputs(device):
    """Kiểm HI-input + temp masking: với has_temp=0 thì RUL phải = đúng nhánh vib-only."""
    import torch
    from src.models.mtoi_model import MTOIModel
    m = MTOIModel(use_temp=True, use_hi=True, use_uncertainty=True, vib_encoder="cnn").to(device).eval()
    with torch.no_grad():
        vib = torch.randn(32, 2, 4096, device=device)
        temp = torch.randn(32, 4, device=device)
        hi = torch.rand(32, 4, device=device)
        ht0 = torch.zeros(32, device=device)                  # KHÔNG có nhiệt độ
        out0 = m(vib, temp, hi, ht0)                          # masking -> phải = vib-only
        out_v = m(vib, None, hi)                              # nhánh vib-only thật
    same = torch.allclose(out0["rul"], out_v["rul"], atol=1e-5)
    rng = bool((out0["rul"] >= 0).all() and (out0["rul"] <= 1).all())
    check("5. temp masking: has_temp=0 -> RUL = vib-only", same and rng,
          f"max|Δrul|={(out0['rul']-out_v['rul']).abs().max().item():.2e}")
    q = out0["rul_quantiles"]
    check("6. HI-input + quantile vẫn đơn điệu", bool((q[:,0]<=q[:,1]+1e-6).all() and (q[:,1]<=q[:,2]+1e-6).all()))


def test_dedup():
    from src.lobo import make_folds, physical_base
    ok_all = True
    for ds, exp in (("pronostia", 17), ("xjtu_sy", 15)):
        folds, _ = make_folds(ds)
        n_ok = len(folds) == exp
        # không fold nào có base(holdout) xuất hiện trong train/val
        leak = any(physical_base(f["holdout"]) in
                   {physical_base(x) for x in f["train"] + f["val"]} for f in folds)
        ok_all = ok_all and n_ok and not leak
        check(f"3. dedup {ds}: {len(folds)} fold (mong {exp}), leak={leak}", n_ok and not leak)
    return ok_all


def _smallest_xjtu_bearings(k=3):
    man = pd.read_csv(proc_dir_for("xjtu_sy") / "_manifest.csv")
    man = man.sort_values("n_windows").reset_index(drop=True)
    return man["bearing"].tolist()[:k]


def test_end_to_end(device):
    from src.lobo import prepare_fold, datasets_from_pack
    import src.train as _t
    from src.train import train_model
    from src.evaluate import evaluate_model

    # Checkpoint smoke -> LOCAL (không ghi Drive).
    _t.CHECKPOINTS_DIR = Path("/content/smoke_ckpts"); _t.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    small = _smallest_xjtu_bearings(k=4)
    holdout, val, train = small[0], [small[1]], small[2:4]   # 1 test + 1 val + 2 train (nhỏ nhất -> nhanh)
    logger = get_logger("smoke")
    HI = ["E_norm", "C_norm", "MTOI_woG", "MTOI_woG_vel", "MTOI_woG_acc"]   # quỹ đạo MTOI vib-only (5)
    pack = prepare_fold("xjtu_sy", holdout, val, train, logger=logger)
    ds = datasets_from_pack(pack, mtoi_target="MTOI_woG", train_subsample=8,
                            rul_col="RUL_capped", hi_cols=HI)
    # Kiểm 1 mẫu có đủ hi (5 chiều: E,C,MTOI,vel,acc) + has_temp.
    s0 = ds["train"][0]
    check("4z. mẫu có 'hi'[5]=quỹ đạo MTOI + 'has_temp'",
          ("hi" in s0 and len(s0["hi"]) == 5 and "has_temp" in s0),
          f"hi={s0.get('hi')}")

    out = train_model({"train": ds["train"], "val": ds["val"], "test": ds["test"]},
                      run_name="smoke_xjtu", device=device,
                      use_temp=False, fusion="gated", vib_encoder="transformer",
                      loss_weights={"lambda1": 0.1, "w_stage": 0.0},
                      use_uncertainty=True, use_hi=True, hi_dim=len(HI),
                      epochs=2, batch_size=128, patience=2, num_workers=2, resume=False, log=logger)
    res = evaluate_model(out["model"], ds["test"], device=device, run_name="smoke_xjtu", save=False)
    M = res["metrics"]; hd = res["hour_df"]

    check("4a. eval_unit == 'hours'", M.get("eval_unit") == "hours", str(M.get("eval_unit")))
    check("4b. rul_mae_hours có giá trị", M.get("rul_mae_hours") is not None,
          f"MAE_h={M.get('rul_mae_hours')}")
    check("4c. rul_phm_score + asym có giá trị",
          M.get("rul_phm_score") is not None and M.get("rul_asym_score") is not None,
          f"PHM={M.get('rul_phm_score')}, Asym={M.get('rul_asym_score')}")
    check("4d. PICP/MPIW có giá trị (uncertainty)",
          M.get("rul_picp") is not None and M.get("rul_mpiw") is not None,
          f"PICP={M.get('rul_picp')}, MPIW={M.get('rul_mpiw')}")
    if {"rul_q10", "rul_q90"}.issubset(hd.columns):
        check("4e. quantile cấp giờ không chéo",
              bool((hd["rul_q10"] <= hd["rul_q90"] + 1e-6).all()))
    # Sanity vòng đời: life_hours khớp H_fail*60/3600.
    if "life_hours" in hd.columns and "H_fail" in hd.columns:
        lh, hf = float(hd["life_hours"].iloc[0]), float(hd["H_fail"].iloc[0])
        check("4f. life_hours khớp H_fail*60s/3600", abs(lh - hf * 60.0 / 3600.0) < 1e-3,
              f"life={lh:.3f}h, H_fail={hf:.0f} snap")
    print(f"    -> Test bearing '{holdout}': RUL MAE={M['rul_mae']:.4f} (norm) | "
          f"MAE={M['rul_mae_hours']:.3f}h RMSE={M['rul_rmse_hours']:.3f}h | "
          f"PICP={M['rul_picp']:.3f} MPIW={M['rul_mpiw']:.3f}")


def main():
    device = setup_environment(seed=42, do_mount=False)
    section(f"SMOKE TEST RUL-CENTRIC (device={device})", get_logger("smoke"))
    test_alias()
    test_quantile_monotone(device)
    test_new_inputs(device)
    test_dedup()
    test_end_to_end(device)
    print("-" * 64)
    if FAILS:
        print(f"❌ SMOKE FAIL ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("✅ SMOKE PASS — code RUL-centric sẵn sàng cho ma trận huấn luyện (KHỐI B).")
    sys.exit(0)


if __name__ == "__main__":
    main()
