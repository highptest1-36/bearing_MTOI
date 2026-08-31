# -*- coding: utf-8 -*-
"""
q1_v2_attribution.py — ĐỊNH LƯỢNG mức độ RUL PHỤ THUỘC vào chỉ số VTOI (Reviewer 2, ý #19).

REVIEWER NÓI GÌ:
  "VTOI is a visible scalar trajectory, but the existence of a visible input does not make the
   final deep RUL predictor transparent... The paper does not quantify how much the RUL output
   depends on VTOI relative to the deep vibration embedding and does not provide attribution,
   perturbation, sensitivity, or feature-importance analysis."

SCRIPT NÀY TRẢ LỜI BẰNG 3 PHÂN TÍCH ĐỘC LẬP:
  (A) GRADIENT SENSITIVITY : |∂RUL/∂c| trung bình so với ||∂RUL/∂e_vib||
  (B) PERTURBATION         : đóng băng c ở mức baseline khoẻ / thay bằng c của bearing KHÁC
  (C) OCCLUSION            : xoá embedding rung (=0) và chỉ dự đoán từ c

Cần checkpoint đã lưu bởi `phase9b_v2.py` (config `vtoi_static`) tại results/checkpoints/v2_keep/.
Chạy: python scripts/q1_v2_attribution.py --seed 42
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
from src.lobo_v2 import prepare_fold_v2                        # noqa: E402
from src.utils.env import setup_environment                    # noqa: E402
from src.utils.logger import get_logger                        # noqa: E402

TAB = ROOT / "results" / "tables" / "v2"
CKPT = ROOT / "results" / "checkpoints" / "v2_keep"
HI_STATIC = ["E_norm", "C_norm", "VTOI"]
PROPOSED = "vtoi_static"


def _mae(model, loader, device, hi_transform=None, zero_emb=False):
    """MAE chuẩn hoá trên tập test, có thể biến đổi vector conditioning hoặc xoá embedding."""
    model.eval()
    errs = []
    with torch.no_grad():
        for b in loader:
            vib = b["vib"].to(device)
            hi = b["hi"].to(device)
            if hi_transform is not None:
                hi = hi_transform(hi)
            e = model.vib_encoder(vib)
            if zero_emb:
                e = torch.zeros_like(e)
            out = model.sigmoid(model.rul_head(torch.cat([e, hi], dim=1))).squeeze(-1)
            errs.append((out - b["rul"].to(device)).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def _gradients(model, loader, device, max_batches=8):
    """|∂RUL/∂c_j| theo từng chiều conditioning, và ||∂RUL/∂e_vib||_2."""
    model.eval()
    g_hi, g_emb = [], []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        vib, hi = b["vib"].to(device), b["hi"].to(device)
        e = model.vib_encoder(vib).detach().requires_grad_(True)
        h = hi.detach().requires_grad_(True)
        rul = model.sigmoid(model.rul_head(torch.cat([e, h], dim=1))).squeeze(-1).sum()
        ge, gh = torch.autograd.grad(rul, [e, h])
        g_hi.append(gh.abs().cpu().numpy())
        g_emb.append(ge.norm(dim=1).cpu().numpy())
    return np.concatenate(g_hi, 0), np.concatenate(g_emb, 0)


def run(seed=42, datasets=("pronostia", "xjtu_sy"), max_folds=None):
    device = setup_environment(seed=seed, do_mount=False)
    logger = get_logger("attribution")
    rows = []

    for ds in datasets:
        folds, _ = make_folds(ds)
        if max_folds:
            folds = folds[:max_folds]
        for f in folds:
            ho = f["holdout"]
            ck = CKPT / f"v2_{ds}_s{seed}_{PROPOSED}_{ho}_best.pt"
            if not ck.exists():
                logger.info(f"  [bỏ qua] thiếu checkpoint {ck.name}")
                continue

            pack = prepare_fold_v2(ds, ho, f["val"], f["train"], seed=seed, logger=logger)
            d = datasets_from_pack(pack, mtoi_target="VTOI", train_subsample=1,
                                   rul_col="RUL_capped", hi_cols=HI_STATIC)
            loader = torch.utils.data.DataLoader(d["test"], batch_size=256, shuffle=False)

            model = MTOIModel(use_temp=False, vib_encoder="transformer",
                              use_hi=True, hi_dim=len(HI_STATIC)).to(device)
            model.load_state_dict(torch.load(ck, map_location=device)["model"])

            base = _mae(model, loader, device)

            # (B) perturbation ---------------------------------------------------------
            hi_healthy = torch.tensor(
                d["test"]._hi[:max(3, len(d["test"]._hi) // 20)].mean(axis=0),
                dtype=torch.float32, device=device)                  # mức baseline khoẻ đầu đời
            frozen = _mae(model, loader, device, hi_transform=lambda h: hi_healthy.expand_as(h))
            shuffled = _mae(model, loader, device,
                            hi_transform=lambda h: h[torch.randperm(h.size(0), device=h.device)])

            # (C) occlusion ------------------------------------------------------------
            hi_only = _mae(model, loader, device, zero_emb=True)

            # (A) gradient -------------------------------------------------------------
            gh, ge = _gradients(model, loader, device)
            rows.append({
                "dataset": ds, "bearing": ho, "seed": seed,
                "mae_base": round(base, 4),
                "mae_hi_frozen_healthy": round(frozen, 4),
                "mae_hi_shuffled": round(shuffled, 4),
                "mae_embedding_occluded": round(hi_only, 4),
                "degradation_if_hi_frozen": round(frozen - base, 4),
                "degradation_if_emb_occluded": round(hi_only - base, 4),
                "grad_hi_mean": round(float(gh.mean()), 6),
                "grad_hi_E": round(float(gh[:, 0].mean()), 6),
                "grad_hi_C": round(float(gh[:, 1].mean()), 6),
                "grad_hi_VTOI": round(float(gh[:, 2].mean()), 6),
                "grad_emb_norm_mean": round(float(ge.mean()), 6),
                "grad_ratio_hi_over_emb": round(float(gh.sum(axis=1).mean() / (ge.mean() + 1e-12)), 4),
            })
            logger.info(f"  [{ds}/{ho}] base={base:.4f} frozen={frozen:.4f} "
                        f"occluded={hi_only:.4f} grad_ratio={rows[-1]['grad_ratio_hi_over_emb']:.3f}")
            del pack, model

    if not rows:
        print("Không có fold nào chạy được — kiểm tra checkpoint trong results/checkpoints/v2_keep/")
        return
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "attribution_per_bearing.csv", index=False)
    summ = (df.groupby("dataset")[["mae_base", "mae_hi_frozen_healthy", "mae_hi_shuffled",
                                   "mae_embedding_occluded", "grad_ratio_hi_over_emb",
                                   "grad_hi_E", "grad_hi_C", "grad_hi_VTOI"]]
              .agg(["mean", "median"]).round(4))
    summ.to_csv(TAB / "attribution.csv")
    print("\n=== ATTRIBUTION & SENSITIVITY (Reviewer 2, ý #19) ===")
    print(summ.to_string())
    print("\nCÁCH ĐỌC:")
    print("  mae_hi_frozen ≫ mae_base   -> RUL PHỤ THUỘC MẠNH vào chỉ số (ủng hộ tuyên bố)")
    print("  mae_hi_frozen ≈ mae_base   -> chỉ số gần như VÔ DỤNG với head (phải thừa nhận)")
    print("  mae_emb_occluded ≈ mae_base-> head chủ yếu dùng chỉ số, embedding ít quan trọng")
    print("  grad_hi_VTOI vs grad_hi_C  -> head dùng scalar hay dùng thành phần? (R2 #1, #26)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", type=str, default="pronostia,xjtu_sy")
    ap.add_argument("--max-folds", type=int, default=0)
    a = ap.parse_args()
    run(seed=a.seed, datasets=tuple(x for x in a.datasets.split(",") if x),
        max_folds=a.max_folds or None)
