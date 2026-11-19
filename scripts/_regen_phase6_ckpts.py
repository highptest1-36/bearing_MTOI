# -*- coding: utf-8 -*-
"""
_regen_phase6_ckpts.py — Tái tạo CHỈ 2 checkpoint mà Phase 8 (robustness) cần,
trong trường hợp checkpoint phase6 đã mất sau disconnect (results/checkpoints/ chỉ còn state.json).

Dùng ĐÚNG hàm run_one() của phase6 -> logic/tham số y hệt, KHÔNG ghi đè proposed_results.csv.
Idempotent: bỏ qua checkpoint đã tồn tại.

Chạy:  python scripts/_regen_phase6_ckpts.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_environment
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import CHECKPOINTS_DIR
from scripts.phase6_train_proposed import run_one

# 2 cấu hình Phase 8 nạp (khớp đúng dòng 85 & 90 trong phase6_train_proposed.py)
NEED = {
    "fusion_concat":      dict(use_temp=True, fusion="concat", vib_encoder="cnn",
                               mtoi_target="MTOI_learnable", loss_override={"lambda1": 0.0}),
    "proposed_learnable": dict(use_temp=True, fusion="gated",  vib_encoder="cnn",
                               mtoi_target="MTOI_learnable", use_uncertainty=True),
}


def main(config="configs/main_mendeley.yaml"):
    device = setup_environment(do_mount=False)   # Drive đã mount sẵn
    cfg = load_config(ROOT / config)
    logger = get_logger("regen_p6")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, params in NEED.items():
        ckpt = CHECKPOINTS_DIR / f"{name}_best.pt"
        if ckpt.exists():
            print(f"[regen] {name}: checkpoint đã có -> BỎ QUA ({ckpt})")
            continue
        print(f"[regen] {name}: thiếu checkpoint -> train lại (run_one của phase6)...")
        run_one(run_name=name, cfg=cfg, device=device, logger=logger, **params)
        ok = ckpt.exists()
        print(f"[regen] {name}: {'OK ' if ok else 'THẤT BẠI '}-> {ckpt} (exists={ok})")

    missing = [n for n in NEED if not (CHECKPOINTS_DIR / f"{n}_best.pt").exists()]
    if missing:
        print(f"[regen] ❌ Vẫn thiếu: {missing}")
        sys.exit(1)
    print("[regen] ✅ Đủ 2 checkpoint cho Phase 8.")


if __name__ == "__main__":
    main()
