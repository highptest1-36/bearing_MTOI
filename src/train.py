# -*- coding: utf-8 -*-
"""
train.py — HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT (có resume chống disconnect + AMP cho A100).

Tính năng quan trọng:
  - Mixed Precision (AMP): tận dụng A100 -> nhanh hơn ~2-3x, tiết kiệm VRAM.
  - CHECKPOINT mỗi epoch: lưu '<run>_last.pt' (epoch hiện tại) + '<run>_best.pt' (val tốt nhất).
        => Nếu Colab rớt giữa chừng, gọi lại train_model() sẽ TỰ NẠP '<run>_last.pt' và chạy tiếp.
  - In tiến trình từng epoch: loss tổng + từng thành phần (stage/mtoi/rul/smooth/mono) + val.
  - Early stopping theo val loss (kiên nhẫn `patience` epoch).
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.paths import CHECKPOINTS_DIR
from src.utils.logger import get_logger, section, format_duration
from src.models.mtoi_model import MTOIModel
from src.losses import total_loss, resolve_weights


def move_batch(batch, device):
    """Chuyển toàn bộ tensor trong batch sang device (GPU/CPU)."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def _eval_loss(model, loader, device, loss_weights, use_unc):
    """Tính loss trung bình trên 1 loader (val) — không cập nhật trọng số."""
    model.eval()                                          # chế độ đánh giá (tắt dropout/bn update)
    tot, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["vib"], batch.get("temp"), batch.get("hi"), batch.get("has_temp"))
        loss, _ = total_loss(out, batch, weights=loss_weights, use_uncertainty=use_unc)
        bs = batch["vib"].size(0)
        tot += loss.item() * bs                           # cộng dồn theo số mẫu
        n += bs
    return tot / max(n, 1)


def train_model(datasets, run_name="proposed_learnable", device="cuda",
                use_temp=True, fusion="gated", use_uncertainty=False, use_attention=False,
                vib_encoder="cnn", use_hi=False, hi_dim=4,
                loss_weights=None, epochs=40, batch_size=128, lr=1e-3, weight_decay=1e-4,
                patience=8, num_workers=2, resume=True, model_cfg=None, log=None,
                sampler_mode="shuffle", seed=42):
    """
    Huấn luyện 1 cấu hình model. 'datasets' = output của datasets.make_datasets().
    Trả về dict {model, history, best_ckpt, last_ckpt}.

    Các cờ phục vụ ablation/baseline:
      use_temp=False        -> biến thể vib-only (w/o temperature)
      fusion='concat'       -> fusion baseline
      loss_weights lambda1=0 -> biến thể w/o MTOI loss (chứng minh MTOI-guided có ích)

    sampler_mode (THÊM Ở v2 — trả lời Reviewer 2 ý #17):
      'shuffle'  -> DataLoader(shuffle=True) như bản cũ (giữ để so sánh, tái lập kết quả cũ).
      'bearing'  -> BearingContiguousBatchSampler: mỗi batch = 1 ổ bi, snapshot LIÊN TIẾP,
                    nên L_smooth/L_mono tính sai phân ĐÚNG NGHĨA thay vì trên chuỗi con thưa.
    """
    logger = log or get_logger("train")
    section(f"HUẤN LUYỆN: {run_name} (temp={use_temp}, fusion={fusion}, "
            f"unc={use_uncertainty}, sampler={sampler_mode})", logger)

    # --- DataLoader ---
    pin = (device == "cuda")                              # pin_memory tăng tốc nạp lên GPU
    train_sampler = None
    if sampler_mode == "bearing":
        from src.samplers import BearingContiguousBatchSampler
        train_sampler = BearingContiguousBatchSampler(datasets["train"],
                                                      batch_size=batch_size, seed=seed)
        train_loader = DataLoader(datasets["train"], batch_sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=pin)
        logger.info(f"Sampler gom-theo-ổ-bi: {train_sampler.describe()}")
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    logger.info(f"Số cửa sổ: train={len(datasets['train'])}, val={len(datasets['val'])}, "
                f"test={len(datasets['test'])}")

    # --- Khởi tạo model ---
    mcfg = model_cfg or {}
    # Cho phép cấu hình YAML (model.use_attention / model.use_uncertainty) GHI ĐÈ tham số mặc định,
    # để bật/tắt 2 cờ này từ configs mà không cần sửa code.
    use_attention = mcfg.get("use_attention", use_attention)
    use_uncertainty = mcfg.get("use_uncertainty", use_uncertainty)
    use_hi = mcfg.get("use_hi", use_hi)
    hi_dim = mcfg.get("hi_dim", hi_dim)
    model = MTOIModel(
        emb_dim=mcfg.get("emb_dim", 128), n_stages=mcfg.get("n_stages", 4),
        use_temp=use_temp, temp_dim=mcfg.get("temp_dim", 4), fusion=fusion,
        use_attention=use_attention, use_uncertainty=use_uncertainty,
        vib_encoder=vib_encoder, use_hi=use_hi, hi_dim=hi_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model có {n_params/1e6:.2f} triệu tham số. (encoder rung: {vib_encoder})")
    # Log trọng số loss ĐÃ GIẢI (RUL-primary: L_RUL là trục chính, không trọng số) — minh bạch & dễ kiểm.
    _rw = resolve_weights(loss_weights)
    logger.info("Loss RUL-primary | L_RUL=1.0(trần) + " +
                ", ".join(f"{k}={v}" for k, v in _rw.items()) + f" | uncertainty={use_uncertainty}")

    # --- Optimizer + AMP scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))   # mixed precision

    # --- Đường dẫn checkpoint ---
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    last_ckpt = CHECKPOINTS_DIR / f"{run_name}_last.pt"
    best_ckpt = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    # --- Trạng thái huấn luyện (để resume) ---
    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "components": []}

    # --- RESUME nếu có checkpoint last ---
    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        history = ckpt.get("history", history)
        logger.info(f"[RESUME] Nạp '{last_ckpt.name}' -> tiếp tục từ epoch {start_epoch} "
                    f"(best_val={best_val:.4f}).")

    if start_epoch >= epochs:                            # đã train đủ -> không cần train lại
        logger.info("Model đã huấn luyện đủ epoch trước đó -> bỏ qua.")
        # GUARD: chỉ nạp best nếu file tồn tại (best chỉ được lưu khi val cải thiện; nếu val toàn NaN
        # hoặc rớt mạng đúng epoch cuối thì best có thể chưa có -> giữ trọng số last đã nạp ở trên).
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        return {"model": model, "history": history, "best_ckpt": best_ckpt, "last_ckpt": last_ckpt}

    # --- VÒNG LẶP HUẤN LUYỆN ---
    for epoch in range(start_epoch, epochs):
        model.train()                                    # chế độ huấn luyện
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)               # đổi thứ tự block mỗi epoch
        t0 = time.time()
        run_loss, n_seen = 0.0, 0
        comp_accum = {}                                  # cộng dồn các thành phần loss

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)        # xoá gradient cũ
            # Tự động chọn float16 trên GPU (AMP) để nhanh & nhẹ.
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(batch["vib"], batch.get("temp"), batch.get("hi"), batch.get("has_temp"))
                loss, comps = total_loss(out, batch, weights=loss_weights,
                                         use_uncertainty=use_uncertainty)
            scaler.scale(loss).backward()                # lan truyền ngược (có scale)
            scaler.step(optimizer)                       # cập nhật trọng số
            scaler.update()                              # cập nhật hệ số scale

            bs = batch["vib"].size(0)
            run_loss += loss.item() * bs
            n_seen += bs
            for k, v in comps.items():                   # cộng dồn thành phần loss
                comp_accum[k] = comp_accum.get(k, 0.0) + v * bs

        train_loss = run_loss / max(n_seen, 1)           # loss train trung bình
        val_loss = _eval_loss(model, val_loader, device, loss_weights, use_uncertainty)
        comp_mean = {k: v / max(n_seen, 1) for k, v in comp_accum.items()}
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["components"].append(comp_mean)

        # In tiến trình epoch (đầy đủ thành phần để bạn theo dõi).
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
            f"RUL={comp_mean.get('rul',0):.3f} mtoi={comp_mean.get('mtoi',0):.3f} "
            f"stage={comp_mean.get('stage',0):.3f} rmono={comp_mean.get('rul_mono',0):.4f} "
            f"mmono={comp_mean.get('mono',0):.4f} | {format_duration(dt)}")

        # --- Lưu checkpoint 'last' mỗi epoch (để resume nếu disconnect) ---
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "best_val": best_val, "history": history,
        }, last_ckpt)

        # --- Lưu 'best' khi val cải thiện + early stopping ---
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss}, best_ckpt)
            logger.info(f"  -> Lưu BEST (val={val_loss:.4f}) tại {best_ckpt.name}")
        else:
            bad_epochs += 1                              # val không cải thiện
            if bad_epochs >= patience:
                logger.info(f"  -> Early stopping (val không cải thiện {patience} epoch).")
                break

    # --- Nạp lại trọng số tốt nhất để trả về ---
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
    logger.info(f"HOÀN TẤT huấn luyện {run_name}. best_val={best_val:.4f}")
    return {"model": model, "history": history, "best_ckpt": best_ckpt, "last_ckpt": last_ckpt,
            "best_val": best_val}
