# -*- coding: utf-8 -*-
"""
fusion.py — HỢP NHẤT CÓ CỔNG (Gated Fusion) theo plan mục 12.4.

Ý tưởng: model TỰ HỌC lúc nào nên tin tín hiệu RUNG, lúc nào nên tin NHIỆT ĐỘ,
thông qua một "cổng" α ∈ (0,1) tính từ cả hai vector:
    u      = [e_vib ; e_temp]                (nối 2 vector -> 256 chiều)
    α      = σ(W_g · u + b_g)                 (cổng, mỗi chiều 1 trọng số)
    e_fuse = α · e_vib + (1-α) · e_temp       (trộn theo cổng)
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, dim=128):
        """dim: chiều của e_vib và e_temp (đều 128). Đầu ra cũng 128."""
        super().__init__()
        # Cổng nhận [e_vib; e_temp] (2*dim) -> sinh α có dim chiều (gating theo từng chiều).
        self.gate = nn.Linear(2 * dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e_vib, e_temp, has_temp=None):
        # Nối 2 vector theo chiều đặc trưng -> [B, 2*dim].
        u = torch.cat([e_vib, e_temp], dim=1)
        alpha = self.sigmoid(self.gate(u))            # cổng α ∈ (0,1), [B, dim]
        # Trộn: tin rung nhiều khi α->1, tin nhiệt độ nhiều khi α->0.
        e_fuse = alpha * e_vib + (1.0 - alpha) * e_temp
        # PER-SAMPLE MASKING: ổ bi KHÔNG có nhiệt độ -> thu về vib-only (bỏ nhánh temp giả 0).
        if has_temp is not None:
            ht = has_temp.view(-1, 1).to(e_fuse.dtype)   # [B,1] 1=có temp, 0=không
            e_fuse = ht * e_fuse + (1.0 - ht) * e_vib
        return e_fuse, alpha                          # trả thêm α để phân tích/biểu đồ


class ConcatFusion(nn.Module):
    """Hợp nhất kiểu nối đơn giản (baseline so sánh với gated fusion)."""
    def __init__(self, dim=128):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)           # nối rồi chiếu về dim
        self.act = nn.ReLU()

    def forward(self, e_vib, e_temp, has_temp=None):
        u = torch.cat([e_vib, e_temp], dim=1)
        e_fuse = self.act(self.proj(u))
        if has_temp is not None:                      # ổ bi không temp -> vib-only
            ht = has_temp.view(-1, 1).to(e_fuse.dtype)
            e_fuse = ht * e_fuse + (1.0 - ht) * e_vib
        return e_fuse, None                           # không có α
