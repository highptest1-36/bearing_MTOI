# -*- coding: utf-8 -*-
"""
temp_encoder.py — BỘ MÃ HOÁ ĐẶC TRƯNG NHIỆT ĐỘ (MLP) theo plan mục 12.3.

Đầu vào : [batch, 4]    ([T_bearing, T_atm, ΔT, G_norm])
Đầu ra  : [batch, 128]  (vector đặc trưng nhiệt độ e^temp)
"""

import torch.nn as nn


class TempEncoder(nn.Module):
    def __init__(self, in_dim=4, emb_dim=128):
        """in_dim: số đặc trưng nhiệt độ (4). emb_dim: chiều đầu ra (128)."""
        super().__init__()
        # MLP 3 lớp: 4 -> 32 -> 64 -> 128.
        # LƯU Ý: lớp CUỐI KHÔNG có ReLU -> embedding nhiệt độ là phép chiếu TUYẾN TÍNH,
        # đối xứng (có cả giá trị âm/dương) với embedding rung. Nếu để ReLU ở cuối, e_temp luôn >=0
        # khiến cổng fusion nhận 2 mô thức ở dải dấu/độ lớn lệch nhau (mất nửa không gian biểu diễn).
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),     # lớp 1 (ẩn) + ReLU
            nn.Linear(32, 64), nn.ReLU(),         # lớp 2 (ẩn) + ReLU
            nn.Linear(64, emb_dim),               # lớp 3 -> emb_dim (KHÔNG ReLU)
        )

    def forward(self, x):
        # x: [B, 4] -> [B, emb_dim]
        return self.net(x)
