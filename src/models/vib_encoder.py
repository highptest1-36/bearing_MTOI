# -*- coding: utf-8 -*-
"""
vib_encoder.py — BỘ MÃ HOÁ TÍN HIỆU RUNG (1D-CNN) theo plan mục 12.2.

Đầu vào : [batch, 2, L]   (2 kênh x,y; L mẫu)
Đầu ra  : [batch, 128]    (vector đặc trưng rung e^vib)

Kiến trúc: 3 khối Conv1D (32 -> 64 -> 128) + BatchNorm + ReLU + MaxPool,
           kết thúc bằng Global Average Pooling để KHÔNG phụ thuộc độ dài L.
           Có tuỳ chọn thêm 1 lớp self-attention (Transformer encoder) cho temporal attention.
"""

import torch
import torch.nn as nn


class VibEncoder(nn.Module):
    def __init__(self, in_ch=2, emb_dim=128, use_attention=False):
        """
        in_ch: số kênh đầu vào (2 = x,y). emb_dim: chiều vector đặc trưng đầu ra (128).
        use_attention: bật thêm 1 lớp attention thời gian (tuỳ chọn, tăng độ biểu diễn).
        """
        super().__init__()
        # ----- Khối Conv 1: 2 -> 32 kênh -----
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=7, padding=3)  # cửa sổ lọc rộng 7
        self.bn1 = nn.BatchNorm1d(32)                                # chuẩn hoá batch
        # ----- Khối Conv 2: 32 -> 64 kênh -----
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        # ----- Khối Conv 3: 64 -> 128 kênh -----
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)                                  # giảm độ dài 1/2 mỗi lần
        self.act = nn.ReLU()                                         # hàm kích hoạt ReLU

        # Lớp attention thời gian tuỳ chọn (1 lớp TransformerEncoder nhẹ).
        self.use_attention = use_attention
        if use_attention:
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=4,
                                               dim_feedforward=256, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=1)

        # Lớp chiếu cuối về emb_dim (nếu emb_dim != 128).
        self.proj = nn.Linear(128, emb_dim) if emb_dim != 128 else nn.Identity()

    def forward(self, x):
        # x: [B, 2, L]
        x = self.pool(self.act(self.bn1(self.conv1(x))))    # -> [B, 32, L/2]
        x = self.pool(self.act(self.bn2(self.conv2(x))))    # -> [B, 64, L/4]
        x = self.pool(self.act(self.bn3(self.conv3(x))))    # -> [B, 128, L/8]

        if self.use_attention:                              # attention trên trục thời gian
            # Đổi sang [B, T, C] cho TransformerEncoder (batch_first=True).
            h = x.transpose(1, 2)                           # [B, L/8, 128]
            h = self.attn(h)                                # tự chú ý theo thời gian
            x = h.transpose(1, 2)                           # về lại [B, 128, L/8]

        # Global Average Pooling theo trục thời gian -> [B, 128].
        x = x.mean(dim=2)
        return self.proj(x)                                 # -> [B, emb_dim]
