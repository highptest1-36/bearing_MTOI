# -*- coding: utf-8 -*-
"""
baseline_encoders.py — BASELINE DEEP HIỆN ĐẠI (phản hồi reviewer Q1: so sánh công bằng).

Bổ sung 3 backbone vib-only mạnh ngoài CNN/LSTM/GRU/Transformer, để paper so sánh dưới
CÙNG protocol LOBO (cùng head, cùng loss, cùng split):
  - TCN                 : Temporal Convolutional Network (dilated causal-style conv + residual).
  - TCN-Transformer     : TCN front (đặc trưng cục bộ) + Transformer (chú ý toàn cục).
  - CNN-BiLSTM-Attention: Conv front + BiLSTM 2 hướng + attention pooling (rất phổ biến cho bearing RUL).

Tất cả nhận đầu vào [B, 2, L] (giống VibEncoder) và trả [B, emb_dim], dùng PatchFront để token-hoá
tín hiệu dài (L lớn) cho nhẹ — y hệt cách RNN/Transformer baseline đang dùng.
"""
import torch
import torch.nn as nn
from src.models.seq_encoders import PatchFront


class _TemporalBlock(nn.Module):
    """1 khối TCN: 2 lớp Conv1d giãn (dilated) + residual + ReLU + Dropout (giữ nguyên độ dài chuỗi)."""
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2          # padding đối xứng -> giữ nguyên độ dài
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                           # x: [B, ch, T]
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.drop(self.act(self.bn2(self.conv2(h))))
        return self.act(x + h)                      # residual


class TCNEncoder(nn.Module):
    """TCN trên chuỗi token (PatchFront). dilations 1,2,4,8 -> trường nhìn rộng dần."""
    def __init__(self, in_ch=2, emb_dim=128, frame_len=128, ch=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.patch = PatchFront(frame_len)
        self.inp = nn.Conv1d(in_ch * frame_len, ch, kernel_size=1)   # token-feat -> ch kênh
        self.blocks = nn.Sequential(*[_TemporalBlock(ch, dilation=d) for d in dilations])
        self.proj = nn.Linear(ch, emb_dim)

    def forward(self, x):
        h = self.patch(x)                           # [B, nf, C*frame_len]
        h = h.transpose(1, 2)                       # [B, C*frame_len, nf]
        h = self.inp(h)                             # [B, ch, nf]
        h = self.blocks(h)                          # [B, ch, nf]
        return self.proj(h.mean(dim=2))             # [B, emb_dim]


class TCNTransformerEncoder(nn.Module):
    """TCN front (đặc trưng cục bộ) -> Transformer encoder (chú ý toàn cục) -> pool."""
    def __init__(self, in_ch=2, emb_dim=128, frame_len=128, d_model=128, nhead=4,
                 layers=2, dilations=(1, 2, 4), max_frames=128):
        super().__init__()
        self.patch = PatchFront(frame_len)
        self.embed = nn.Linear(in_ch * frame_len, d_model)
        self.tcn = nn.Sequential(*[_TemporalBlock(d_model, dilation=d) for d in dilations])
        self.pos = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, emb_dim)

    def forward(self, x):
        h = self.patch(x)                           # [B, nf, C*frame_len]
        h = self.embed(h)                           # [B, nf, d_model]
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)   # TCN cục bộ -> [B, nf, d_model]
        nf = h.size(1)
        h = h + self.pos[:, :nf, :]
        h = self.encoder(h)                         # chú ý toàn cục
        return self.proj(h.mean(dim=1))             # [B, emb_dim]


class CNNBiLSTMAttnEncoder(nn.Module):
    """Conv front -> BiLSTM 2 hướng -> attention pooling (baseline bearing-RUL phổ biến)."""
    def __init__(self, in_ch=2, emb_dim=128, frame_len=128, ch=64, hidden=128, layers=2):
        super().__init__()
        self.patch = PatchFront(frame_len)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch * frame_len, ch, kernel_size=3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
        )
        self.lstm = nn.LSTM(ch, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        out_dim = hidden * 2
        self.attn = nn.Linear(out_dim, 1)           # trọng số chú ý theo thời gian
        self.proj = nn.Linear(out_dim, emb_dim)

    def forward(self, x):
        h = self.patch(x)                           # [B, nf, C*frame_len]
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)   # [B, nf, ch]
        out, _ = self.lstm(h)                       # [B, nf, hidden*2]
        w = torch.softmax(self.attn(out), dim=1)    # [B, nf, 1] trọng số chú ý
        feat = (w * out).sum(dim=1)                 # gộp có trọng số -> [B, hidden*2]
        return self.proj(feat)                      # [B, emb_dim]
