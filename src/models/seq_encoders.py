# -*- coding: utf-8 -*-
"""
seq_encoders.py — CÁC BỘ MÃ HOÁ TUẦN TỰ làm BASELINE deep (LSTM / GRU / Transformer thuần).

Mục tiêu (plan mục 15.2 "Deep vibration-only: 1D-CNN, LSTM/GRU, Transformer"):
  cung cấp baseline mạnh hơn ngoài 1D-CNN, để paper so sánh công bằng. Tất cả nhận đầu vào
  giống VibEncoder ([B, 2, L]) và trả ra [B, emb_dim] -> cắm thẳng vào các head của MTOIModel.

Vấn đề: L=4096 quá dài để đưa thẳng vào RNN/Transformer theo từng mẫu. Giải pháp CHUẨN:
  "patchify" — cắt tín hiệu thành n_frames khung ngắn (mỗi khung frame_len mẫu), mỗi khung là
  1 "token" có chiều (2 * frame_len). Khi đó RNN/Transformer chạy trên n_frames token (vd 32),
  vừa nhẹ vừa giữ thông tin thời gian.
"""

import torch
import torch.nn as nn


class PatchFront(nn.Module):
    """Biến [B, 2, L] -> [B, n_frames, 2*frame_len] (cắt tín hiệu thành các khung làm token)."""
    def __init__(self, frame_len=128):
        super().__init__()
        self.frame_len = frame_len                       # số mẫu trong 1 khung

    def forward(self, x):
        B, C, L = x.shape                                # B batch, C=2 kênh, L độ dài
        nf = L // self.frame_len                         # số khung lấy được
        x = x[:, :, :nf * self.frame_len]                # cắt bỏ phần dư cuối (không đủ 1 khung)
        # [B, C, nf, frame_len] -> [B, nf, C, frame_len] -> [B, nf, C*frame_len]
        x = x.reshape(B, C, nf, self.frame_len).permute(0, 2, 1, 3).reshape(B, nf, C * self.frame_len)
        return x                                         # mỗi khung là 1 token chiều C*frame_len


class RNNEncoder(nn.Module):
    """Baseline LSTM hoặc GRU 2 lớp (2 hướng) trên chuỗi khung. rnn='lstm' hoặc 'gru'."""
    def __init__(self, in_ch=2, emb_dim=128, frame_len=128, hidden=128, layers=2,
                 rnn="lstm", bidirectional=True):
        super().__init__()
        self.patch = PatchFront(frame_len)               # tạo token từ tín hiệu
        rnn_cls = nn.LSTM if rnn == "lstm" else nn.GRU   # chọn LSTM hay GRU
        self.rnn = rnn_cls(
            input_size=in_ch * frame_len,                # chiều mỗi token
            hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.1 if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)   # chiều đầu ra RNN
        self.proj = nn.Linear(out_dim, emb_dim)          # chiếu về emb_dim

    def forward(self, x):
        h = self.patch(x)                                # [B, nf, C*frame_len]
        out, _ = self.rnn(h)                             # [B, nf, hidden*dir]
        feat = out.mean(dim=1)                           # gộp trung bình theo thời gian
        return self.proj(feat)                           # [B, emb_dim]


class TransformerSeqEncoder(nn.Module):
    """Baseline Transformer THUẦN (patch embedding + positional + self-attention) trên chuỗi khung."""
    def __init__(self, in_ch=2, emb_dim=128, frame_len=128, d_model=128, nhead=4,
                 layers=2, max_frames=128):
        super().__init__()
        self.patch = PatchFront(frame_len)               # tạo token
        self.embed = nn.Linear(in_ch * frame_len, d_model)   # nhúng token -> d_model
        # Positional embedding học được (cho tối đa max_frames token).
        self.pos = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, emb_dim)

    def forward(self, x):
        h = self.patch(x)                                # [B, nf, C*frame_len]
        h = self.embed(h)                                # [B, nf, d_model]
        nf = h.size(1)
        h = h + self.pos[:, :nf, :]                      # cộng positional (cắt đúng số token)
        h = self.encoder(h)                              # self-attention [B, nf, d_model]
        return self.proj(h.mean(dim=1))                  # gộp token -> [B, emb_dim]
