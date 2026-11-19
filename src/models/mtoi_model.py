# -*- coding: utf-8 -*-
"""
mtoi_model.py — MÔ HÌNH ĐỀ XUẤT (multimodal multi-task) theo plan mục 12.

Luồng dữ liệu:
   vib [B,2,L] ──► VibEncoder ──► e_vib [B,128] ┐
                                                 ├─► (Gated)Fusion ──► e_fuse [B,128] ─┬─► Stage head  (4 lớp)
   temp [B,4] ──► TempEncoder ─► e_temp [B,128] ┘                                       ├─► MTOI head   (1, sigmoid)
                                                                                        ├─► RUL head    (1, sigmoid)
                                                                                        └─► (tuỳ chọn) Uncertainty (q10,q50,q90)

Các CỜ cấu hình phục vụ ablation:
   use_temp=False        -> bỏ nhánh nhiệt độ (chỉ dùng e_vib) — biến thể "w/o temperature".
   fusion='gated'/'concat' -> chọn kiểu hợp nhất.
   use_uncertainty=True  -> bật đầu ra phân vị RUL (q10/q50/q90).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vib_encoder import VibEncoder
from src.models.temp_encoder import TempEncoder
from src.models.fusion import GatedFusion, ConcatFusion
from src.models.seq_encoders import RNNEncoder, TransformerSeqEncoder


def build_vib_encoder(kind="cnn", emb_dim=128, use_attention=False):
    """
    Nhà máy tạo bộ mã hoá rung theo loại (để CNN/LSTM/GRU/Transformer dùng chung các head).
      kind='cnn'         -> 1D-CNN (đề xuất, plan mục 12.2)
      kind='lstm'/'gru'  -> baseline RNN 2 hướng trên chuỗi khung
      kind='transformer' -> baseline Transformer thuần trên chuỗi khung
    Mọi encoder đều trả ra [B, emb_dim].
    """
    if kind == "cnn":
        return VibEncoder(in_ch=2, emb_dim=emb_dim, use_attention=use_attention)
    if kind in ("lstm", "gru"):
        return RNNEncoder(in_ch=2, emb_dim=emb_dim, rnn=kind)
    if kind == "transformer":
        return TransformerSeqEncoder(in_ch=2, emb_dim=emb_dim)
    # --- Baseline hiện đại (reviewer Q1): TCN / TCN-Transformer / CNN-BiLSTM-Attention ---
    if kind in ("tcn", "tcn_transformer", "cnn_bilstm_attn"):
        from src.models.baseline_encoders import (TCNEncoder, TCNTransformerEncoder,
                                                  CNNBiLSTMAttnEncoder)
        if kind == "tcn":
            return TCNEncoder(in_ch=2, emb_dim=emb_dim)
        if kind == "tcn_transformer":
            return TCNTransformerEncoder(in_ch=2, emb_dim=emb_dim)
        return CNNBiLSTMAttnEncoder(in_ch=2, emb_dim=emb_dim)
    raise ValueError(f"vib_encoder không hợp lệ: {kind} "
                     f"(chọn cnn|lstm|gru|transformer|tcn|tcn_transformer|cnn_bilstm_attn)")


class MTOIModel(nn.Module):
    def __init__(self, emb_dim=128, n_stages=4, use_temp=True, temp_dim=4,
                 fusion="gated", use_attention=False, use_uncertainty=False,
                 vib_encoder="cnn", use_hi=False, hi_dim=4):
        super().__init__()
        self.use_temp = use_temp                         # có dùng nhiệt độ không
        self.use_uncertainty = use_uncertainty           # có đầu ra bất định không
        self.vib_encoder_kind = vib_encoder              # loại encoder rung (cnn/lstm/gru/transformer)
        self.use_hi = use_hi                             # đưa MTOI/E/C/G làm ĐẦU VÀO cho RUL head
        self.hi_dim = hi_dim if use_hi else 0            # số chiều HI cộng thêm

        # --- Bộ mã hoá rung (chọn theo loại) ---
        self.vib_encoder = build_vib_encoder(vib_encoder, emb_dim=emb_dim, use_attention=use_attention)
        if use_temp:
            self.temp_encoder = TempEncoder(in_dim=temp_dim, emb_dim=emb_dim)
            # Chọn khối hợp nhất.
            self.fusion = GatedFusion(emb_dim) if fusion == "gated" else ConcatFusion(emb_dim)

        # --- Các đầu ra (heads) ---
        self.stage_head = nn.Linear(emb_dim, n_stages)   # phân loại 4 giai đoạn (phụ trợ)
        self.mtoi_head = nn.Linear(emb_dim, 1)           # hồi quy MTOI (qua sigmoid) — chỉ số giải thích
        # RUL là ĐẦU RA CHÍNH -> MLP 2 lớp. Khi use_hi: RUL head ĐỌC THÊM [E,C,G,MTOI] (HI->RUL hai tầng)
        # -> MTOI thật sự DẪN DẮT dự đoán RUL (không chỉ là loss phụ).
        rul_in = emb_dim + self.hi_dim
        self.rul_head = nn.Sequential(
            nn.Linear(rul_in, emb_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(emb_dim // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()
        if use_uncertainty:                              # 3 phân vị RUL (q10,q50,q90) — đầu ra bất định
            self.unc_head = nn.Sequential(
                nn.Linear(rul_in, emb_dim // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(emb_dim // 2, 3),
            )

    def forward(self, vib, temp=None, hi=None, has_temp=None):
        # --- Mã hoá rung ---
        e_vib = self.vib_encoder(vib)                    # [B, 128]
        alpha = None                                     # cổng fusion (nếu có)

        # --- Hợp nhất với nhiệt độ (nếu dùng) + per-sample masking ---
        if self.use_temp and temp is not None:
            e_temp = self.temp_encoder(temp)             # [B, 128]
            e_fuse, alpha = self.fusion(e_vib, e_temp, has_temp=has_temp)  # ổ bi không temp -> vib-only
        else:
            e_fuse = e_vib                               # biến thể vib-only

        # --- Đầu vào RUL head: nối HI [E,C,G,MTOI] nếu use_hi (HI dẫn dắt RUL) ---
        if self.use_hi and hi is not None:
            rul_in = torch.cat([e_fuse, hi], dim=1)      # [B, 128+hi_dim]
        else:
            rul_in = e_fuse

        # --- Các đầu ra ---
        out = {
            "stage_logits": self.stage_head(e_fuse),                  # [B, 4] (logits) — KHÔNG dùng HI
            "mtoi": self.sigmoid(self.mtoi_head(e_fuse)).squeeze(-1), # [B] — KHÔNG dùng HI (tránh tự dự đoán MTOI tầm thường)
            "rul": self.sigmoid(self.rul_head(rul_in)).squeeze(-1),   # [B] trong (0,1) — ĐIỂM CHÍNH
            "embedding": e_fuse,
            "alpha": alpha,
        }
        if self.use_uncertainty:
            # 3 phân vị RUL ĐƠN ĐIỆU: q10 <= q50 <= q90, đều trong (0,1).
            raw = self.unc_head(rul_in)                               # [B, 3]
            q10 = self.sigmoid(raw[:, 0])
            q50 = q10 + F.softplus(raw[:, 1]) * (1.0 - q10)
            q90 = q50 + F.softplus(raw[:, 2]) * (1.0 - q50)
            out["rul_quantiles"] = torch.stack([q10, q50, q90], dim=1)  # [B, 3]
        return out


def build_model_from_config(cfg, use_temp=True):
    """Tạo MTOIModel từ dict cấu hình (đọc các khoá trong configs/*.yaml)."""
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    return MTOIModel(
        emb_dim=m.get("emb_dim", 128),
        n_stages=m.get("n_stages", 4),
        use_temp=use_temp,
        temp_dim=m.get("temp_dim", 4),
        fusion=m.get("fusion", "gated"),
        use_attention=m.get("use_attention", False),
        use_uncertainty=m.get("use_uncertainty", False),
        vib_encoder=m.get("vib_encoder", "cnn"),
        use_hi=m.get("use_hi", False),
        hi_dim=m.get("hi_dim", 4),
    )
