# -*- coding: utf-8 -*-
"""
losses.py — HÀM MẤT MÁT ĐA NHIỆM (RUL-PRIMARY).

Sau khi tái cấu trúc sang hướng RUL-centric, L_RUL là TRỤC CHÍNH (số hạng TRẦN, không trọng số);
stage / MTOI / các regularizer là PHỤ TRỢ có trọng số:

  L_total = L_RUL                                   (chính: SmoothL1 RUL chuẩn hoá)
          + w_stage·L_stage                          (CE phân loại giai đoạn — phụ nhẹ)
          + w_mtoi·L_MTOI                             (MSE — giám sát phụ; tắt qua alias lambda1=0)
          + w_rul_smooth·smooth(RUL) + w_rul_mono·mono↓(RUL)   (RUL phải GIẢM, mượt)
          + w_mtoi_smooth·smooth(MTOI) + w_mtoi_mono·mono↑(MTOI) (MTOI phải TĂNG, mượt)
          [+ w_asym·L_asym]                           (phạt over-predict RUL — tuỳ chọn)
          [+ w_unc·L_unc]                             (pinball quantile RUL — khi bật uncertainty)

GHI CHÚ QUAN TRỌNG (chống lỗi LOBO): smooth/mono cần chuỗi theo GIỜ TRONG TỪNG ổ bi. Một batch
LOBO trộn nhiều ổ bi (hour_id offset toàn cục), nên regularizer được tính RIÊNG theo `bearing_idx`
rồi trung bình — KHÔNG tính mono/smooth xuyên biên ổ bi. Mendeley single-run không có bearing_idx
-> coi như 1 ổ bi (đúng).

TƯƠNG THÍCH NGƯỢC: resolve_weights() chấp nhận cả khóa cũ (lambda1..lambda5) lẫn mới (w_*), nên
ablation cũ loss_override={'lambda1':0} VẪN tắt nhánh MTOI (lambda1 -> w_mtoi).
"""

import torch
import torch.nn.functional as F


# ============================= GỘP & PHẠT THEO CHUỖI =============================

def aggregate_by_hour(values, hour_ids):
    """
    Gộp 'values' (theo cửa sổ) -> trung bình theo từng hour_id (tăng dần). Trả (uniq_hours, means).
    VECTOR HOÁ bằng scatter (O(N), KHÔNG vòng lặp Python) — quan trọng vì PRONOSTIA có 1 cửa sổ/giờ
    nên 1 batch có thể chứa hàng trăm giờ khác nhau (vòng lặp Python sẽ rất chậm).
    """
    uniq, inv = torch.unique(hour_ids, sorted=True, return_inverse=True)
    n = uniq.numel()
    sums = torch.zeros(n, device=values.device, dtype=values.dtype).scatter_add_(0, inv, values)
    cnts = torch.zeros(n, device=values.device, dtype=values.dtype).scatter_add_(
        0, inv, torch.ones_like(values))
    return uniq, sums / cnts.clamp_min(1.0)


def _second_diff(m):
    """Phạt gồ ghề: mean sai phân bậc 2 bình phương (cần >=3 điểm)."""
    if m.numel() < 3:
        return torch.zeros((), device=m.device)
    d2 = m[2:] - 2 * m[1:-1] + m[:-2]
    return (d2 ** 2).mean()


def _penalize_decrease(m):
    """Phạt khi chuỗi GIẢM — dùng cho MTOI (nên TĂNG theo degradation)."""
    if m.numel() < 2:
        return torch.zeros((), device=m.device)
    return F.relu(m[:-1] - m[1:]).mean()


def _penalize_increase(m):
    """Phạt khi chuỗi TĂNG — dùng cho RUL (nên GIẢM theo thời gian)."""
    if m.numel() < 2:
        return torch.zeros((), device=m.device)
    return F.relu(m[1:] - m[:-1]).mean()


def _traj_penalty(values, hour_ids, bearing_ids, seq_fn):
    """Áp seq_fn lên chuỗi gộp-theo-giờ TRONG TỪNG bearing rồi trung bình. bearing_ids=None -> 1 bearing."""
    if bearing_ids is None:
        _, m = aggregate_by_hour(values, hour_ids)
        return seq_fn(m)
    outs = []
    for b in torch.unique(bearing_ids):
        mask = (bearing_ids == b)
        _, m = aggregate_by_hour(values[mask], hour_ids[mask])
        outs.append(seq_fn(m))
    if not outs:
        return torch.zeros((), device=values.device)
    return torch.stack(outs).mean()


# Giữ TÊN CŨ để tương thích (nếu nơi khác import smoothness_loss/monotonic_loss).
def smoothness_loss(mtoi_pred, hour_ids):
    _, m = aggregate_by_hour(mtoi_pred, hour_ids)
    return _second_diff(m)


def monotonic_loss(mtoi_pred, hour_ids):
    _, m = aggregate_by_hour(mtoi_pred, hour_ids)
    return _penalize_decrease(m)


def quantile_loss(pred_q, target, quantiles=(0.1, 0.5, 0.9)):
    """Pinball loss cho các phân vị RUL (q10/q50/q90)."""
    losses = []
    for i, qv in enumerate(quantiles):
        err = target - pred_q[:, i]
        losses.append(torch.max(qv * err, (qv - 1) * err).mean())
    return torch.stack(losses).mean()


# ============================= TRỌNG SỐ RUL-PRIMARY =============================

DEFAULT_WEIGHTS = {
    "w_stage": 0.2,         # stage = phụ trợ nhẹ (lượng tử hoá vòng đời)
    "w_mtoi": 0.3,          # MTOI = giám sát phụ (auxiliary). Tắt qua alias lambda1=0 (ablation w/o MTOI).
    "w_rul_smooth": 0.05,   # mượt quỹ đạo RUL dự đoán
    "w_rul_mono": 0.05,     # RUL nên GIẢM dần
    "w_mtoi_smooth": 0.05,  # mượt quỹ đạo MTOI dự đoán
    "w_mtoi_mono": 0.05,    # MTOI nên TĂNG dần
    "w_unc": 0.1,           # pinball quantile (khi bật uncertainty)
    "w_asym": 0.0,          # phạt over-predict RUL (tắt mặc định)
    "asym_late": 2.0,       # hệ số phạt khi dự đoán CÒN NHIỀU thời gian hơn thực (nguy hiểm)
}

# Ánh xạ khóa CŨ -> MỚI (tương thích loss_override cũ).
_ALIAS = {"lambda1": "w_mtoi", "lambda3": "w_mtoi_smooth", "lambda4": "w_mtoi_mono",
          "lambda5": "w_unc"}


def resolve_weights(weights=None):
    """Hợp nhất mặc định + override; hỗ trợ cả khóa cũ (lambda*) lẫn mới (w_*). lambda2 (RUL cũ) bỏ qua vì RUL nay là trục chính."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for old, new in _ALIAS.items():
            if old in weights:
                w[new] = weights[old]
        w.update({k: v for k, v in weights.items() if k in w})
    return w


def total_loss(outputs, batch, weights=None, use_uncertainty=False):
    """
    Tính tổng loss RUL-primary. Trả (loss_tổng, dict_thành_phần).
      outputs: dict từ MTOIModel.forward (stage_logits, mtoi, rul, [rul_quantiles]).
      batch  : dict từ DataLoader (stage, mtoi, rul, hour_id, [bearing_idx]).
      weights: dict ghi đè (khóa cũ lambda* hoặc mới w_*).
    """
    w = resolve_weights(weights)
    hour_ids = batch["hour_id"]
    bearing_ids = batch.get("bearing_idx")          # None ở Mendeley single-run

    # --- L_RUL: TRỤC CHÍNH (SmoothL1, không trọng số) ---
    rul_pred = outputs["rul"]
    l_rul = F.smooth_l1_loss(rul_pred, batch["rul"])

    # --- Phụ trợ: stage + MTOI ---
    l_stage = F.cross_entropy(outputs["stage_logits"], batch["stage"])
    l_mtoi = F.mse_loss(outputs["mtoi"], batch["mtoi"])

    # --- Regularizer quỹ đạo (RIÊNG theo từng bearing) ---
    l_rul_smooth = _traj_penalty(rul_pred, hour_ids, bearing_ids, _second_diff)
    l_rul_mono = _traj_penalty(rul_pred, hour_ids, bearing_ids, _penalize_increase)
    l_mtoi_smooth = _traj_penalty(outputs["mtoi"], hour_ids, bearing_ids, _second_diff)
    l_mtoi_mono = _traj_penalty(outputs["mtoi"], hour_ids, bearing_ids, _penalize_decrease)

    loss = (l_rul
            + w["w_stage"] * l_stage
            + w["w_mtoi"] * l_mtoi
            + w["w_rul_smooth"] * l_rul_smooth + w["w_rul_mono"] * l_rul_mono
            + w["w_mtoi_smooth"] * l_mtoi_smooth + w["w_mtoi_mono"] * l_mtoi_mono)

    comps = {"rul": l_rul.item(), "stage": l_stage.item(), "mtoi": l_mtoi.item(),
             "rul_smooth": float(l_rul_smooth.detach()), "rul_mono": float(l_rul_mono.detach()),
             "smooth": float(l_mtoi_smooth.detach()), "mono": float(l_mtoi_mono.detach())}

    # --- L_asym: phạt over-predict RUL (tuỳ chọn) ---
    if w["w_asym"] > 0:
        e = rul_pred - batch["rul"]                 # >0: dự đoán còn NHIỀU thời gian hơn thực
        l_asym = (w["asym_late"] * F.relu(e) + F.relu(-e)).mean()
        loss = loss + w["w_asym"] * l_asym
        comps["asym"] = float(l_asym.detach())

    # --- L_unc: pinball quantile RUL (tuỳ chọn) ---
    if use_uncertainty and "rul_quantiles" in outputs:
        l_unc = quantile_loss(outputs["rul_quantiles"], batch["rul"])
        loss = loss + w["w_unc"] * l_unc
        comps["unc"] = l_unc.item()

    comps["total"] = loss.item()
    return loss, comps
