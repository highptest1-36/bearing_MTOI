# -*- coding: utf-8 -*-
"""
mtoi.py — PHASE 4: XÂY DỰNG CHỈ SỐ MTOI++ (đóng góp chính của bài).

MTOI gồm 3 thành phần (plan mục 10):
  E_h : khác trạng thái normal bao nhiêu (Mahalanobis so với baseline khoẻ).
  C_h : trạng thái thay đổi nhanh hay chậm (chênh lệch giữa 2 giờ liên tiếp).
  G_h : nhiệt độ có xu hướng tăng do ma sát/suy giảm không (độ dốc ΔT).

Quy trình:
  1) Chuẩn hoá leakage-free: z_h = (f_h - median(baseline)) / (IQR(baseline) + eps).
     baseline = đoạn NORMAL đầu đời (H0 = baseline_frac đầu, vd 20%).
  2) E_h = sqrt( (z_h-μ0)^T (Σ0+εI)^-1 (z_h-μ0) )   (Mahalanobis).
  3) C_h = ||z_h - z_{h-1}||_2.
  4) G_h = (ΔT̄_h - ΔT̄_{h-k}) / (k·Δh),  k mặc định 5.
  5) Chuẩn hoá min-max E,C,G theo đoạn TRAIN (leakage-free) -> Ẽ, C̃, G̃ ∈ [0,1].
  6) MTOI_fixed   = combine(0.5 Ẽ + 0.3 C̃ + 0.2 G̃).
     MTOI_learn   = combine(a Ẽ + b C̃ + c G̃) với (a,b,c) HỌC bằng grid-search trên simplex
                    để TỐI ĐA hoá (monotonicity + |spearman với độ suy giảm|).

GHI CHÚ KỸ THUẬT (quan trọng): plan viết công thức σ(·) (sigmoid).
  Nhưng vì Ẽ,C̃,G̃ ∈ [0,1] và a+b+c=1 nên tổng có trọng số đã nằm trong [0,1];
  bọc thêm sigmoid sẽ NÉN dải về ~[0.5, 0.73] -> chỉ số mất độ trải, khó làm HI tốt.
  => Mặc định ta dùng TRỰC TIẾP tổng có trọng số (combine = identity) để MTOI trải đủ 0→1.
     Có cờ use_sigmoid=True để bật bản σ đúng như plan nếu bạn muốn so sánh.

Đầu ra: mtoi_by_hour.csv (hour_id, E_h, C_h, G_h, E_norm, C_norm, G_norm, MTOI_fixed, MTOI_learnable)
        và results/tables/learned_mtoi_weights.csv (trọng số a,b,c đã học — Table 7 của paper).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import PROC_MENDELEY, TABLES_DIR
from src.utils.logger import get_logger, section
from src import metrics

EPS = 1e-9

# 10 đặc trưng rung + 3 nhiệt độ tạo thành vector trạng thái f_h.
VIB_FEATURES = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y",
                "SE_x", "SE_y", "ESE_x", "ESE_y"]
TEMP_FEATURES = ["T_bearing_mean", "T_atm_mean", "DeltaT_mean"]


def _sigmoid(z):
    """Hàm sigmoid σ(z) = 1/(1+e^-z)."""
    return 1.0 / (1.0 + np.exp(-z))


def robust_scale(F, baseline_idx):
    """
    Chuẩn hoá robust theo baseline (leakage-free):
        z = (f - median(baseline)) / (IQR(baseline) + eps).
    F: ma trận [H, D] (H giờ, D đặc trưng). baseline_idx: chỉ số các giờ baseline khoẻ.
    """
    base = F[baseline_idx]                                  # lấy phần baseline
    med = np.median(base, axis=0)                          # trung vị từng cột
    q75, q25 = np.percentile(base, [75, 25], axis=0)       # tứ phân vị
    iqr = (q75 - q25)                                       # khoảng tứ phân vị (IQR)
    return (F - med) / (iqr + EPS)                          # chuẩn hoá toàn bộ giờ


def compute_E(z, baseline_idx):
    """E_h = khoảng cách Mahalanobis của z_h so với phân phối baseline khoẻ."""
    base = z[baseline_idx]                                  # z của các giờ baseline
    mu0 = np.mean(base, axis=0)                             # vector trung bình μ0
    # Hiệp phương sai Σ0 + εI (cộng εI để khả nghịch khi D lớn / baseline ít điểm).
    Sigma0 = np.cov(base, rowvar=False)
    Sigma0 = np.atleast_2d(Sigma0)
    Sigma0_inv = np.linalg.pinv(Sigma0 + EPS * np.eye(Sigma0.shape[0]))  # nghịch đảo giả
    # Tính Mahalanobis cho từng giờ.
    diff = z - mu0                                          # (H, D)
    # m_h = sqrt( diff_h^T Σ0^-1 diff_h ) — tính vector hoá.
    E = np.sqrt(np.einsum("hi,ij,hj->h", diff, Sigma0_inv, diff) + EPS)
    return E


def compute_C(z):
    """C_h = ||z_h - z_{h-1}||_2 (tốc độ thay đổi trạng thái). C_0 := C_1."""
    diff = np.diff(z, axis=0)                               # z[h]-z[h-1] cho h>=1
    C = np.sqrt(np.sum(diff ** 2, axis=1) + EPS)            # chuẩn L2 theo từng giờ
    C = np.concatenate([[C[0]] if len(C) else [0.0], C])   # gán C_0 = C_1 để đủ độ dài H
    return C


def compute_G(deltaT, k=5):
    """
    G_h = (ΔT̄_h - ΔT̄_{h-k}) / (k·Δh), Δh=1 -> độ dốc nhiệt độ chênh lệch trong k giờ.
    Với h < k -> dùng cửa sổ ngắn hơn (k hiệu dụng = h) để vẫn có giá trị.
    """
    H = len(deltaT)
    G = np.zeros(H)
    for h in range(H):
        kk = min(k, h)                                     # cửa sổ thực tế (đầu đời thì ngắn)
        if kk == 0:
            G[h] = 0.0                                     # giờ đầu chưa có quá khứ -> 0
        else:
            G[h] = (deltaT[h] - deltaT[h - kk]) / kk       # độ dốc trung bình kk giờ
    return G


def minmax_norm(u, train_idx):
    """Chuẩn hoá min-max theo đoạn TRAIN (leakage-free); clip về [0,1]."""
    lo = np.min(u[train_idx]); hi = np.max(u[train_idx])   # min/max trên đoạn train
    un = (u - lo) / (hi - lo + EPS)                        # đưa về [0,1] theo train
    return np.clip(un, 0.0, 1.0)                           # cắt ngoài [0,1] (đoạn test có thể vượt)


def combine(E_n, C_n, G_n, w, use_sigmoid=False):
    """Kết hợp Ẽ,C̃,G̃ với trọng số w=(a,b,c). use_sigmoid=True -> bọc σ theo plan."""
    s = w[0] * E_n + w[1] * C_n + (w[2] * G_n if G_n is not None else 0.0)
    return _sigmoid(s) if use_sigmoid else s


def fit_learnable_weights_gradient(E_n, C_n, G_n, deg, has_temp=True,
                                   steps=800, lr=0.05, mono_lambda=1.0, use_sigmoid=False):
    """
    HỌC trọng số bằng SOFTMAX + GRADIENT (đúng plan mục 10.6: [a,b,c]=Softmax(θ_E,θ_C,θ_G)).

    Tham số học là θ; trọng số = softmax(θ) -> TỰ ĐỘNG thoả a,b,c>=0 và a+b+c=1 (ràng buộc simplex).
    Hàm mục tiêu KHẢ VI (để gradient hoạt động): tối đa hoá tương quan Pearson(MTOI, deg)
    và phạt phần MTOI GIẢM (đại diện cho monotonicity). Tối ưu bằng Adam.
    Trả về (w=(a,b,c), fitness) với fitness = monotonicity + |Spearman| (để so sánh/ghi bảng).
    """
    import torch
    E = torch.tensor(np.asarray(E_n), dtype=torch.float32)
    C = torch.tensor(np.asarray(C_n), dtype=torch.float32)
    G = torch.tensor(np.asarray(G_n), dtype=torch.float32) if has_temp else None
    d = torch.tensor(np.asarray(deg), dtype=torch.float32)
    dc = d - d.mean()                                      # deg đã trừ trung bình (cho Pearson)

    n_theta = 3 if has_temp else 2                         # 3 thành phần nếu có nhiệt độ, 2 nếu không
    theta = torch.zeros(n_theta, requires_grad=True)       # tham số học, khởi tạo 0 -> softmax đều
    opt = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):                                 # vòng lặp tối ưu gradient
        opt.zero_grad()
        w = torch.softmax(theta, dim=0)                    # trọng số trên simplex
        m = w[0] * E + w[1] * C + (w[2] * G if has_temp else 0.0)  # MTOI (tổng có trọng số)
        if use_sigmoid:
            m = torch.sigmoid(m)
        mc = m - m.mean()                                  # MTOI đã trừ trung bình
        pearson = (mc * dc).sum() / (torch.sqrt((mc ** 2).sum() * (dc ** 2).sum()) + 1e-8)  # tương quan
        dec = torch.relu(m[:-1] - m[1:]).mean()            # phần GIẢM (đại diện vi phạm đơn điệu)
        loss = -pearson + mono_lambda * dec                # tối đa tương quan + tối thiểu giảm
        loss.backward(); opt.step()

    w = torch.softmax(theta.detach(), dim=0).numpy()       # trọng số cuối
    w = (float(w[0]), float(w[1]), float(w[2]) if has_temp else 0.0)
    mtoi = combine(E_n, C_n, G_n if has_temp else None, w, use_sigmoid=use_sigmoid)
    fit = metrics.monotonicity(mtoi) + abs(metrics.spearman_corr(mtoi, deg))
    return w, fit


def fit_learnable_weights_grid(E_n, C_n, G_n, deg, step=0.05, has_temp=True, use_sigmoid=False):
    """Bản dự phòng: grid-search trên simplex (đơn giản, không gradient). Cùng fitness."""
    best_w, best_fit = None, -1e9
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < -1e-9 or c > 1.0 + 1e-9:
                continue
            c = max(0.0, c)
            if not has_temp and c > 1e-9:
                continue
            w = (a, b, c)
            mtoi = combine(E_n, C_n, G_n if has_temp else None, w, use_sigmoid=use_sigmoid)
            fit = metrics.monotonicity(mtoi) + abs(metrics.spearman_corr(mtoi, deg))
            if fit > best_fit:
                best_fit, best_w = fit, w
    return best_w, best_fit


def fit_learnable_weights(E_n, C_n, G_n, deg, has_temp=True, method="gradient",
                          use_sigmoid=False, step=0.05):
    """
    HỌC trọng số (a,b,c) cho MTOI.
      method='gradient' (mặc định, ĐÚNG plan): Softmax(θ) + Adam.
      method='grid'    : grid-search trên simplex (dự phòng/đối chiếu).
    Trả về (w, fitness).
    """
    if method == "grid":
        return fit_learnable_weights_grid(E_n, C_n, G_n, deg, step=step,
                                          has_temp=has_temp, use_sigmoid=use_sigmoid)
    return fit_learnable_weights_gradient(E_n, C_n, G_n, deg, has_temp=has_temp,
                                          use_sigmoid=use_sigmoid)


def drop_component_mtoi(E_n, C_n, G_n, w_learn, drop, has_temp, use_sigmoid):
    """
    Tạo MTOI khi BỎ 1 thành phần (cho ablation w/o E / w/o C / w/o G).
    Cách làm: đặt trọng số thành phần bị bỏ = 0 rồi CHUẨN HOÁ LẠI 2 trọng số còn lại cho tổng=1.
    Trả về mảng MTOI, hoặc None nếu không áp dụng (vd bỏ G mà dataset không có nhiệt độ).
    """
    ws = {"E": w_learn[0], "C": w_learn[1], "G": w_learn[2]}
    if drop == "G" and not has_temp:                       # vib-only không có G -> bỏ qua
        return None
    ws[drop] = 0.0                                          # tắt thành phần bị bỏ
    s = sum(ws.values())
    if s < 1e-9:                                            # tất cả = 0 -> không hợp lệ
        return None
    ws = {k: v / s for k, v in ws.items()}                 # chuẩn hoá lại tổng = 1
    return combine(E_n, C_n, G_n if has_temp else None,
                   (ws["E"], ws["C"], ws["G"]), use_sigmoid=use_sigmoid)


def build_mtoi(proc_dir=None, baseline_frac=0.2, train_frac=0.6, k_temp=5,
               has_temp=True, use_sigmoid=False, dataset_name="mendeley", method="gradient"):
    """
    PHASE 4: đọc hour_features.csv -> tính E,C,G, chuẩn hoá, MTOI_fixed & MTOI_learnable.
    baseline_frac: tỉ lệ giờ đầu coi là NORMAL baseline (cho robust scaling & Mahalanobis).
    train_frac   : tỉ lệ giờ đầu coi là TRAIN (cho min-max norm leakage-free & học trọng số).
    has_temp     : dataset có nhiệt độ không (XJTU-SY = False -> MTOI rút gọn 2 thành phần).
    """
    logger = get_logger("mtoi")
    section("PHASE 4 — XÂY DỰNG MTOI++ (E, C, G -> fixed & learnable)", logger)
    proc_dir = Path(proc_dir or PROC_MENDELEY)

    # --- Nạp đặc trưng cấp giờ ---
    df = pd.read_csv(proc_dir / "hour_features.csv").sort_values("hour_id").reset_index(drop=True)
    H = len(df)                                            # số giờ
    logger.info(f"Số giờ: {H} | có nhiệt độ: {has_temp}")

    # Chọn danh sách đặc trưng cho vector trạng thái f_h.
    feat_cols = VIB_FEATURES + (TEMP_FEATURES if has_temp else [])
    F = df[feat_cols].to_numpy(dtype=float)               # ma trận [H, D]

    # Chỉ số baseline (normal) và train.
    n_base = max(2, int(baseline_frac * H))               # ít nhất 2 giờ baseline
    n_train = max(n_base, int(train_frac * H))            # train bao trùm baseline
    baseline_idx = np.arange(n_base)
    train_idx = np.arange(n_train)
    logger.info(f"Baseline normal = {n_base} giờ đầu | Train (cho norm/học trọng số) = {n_train} giờ đầu")

    # --- (1) Chuẩn hoá robust theo baseline ---
    z = robust_scale(F, baseline_idx)                     # z_h [H, D]

    # --- (2)(3) Tính E_h và C_h ---
    E = compute_E(z, baseline_idx)                        # Mahalanobis
    C = compute_C(z)                                      # tốc độ đổi trạng thái

    # --- (4) Tính G_h (chỉ khi có nhiệt độ) ---
    if has_temp:
        deltaT = df["DeltaT_mean"].to_numpy(float)
        G = compute_G(deltaT, k=k_temp)
    else:
        G = np.zeros(H)                                   # không có nhiệt độ -> G = 0

    # --- (5) Chuẩn hoá min-max leakage-free ---
    E_n = minmax_norm(E, train_idx)
    C_n = minmax_norm(C, train_idx)
    G_n = minmax_norm(G, train_idx) if has_temp else np.zeros(H)

    # Độ suy giảm tham chiếu (0 đầu đời -> 1 lúc hỏng) để học trọng số & đánh giá.
    deg = np.arange(H) / max(H - 1, 1)

    # --- (6a) MTOI cố định ---
    if has_temp:
        w_fixed = (0.5, 0.3, 0.2)
    else:
        w_fixed = (0.625, 0.375, 0.0)                     # quy 0.5:0.3 về tổng 1 khi bỏ G
    mtoi_fixed = combine(E_n, C_n, G_n if has_temp else None, w_fixed, use_sigmoid=use_sigmoid)

    # --- (6b) MTOI học trọng số bằng SOFTMAX + GRADIENT (đúng plan mục 10.6) ---
    w_learn, fit = fit_learnable_weights(E_n, C_n, G_n, deg, has_temp=has_temp,
                                         method=method, use_sigmoid=use_sigmoid)
    mtoi_learn = combine(E_n, C_n, G_n if has_temp else None, w_learn, use_sigmoid=use_sigmoid)
    logger.info(f"Trọng số CỐ ĐỊNH (a,b,c) = {w_fixed}")
    logger.info(f"Trọng số HỌC ĐƯỢC ({method}) (a,b,c) = "
                f"{tuple(round(x,3) for x in w_learn)} | fitness={fit:.3f}")

    # --- (6c) MTOI bỏ từng thành phần (cho ablation w/o E / w/o C / w/o G) ---
    mtoi_woE = drop_component_mtoi(E_n, C_n, G_n, w_learn, "E", has_temp, use_sigmoid)
    mtoi_woC = drop_component_mtoi(E_n, C_n, G_n, w_learn, "C", has_temp, use_sigmoid)
    mtoi_woG = drop_component_mtoi(E_n, C_n, G_n, w_learn, "G", has_temp, use_sigmoid)

    # --- Ghi mtoi_by_hour.csv ---
    out = pd.DataFrame({
        "hour_id": df["hour_id"].to_numpy(),
        "E_h": E, "C_h": C, "G_h": G,
        "E_norm": E_n, "C_norm": C_n, "G_norm": G_n,
        "MTOI_fixed": mtoi_fixed, "MTOI_learnable": mtoi_learn,
        "deg_ref": deg,
    })
    # Thêm các cột MTOI bỏ-thành-phần (nếu áp dụng) -> phục vụ ablation tự động ở Phase 7.
    out["MTOI_woE"] = mtoi_woE if mtoi_woE is not None else mtoi_learn
    out["MTOI_woC"] = mtoi_woC if mtoi_woC is not None else mtoi_learn
    out["MTOI_woG"] = mtoi_woG if mtoi_woG is not None else mtoi_learn
    out_path = proc_dir / "mtoi_by_hour.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Đã ghi {out_path}")

    # --- Ghi bảng trọng số đã học (Table 7) ---
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    wdf = pd.DataFrame([
        {"dataset": dataset_name, "type": "fixed",     "a_E": w_fixed[0], "b_C": w_fixed[1], "c_G": w_fixed[2]},
        {"dataset": dataset_name, "type": "learnable", "a_E": round(w_learn[0], 4),
         "b_C": round(w_learn[1], 4), "c_G": round(w_learn[2], 4), "fitness": round(fit, 4)},
    ])
    wdf.to_csv(TABLES_DIR / f"learned_mtoi_weights_{dataset_name}.csv", index=False)

    # --- In nhanh chất lượng MTOI để bạn thấy ngay ---
    logger.info(f"[Chất lượng] Monotonicity fixed={metrics.monotonicity(mtoi_fixed):.3f} | "
                f"learnable={metrics.monotonicity(mtoi_learn):.3f}")
    logger.info(f"[Chất lượng] |Spearman vs deg| fixed={abs(metrics.spearman_corr(mtoi_fixed, deg)):.3f} | "
                f"learnable={abs(metrics.spearman_corr(mtoi_learn, deg)):.3f}")
    return {"mtoi_path": out_path, "weights_fixed": w_fixed, "weights_learn": w_learn}
