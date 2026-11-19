# -*- coding: utf-8 -*-
"""
vtoi.py — XÂY DỰNG CHỈ SỐ VTOI THEO GIAO THỨC LEAKAGE-FREE (bản v2 cho vòng nộp lại).

VÌ SAO CÓ FILE NÀY (đọc kỹ — đây là sửa lỗi CHÍ MẠNG B1 của bản đã nộp):
------------------------------------------------------------------------------------
Bản cũ (`src/mtoi.py:build_mtoi`) tính MTOI cho TỪNG bearing một cách ĐỘC LẬP, TRƯỚC khi
chia fold. Bên trong nó:
    deg      = arange(H)/(H-1)          <-- DÙNG H_fail CỦA CHÍNH BEARING ĐÓ
    (a,b)    = argmax fitness(vtoi, deg) <-- TRỌNG SỐ FIT TRÊN NHÃN CỦA CHÍNH NÓ
    E_norm   = minmax(E, 60% đầu đời)    <-- BIÊN CHUẨN HOÁ TỪ CHÍNH NÓ
=> Khi bearing đó làm HELD-OUT TEST, vector conditioning đưa vào RUL head ĐÃ NHÌN THẤY
   nhãn của chính nó. Đây là target leakage, và nó mâu thuẫn với chính câu paper viết
   ("estimated per fold on the training bearings only").

Bản v2 tách bạch FIT / APPLY:
    compute_raw_components(bearing)      -> E_raw, C_raw   [KHÔNG dùng nhãn, KHÔNG dùng H_fail]
    fit_vtoi_params(train_bearings)      -> a, b, biên min-max  [CHỈ trên bearing TRAIN của fold]
    apply_vtoi(bearing, params)          -> E_norm, C_norm, VTOI [tham số ĐÓNG BĂNG]

TÍNH HỢP LỆ CỦA BASELINE KHOẺ (μ0, Σ0, median/IQR):
    Vẫn lấy từ 20% ĐẦU ĐỜI CỦA CHÍNH BEARING đó — điều này HỢP LỆ và TRIỂN KHAI ĐƯỢC,
    vì dữ liệu run-in của một ổ bi mới LUÔN có sẵn khi vận hành. Nó KHÔNG dùng H_fail.
    (Cần nói rõ điều này trong paper — reviewer sẽ hỏi.)

TIN TỐT VỀ CHI PHÍ: E_raw/C_raw chỉ cần `hour_features.csv` (đã có sẵn trên Drive cho
mọi bearing) => KHÔNG cần chạy lại phase9a, KHÔNG cần đụng tới tín hiệu rung thô.

Trả lời trực tiếp các ý kiến: R2 #6, #7, #8, #22, #14, #15, #16, #23, #25 ; R1 #4 ; R3 #3.
"""

import hashlib

import numpy as np
import pandas as pd

EPS = 1e-9

# 10 đặc trưng rung tạo thành vector trạng thái f_h (KHÔNG có nhiệt độ — paper là vibration-only).
VIB_FEATURES = ["RMS_x", "RMS_y", "Kurt_x", "Kurt_y", "CF_x", "CF_y",
                "SE_x", "SE_y", "ESE_x", "ESE_y"]


# =========================================================================================
# (1) THÀNH PHẦN THÔ — per-bearing, KHÔNG dùng nhãn, KHÔNG dùng H_fail
# =========================================================================================

def _robust_scale(F, baseline_idx):
    """z = (f - median(baseline)) / (IQR(baseline) + eps). Chỉ dùng đoạn baseline khoẻ."""
    base = F[baseline_idx]
    med = np.median(base, axis=0)
    q75, q25 = np.percentile(base, [75, 25], axis=0)
    return (F - med) / ((q75 - q25) + EPS)


def _mahalanobis_shrunk(z, baseline_idx, min_ratio=3.0):
    """
    E_h = khoảng cách Mahalanobis của z_h so với phân phối baseline khoẻ (μ0, Σ0).

    KHÁC BẢN CŨ (trả lời R1 #4): thay `pinv(Σ0 + 1e-9·I)` bằng LEDOIT–WOLF SHRINKAGE, vì
    baseline có thể chỉ có vài chục snapshot cho một hiệp phương sai 10 chiều -> ill-conditioned.
    Nếu số mẫu baseline < min_ratio·D thì rơi về hiệp phương sai CHÉO (an toàn tuyệt đối).

    Trả về (E, info) với info ghi lại n_baseline / shrinkage / estimator để đưa vào bảng paper.
    """
    base = z[baseline_idx]
    n, D = base.shape
    mu0 = base.mean(axis=0)
    d = z - mu0

    if n < min_ratio * D:
        var = base.var(axis=0) + 1e-6
        E = np.sqrt((d ** 2 / var).sum(axis=1) + EPS)
        return E, {"n_baseline": int(n), "estimator": "diagonal", "shrinkage": None}

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=False).fit(base)
        Sigma, shrink = lw.covariance_, float(lw.shrinkage_)
        est = "ledoit_wolf"
    except Exception:                                   # fallback nếu sklearn thiếu
        Sigma = np.cov(base, rowvar=False) + 1e-6 * np.eye(D)
        shrink, est = None, "empirical+ridge"

    Sinv = np.linalg.pinv(np.atleast_2d(Sigma))
    E = np.sqrt(np.maximum(np.einsum("hi,ij,hj->h", d, Sinv, d), 0.0) + EPS)
    return E, {"n_baseline": int(n), "estimator": est, "shrinkage": shrink}


def compute_raw_components(hour_features_df, baseline_frac=0.2, min_baseline_mult=3.0,
                           log_compress=True):
    """
    Tính E_raw (abnormality) và C_raw (state-change magnitude) cho MỘT bearing.

    KHÔNG dùng: H_fail, RUL, Deg, hay bất kỳ nhãn nào. Chỉ dùng baseline_frac đầu đời.
    -> Hàm này chạy được cho cả bearing held-out mà KHÔNG gây leakage.

    C_h = ||z_h - z_{h-1}||_2 (biên: C_0 := C_1). Đây là ĐỘ LỚN thay đổi trạng thái, không
    phải "rate" theo nghĩa chia cho Δt; vì Δt hằng số trong mỗi dataset nên nó tỉ lệ với rate
    (xem R2 #16 — paper phải nói rõ điều này).

    ------------------------------------------------------------------------------------
    log_compress=True (MẶC ĐỊNH — sửa lỗi thiết kế phát hiện 2026-08-05, xem R2 #15):
    ------------------------------------------------------------------------------------
    Khoảng cách Mahalanobis là đại lượng NHÂN TÍNH: trên một quỹ đạo run-to-failure nó trải
    3-4 BẬC ĐỘ LỚN (đo được: E_hi của biên train = 342 ... 6354 tuỳ fold). Chuẩn hoá min-max
    TUYẾN TÍNH trên đại lượng như vậy nghiền 98.8% số snapshot xuống dưới 0.1 -> chỉ số phẳng
    gần như suốt đời rồi vọt lên 1 ở cuối, và KHÔNG BAO GIỜ vượt ngưỡng cảnh báo τ=0.6.

    Nén log trước khi chuẩn hoá tuyến tính là thực hành CHUẨN cho đại lượng khoảng cách/năng
    lượng (chính là lý do ngành rung động dùng thang dB). Đo được trên dữ liệu thật:

        chỉ tiêu                      tuyến tính   log
        PRONOSTIA IQR                  0.0008      0.0603   (rộng gấp 75x)
        PRONOSTIA % snapshot < 0.1     98.8%       11.4%
        PRONOSTIA monotonicity         0.0637      0.0715   (tốt hơn)
        PRONOSTIA rho vs Deg           0.805       0.824    (tốt hơn)
        PRONOSTIA cảnh báo nổ trên     2/17        13/17
        PRONOSTIA precision (τ=0.6)    0.118       0.753
        XJTU      precision (τ=0.6)    0.457       0.815
        XJTU      recall / lead        0.101/-187  0.488/-10

    Đặt log_compress=False để tái lập biến thể tuyến tính (dùng cho bảng ablation chuẩn hoá).
    """
    df = hour_features_df.sort_values("hour_id").reset_index(drop=True)
    F = df[VIB_FEATURES].to_numpy(dtype=float)
    H = len(F)
    D = F.shape[1]

    # Baseline khoẻ: đủ lớn để ước lượng hiệp phương sai, nhưng không vượt quá nửa vòng đời.
    n_base = int(max(min_baseline_mult * D, baseline_frac * H))
    n_base = int(min(max(n_base, 5), max(H // 2, 5), H))
    baseline_idx = np.arange(n_base)

    z = _robust_scale(F, baseline_idx)
    E_raw, info = _mahalanobis_shrunk(z, baseline_idx, min_ratio=min_baseline_mult)

    diff = np.diff(z, axis=0)
    C_raw = np.sqrt(np.sum(diff ** 2, axis=1) + EPS)
    C_raw = np.concatenate([[C_raw[0]] if len(C_raw) else [0.0], C_raw])   # C_0 := C_1

    # Nén log: đại lượng khoảng cách là NHÂN TÍNH, trải nhiều bậc độ lớn (xem docstring).
    if log_compress:
        E_raw = np.log1p(E_raw)
        C_raw = np.log1p(C_raw)

    info.update({"H": int(H), "baseline_frac_effective": round(n_base / max(H, 1), 4),
                 "log_compress": bool(log_compress)})
    return {"hour_id": df["hour_id"].to_numpy(), "E_raw": E_raw, "C_raw": C_raw,
            "z": z, "info": info}


# =========================================================================================
# (2) FIT THAM SỐ — CHỈ TRÊN BEARING TRAIN CỦA FOLD
# =========================================================================================

def _monotonicity(x):
    """Coble & Hines: |#(Δ>0) - #(Δ<0)| / (K-1). 1 = đơn điệu hoàn toàn."""
    dx = np.diff(np.asarray(x, float))
    if len(dx) == 0:
        return 0.0
    return float(abs((dx > 0).sum() - (dx < 0).sum()) / len(dx))


def _spearman(x, y):
    """Spearman CÓ DẤU (không lấy trị tuyệt đối — xem R2 #23)."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else 0.0


def _fitness(vtoi_list, deg_list, objective):
    """
    Fitness = TRUNG BÌNH QUA CÁC BEARING (không nối chuỗi — nối sẽ tạo bước nhảy giả ở biên).
      objective='mono+corr' : monotonicity + spearman CÓ DẤU với deg  (mặc định)
      objective='mono'      : chỉ monotonicity — KHÔNG dùng nhãn deg  (control ctl_nodeg, R2 #6)
    """
    vals = []
    for v, d in zip(vtoi_list, deg_list):
        m = _monotonicity(v)
        vals.append(m if objective == "mono" else m + _spearman(v, d))
    return float(np.mean(vals)) if vals else -1e9


def fit_vtoi_params(train_components, train_degs, lo_pct=1.0, hi_pct=99.0,
                    objective="mono+corr", grid_step=0.01):
    """
    Học tham số chỉ số TỪ CÁC BEARING TRAIN CỦA FOLD.

    train_components : list dict trả về bởi compute_raw_components() — CHỈ bearing train.
    train_degs       : list mảng deg = h/H_fail của các bearing TRAIN (được phép: train có nhãn).
    lo_pct/hi_pct    : PERCENTILE dùng làm biên min-max (thay vì min/max tuyệt đối) -> bền với
                       một snapshot ngoại lai duy nhất (R1 #4).
    objective        : 'mono+corr' | 'mono'.

    Trả về dict tham số ĐÓNG BĂNG. Tuyệt đối KHÔNG được nhận dữ liệu của bearing held-out.
    """
    assert len(train_components) >= 2, "Cần >= 2 bearing train để fit tham số VTOI."
    if objective != "mono":
        assert len(train_degs) == len(train_components), "train_degs phải khớp train_components."

    E_all = np.concatenate([c["E_raw"] for c in train_components])
    C_all = np.concatenate([c["C_raw"] for c in train_components])
    E_lo, E_hi = np.percentile(E_all, [lo_pct, hi_pct])
    C_lo, C_hi = np.percentile(C_all, [lo_pct, hi_pct])

    Es = [np.clip((c["E_raw"] - E_lo) / (E_hi - E_lo + EPS), 0.0, 1.0) for c in train_components]
    Cs = [np.clip((c["C_raw"] - C_lo) / (C_hi - C_lo + EPS), 0.0, 1.0) for c in train_components]
    degs = train_degs if objective != "mono" else [None] * len(Es)

    # Grid search xác định (deterministic, tái lập 100% — không phụ thuộc seed/optimizer).
    best_a, best_fit = 0.5, -1e9
    sweep = []
    for a in np.arange(0.0, 1.0 + 1e-12, grid_step):
        vt = [a * e + (1.0 - a) * c for e, c in zip(Es, Cs)]
        f = _fitness(vt, degs, objective)
        sweep.append((round(float(a), 4), round(float(f), 6)))
        if f > best_fit:
            best_fit, best_a = f, float(a)

    return {
        "a": round(best_a, 6), "b": round(1.0 - best_a, 6),
        "E_lo": float(E_lo), "E_hi": float(E_hi),
        "C_lo": float(C_lo), "C_hi": float(C_hi),
        "objective": objective, "fitness": round(float(best_fit), 6),
        "n_train_bearings": len(train_components),
        "lo_pct": lo_pct, "hi_pct": hi_pct,
        "sweep": sweep,                       # -> Fig. quét trọng số (R3 #3)
    }


def apply_vtoi(components, params):
    """
    Áp tham số ĐÃ ĐÓNG BĂNG cho MỘT bearing bất kỳ (train / val / held-out test).
    KHÔNG dùng nhãn. Giá trị ngoài dải train bị CLIP về [0,1] -> chỉ số BÃO HOÀ ở 1 trong
    cuối đời; `sat_frac` ghi lại tỉ lệ bão hoà để công bố trong paper (R1 #4, R2 #15).
    """
    E_n = (components["E_raw"] - params["E_lo"]) / (params["E_hi"] - params["E_lo"] + EPS)
    C_n = (components["C_raw"] - params["C_lo"]) / (params["C_hi"] - params["C_lo"] + EPS)
    sat = float(np.mean((E_n > 1.0) | (E_n < 0.0)))
    E_n = np.clip(E_n, 0.0, 1.0)
    C_n = np.clip(C_n, 0.0, 1.0)
    vt = params["a"] * E_n + params["b"] * C_n
    return {"E_norm": E_n, "C_norm": C_n, "VTOI": vt, "sat_frac": sat}


# =========================================================================================
# (3) CỘT CONDITIONING + CÁC CONTROL (R2 #25)
# =========================================================================================

def _stable_rng(*keys, seed=0):
    """RNG xác định theo (tên fold, tên bearing, seed) -> tái lập 100% qua các lần chạy."""
    h = hashlib.md5(("|".join(map(str, keys))).encode()).hexdigest()
    return np.random.default_rng((int(h[:8], 16) + int(seed)) % (2 ** 32))


def _vel_acc(x):
    """Sai phân lùi bậc 1 và bậc 2 TRONG MỘT bearing (đệm 0 ở đầu chuỗi)."""
    x = np.asarray(x, float)
    vel = np.zeros_like(x); acc = np.zeros_like(x)
    if len(x) >= 2:
        vel[1:] = x[1:] - x[:-1]
    if len(x) >= 3:
        acc[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    return vel, acc


def build_conditioning_columns(components, params, params_mono, bearing_name, fold_name,
                               H_max_train, deg=None, hc_features=None, hc_stats=None, seed=42):
    """
    Sinh TOÀN BỘ cột conditioning (đề xuất + 10 control của R2 #25) cho MỘT bearing.

    H_max_train : max H_fail QUA CÁC BEARING TRAIN -> dùng cho `elapsed_norm`.
                  TUYỆT ĐỐI không dùng H_fail của bearing test (nếu dùng sẽ thành oracle).
    deg         : h/H_fail của chính bearing này — CHỈ dùng cho cột ORACLE `Deg` (ghi rõ trong bảng).
    hc_features : DataFrame 10 đặc trưng handcrafted của bearing này (control ctl_hc10).
    hc_stats    : (mean, std) tính TRÊN BEARING TRAIN -> chuẩn hoá hc leakage-free.

    Trả về DataFrame index theo hour_id, sẵn sàng merge vào bảng targets cấp snapshot.
    """
    out = pd.DataFrame({"hour_id": components["hour_id"]})
    H = len(out)

    # ---- Đề xuất ----
    ap = apply_vtoi(components, params)
    out["E_norm"] = ap["E_norm"]
    out["C_norm"] = ap["C_norm"]
    out["VTOI"] = ap["VTOI"]
    v, a = _vel_acc(ap["VTOI"])
    out["VTOI_vel"], out["VTOI_acc"] = v, a

    # ---- ctl_nodeg: trọng số fit CHỈ theo monotonicity (không dùng nhãn deg) — R2 #6 ----
    out["VTOI_mono"] = apply_vtoi(components, params_mono)["VTOI"]

    # ---- ctl_random: trọng số NGẪU NHIÊN, cố định theo (fold, bearing, seed) — R2 #25 ----
    rng = _stable_rng(fold_name, bearing_name, "rand", seed=seed)
    a_r = float(rng.uniform(0.0, 1.0))
    out["VTOI_rand"] = a_r * ap["E_norm"] + (1.0 - a_r) * ap["C_norm"]

    # ---- ctl_shuffled: hoán vị VTOI TRONG bearing -> giữ phân phối biên, PHÁ cấu trúc thời gian ----
    rng2 = _stable_rng(fold_name, bearing_name, "shuf", seed=seed)
    out["VTOI_shuf"] = ap["VTOI"][rng2.permutation(H)]

    # ---- ctl_elapsed: đồng hồ thuần, chuẩn hoá bằng H_max của TRAIN (không phải H_fail test) ----
    out["elapsed_norm"] = np.arange(H, dtype=float) / max(float(H_max_train), 1.0)

    # ---- ctl_lifefrac: ORACLE (dùng H_fail thật) — trần trên, phải ghi rõ nhãn ORACLE ----
    out["Deg_oracle"] = (np.asarray(deg, float) if deg is not None
                         else np.arange(H, dtype=float) / max(H - 1, 1))

    # ---- ctl_hc10: 10 đặc trưng handcrafted, chuẩn hoá bằng thống kê TRAIN ----
    if hc_features is not None and hc_stats is not None:
        mu, sd = hc_stats
        hc = hc_features.sort_values("hour_id")[VIB_FEATURES].to_numpy(float)
        hc = (hc - mu) / (sd + EPS)
        hc = np.clip(hc, -5.0, 5.0)                     # chặn đuôi nặng (kurtosis có thể rất lớn)
        for i, name in enumerate(VIB_FEATURES):
            out[f"hc_{name}"] = hc[:, i]

    out["vtoi_sat_frac"] = ap["sat_frac"]
    return out


HC_COLS = [f"hc_{n}" for n in VIB_FEATURES]


# =========================================================================================
# (4) KIỂM TRA AN TOÀN — gọi trong mỗi fold, FAIL LOUD nếu có leakage
# =========================================================================================

def assert_no_leakage(params, holdout_name, train_names, val_names):
    """Chốt chặn: tham số phải được fit từ đúng số bearing train, và holdout không nằm trong đó."""
    assert holdout_name not in train_names, f"LEAKAGE: holdout '{holdout_name}' nằm trong train!"
    assert holdout_name not in val_names, f"LEAKAGE: holdout '{holdout_name}' nằm trong val!"
    assert params["n_train_bearings"] == len(train_names), (
        f"LEAKAGE: tham số fit trên {params['n_train_bearings']} bearing "
        f"nhưng fold có {len(train_names)} bearing train.")
    return True
