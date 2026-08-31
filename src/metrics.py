# -*- coding: utf-8 -*-
"""
metrics.py — CÁC HÀM ĐO LƯỜNG cho paper (HI quality + RUL + stage + early warning).

Các định nghĩa metric HI lấy theo Coble & Hines (2009) — đã được kiểm chứng trong phần
nghiên cứu tài liệu (xem paper/manuscript/02_related_work_and_references.md):
  - Monotonicity : |#(Δ>0) - #(Δ<0)| / (K-1)
  - Trendability : min_i |corr(x_i, t_i)|  (dạng tương quan)
  - Prognosability: exp( -std_i(x_fail) / mean_i|x_start - x_fail| )
  - Spearman với RUL: tương quan hạng.
Ngoài ra: MAE/RMSE/R2 (RUL), Accuracy/Macro-F1/Balanced-Acc (stage),
          smoothness, first-warning-time, lead-time, false-alarm-rate (early warning).
"""

import numpy as np
from scipy.stats import spearmanr           # tương quan hạng Spearman


# ============================= HI QUALITY METRICS =============================

def monotonicity(x):
    """
    Monotonicity của 1 chuỗi HI x: |#(Δ>0) - #(Δ<0)| / (K-1).
    Bằng 1 nếu HI tăng (hoặc giảm) đơn điệu hoàn toàn; 0 nếu lên xuống cân bằng.
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)                                   # sai phân bậc 1: x[k+1]-x[k]
    if len(dx) == 0:
        return 0.0
    n_pos = np.sum(dx > 0)                            # số bước tăng
    n_neg = np.sum(dx < 0)                            # số bước giảm
    return float(abs(n_pos - n_neg) / len(dx))       # chuẩn hoá theo (K-1)


def trendability(series_list, time_list=None):
    """
    Trendability (dạng tương quan): min trên các run của |corr(x_i, t_i)|.
    series_list: danh sách các chuỗi HI (mỗi run 1 chuỗi).
    time_list: danh sách trục thời gian tương ứng (mặc định 0,1,2,...).
    """
    corrs = []
    for i, x in enumerate(series_list):
        x = np.asarray(x, dtype=float)
        t = np.arange(len(x)) if time_list is None else np.asarray(time_list[i], dtype=float)
        if len(x) < 2 or np.std(x) < 1e-12:          # chuỗi quá ngắn hoặc phẳng -> tương quan 0
            corrs.append(0.0)
            continue
        c = np.corrcoef(x, t)[0, 1]                  # hệ số tương quan Pearson(x, t)
        corrs.append(abs(c))
    return float(np.min(corrs)) if corrs else 0.0    # lấy MIN trên các run (yêu cầu nhất quán)


def prognosability(series_list):
    """
    Prognosability: exp( -std_i(x_fail) / mean_i|x_start - x_fail| ).
    Đo mức độ giá trị HI tại thời điểm HỎNG có cụm chặt giữa các run hay không.
    """
    x_fail = np.array([np.asarray(x, float)[-1] for x in series_list])   # giá trị cuối (lúc hỏng)
    ranges = np.array([abs(np.asarray(x, float)[0] - np.asarray(x, float)[-1])
                       for x in series_list])                            # |đầu - cuối| mỗi run
    denom = np.mean(ranges) + 1e-12                                      # mẫu số
    return float(np.exp(-np.std(x_fail) / denom))


def spearman_corr(x, y):
    """Tương quan hạng Spearman giữa 2 chuỗi (trả về hệ số rho)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho, _ = spearmanr(x, y)                          # rho ∈ [-1, 1]
    return float(rho) if np.isfinite(rho) else 0.0


def smoothness(x):
    """
    Độ mượt: nghịch đảo của trung bình sai phân bậc 2 bình phương (càng LỚN càng mượt).
    Dùng để so sánh HI nào ít gồ ghề hơn.
    """
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    d2 = np.diff(x, n=2)                              # sai phân bậc 2
    return float(1.0 / (np.mean(d2 ** 2) + 1e-12))


# ============================= RUL REGRESSION METRICS =============================

def mae(y_true, y_pred):
    """Sai số tuyệt đối trung bình (Mean Absolute Error)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    """Căn của sai số bình phương trung bình (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score(y_true, y_pred):
    """Hệ số xác định R^2 = 1 - SS_res/SS_tot."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)          # tổng bình phương phần dư
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # tổng bình phương quanh trung bình
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# --------------------- RUL-CENTRIC: score bất đối xứng + khoảng tin cậy ---------------------

def asymmetric_score(y_true, y_pred, a_late=2.0, a_early=1.0):
    """
    Score bất đối xứng (TRUNG BÌNH tuyến tính): phạt OVER-PREDICT RUL (dự đoán còn NHIỀU thời gian
    hơn thực -> nguy hiểm: bỏ lỡ bảo trì) NẶNG hơn under-predict. Cùng đơn vị với đầu vào (giờ).
      e = y_pred - y_true ; pen = a_late·max(e,0) + a_early·max(-e,0).  Càng NHỎ càng tốt.
    """
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    pen = a_late * np.maximum(e, 0.0) + a_early * np.maximum(-e, 0.0)
    return float(np.mean(pen))


def phm_score(y_true_hours, y_pred_hours, a_early=13.0, a_late=10.0):
    """
    Score luỹ thừa kiểu PHM08/C-MAPSS (TỔNG): d = pred - true (giờ).
      d<0 (under/early): exp(-d/a_early) - 1   |  d>=0 (over/late): exp(d/a_late) - 1.
    a_late < a_early -> phạt LATE (over-predict) nặng hơn. Càng NHỎ càng tốt.
    """
    d = np.asarray(y_pred_hours, float) - np.asarray(y_true_hours, float)
    s = np.where(d < 0, np.exp(-d / a_early) - 1.0, np.exp(d / a_late) - 1.0)
    return float(np.sum(s))


def picp(y_true, q_lo, q_hi):
    """PICP — tỉ lệ điểm thật nằm TRONG khoảng [q_lo, q_hi] (mong đợi ~ mức danh nghĩa, vd 0.8)."""
    y = np.asarray(y_true, float); lo = np.asarray(q_lo, float); hi = np.asarray(q_hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(q_lo, q_hi):
    """MPIW — bề rộng trung bình của khoảng dự đoán (càng HẸP càng tốt, với PICP đạt danh nghĩa)."""
    return float(np.mean(np.asarray(q_hi, float) - np.asarray(q_lo, float)))


# ============================= STAGE CLASSIFICATION METRICS =============================

def classification_metrics(y_true, y_pred):
    """Trả về dict {accuracy, macro_f1, balanced_accuracy} cho bài phân loại stage."""
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# ============================= EARLY WARNING METRICS =============================

def warning_from_threshold(hi, tau, q=3):
    """
    Sinh cảnh báo theo luật: cảnh báo BẬT khi HI > tau LIÊN TỤC q giờ.
    Trả về (first_warning_hour hoặc None, mảng cờ cảnh báo theo giờ).

    NHẤT QUÁN HOÁ (đã sửa): cờ flags được BẬT từ giờ BẮT ĐẦU chuỗi vượt ngưỡng (h-q+1),
    không phải từ giờ thứ q. Nhờ đó np.argmax(flags) == first, và lead_time (dùng first) cùng
    false_alarm_rate (đếm flags) ở CÙNG một mốc thời gian.
    """
    hi = np.asarray(hi, float)
    flags = np.zeros(len(hi), dtype=int)             # cờ cảnh báo từng giờ
    first = None                                     # giờ cảnh báo đầu tiên
    run = 0                                          # đếm số giờ liên tục vượt ngưỡng
    for h in range(len(hi)):
        if hi[h] > tau:                              # nếu vượt ngưỡng
            run += 1
        else:
            run = 0                                  # đứt chuỗi -> đếm lại
        if run >= q:                                 # đủ q giờ liên tục -> đang trong vùng cảnh báo
            start = h - q + 1                        # giờ bắt đầu chuỗi vượt ngưỡng hiện tại
            flags[start:h + 1] = 1                   # bật cờ TỪ đầu chuỗi tới hiện tại (back-fill)
            if first is None:
                first = start                        # ghi nhận giờ cảnh báo đầu tiên
    return first, flags


def early_warning_metrics(hi, tau, failure_hour, onset_hour=None, q=3):
    """
    Tính các metric cảnh báo sớm cho 1 run:
      - first_warning : giờ cảnh báo đầu tiên
      - lead_time     : failure_hour - first_warning (càng lớn càng cảnh báo sớm)
      - false_alarm_rate: tỉ lệ cảnh báo sai trong vùng "khoẻ" (trước onset_hour)
    onset_hour: giờ bắt đầu suy giảm thật (nếu None -> ước lượng = 0.6*failure_hour).
    """
    first, flags = warning_from_threshold(hi, tau, q=q)     # chạy luật cảnh báo
    if onset_hour is None:
        onset_hour = int(0.6 * failure_hour)                # giả định vùng khoẻ ~ 60% đầu đời
    # Lead time: thời gian báo trước khi hỏng.
    lead_time = (failure_hour - first) if first is not None else 0
    # False alarm: số giờ bị cảnh báo trong vùng khoẻ / tổng số giờ vùng khoẻ.
    healthy = flags[:max(onset_hour, 1)]
    far = float(np.sum(healthy) / max(len(healthy), 1))
    return {
        "first_warning": first if first is not None else -1,
        "lead_time": int(lead_time),
        "false_alarm_rate": far,
    }
