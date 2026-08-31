# -*- coding: utf-8 -*-
"""
robustness.py — PHASE 8: KIỂM TRA ĐỘ BỀN (plan mục 16, Experiment 5).

Thử model (đã huấn luyện) dưới các điều kiện khắc nghiệt và so với fusion baseline:
  - Nhiễu Gaussian trên rung theo SNR: 10dB, 5dB, 0dB, -5dB.
  - Thiếu nhiệt độ (sensor mất tín hiệu).
  - Thiếu 1 trục rung (vib_y = 0).
  - Trôi nhiệt độ ổ bi (+3°C — trong ĐƠN VỊ VẬT LÝ thật, không phải đơn vị chuẩn hoá).
Mục tiêu chứng minh: model đề xuất GIẢM ÍT HƠN baseline khi dữ liệu xấu.

QUAN TRỌNG (đã sửa theo review): dataset gốc trả về 'temp' ĐÃ chuẩn hoá (z-score). Nếu cộng thẳng
drift=3 vào temp chuẩn hoá thì hoá ra +3 ĐỘ LỆCH CHUẨN, không phải +3°C. Vì vậy ta nhận thêm
temp_mean/temp_std (thống kê chuẩn hoá từ train) để quy mọi phép can thiệp nhiệt độ về ĐÚNG đơn vị °C:
  - Trôi +d°C:  temp_norm += d_raw / std   (chỉ tác động kênh nhiệt độ bị trôi).
  - Thiếu nhiệt độ: đặt giá trị THÔ = 0 (sensor đứt) rồi chuẩn hoá lại -> giá trị OOD thật.
"""

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

from src.utils.paths import TABLES_DIR
from src.utils.logger import get_logger, section
from src.evaluate import evaluate_model


class PerturbedDataset(Dataset):
    """Bọc 1 dataset gốc và áp 1 phép nhiễu lên 'vib'/'temp' khi lấy mẫu."""
    def __init__(self, base, mode="clean", snr_db=None, drift=0.0, seed=42,
                 temp_mean=None, temp_std=None):
        self.base = base                                  # dataset gốc
        self.mode = mode                                  # loại nhiễu
        self.snr_db = snr_db                              # mức SNR (nếu mode='noise')
        self.drift = drift                                # độ trôi nhiệt độ (°C)
        self.rng = np.random.default_rng(seed)            # bộ sinh nhiễu cố định seed

        # Lưu thống kê chuẩn hoá nhiệt độ (mỗi cái mảng [4]) để quy phép can thiệp về đơn vị °C.
        self.temp_std = None if temp_std is None else np.asarray(temp_std, dtype=np.float32)
        self.temp_mean = None if temp_mean is None else np.asarray(temp_mean, dtype=np.float32)
        if self.temp_std is not None:
            # Giá trị nhiệt độ THÔ=0 sau khi chuẩn hoá lại (dùng cho 'missing_temp').
            self._zero_norm = torch.from_numpy((0.0 - self.temp_mean) / self.temp_std)
            # Vector trôi trong KHÔNG GIAN CHUẨN HOÁ: chỉ ổ bi (kênh 0) và ΔT (kênh 2) bị trôi +d°C.
            drift_raw = np.array([self.drift, 0.0, self.drift, 0.0], dtype=np.float32)
            self._drift_norm = torch.from_numpy(drift_raw / self.temp_std)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]                             # lấy mẫu gốc (dict tensor)
        vib = sample["vib"].clone()                       # sao chép để không sửa gốc
        temp = sample["temp"].clone() if "temp" in sample else None

        if self.mode == "noise" and self.snr_db is not None:
            # Thêm nhiễu Gaussian theo SNR (tính trên công suất tín hiệu đã chuẩn hoá — SNR là tỉ số tương đối).
            sig_power = float((vib ** 2).mean())          # công suất tín hiệu
            snr = 10 ** (self.snr_db / 10.0)              # đổi dB -> tỉ số tuyến tính
            noise_power = sig_power / max(snr, 1e-9)       # công suất nhiễu cần thêm
            noise = torch.from_numpy(
                self.rng.normal(0, np.sqrt(noise_power), size=tuple(vib.shape)).astype("float32"))
            vib = vib + noise

        elif self.mode == "missing_temp" and temp is not None:
            # Sensor nhiệt độ mất tín hiệu: giá trị thô = 0 -> chuẩn hoá lại thành giá trị OOD.
            temp = self._zero_norm.clone() if self.temp_std is not None else torch.zeros_like(temp)

        elif self.mode == "missing_vib_y":
            vib[1] = 0.0                                   # xoá trục y

        elif self.mode == "temp_drift" and temp is not None:
            # Trôi nhiệt độ ổ bi +drift°C (quy về đơn vị chuẩn hoá nếu có thống kê).
            temp = temp + (self._drift_norm if self.temp_std is not None else self.drift)

        sample = dict(sample)                              # tạo bản sao dict
        sample["vib"] = vib
        if temp is not None:
            sample["temp"] = temp
        return sample


def run_robustness(model, base_test, device="cuda", run_name="proposed",
                   temp_mean=None, temp_std=None):
    """Chạy toàn bộ kịch bản robustness cho 1 model, trả về DataFrame kết quả."""
    logger = get_logger("robustness")
    section(f"PHASE 8 — ROBUSTNESS cho {run_name}", logger)

    # Danh sách kịch bản (tên hiển thị -> tham số PerturbedDataset).
    scenarios = [
        ("Clean", dict(mode="clean")),
        ("SNR_10dB", dict(mode="noise", snr_db=10)),
        ("SNR_5dB", dict(mode="noise", snr_db=5)),
        ("SNR_0dB", dict(mode="noise", snr_db=0)),
        ("SNR_-5dB", dict(mode="noise", snr_db=-5)),
        ("Missing_temp", dict(mode="missing_temp")),
        ("Missing_vib_y", dict(mode="missing_vib_y")),
        ("Temp_drift_+3C", dict(mode="temp_drift", drift=3.0)),
    ]

    rows = []
    for sc_name, kwargs in scenarios:
        # Truyền thống kê chuẩn hoá nhiệt độ để can thiệp đúng đơn vị °C.
        ds = PerturbedDataset(base_test, temp_mean=temp_mean, temp_std=temp_std, **kwargs)
        res = evaluate_model(model, ds, device=device, run_name=f"{run_name}_{sc_name}",
                             save=False)["metrics"]
        rows.append({
            "scenario": sc_name,
            "stage_macro_f1": round(res["stage_macro_f1"], 4),
            "rul_rmse": round(res["rul_rmse"], 4),
            "mtoi_spearman": round(res["mtoi_spearman"], 4),
            "lead_time": res["lead_time"],
        })
        logger.info(f"  {sc_name}: F1={rows[-1]['stage_macro_f1']} rulRMSE={rows[-1]['rul_rmse']}")

    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / f"table5_robustness_{run_name}.csv"
    table.to_csv(out, index=False)
    logger.info(f"Đã ghi {out}")
    print(table.to_string(index=False))
    return table
