# -*- coding: utf-8 -*-
"""
datasets.py — GỘP DỮ LIỆU & TẠO PyTorch Dataset + CHIA TRAIN/VAL/TEST chống leakage.

Nhiệm vụ:
  1) build_hour_targets(): gộp temp_by_hour + mtoi_by_hour + labels_by_hour -> 1 bảng đích/giờ.
  2) chronological_split(): chia theo THỜI GIAN (không random theo cửa sổ) để TRÁNH LEAKAGE
     (plan mục 14: cửa sổ cùng 1 giờ KHÔNG được rơi cả train lẫn test).
  3) MTOIWindowDataset: Dataset trả về mỗi mẫu (1 cửa sổ rung) kèm nhãn của GIỜ chứa nó.

Mỗi mẫu gồm:
  vib    : tensor [2, L]   (cửa sổ rung, đã chuẩn hoá nếu truyền mean/std)
  temp   : tensor [4]      ([T_bearing, T_atm, ΔT, G_norm], đã chuẩn hoá)
  stage  : int (0..3)      nhãn giai đoạn
  mtoi   : float           nhãn MTOI mục tiêu (auxiliary supervision)
  rul    : float           nhãn RUL mục tiêu
  warning: int (0/1)       nhãn cảnh báo
  hour_id: int             giờ chứa cửa sổ (để gộp dự đoán theo giờ khi đánh giá)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.paths import PROC_MENDELEY


# 4 đặc trưng nhiệt độ đưa vào temp-encoder (plan: [T̄_b, T̄_a, ΔT̄, G]).
TEMP_INPUT_COLS = ["T_bearing_mean", "T_atm_mean", "DeltaT_mean", "G_norm"]


def build_hour_targets(proc_dir=None):
    """
    Gộp 3 file theo hour_id -> 1 bảng đích cấp giờ, gồm:
      nhiệt độ (cho temp encoder) + MTOI mục tiêu + RUL + stage + warning + G_norm.
    """
    proc_dir = Path(proc_dir or PROC_MENDELEY)
    temp = pd.read_csv(proc_dir / "temp_by_hour.csv")
    mtoi = pd.read_csv(proc_dir / "mtoi_by_hour.csv")
    labels = pd.read_csv(proc_dir / "labels_by_hour.csv")
    # Ghép lần lượt theo hour_id.
    df = temp.merge(mtoi, on="hour_id", how="inner").merge(labels, on="hour_id", how="inner")
    df = df.sort_values("hour_id").reset_index(drop=True)
    return df


def chronological_split(hour_ids_unique, train=0.6, val=0.2, test_end=1.0):
    """
    Chia tập giờ theo THỜI GIAN: train% đầu -> train, kế tiếp val% -> val, sau đó tới test_end% -> test.
    test_end < 1.0 dùng cho ROLLING evaluation (test chỉ là 1 CỬA SỔ, không phải toàn bộ phần cuối).
      Ví dụ rolling "Train 0-70% -> Test 70-85%": train+val=0.7, test_end=0.85.
    Trả về (train_hours, val_hours, test_hours) là các tập hour_id.
    """
    hrs = np.sort(np.unique(hour_ids_unique))            # các giờ, tăng dần
    H = len(hrs)
    n_tr = int(train * H)                                # số giờ train
    n_va = int(val * H)                                  # số giờ val
    n_end = int(test_end * H)                            # mốc kết thúc test
    train_hours = set(hrs[:n_tr].tolist())
    val_hours = set(hrs[n_tr:n_tr + n_va].tolist())
    test_hours = set(hrs[n_tr + n_va:n_end].tolist())    # test = [train+val, test_end)
    return train_hours, val_hours, test_hours


def compute_norm_stats(X, idx):
    """
    Tính mean/std theo từng kênh của tín hiệu rung trên TẬP TRAIN (idx).
    Trả về (mean[2,1], std[2,1]) để chuẩn hoá vib leakage-free.
    """
    sub = X[idx]                                         # [n, 2, L] phần train
    mean = sub.mean(axis=(0, 2), keepdims=True)[0]       # trung bình mỗi kênh -> [2,1]
    std = sub.std(axis=(0, 2), keepdims=True)[0] + 1e-8  # độ lệch chuẩn mỗi kênh
    return mean.astype(np.float32), std.astype(np.float32)


class MTOIWindowDataset(Dataset):
    """Dataset cấp cửa sổ cho mô hình đề xuất. Lọc theo tập giờ (split) để chống leakage."""

    def __init__(self, X, window_hour_ids, hour_targets, keep_hours,
                 vib_mean=None, vib_std=None, temp_mean=None, temp_std=None,
                 mtoi_col="MTOI_learnable", rul_col="RUL", cap_onset_frac=0.4, hi_cols=None):
        """
        X: mảng [N,2,L] tất cả cửa sổ. window_hour_ids: [N] giờ của từng cửa sổ.
        hour_targets: DataFrame đích cấp giờ (từ build_hour_targets).
        keep_hours: tập hour_id thuộc split này (train/val/test).
        vib_mean/std, temp_mean/std: thống kê chuẩn hoá (tính từ TRAIN, truyền vào để tránh leakage).
        mtoi_col: cột MTOI dùng làm nhãn ('MTOI_learnable' hoặc 'MTOI_fixed').
        rul_col : cột RUL dùng làm nhãn ('RUL' tuyến tính hoặc 'RUL_capped' piecewise).
        """
        self.mtoi_col = mtoi_col                          # tên cột MTOI mục tiêu
        self.rul_col = rul_col                            # tên cột RUL mục tiêu
        # Chỉ giữ các cửa sổ có hour_id nằm trong split.
        mask = np.isin(window_hour_ids, list(keep_hours))
        self.idx = np.where(mask)[0]                     # chỉ số cửa sổ được giữ
        self.X = X                                       # tham chiếu (không copy để tiết kiệm RAM)
        self.window_hour_ids = window_hour_ids

        # PRECOMPUTE tra cứu O(1) theo hour_id -> mảng numpy. TRÁNH pandas.loc mỗi cửa sổ
        # (vốn là nút thắt: chậm ~5-10x khi train hàng chục nghìn cửa sổ/epoch).
        ht = hour_targets.set_index("hour_id")
        self._cols = set(ht.columns)
        self._row_of = {int(h): i for i, h in enumerate(ht.index.to_numpy())}   # hour_id -> chỉ số hàng
        self._stage = ht["stage_label"].to_numpy(np.int64)
        self._mtoi = ht[mtoi_col].to_numpy(np.float32)
        self._rul = ht[rul_col].to_numpy(np.float32)
        self._warn = ht["warning_label"].to_numpy(np.int64)
        self._temp = ht[TEMP_INPUT_COLS].to_numpy(np.float32)                   # [H,4]
        self._bidx = ht["bearing_idx"].to_numpy(np.int64) if "bearing_idx" in self._cols else None
        self._hfail = ht["H_fail"].to_numpy(np.float32) if "H_fail" in self._cols else None
        self._life = ht["life_hours"].to_numpy(np.float32) if "life_hours" in self._cols else None

        # --- HI features -> ĐẦU VÀO RUL head (GĐ2.2). hi_cols = danh sách cột LINH HOẠT, vd:
        #   vib-only:  ['E_norm','C_norm','MTOI_woG','MTOI_woG_vel','MTOI_woG_acc']
        #   full:      ['E_norm','C_norm','G_norm','MTOI_learnable','MTOI_learnable_vel','MTOI_learnable_acc']
        # Cột thiếu -> 0 (guard). hi_dim = len(hi_cols).
        def _col(c):
            return ht[c].to_numpy(np.float32) if c in self._cols else np.zeros(len(ht), np.float32)
        self.hi_cols = list(hi_cols) if hi_cols else None
        if self.hi_cols:
            self._hi = np.stack([_col(c) for c in self.hi_cols], axis=1).astype(np.float32)  # [H,k]
            self.hi_dim = len(self.hi_cols)
        else:
            self._hi = None; self.hi_dim = 0
        # has_temp (1=có nhiệt độ thật). Vắng cột -> mặc định 1 (Mendeley có temp).
        self._has_temp = (ht["has_temp"].to_numpy(np.float32) if "has_temp" in self._cols
                          else np.ones(len(ht), np.float32))
        # Hệ số quy RUL(chuẩn hoá) -> GIỜ. RUL_capped saturate ở vùng khoẻ -> dùng R_early=(1-onset)*life.
        if self._life is not None:
            self._scale = self._life * ((1.0 - cap_onset_frac) if rul_col == "RUL_capped" else 1.0)
        else:
            self._scale = None

        # Thống kê chuẩn hoá (nếu None -> không chuẩn hoá).
        self.vib_mean = vib_mean; self.vib_std = vib_std
        self.temp_mean = temp_mean; self.temp_std = temp_std

    def __len__(self):
        return len(self.idx)                             # số cửa sổ trong split

    def __getitem__(self, i):
        j = self.idx[i]                                  # chỉ số cửa sổ thực trong X
        hid = int(self.window_hour_ids[j])               # giờ chứa cửa sổ này
        r = self._row_of[hid]                            # chỉ số hàng đích (O(1), không pandas.loc)

        # --- Tín hiệu rung ---
        vib = self.X[j].astype(np.float32)               # [2, L]
        if self.vib_mean is not None:                    # chuẩn hoá theo kênh (leakage-free)
            vib = (vib - self.vib_mean) / self.vib_std

        # --- Đặc trưng nhiệt độ [4] ---
        temp = self._temp[r].copy()
        if self.temp_mean is not None:
            temp = (temp - self.temp_mean) / self.temp_std

        # --- Trả về mẫu dạng dict tensor ---
        sample = {
            "vib": torch.from_numpy(vib),                            # [2, L]
            "temp": torch.from_numpy(temp),                          # [4]
            "stage": torch.tensor(int(self._stage[r]), dtype=torch.long),
            "mtoi": torch.tensor(float(self._mtoi[r]), dtype=torch.float32),
            "rul": torch.tensor(float(self._rul[r]), dtype=torch.float32),
            "warning": torch.tensor(int(self._warn[r]), dtype=torch.long),
            "hour_id": torch.tensor(hid, dtype=torch.long),
            # bearing_idx: định danh ổ bi để regularizer KHÔNG tính xuyên biên (LOBO). 0 nếu single-run.
            "bearing_idx": torch.tensor(int(self._bidx[r]) if self._bidx is not None else 0,
                                        dtype=torch.long),
            # has_temp -> per-sample masking (temp-adaptive fusion).
            "has_temp": torch.tensor(float(self._has_temp[r]), dtype=torch.float32),
        }
        # HI (E/C/[G]/MTOI/vel/acc) -> đầu vào RUL head (chỉ khi có hi_cols).
        if self._hi is not None:
            sample["hi"] = torch.from_numpy(self._hi[r].copy())
        # --- Đơn vị thời gian tuyệt đối (guard: chỉ có khi labels đã regen) ---
        if self._hfail is not None:
            sample["H_fail"] = torch.tensor(float(self._hfail[r]), dtype=torch.float32)
        if self._scale is not None:                       # hệ số quy RUL->giờ (đúng cho linear/capped)
            sample["rul_scale_hours"] = torch.tensor(float(self._scale[r]), dtype=torch.float32)
        return sample


def make_datasets(proc_dir=None, train=0.6, val=0.2, mtoi_target="MTOI_learnable", test_end=1.0,
                  rul_col="RUL", hi_cols=None):
    """
    Tiện ích tạo nhanh 3 Dataset (train/val/test) đã chuẩn hoá đúng cách (stats từ train).
    Trả về dict gồm datasets + thống kê chuẩn hoá + bảng đích.
    """
    proc_dir = Path(proc_dir or PROC_MENDELEY)
    X = np.load(proc_dir / "X_vib_windows.npy")          # [N,2,L]
    wh = np.load(proc_dir / "window_hour_ids.npy")       # [N]
    targets = build_hour_targets(proc_dir)               # bảng đích cấp giờ

    # Chia theo thời gian (test_end<1.0 -> rolling evaluation).
    train_hours, val_hours, test_hours = chronological_split(wh, train=train, val=val, test_end=test_end)

    # Tính thống kê chuẩn hoá vib trên TRAIN.
    train_mask = np.isin(wh, list(train_hours))
    vib_mean, vib_std = compute_norm_stats(X, np.where(train_mask)[0])

    # Thống kê chuẩn hoá temp trên TRAIN (theo giờ train).
    temp_train = targets[targets["hour_id"].isin(train_hours)][TEMP_INPUT_COLS].to_numpy(np.float32)
    temp_mean = temp_train.mean(axis=0); temp_std = temp_train.std(axis=0) + 1e-8

    # Tạo 3 dataset, dùng chung 1 bộ thống kê (từ train) + cùng cột MTOI/RUL mục tiêu.
    common = dict(vib_mean=vib_mean, vib_std=vib_std, temp_mean=temp_mean, temp_std=temp_std,
                  mtoi_col=mtoi_target, rul_col=rul_col, hi_cols=hi_cols)
    ds_train = MTOIWindowDataset(X, wh, targets, train_hours, **common)
    ds_val = MTOIWindowDataset(X, wh, targets, val_hours, **common)
    ds_test = MTOIWindowDataset(X, wh, targets, test_hours, **common)

    return {
        "train": ds_train, "val": ds_val, "test": ds_test,
        "hi_dim": ds_train.hi_dim,
        "targets": targets,
        "splits": {"train": train_hours, "val": val_hours, "test": test_hours},
        "norm": {"vib_mean": vib_mean, "vib_std": vib_std,
                 "temp_mean": temp_mean, "temp_std": temp_std},
    }
