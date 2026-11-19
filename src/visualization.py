# -*- coding: utf-8 -*-
"""
visualization.py — VẼ HÌNH CHO PAPER (plan mục 17).

Các hàm vẽ và LƯU hình ra results/figures/:
  - plot_mtoi_curve()      : Figure 3 — đường MTOI theo vòng đời (fixed vs learnable).
  - plot_hi_comparison()   : so sánh các HI đã chuẩn hoá theo thời gian.
  - plot_early_warning()   : Figure 4 — case study cảnh báo sớm (ngưỡng + thời điểm báo).
  - plot_training_history(): đường loss train/val theo epoch.
  - plot_learned_weights() : Figure 7 — trọng số a,b,c đã học.
  - plot_robustness()      : Figure 5 — đường suy giảm hiệu năng theo mức nhiễu.

Dùng matplotlib (không cần seaborn). Mỗi hàm trả về đường dẫn file đã lưu.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # backend không cần màn hình (an toàn trên Colab/headless)
import matplotlib.pyplot as plt
from pathlib import Path

# ---- STYLE Q1: chữ to + in đậm, nét dày (dễ đọc khi in) ----
plt.rcParams.update({
    "font.size": 15, "font.weight": "bold",
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 17, "axes.labelweight": "bold",
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.titlesize": 19, "figure.titleweight": "bold",
    "axes.linewidth": 1.8, "lines.linewidth": 2.6, "patch.linewidth": 1.4,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
})

from src.utils.paths import FIGURES_DIR, PROC_MENDELEY, TABLES_DIR


def _save(fig, name):
    """Lưu figure ra results/figures/<name>.png (300 DPI cho chất lượng paper)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Đã lưu {path}")
    return path


def plot_mtoi_curve(proc_dir=None, name="fig3_mtoi_curve.png"):
    """Figure 3: vẽ MTOI_fixed và MTOI_learnable theo hour_id, kèm các mốc stage."""
    proc_dir = Path(proc_dir or PROC_MENDELEY)
    mtoi = pd.read_csv(proc_dir / "mtoi_by_hour.csv")
    labels = pd.read_csv(proc_dir / "labels_by_hour.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(mtoi["hour_id"], mtoi["MTOI_fixed"], label="MTOI (fixed)", lw=1.8, alpha=0.8)
    ax.plot(mtoi["hour_id"], mtoi["MTOI_learnable"], label="MTOI (learnable)", lw=2.2)
    # Tô màu vùng theo stage để thấy giai đoạn suy giảm.
    colors = {0: "#e8f5e9", 1: "#fff9c4", 2: "#ffe0b2", 3: "#ffcdd2"}
    for h, st in zip(labels["hour_id"], labels["stage_label"]):
        ax.axvspan(h - 0.5, h + 0.5, color=colors.get(int(st), "white"), alpha=0.3, lw=0)
    ax.set_xlabel("Snapshot index (hour)"); ax.set_ylabel("MTOI")
    ax.set_title("MTOI trajectory over bearing life (shaded = degradation stage)")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_hi_comparison(proc_dir=None, name="fig_hi_comparison.png"):
    """Vẽ vài HI tiêu biểu (đã chuẩn hoá 0-1) theo thời gian để so trực quan với MTOI."""
    proc_dir = Path(proc_dir or PROC_MENDELEY)
    hf = pd.read_csv(proc_dir / "hour_features.csv")
    mtoi = pd.read_csv(proc_dir / "mtoi_by_hour.csv")

    def norm(x):
        x = np.asarray(x, float); return (x - x.min()) / (x.max() - x.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hf["hour_id"], norm(hf["RMS_x"] + hf["RMS_y"]), label="RMS", alpha=0.7)
    ax.plot(hf["hour_id"], norm(hf["Kurt_x"] + hf["Kurt_y"]), label="Kurtosis", alpha=0.7)
    ax.plot(hf["hour_id"], norm(hf["T_bearing_mean"]), label="Temperature", alpha=0.7)
    ax.plot(mtoi["hour_id"], norm(mtoi["MTOI_learnable"]), label="MTOI (ours)", lw=2.5, color="black")
    ax.set_xlabel("Snapshot index (hour)"); ax.set_ylabel("HI (normalized 0-1)")
    ax.set_title("Health-indicator comparison over time")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_early_warning(proc_dir=None, tau=0.6, q=3, name="fig4_early_warning.png"):
    """Figure 4: case study cảnh báo sớm — vẽ MTOI, ngưỡng τ, và thời điểm báo đầu tiên."""
    from src import metrics
    proc_dir = Path(proc_dir or PROC_MENDELEY)
    mtoi = pd.read_csv(proc_dir / "mtoi_by_hour.csv")
    hi = mtoi["MTOI_learnable"].to_numpy()
    first, _ = metrics.warning_from_threshold(hi, tau=tau, q=q)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(mtoi["hour_id"], hi, label="MTOI (learnable)", lw=2)
    ax.axhline(tau, color="red", ls="--", label=f"Threshold τ={tau}")
    if first is not None:                                  # mark the first warning time
        ax.axvline(first, color="green", ls=":", lw=2, label=f"First warning (hour {first})")
    ax.set_xlabel("Snapshot index (hour)"); ax.set_ylabel("MTOI")
    ax.set_title("MTOI-based early-warning case study")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_training_history(history, name="fig_training_history.png"):
    """Vẽ loss train/val theo epoch (theo dõi hội tụ)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training history"); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_learned_weights(dataset_name="mendeley", name="fig7_learned_weights.png"):
    """Figure 7: cột trọng số a(E), b(C), c(G) — cố định vs học được."""
    wpath = TABLES_DIR / f"learned_mtoi_weights_{dataset_name}.csv"
    w = pd.read_csv(wpath)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["a (E - abnormality)", "b (C - state-change)", "c (G - temp trend)"]
    x = np.arange(3)
    for i, t in enumerate(["fixed", "learnable"]):         # 2 nhóm cột
        row = w[w["type"] == t].iloc[0]
        vals = [row["a_E"], row["b_C"], row["c_G"]]
        ax.bar(x + i * 0.35, vals, width=0.35, label=t)
    ax.set_xticks(x + 0.175); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Weight"); ax.set_title("MTOI component weights: fixed vs learnable")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    return _save(fig, name)


def plot_robustness(table_path, name="fig5_robustness.png"):
    """Figure 5: hiệu năng (RUL RMSE thấp=tốt; MTOI Spearman cao=tốt) theo kịch bản nhiễu/mất cảm biến.
    Chú thích: dùng MTOI Spearman thay cho stage-F1 (vốn = 0 ở mọi kịch bản trong cấu hình RUL-primary)."""
    t = pd.read_csv(table_path)
    x = np.arange(len(t))
    fig, ax1 = plt.subplots(figsize=(11, 6.0))
    l1, = ax1.plot(x, t["rul_rmse"], "s--", color="tab:red", ms=11, mec="black",
                   mew=1.2, label="RUL RMSE (lower better)")
    ax1.set_ylabel("RUL RMSE", color="tab:red")
    ax1.tick_params(axis="y", colors="tab:red")
    # nhãn SỐ trên đường RMSE
    for xi, v in zip(x, t["rul_rmse"]):
        ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=12, fontweight="bold", color="tab:red")
    ax2 = ax1.twinx()                                      # trục y thứ 2 cho MTOI Spearman
    spear = t["mtoi_spearman"] if "mtoi_spearman" in t else t.get("stage_macro_f1")
    l2, = ax2.plot(x, spear, "o-", color="tab:blue", ms=11, mec="black",
                   mew=1.2, label="MTOI |rho| (higher better)")
    ax2.set_ylabel("MTOI rank correlation", color="tab:blue")
    ax2.tick_params(axis="y", colors="tab:blue")
    ax2.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(t["scenario"], rotation=30, ha="right", fontweight="bold")
    ax1.set_title("Model robustness under noise and sensor-dropout scenarios")
    ax1.grid(alpha=0.35, axis="y", linewidth=1.2)
    ax1.legend(handles=[l1, l2], loc="lower left", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, name)
