# -*- coding: utf-8 -*-
"""
q1_tier_d_figures.py — TIER D (reviewer Q1 v2): hình chuẩn journal từ DỮ LIỆU CÓ SẴN.
  D1 per-bearing LOBO boxplot (RUL MAE) proposed vs baselines.
  D2 paired-improvement plot (proposed vs best baseline, mỗi bearing).
  D3 PR curve early-warning (MTOI, gộp 2 dataset).
  D4 ablation bar chart trên LOBO.
Hình nhãn tiếng Anh. CHỮ TO + ĐẬM, có nhãn SỐ trên cột, highlight đề xuất nổi bật.
Chạy: python scripts/q1_tier_d_figures.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- STYLE Q1: chữ to, in đậm, nét dày, độ phân giải cao (dễ đọc khi in) ----
plt.rcParams.update({
    "font.size": 16, "font.weight": "bold",
    "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.labelsize": 18, "axes.labelweight": "bold",
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "legend.title_fontsize": 14,
    "figure.titlesize": 20, "figure.titleweight": "bold",
    "axes.linewidth": 1.8, "lines.linewidth": 2.8, "patch.linewidth": 1.5,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "savefig.dpi": 220, "figure.dpi": 220, "savefig.bbox": "tight",
})
PROP = "#1565c0"   # xanh đậm = đề xuất (nổi bật)
GREY = "#cfcfcf"   # xám = baseline
EDGE = "black"

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

BASE = ["mtoi_vib_static", "transformer_vib", "tcn_vib", "tcn_transformer_vib", "cnn_bilstm_attn_vib"]
LAB = {"mtoi_vib_static": "VTOI\n(proposed)", "transformer_vib": "Transf.", "tcn_vib": "TCN",
       "tcn_transformer_vib": "TCN-Tr.", "cnn_bilstm_attn_vib": "CNN-BiLSTM"}


def _bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def d1_boxplot():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (ds, title) in zip(axes, [("pronostia", "PRONOSTIA"), ("xjtu_sy", "XJTU-SY")]):
        d = pd.read_csv(TAB / f"lobo_{ds}_perfold.csv")
        data = [d[d.config == c]["rul_mae_hours"].dropna().to_numpy() for c in BASE]
        bp = ax.boxplot(
            data, labels=[LAB[c] for c in BASE], showmeans=True, patch_artist=True,
            widths=0.62,
            medianprops=dict(color="#d84315", linewidth=3.0),
            whiskerprops=dict(linewidth=2.0), capprops=dict(linewidth=2.0),
            boxprops=dict(linewidth=1.8),
            meanprops=dict(marker="^", markerfacecolor="#2e7d32",
                           markeredgecolor="black", markersize=12, markeredgewidth=1.5),
        )
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(PROP if i == 0 else GREY)
            box.set_edgecolor(EDGE)
        # nhãn SỐ mean căn giữa phía trên mỗi box (tránh cắt mép)
        for i, arr in enumerate(data):
            if len(arr):
                m = float(np.mean(arr))
                ax.annotate(f"{m:.3f}", (i + 1, m), textcoords="offset points",
                            xytext=(0, 13), ha="center", fontsize=13, fontweight="bold",
                            color=(PROP if i == 0 else "#333333"),
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_title(f"{title} (leave-one-bearing-out)")
        ax.set_ylabel("RUL MAE (hours)")
        ax.grid(alpha=0.35, axis="y", linewidth=1.2)
        ax.tick_params(axis="x", labelsize=12)
        _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(FIG / "q1_fig_lobo_boxplot.png")
    plt.close(fig)
    print("[D1] q1_fig_lobo_boxplot.png")


def d2_paired():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (ds, title) in zip(axes, [("pronostia", "PRONOSTIA"), ("xjtu_sy", "XJTU-SY")]):
        d = pd.read_csv(TAB / f"lobo_{ds}_perfold.csv")
        prop = d[d.config == "mtoi_vib_static"].set_index("holdout")["rul_mae_hours"]
        bases = [c for c in BASE if c != "mtoi_vib_static"]
        bb = d[d.config.isin(bases)].groupby("holdout")["rul_mae_hours"].min()
        common = sorted(set(prop.index) & set(bb.index))
        diff = (bb.loc[common] - prop.loc[common]).sort_values()  # >0 = proposed tốt hơn
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in diff.values]
        ax.bar(range(len(diff)), diff.values, color=colors, edgecolor=EDGE, linewidth=1.2)
        ax.axhline(0, color="k", lw=1.5)
        ax.set_title(f"{title} (leave-one-bearing-out)")
        ax.set_xlabel("Held-out bearing (sorted)")
        ax.set_ylabel("Best-baseline $-$ proposed MAE (h)")
        won = int((diff.values > 0).sum())
        ax.text(0.03, 0.92, f"proposed better on {won}/{len(diff)}", transform=ax.transAxes,
                fontsize=16, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#e8f5e9", ec="#2e7d32", lw=1.8))
        ax.grid(alpha=0.35, axis="y", linewidth=1.2)
        _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(FIG / "q1_fig_paired_improvement.png")
    plt.close(fig)
    print("[D2] q1_fig_paired_improvement.png")


def d3_pr_curve():
    pr = pd.read_csv(TAB / "q1_c2_pr_curve.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # vùng y hiển thị: 0.4 -> 1.0 (phóng to để thấy rõ trade-off; mọi điểm đều >= 0.48)
    Y_LO = 0.4
    ax.fill_between(pr["recall"], pr["precision"], Y_LO, color=PROP, alpha=0.08, zorder=0)
    ax.plot(pr["recall"], pr["precision"], "o-", color=PROP, ms=8, mec="black", mew=1.0)
    for _, r in pr.iterrows():
        if abs(r["tau"] - 0.6) < 0.03:
            ax.scatter([r["recall"]], [r["precision"]], color="#c62828", s=180, zorder=5,
                       edgecolor="black", linewidth=1.5)
            ax.annotate(rf"  $\tau$=0.6 (P={r['precision']:.2f}, R={r['recall']:.2f})",
                        (r["recall"], r["precision"]), fontsize=15, fontweight="bold",
                        color="#c62828")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Early-warning precision-recall (VTOI, pooled)")
    ax.set_xlim(0, 1.02); ax.set_ylim(Y_LO, 1.01)
    ax.set_yticks(np.arange(Y_LO, 1.001, 0.1))
    ax.grid(alpha=0.35, linewidth=1.2)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(FIG / "q1_fig_pr_curve.png")
    plt.close(fig)
    print("[D3] q1_fig_pr_curve.png")


def d4_ablation():
    abl = pd.read_csv(TAB / "q1_lobo_ablation.csv")
    order = ["no index (baseline)", "VTOI loss only, not fed to head", "E-only", "E+C (proposed)",
             "E+C, no smoothness", "E+C, no monotonicity"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, (ds, title) in zip(axes, [("pronostia", "PRONOSTIA"), ("xjtu_sy", "XJTU-SY")]):
        present = abl[abl.dataset == ds]["config"].values
        sub = abl[abl.dataset == ds].set_index("config").reindex([o for o in order if o in present])
        colors = [PROP if "proposed" in c else GREY for c in sub.index]
        vals = sub["rul_mae_h"].values
        se = sub["rul_mae_std"].values / np.sqrt(sub["n"].values)   # standard error of the mean
        ax.barh(range(len(sub)), vals, color=colors, edgecolor=EDGE, linewidth=1.3,
                xerr=se, capsize=5, error_kw=dict(elinewidth=1.6, capthick=1.6, ecolor="#555555"))
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("RUL MAE (hours)")
        ax.set_title(f"{title} ablation (leave-one-bearing-out)")
        ax.grid(alpha=0.35, axis="x", linewidth=1.2)
        # nhãn SỐ đặt SAU đầu mút error bar -> không đè lên cột/error bar
        xmax = float(np.nanmax(vals + se)) if len(vals) else 1.0
        ax.set_xlim(0, xmax * 1.30)
        for i, (v, e) in enumerate(zip(vals, se)):
            ax.text(v + e + xmax * 0.025, i, f"{v:.3f}", va="center", ha="left", fontsize=13,
                    fontweight="bold", color=(PROP if "proposed" in sub.index[i] else "#333333"))
        _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(FIG / "q1_fig_ablation.png")
    plt.close(fig)
    print("[D4] q1_fig_ablation.png")


def main():
    d1_boxplot(); d2_paired(); d3_pr_curve(); d4_ablation()
    print("Tier D figures done.")


if __name__ == "__main__":
    main()
