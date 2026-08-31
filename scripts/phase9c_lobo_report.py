# -*- coding: utf-8 -*-
"""
phase9c_lobo_report.py — BẢNG + HÌNH cho kết quả LOBO (chạy sau phase9b).

Đọc:
  results/tables/lobo_<dataset>_perfold.csv   (1 dòng/fold)
  results/tables/lobo_<dataset>_summary.csv   (mean±std qua fold)
  results/predictions/lobo_<dataset>_<bearing>_proposed.csv  (đường dự đoán bearing held-out)

Sinh:
  results/tables/lobo_main_performance.csv     (bảng chính: mean±std, mỗi (dataset,config) 1 dòng — định dạng đẹp)
  results/tables/lobo_main_performance.md      (markdown để dán vào paper)
  results/figures/lobo_<dataset>_hi_trajectories.png   (MTOI dự đoán vs RUL trên vài bearing held-out)

An toàn với kết quả CHƯA đủ: bỏ qua phần thiếu, không lỗi.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- STYLE Q1: chữ to + in đậm, nét dày, độ phân giải cao (dễ đọc khi in) ----
plt.rcParams.update({
    "font.size": 15, "font.weight": "bold",
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 17, "axes.labelweight": "bold",
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.titlesize": 19, "figure.titleweight": "bold",
    "axes.linewidth": 1.8, "lines.linewidth": 2.6, "patch.linewidth": 1.4,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "savefig.dpi": 220, "figure.dpi": 220, "savefig.bbox": "tight",
})

from src.utils.paths import TABLES_DIR, FIGURES_DIR, PREDICTIONS_DIR
from src.utils.logger import get_logger, section

DATASETS = ("pronostia", "xjtu_sy")

# Các metric trình bày trong bảng chính (RUL-FIRST). RUL theo giờ + score + PI lên đầu.
SHOW = [
    ("rul_mae_hours",  "RUL MAE (h)",     "↓"),
    ("rul_rmse_hours", "RUL RMSE (h)",    "↓"),
    ("rul_rmse",       "RUL RMSE (norm)", "↓"),
    ("rul_phm_score",  "RUL PHM-Score",   "↓"),
    ("rul_asym_score", "RUL Asym",        "↓"),
    ("lead_time",      "Lead time",       "↑"),
    ("false_alarm_rate", "False-alarm",   "↓"),
    ("rul_picp",       "PICP",            "↑"),
    ("rul_mpiw",       "MPIW",            "↓"),
    ("mtoi_spearman",  "MTOI |Spearman|", "↑"),
    ("stage_macro_f1", "Stage Macro-F1",  "↑"),
]


def build_main_table(logger):
    """Gộp summary của các dataset -> bảng chính mean±std (csv + md)."""
    rows = []
    for ds in DATASETS:
        f = TABLES_DIR / f"lobo_{ds}_summary.csv"
        if not f.exists():
            logger.info(f"  (bỏ qua) chưa có {f.name}")
            continue
        s = pd.read_csv(f)
        for _, r in s.iterrows():
            row = {"dataset": ds, "config": r["config"], "n_folds": int(r.get("n_folds", 0))}
            for key, nice, _arrow in SHOW:
                m, sd = r.get(f"{key}_mean"), r.get(f"{key}_std")
                row[nice] = (f"{m:.3f}±{sd:.3f}" if pd.notna(m) else "—")
            rows.append(row)
    if not rows:
        logger.info("  Chưa có summary nào -> bỏ qua bảng chính.")
        return None
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES_DIR / "lobo_main_performance.csv", index=False)
    # Markdown
    md = ["# LOBO main performance (mean ± std qua các fold)\n",
          "| dataset | config | folds | " + " | ".join(n for _, n, _ in SHOW) + " |",
          "|" + "---|" * (3 + len(SHOW))]
    for _, r in tbl.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in
                  ["dataset", "config", "n_folds"] + [n for _, n, _ in SHOW]) + " |")
    (TABLES_DIR / "lobo_main_performance.md").write_text("\n".join(md), encoding="utf-8")
    logger.info(f"  Ghi {TABLES_DIR/'lobo_main_performance.csv'} + .md")
    print("\n" + "\n".join(md))
    return tbl


def plot_rul_trajectories(ds, k=3, logger=None):
    """HÌNH HEADLINE: RUL dự đoán vs RUL thật (+ dải PI q10-q90 nếu có) trên k bearing held-out;
    MTOI dự đoán vẽ mờ làm chỉ số giải thích phụ. Quy về GIỜ nếu có life_hours."""
    preds = sorted(PREDICTIONS_DIR.glob(f"lobo_{ds}_*_proposed.csv"))
    if not preds:
        if logger: logger.info(f"  ({ds}) chưa có file dự đoán -> bỏ qua hình.")
        return
    sized = []
    for p in preds:
        try:
            df = pd.read_csv(p); sized.append((len(df), p, df))
        except Exception:
            continue
    sized.sort(reverse=True)
    sized = sized[:k]
    if not sized:
        return
    fig, axes = plt.subplots(1, len(sized), figsize=(6.2 * len(sized), 5.0), squeeze=False)
    for ax, (_, p, df) in zip(axes[0], sized):
        name = p.stem.replace(f"lobo_{ds}_", "").replace("_proposed", "")
        x = df["hour_id"] if "hour_id" in df else np.arange(len(df))
        # Quy RUL về GIỜ nếu có life_hours (đơn vị "giờ còn lại tới hỏng").
        scale = df["life_hours"] if "life_hours" in df else 1.0
        ylab = "RUL (hours)" if "life_hours" in df else "RUL (normalized)"
        if "rul_q10" in df and "rul_q90" in df:
            ax.fill_between(x, df["rul_q10"] * scale, df["rul_q90"] * scale,
                            alpha=.20, color="tab:blue", label="PI q10-q90")
        if "rul_pred" in df: ax.plot(x, df["rul_pred"] * scale, color="tab:blue", lw=3.0, label="RUL predicted")
        if "rul_true" in df: ax.plot(x, df["rul_true"] * scale, "--", color="k", lw=2.2, label="RUL true")
        if "mtoi_pred" in df:  # auxiliary interpretable index (right axis, faded)
            ax2 = ax.twinx(); ax2.plot(x, df["mtoi_pred"], ":", color="tab:red", lw=2.0, alpha=.6)
            ax2.set_ylim(-0.05, 1.05); ax2.set_ylabel("MTOI (aux.)", color="tab:red", fontsize=14, fontweight="bold")
            ax2.tick_params(axis="y", labelsize=12, colors="tab:red")
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_xlabel("Snapshot"); ax.set_ylabel(ylab)
        ax.legend(fontsize=12, loc="upper right")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
    fig.suptitle(f"LOBO - predicted vs true RUL on held-out bearings ({ds})")
    fig.tight_layout()
    out = FIGURES_DIR / f"lobo_{ds}_rul_trajectories.png"
    fig.savefig(out); plt.close(fig)
    if logger: logger.info(f"  Ghi {out}")


def main():
    logger = get_logger("phase9c")
    section("PHASE 9C — BÁO CÁO LOBO (bảng + hình)", logger)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    build_main_table(logger)
    for ds in DATASETS:
        plot_rul_trajectories(ds, k=3, logger=logger)
    # Gộp bảng trọng số học (nếu có) cho tiện.
    wfiles = sorted(TABLES_DIR.glob("learned_mtoi_weights_*.csv"))
    logger.info(f"  Có {len(wfiles)} file trọng số học MTOI trong results/tables/.")
    logger.info("HOÀN TẤT Phase 9C.")


if __name__ == "__main__":
    main()
