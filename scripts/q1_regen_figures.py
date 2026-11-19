# -*- coding: utf-8 -*-
"""
q1_regen_figures.py — TÁI TẠO TẤT CẢ 8 HÌNH dùng trong paper với CHỮ TO + IN ĐẬM,
rồi COPY sang thư mục template paper với đúng tên file.

Chạy (sau khi mount Drive, cd vào MTOI-Bearing):
    python scripts/q1_regen_figures.py

Không cần GPU, không train lại — chỉ đọc CSV/predictions/processed-data đã có.
"""
import sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
PAPER = ROOT.parent / "paper" / "ACCESS_latex_template_20240429"

# map: file nguồn trong results/figures  ->  tên file trong thư mục paper
COPY_MAP = {
    "q1_fig_lobo_boxplot.png": "q1_fig_lobo_boxplot.png",
    "q1_fig_paired_improvement.png": "q1_fig_paired_improvement.png",
    "q1_fig_ablation.png": "q1_fig_ablation.png",
    "q1_fig_pr_curve.png": "q1_fig_pr_curve.png",
    "q1_physics_mtoi_vs_defect.png": "fig_physics_defect.png",
    "q1_physics_envelope_spectrum.png": "fig_physics_envspec.png",
    "fig5_robustness.png": "fig_robustness.png",
    "lobo_pronostia_rul_trajectories.png": "fig_lobo_pronostia.png",
}


def main():
    print("=== [1/4] Tier-D figures (boxplot / paired / PR / ablation) ===")
    import q1_tier_d_figures as d
    d.d1_boxplot(); d.d2_paired(); d.d3_pr_curve(); d.d4_ablation()

    print("=== [2/4] Physics figures (defect overlay / envelope spectrum) ===")
    import q1_tier3_physics as ph
    ph.main()

    print("=== [3/4] LOBO RUL trajectories ===")
    from scripts.phase9c_lobo_report import plot_rul_trajectories
    for ds in ("pronostia", "xjtu_sy"):
        try:
            plot_rul_trajectories(ds)
        except Exception as ex:
            print(f"  (bỏ {ds}: {ex})")

    print("=== [4/4] Robustness ===")
    from src.visualization import plot_robustness
    rob = TAB / "table5_robustness_proposed_learnable.csv"
    if rob.exists():
        plot_robustness(rob)
    else:
        print(f"  (thiếu {rob.name}, bỏ qua robustness)")

    print("=== COPY sang thư mục paper ===")
    PAPER.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_name, dst_name in COPY_MAP.items():
        src = FIG / src_name
        if src.exists():
            shutil.copy2(src, PAPER / dst_name)
            print(f"  {src_name}  ->  {dst_name}")
            n += 1
        else:
            print(f"  [THIẾU] {src_name} (chưa tạo được)")
    print(f"Hoàn tất: đã copy {n}/{len(COPY_MAP)} hình sang {PAPER}")


if __name__ == "__main__":
    main()
