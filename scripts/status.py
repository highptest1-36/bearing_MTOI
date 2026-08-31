# -*- coding: utf-8 -*-
"""
status.py — XEM ĐANG Ở ĐÂU TRONG PIPELINE + GỢI Ý LỆNH TIẾP THEO.

Chạy:  python scripts/status.py
An toàn với độ trễ Drive: đọc file/manifest cụ thể thay vì liệt kê thư mục.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.paths import (PROC_PRONOSTIA, PROC_XJTU, PROC_MENDELEY,
                             TABLES_DIR, CHECKPOINTS_DIR)


def _n_manifest(p):
    f = Path(p) / "_manifest.csv"
    try:
        return len(pd.read_csv(f))
    except Exception:
        return 0


def _n_perfold(dataset):
    f = TABLES_DIR / f"lobo_{dataset}_perfold.csv"
    try:
        return len(pd.read_csv(f))
    except Exception:
        return 0


def _expected_trainings(dataset):
    """folds × configs theo manifest + danh sách config (an toàn nếu chưa đủ artifact)."""
    try:
        from src.lobo import make_folds
        from scripts.phase9b_lobo_train import configs_for
        folds, has_temp = make_folds(dataset)
        return len(folds) * len(configs_for(dataset, has_temp))
    except Exception:
        return None


def main():
    print("=" * 64)
    print(" MTOI-Bearing — TRẠNG THÁI PIPELINE")
    print("=" * 64)

    # --- PhaseTracker ---
    try:
        st = json.load(open(CHECKPOINTS_DIR / "state.json"))
        print("\n[Phase tracker]")
        for k, v in st.items():
            print(f"  {k:34s} {v.get('status')}")
    except Exception:
        st = {}
        print("\n[Phase tracker] (chưa có state.json)")

    # --- Artifact per-bearing ---
    npr, nxj = _n_manifest(PROC_PRONOSTIA), _n_manifest(PROC_XJTU)
    print("\n[Artifact per-bearing trên Drive]")
    print(f"  PRONOSTIA : {npr} bearing (mong đợi ~28)")
    print(f"  XJTU-SY   : {nxj} bearing (mong đợi ~15)")
    mendeley_ok = (PROC_MENDELEY / 'X_vib_windows.npy').exists()
    print(f"  Mendeley  : {'OK' if mendeley_ok else 'THIẾU'} (X_vib_windows.npy)")

    # --- LOBO progress ---
    print("\n[LOBO training]")
    suggestion = None
    for ds in ("pronostia", "xjtu_sy"):
        done = _n_perfold(ds)
        exp = _expected_trainings(ds)
        summ = (TABLES_DIR / f"lobo_{ds}_summary.csv").exists()
        print(f"  {ds:10s}: {done}/{exp if exp else '?'} lần-train | summary={'có' if summ else 'chưa'}")

    # --- GỢI Ý lệnh tiếp theo ---
    print("\n" + "-" * 64)
    if npr < 28 or nxj < 15:
        suggestion = "python scripts/phase9a_build_bearing_artifacts.py   # build artifact còn thiếu"
    else:
        pr_done, pr_exp = _n_perfold("pronostia"), _expected_trainings("pronostia")
        xj_done, xj_exp = _n_perfold("xjtu_sy"), _expected_trainings("xjtu_sy")
        if (pr_exp and pr_done < pr_exp) or (xj_exp and xj_done < xj_exp):
            suggestion = ("nohup python scripts/phase9b_lobo_train.py "
                          "> results/logs/phase9b.log 2>&1 &   # tiếp tục LOBO (tự bỏ qua fold đã xong)")
        elif not (TABLES_DIR / "lobo_pronostia_summary.csv").exists():
            suggestion = "python scripts/phase9b_lobo_train.py   # tổng hợp summary"
        elif (ROOT / "scripts/phase9c_lobo_report.py").exists():
            suggestion = "python scripts/phase9c_lobo_report.py   # bảng + hình LOBO"
        else:
            suggestion = "LOBO xong. Tiếp: phase5→6→7→8 (Mendeley) — xem RESUME_RUNBOOK.md mục 3 BƯỚC 4."
    print(f"👉 LỆNH TIẾP THEO:\n   {suggestion}")
    print("   (hoặc: bash resume.sh — tự chạy tiếp toàn bộ)")
    print("-" * 64)


if __name__ == "__main__":
    main()
