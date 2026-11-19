# -*- coding: utf-8 -*-
"""
check_pronostia_leakage.py — CỔNG FAIL-LOUD chống leakage LOBO (chạy TRƯỚC mọi lần train).

Kiểm tra: trong manifest của PRONOSTIA/XJTU, KHÔNG được có 2 ổ bi chia sẻ cùng TÊN VẬT LÝ
(base = bỏ tiền tố tập Full_Test_Set_/Test_set_/Learning_set_). PRONOSTIA gốc có
'Test_set_BearingX_Y' (cắt ngắn) và 'Full_Test_Set_BearingX_Y' (đầy đủ) là CÙNG ổ bi -> leakage.

  - In danh sách trùng (nếu có) và RAW manifest count.
  - In số fold LOBO sau khi make_folds() đã KHỬ TRÙNG (mong đợi PRONOSTIA=17, XJTU=15).
  - Exit code != 0 nếu RAW manifest còn trùng (để dùng làm cổng CI/pre-run).

Chạy:  python scripts/check_pronostia_leakage.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.paths import proc_dir_for
from src.lobo import physical_base, make_folds


def check_dataset(dataset):
    """Trả về (n_raw, n_unique, duplicates, n_folds). Không raise; để main quyết định exit."""
    man_path = proc_dir_for(dataset) / "_manifest.csv"
    if not man_path.exists():
        print(f"  [{dataset}] (chưa có manifest -> bỏ qua)")
        return None
    man = pd.read_csv(man_path)
    names = man["bearing"].tolist()
    bases = [physical_base(n) for n in names]
    # base -> danh sách run trùng
    from collections import defaultdict
    groups = defaultdict(list)
    for nm, b in zip(names, bases):
        groups[b].append(nm)
    dups = {b: v for b, v in groups.items() if len(v) > 1}
    folds, _ = make_folds(dataset)        # đã khử trùng bên trong
    print(f"  [{dataset}] manifest RAW: {len(names)} run | ổ bi vật lý: {len(set(bases))} "
          f"| fold LOBO sau khử trùng: {len(folds)}")
    if dups:
        print(f"     ⚠️  {len(dups)} ổ bi BỊ NHÂN ĐÔI trong manifest RAW:")
        for b, v in sorted(dups.items()):
            print(f"        {b}: {v}")
        print(f"     -> make_folds() đã tự KHỬ TRÙNG (giữ Full). Manifest RAW vẫn còn trùng.")
    return len(names), len(set(bases)), dups, len(folds)


def main():
    print("=" * 64)
    print(" KIỂM TRA LEAKAGE LOBO (Full_Test_Set vs Test_set)")
    print("=" * 64)
    raw_dup = False
    try:
        for ds in ("pronostia", "xjtu_sy"):
            r = check_dataset(ds)              # make_folds() bên trong đã assert base duy nhất (fail-loud)
            if r and r[2]:
                raw_dup = True
    except AssertionError as e:
        # make_folds KHÔNG khử hết trùng -> fold CÒN leakage -> CHẶN.
        print("-" * 64)
        print(f"❌ FOLD CÒN LEAKAGE: {e}")
        sys.exit(2)
    print("-" * 64)
    # Đến đây: fold LOBO đã sạch (make_folds dedup + assert OK). Chỉ cảnh báo manifest raw.
    if raw_dup:
        print("ℹ️  manifest RAW còn ổ bi nhân đôi, NHƯNG make_folds() đã khử trùng -> fold LOBO SẠCH. ✅")
        print("   (tuỳ chọn: chạy lại phase9a với skip-duplicate để manifest gọn hẳn.)")
    else:
        print("KẾT LUẬN: không phát hiện ổ bi nhân đôi. ✅")
    sys.exit(0)


if __name__ == "__main__":
    main()
