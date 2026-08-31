# -*- coding: utf-8 -*-
"""
lobo.py — PHASE 9-LOBO: PIPELINE HUẤN LUYỆN LEAVE-ONE-BEARING-OUT (đa-bearing).

Đây là phần BỔ SUNG để bài mạnh chuẩn Q1: huấn luyện mô hình đề xuất (MTOIModel)
theo kiểu LEAVE-ONE-BEARING-OUT trên PRONOSTIA (đa bearing) & XJTU-SY (đa bearing),
nghĩa là: train trên N-1 bearing, test trên 1 bearing CHƯA TỪNG THẤY -> bảng main-performance
+ phân loại stage CÂN BẰNG (vì train thấy đủ vòng đời nhiều bearing).

Gồm 2 lớp chức năng:
  (A) build_bearing_artifacts(): xử lý 1 bearing -> đủ 6 file (giống hợp đồng Mendeley) GHI LÊN DRIVE:
        X_vib_windows.npy [N,2,L], window_hour_ids.npy [N],
        temp_by_hour.csv, hour_features.csv, mtoi_by_hour.csv, labels_by_hour.csv
      process_dataset(): lặp toàn bộ bearing của 1 dataset -> ghi PROC_<X>/<bearing>/ + manifest.
  (B) Assembler LOBO: gom nhiều bearing (offset hour_id để TOÀN CỤC duy nhất), tính thống kê
      chuẩn hoá CHỈ từ bearing train (leakage-free), tạo {train,val,test} MTOIWindowDataset.

Quyết định lưu trữ (để KHÔNG mất khi Colab disconnect):
  - RAW giải nén ở /content (local, nhanh, giải nén lại được từ zip trên Drive).
  - ARTIFACT đã xử lý (per-bearing) GHI LÊN DRIVE: PROC_PRONOSTIA/<bearing>, PROC_XJTU/<bearing>.
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import (
    PROC_PRONOSTIA, PROC_XJTU, proc_dir_for, raw_dir_for,
)
from src.utils.logger import get_logger, section, Timer
from src.segmentation import make_windows_from_signal
from src.features import window_feature_vector, FEATURE_NAMES
from src.mtoi import build_mtoi
from src.labels import build_labels, SAMPLE_INTERVAL_S
from src.datasets import MTOIWindowDataset, build_hour_targets, TEMP_INPUT_COLS


# Cấu hình mặc định MỖI dataset (khớp configs/pronostia.yaml & xjtu_sy.yaml).
DATASET_CFG = {
    "pronostia": dict(L=2560, overlap=0.0, max_windows_per_hour=1, has_temp_default=True),
    "xjtu_sy":   dict(L=4096, overlap=0.5, max_windows_per_hour=16, has_temp_default=False),
    # IMS/NASA (R3 #2): 20480 mẫu @20 kHz mỗi snapshot -> cùng L/overlap với XJTU cho đối xứng.
    "ims":       dict(L=4096, overlap=0.5, max_windows_per_hour=16, has_temp_default=False),
}

# Bỏ qua bearing quá ngắn (không đủ để tạo MTOI/stage có nghĩa).
MIN_SNAPSHOTS = 10


# =========================================================================================
# (A) XỬ LÝ 1 BEARING -> ĐỦ 6 FILE GHI LÊN DRIVE
# =========================================================================================
def build_bearing_artifacts(name, hour_iter, out_dir, L, overlap, max_windows_per_hour,
                            temp_array=None, has_temp=False, dataset_name="lobo",
                            baseline_frac=0.2, train_frac=0.6, k_temp=5,
                            stage_mode="time", tau=0.6, q=3,
                            min_snapshots=MIN_SNAPSHOTS, band=None, logger=None, force=False):
    """
    Xử lý ĐẦY ĐỦ 1 bearing run-to-failure -> ghi 6 file vào out_dir (trên Drive).
    Trả về dict tóm tắt (n_snapshots, n_windows, has_temp) hoặc None nếu bearing quá ngắn.

    has_temp: nếu True và temp_array != None -> dùng nhiệt độ (PRONOSTIA có nhiệt độ) cho MTOI đầy đủ.
              nếu False -> MTOI rút gọn (E+C); vẫn ghi cột nhiệt độ = 0 để Dataset đồng nhất.
    """
    logger = logger or get_logger("lobo")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / "labels_by_hour.csv"
    if done_flag.exists() and not force:
        # Đã xử lý xong bearing này -> đọc lại tóm tắt (idempotent, sống sót disconnect).
        try:
            wh = np.load(out_dir / "window_hour_ids.npy")
            n_win = int(len(wh)); n_snap = int(len(np.unique(wh)))
            logger.info(f"  [skip] {name}: đã có artifact ({n_snap} snapshot, {n_win} cửa sổ).")
            return {"bearing": name, "n_snapshots": n_snap, "n_windows": n_win,
                    "has_temp": bool(has_temp), "out_dir": str(out_dir)}
        except Exception:
            pass  # file hỏng -> xử lý lại

    use_temp = bool(has_temp and temp_array is not None)

    # --- Duyệt từng snapshot: cắt cửa sổ + tính đặc trưng median/snapshot ---
    all_windows, all_hids, feat_rows = [], [], []
    for hid, vib in hour_iter:
        wins = make_windows_from_signal(vib, L=L, overlap=overlap, max_windows=max_windows_per_hour)
        if len(wins) == 0:
            continue
        feats = np.stack([window_feature_vector(w, band=band) for w in wins], axis=0)
        med = np.median(feats, axis=0)
        all_windows.append(wins.astype(np.float32))
        all_hids.append(np.full(len(wins), int(hid), dtype=np.int32))
        row = {"hour_id": int(hid)}
        row.update({fn: med[i] for i, fn in enumerate(FEATURE_NAMES)})
        feat_rows.append(row)

    n_snap = len(feat_rows)
    if n_snap < min_snapshots:
        logger.info(f"  [bỏ qua] {name}: chỉ {n_snap} snapshot (< {min_snapshots}).")
        return None

    X = np.concatenate(all_windows, axis=0)                # [N,2,L]
    hour_ids = np.concatenate(all_hids, axis=0)            # [N]
    hf = pd.DataFrame(feat_rows).sort_values("hour_id").reset_index(drop=True)

    # --- Bảng nhiệt độ theo snapshot (đồng nhất cột với Mendeley) ---
    H = len(hf)
    if use_temp:
        # Nội suy nhiệt độ về đúng số snapshot (PRONOSTIA: chỉ có nhiệt độ Ổ BI, không ambient).
        t = np.asarray(temp_array, dtype=float)
        xq = np.linspace(0.0, 1.0, H); xp = np.linspace(0.0, 1.0, len(t))
        t_aligned = np.interp(xq, xp, t)
        T_bearing = t_aligned
        T_atm = np.zeros(H)                                # không có ambient -> 0
        DeltaT = t_aligned                                 # G dùng độ dốc nhiệt độ ổ bi
    else:
        T_bearing = np.zeros(H); T_atm = np.zeros(H); DeltaT = np.zeros(H)

    temp_df = pd.DataFrame({
        "hour_id": hf["hour_id"].to_numpy(),
        "T_bearing_mean": T_bearing, "T_atm_mean": T_atm, "DeltaT_mean": DeltaT,
    })
    # Gắn nhiệt độ vào hour_features để build_mtoi dùng (khi has_temp=True).
    hf["T_bearing_mean"] = T_bearing; hf["T_atm_mean"] = T_atm; hf["DeltaT_mean"] = DeltaT

    # --- Ghi các file gốc lên Drive ---
    np.save(out_dir / "X_vib_windows.npy", X)
    np.save(out_dir / "window_hour_ids.npy", hour_ids.astype(np.int32))
    temp_df.to_csv(out_dir / "temp_by_hour.csv", index=False)
    hf.to_csv(out_dir / "hour_features.csv", index=False)

    # --- MTOI + nhãn (per-bearing; mỗi bearing có baseline/khởi đầu riêng) ---
    build_mtoi(proc_dir=out_dir, baseline_frac=baseline_frac, train_frac=train_frac,
               k_temp=k_temp, has_temp=use_temp, dataset_name=f"{dataset_name}_perbearing")
    # Đơn vị thời gian tuyệt đối cho RUL-centric: truyền cadence theo dataset (10s/60s/3600s).
    build_labels(proc_dir=out_dir, stage_mode=stage_mode, tau=tau, q=q,
                 mtoi_col="MTOI_learnable",
                 sample_interval_s=SAMPLE_INTERVAL_S.get(dataset_name))

    logger.info(f"  [ok] {name}: {n_snap} snapshot, {X.shape[0]} cửa sổ, "
                f"has_temp={use_temp} -> {out_dir}")
    return {"bearing": name, "n_snapshots": int(n_snap), "n_windows": int(X.shape[0]),
            "has_temp": bool(use_temp), "out_dir": str(out_dir)}


def process_dataset(dataset, max_windows_per_hour=None, L=None, overlap=None,
                    max_bearings=None, force=False, logger=None):
    """
    Xử lý TOÀN BỘ bearing của 1 dataset ('pronostia' | 'xjtu_sy') -> ghi per-bearing artifact lên Drive
    + manifest PROC_<X>/_manifest.csv. Trả về DataFrame manifest.
    """
    logger = logger or get_logger("lobo")
    assert dataset in DATASET_CFG, f"dataset phải thuộc {list(DATASET_CFG)}"
    cfg = DATASET_CFG[dataset]
    L = L or cfg["L"]; overlap = cfg["overlap"] if overlap is None else overlap
    mwph = max_windows_per_hour or cfg["max_windows_per_hour"]
    proc_root = proc_dir_for(dataset)
    proc_root.mkdir(parents=True, exist_ok=True)

    section(f"LOBO build artifacts — {dataset.upper()} "
            f"(L={L}, overlap={overlap}, max_win/snapshot={mwph})", logger)

    # Nạp loader + danh sách bearing tương ứng.
    if dataset == "pronostia":
        from src.data import pronostia_loader as ld
        bearings = ld.find_bearing_dirs(raw_dir_for(dataset))
    elif dataset == "ims":
        from src.data import ims_loader as ld
        bearings = ld.find_bearing_dirs(raw_dir_for(dataset))
    else:
        from src.data import xjtu_loader as ld
        bearings = ld.find_bearing_dirs(raw_dir_for(dataset))

    items = list(bearings.items())
    if max_bearings:
        items = items[:max_bearings]
    logger.info(f"Tìm thấy {len(bearings)} bearing; xử lý {len(items)}.")

    rows = []
    for i, (name, bdir) in enumerate(items):
        out_dir = proc_root / name
        with Timer(f"{dataset} [{i+1}/{len(items)}] {name}", logger):
            if dataset == "pronostia":
                n_acc = ld.count_acc_files(bdir)
                temp_arr = ld.load_pronostia_temperature_aligned(bdir, n_acc)
                has_temp = temp_arr is not None
                info = build_bearing_artifacts(
                    name, ld.iter_pronostia_bearing(bdir), out_dir,
                    L=L, overlap=overlap, max_windows_per_hour=mwph,
                    temp_array=temp_arr, has_temp=has_temp, dataset_name=dataset,
                    force=force, logger=logger)
            elif dataset == "ims":
                # IMS gom 4 bearing trong CÙNG một thư mục test -> phải truyền thêm uid
                # để loader biết lấy đúng kênh cảm biến của bearing đó.
                info = build_bearing_artifacts(
                    name, ld.iter_ims_bearing(bdir, name), out_dir,
                    L=L, overlap=overlap, max_windows_per_hour=mwph,
                    temp_array=None, has_temp=False, dataset_name=dataset,
                    force=force, logger=logger)
            else:
                info = build_bearing_artifacts(
                    name, ld.iter_xjtu_bearing(bdir), out_dir,
                    L=L, overlap=overlap, max_windows_per_hour=mwph,
                    temp_array=None, has_temp=False, dataset_name=dataset,
                    force=force, logger=logger)
        if info:
            rows.append(info)

    manifest = pd.DataFrame(rows)
    man_path = proc_root / "_manifest.csv"
    manifest.to_csv(man_path, index=False)
    logger.info(f"Ghi manifest {man_path} ({len(manifest)} bearing).")
    if len(manifest):
        n_temp = int(manifest["has_temp"].sum())
        logger.info(f"  Tổng: {len(manifest)} bearing | có nhiệt độ: {n_temp} | "
                    f"tổng cửa sổ: {int(manifest['n_windows'].sum())}")
    return manifest


# =========================================================================================
# (B) ASSEMBLER LOBO: GOM NHIỀU BEARING -> {train,val,test} MTOIWindowDataset
# =========================================================================================
HID_STRIDE = 1_000_000  # offset hour_id theo bearing để TOÀN CỤC duy nhất (tránh đụng độ).


def load_bearing_pack(out_dir, bearing_idx, has_temp=1, mtoi_col="MTOI_learnable"):
    """
    Nạp artifact 1 bearing và OFFSET hour_id để duy nhất toàn cục.
    has_temp: 1 nếu ổ bi có nhiệt độ thật (per-sample masking ở fusion).
    Trả về (X[N,2,L], gh[N] hour_id toàn cục, targets DataFrame với hour_id toàn cục).
    """
    out_dir = Path(out_dir)
    X = np.load(out_dir / "X_vib_windows.npy")
    wh = np.load(out_dir / "window_hour_ids.npy").astype(np.int64)
    targets = build_hour_targets(out_dir)                  # gộp temp+mtoi+labels theo hour_id
    off = bearing_idx * HID_STRIDE
    gh = wh + off
    targets = targets.copy()
    targets["hour_id"] = targets["hour_id"].astype(np.int64) + off
    targets["bearing_idx"] = bearing_idx
    targets["has_temp"] = int(has_temp)                    # cờ nhiệt độ (per-bearing) cho masking
    return X.astype(np.float32), gh, targets


def prepare_fold(dataset, holdout_name, val_names, train_names, logger=None):
    """
    NẠP DỮ LIỆU 1 fold LOBO TỪ DRIVE ĐÚNG 1 LẦN (dùng lại cho mọi config -> tiết kiệm I/O Drive 4x).
    Gom cửa sổ + đích cấp giờ (offset hour_id theo bearing), tính thống kê chuẩn hoá CHỈ trên TRAIN.
    Trả về 'pack' dict (mảng + thống kê) -> datasets_from_pack() tạo view nhẹ theo từng mtoi_target.
    """
    logger = logger or get_logger("lobo")
    proc_root = proc_dir_for(dataset)

    # GUARD chống leakage: holdout (theo BASE ổ bi vật lý) KHÔNG được xuất hiện trong train/val.
    b_ho = physical_base(holdout_name)
    for nm in list(train_names) + list(val_names):
        assert physical_base(nm) != b_ho, (
            f"LEAKAGE: '{nm}' cùng ổ bi vật lý với holdout '{holdout_name}' "
            f"(base='{b_ho}'). Chạy make_folds() đã khử trùng.")

    # Gán chỉ số bearing ổn định (theo thứ tự tên) để offset hour_id nhất quán.
    all_names = list(train_names) + list(val_names) + [holdout_name]
    name2idx = {n: i for i, n in enumerate(sorted(set(all_names)))}

    # Cờ has_temp theo từng ổ bi (đọc manifest) -> per-sample masking + chuẩn hoá temp đúng.
    man = pd.read_csv(proc_root / "_manifest.csv")
    has_temp_map = dict(zip(man["bearing"], man["has_temp"].astype(int)))

    def pack_group(names):
        Xs, ghs, tgs = [], [], []
        for n in names:
            X, gh, tg = load_bearing_pack(proc_root / n, name2idx[n],
                                          has_temp=has_temp_map.get(n, 0))
            Xs.append(X); ghs.append(gh); tgs.append(tg)
        if not Xs:
            return None
        return (np.concatenate(Xs, 0), np.concatenate(ghs, 0),
                pd.concat(tgs, ignore_index=True))

    tr = pack_group(list(train_names))
    va = pack_group(list(val_names))
    te = pack_group([holdout_name])
    if tr is None or te is None:
        raise RuntimeError(f"Fold {holdout_name}: thiếu dữ liệu train/test.")

    X_tr, gh_tr, tg_tr = tr
    # --- Thống kê chuẩn hoá CHỈ TỪ TRAIN (leakage-free) ---
    vib_mean = X_tr.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)   # [2,1]
    vib_std = (X_tr.std(axis=(0, 2), keepdims=True)[0] + 1e-8).astype(np.float32)
    # Chuẩn hoá nhiệt độ CHỈ trên ổ bi CÓ temp (tránh thiên lệch vì ổ bi temp=0).
    tmask = tg_tr["has_temp"].to_numpy() > 0.5 if "has_temp" in tg_tr.columns else np.ones(len(tg_tr), bool)
    temp_real = tg_tr.loc[tmask, TEMP_INPUT_COLS].to_numpy(np.float32)
    if len(temp_real) >= 2:
        temp_mean = temp_real.mean(axis=0); temp_std = temp_real.std(axis=0) + 1e-8
    else:                                                  # fold không có ổ bi temp -> stats trung tính
        temp_mean = np.zeros(len(TEMP_INPUT_COLS), np.float32)
        temp_std = np.ones(len(TEMP_INPUT_COLS), np.float32)

    logger.info(f"  Fold[{holdout_name}]: train {len(train_names)}b/{len(gh_tr)}w | "
                f"val {len(val_names)}b | test 1b/{len(te[1])}w")
    return {
        "tr": tr, "va": va, "te": te,
        "norm": dict(vib_mean=vib_mean, vib_std=vib_std, temp_mean=temp_mean, temp_std=temp_std),
        "n_train_bearings": len(train_names), "n_val_bearings": len(val_names),
        "holdout": holdout_name,
    }


def datasets_from_pack(pack, mtoi_target="MTOI_learnable", train_subsample=1, rul_col="RUL",
                       hi_cols=None):
    """Tạo {train,val,test} MTOIWindowDataset (view nhẹ, KHÔNG nạp lại Drive) cho 1 config."""
    common = dict(**pack["norm"], mtoi_col=mtoi_target, rul_col=rul_col, hi_cols=hi_cols)

    def make_ds(grp, sub=1):
        if grp is None:
            return None
        X, gh, tg = grp
        keep = set(np.unique(gh).tolist())
        ds = MTOIWindowDataset(X, gh, tg, keep, **common)
        if sub and sub > 1:                               # lấy thưa cửa sổ TRAIN để tăng tốc
            ds.idx = ds.idx[::sub]
        return ds

    return {
        "train": make_ds(pack["tr"], sub=train_subsample),
        "val": make_ds(pack["va"]) if pack["va"] is not None else make_ds(pack["te"]),
        "test": make_ds(pack["te"]),
        "n_train_bearings": pack["n_train_bearings"], "holdout": pack["holdout"],
    }


def assemble_lobo_fold(dataset, holdout_name, val_names, train_names,
                       mtoi_target="MTOI_learnable", train_subsample=1, logger=None):
    """Tiện ích 1 phát: prepare_fold + datasets_from_pack (cho dùng đơn lẻ/kiểm thử)."""
    pack = prepare_fold(dataset, holdout_name, val_names, train_names, logger=logger)
    ds = datasets_from_pack(pack, mtoi_target=mtoi_target, train_subsample=train_subsample)
    ds["norm"] = pack["norm"]; ds["n_val_bearings"] = pack["n_val_bearings"]
    return ds


# Tiền tố TẬP của PRONOSTIA/FEMTO — strip để lấy TÊN Ổ BI VẬT LÝ (base) chống trùng.
_SET_PREFIXES = ("Full_Test_Set_", "Test_set_", "Learning_set_")


def physical_base(name):
    """Tên ổ bi VẬT LÝ: bỏ tiền tố tập (Full_Test_Set_/Test_set_/Learning_set_) nếu có.
    PRONOSTIA lặp cùng ổ bi giữa Test_set (cắt ngắn) và Full_Test_Set (đầy đủ); XJTU không có tiền tố."""
    for pref in _SET_PREFIXES:
        if name.startswith(pref):
            return name[len(pref):]
    return name


# IMS: 4 ổ bi nằm trên CÙNG một trục, đo ĐỒNG THỜI trong cùng một lần chạy đến hỏng.
# Rung động của giàn thử là chung, nên hai ổ bi cùng test KHÔNG độc lập về mặt thống kê.
_IMS_RUN_RE = re.compile(r"^(Test\d+)_Bearing\d+$")


def group_key(name):
    """
    Khoá GOM NHÓM cho LOBO — đơn vị nhỏ nhất được coi là ĐỘC LẬP.

    Khác `physical_base` (dùng để khử trùng bản ghi của CÙNG một ổ bi):
      - PRONOSTIA / XJTU-SY : mỗi ổ bi là một lần chạy riêng -> khoá = physical_base.
      - IMS                 : 4 ổ bi dùng chung trục, chung tải, đo cùng lúc. Nếu tách
                              Test1_Bearing3 (test) khỏi Test1_Bearing4 (train) thì mô hình
                              vẫn thấy rung động của CHÍNH giàn thử đó tại CHÍNH khoảng thời
                              gian đó -> đúng loại leakage mà bài này phê phán.
                              Vì vậy khoá = LẦN CHẠY ('Test1'), tức leave-one-RUN-out.

    Đây là lựa chọn bảo thủ và phải công bố rõ trong bài: nó làm tập train của IMS rất nhỏ
    (1-2 ổ bi/fold), nên IMS được dùng làm BẰNG CHỨNG BỔ TRỢ ở phụ lục, không phải benchmark.
    """
    m = _IMS_RUN_RE.match(name)
    if m:
        return m.group(1)
    return physical_base(name)


def dedup_bearings(man, logger=None):
    """
    KHỬ TRÙNG ổ bi vật lý: PRONOSTIA có cả 'Test_set_BearingX_Y' (cắt ngắn) lẫn
    'Full_Test_Set_BearingX_Y' (run ĐẦY ĐỦ) của CÙNG một ổ bi -> giữ run ĐẦY ĐỦ, bỏ run cắt ngắn.
    Nếu không khử, LOBO sẽ train trên ổ bi đem test (leakage). Trả về manifest đã khử trùng.
    """
    man = man.copy()
    man["_base"] = man["bearing"].map(physical_base)
    rows, dropped = [], []
    for base, g in man.groupby("_base", sort=True):
        if len(g) == 1:
            rows.append(g.iloc[0]); continue
        full = g[g["bearing"].str.startswith("Full_Test_Set_")]
        # Ưu tiên Full_Test_Set; nếu không có thì giữ run NHIỀU snapshot nhất (đầy đủ nhất).
        chosen = (full.sort_values("n_snapshots").iloc[-1] if len(full)
                  else g.sort_values("n_snapshots").iloc[-1])
        rows.append(chosen)
        dropped += [b for b in g["bearing"].tolist() if b != chosen["bearing"]]
    out = pd.DataFrame(rows).drop(columns=["_base"]).reset_index(drop=True)
    if logger and dropped:
        logger.info(f"  [dedup] bỏ {len(dropped)} run TRÙNG ổ bi vật lý (giữ Full): {dropped}")
    # FAIL-LOUD: sau khử trùng, mỗi base phải duy nhất.
    bases = [physical_base(b) for b in out["bearing"]]
    assert len(set(bases)) == len(bases), f"LEAKAGE: còn base trùng sau dedup: {sorted(bases)}"
    return out


def make_folds(dataset, min_snapshots=MIN_SNAPSHOTS, logger=None):
    """
    Đọc manifest -> KHỬ TRÙNG ổ bi vật lý -> tạo danh sách fold LOBO. Mỗi fold: 1 bearing holdout
    (test), 1 bearing val (bearing kế tiếp có BASE KHÁC holdout), còn lại là train.
    Trả về (folds, has_temp_dataset) với folds = list[dict{holdout,val,train}].
    """
    proc_root = proc_dir_for(dataset)
    man = pd.read_csv(proc_root / "_manifest.csv")
    man = man[man["n_snapshots"] >= min_snapshots].reset_index(drop=True)
    man = dedup_bearings(man, logger=logger)               # CHỐNG LEAKAGE: 1 run/ổ bi vật lý
    names = sorted(man["bearing"].tolist())
    n = len(names)
    base = {nm: group_key(nm) for nm in names}      # IMS gom theo LẦN CHẠY, xem group_key()
    folds = []
    for i, ho in enumerate(names):
        # val = bearing kế tiếp có BASE KHÁC holdout (đảm bảo val != test, không cùng ổ bi vật lý).
        val = next((names[(i + k) % n] for k in range(1, n)
                    if base[names[(i + k) % n]] != base[ho]), None)
        train = [x for x in names if base[x] not in (base[ho], base[val])]
        folds.append({"holdout": ho, "val": [val], "train": train})
    has_temp_ds = bool(man["has_temp"].any())
    return folds, has_temp_ds
