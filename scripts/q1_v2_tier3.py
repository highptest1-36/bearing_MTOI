# -*- coding: utf-8 -*-
"""
q1_v2_tier3.py — PHÂN TÍCH LẠI (CPU, KHÔNG train lại) cho bản nộp lại.

Chạy được NGAY sau khi Tier 0 xong; không cần chờ Tier 1/2.

Sinh (results/tables/v2/):
  1. weights_distribution.csv  — phân bố (a,b) TOÀN BỘ fold: median/IQR/min-max   [R2 #14, R3 #3]
  2. weight_sweep.csv          — quét a∈[0,1] × fold -> vùng phẳng của fitness     [R3 #3]
  3. hi_quality.csv            — monotonicity / trendability / prognosability      [R3 #1, #4]
  4. vtoi_range.csv            — dải thực nghiệm + % snapshot bão hoà              [R2 #15, R1 #4]
  5. early_warning_v2.csv      — 17 bearing dedup, 2 loại onset, τ chọn trên TRAIN [A8, R1 #3, R2 #21,#22]
  6. onset_sensitivity.csv     — onset ∈ {0.5,0.6,0.7,0.8}                          [R1 #3, R2 #21]
  7. conformal.csv             — split-conformal PICP/MPIW ở 80%/90%                [R3 #6]
  8. deployable_hours.csv      — quy giờ bằng vòng đời TRUNG BÌNH CỦA TRAIN         [R2 #7]

Chạy: python scripts/q1_v2_tier3.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import vtoi as V                                        # noqa: E402
from src.lobo import make_folds                                  # noqa: E402
from src.utils.paths import proc_dir_for                         # noqa: E402

TAB = ROOT / "results" / "tables" / "v2"
PAR = ROOT / "results" / "tables" / "v2_vtoi_params"
PRED = ROOT / "results" / "predictions" / "v2"
TAB.mkdir(parents=True, exist_ok=True)
PROPOSED = "vtoi_static"
DATASETS = [("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy")]


# ==================================================================== 1. phân bố trọng số
def t1_weight_distribution():
    """R2 #14: bản cũ báo cáo trọng số của MỘT bearing (file bị ghi đè). Nay báo cáo PHÂN BỐ."""
    fs = sorted(PAR.glob("params_*.csv"))
    if not fs:
        print("[bỏ qua 1] chưa có params fold."); return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    rows = []
    for (ds, sd), g in df.groupby(["dataset", "seed"]):
        rows.append({"dataset": ds, "seed": sd, "n_folds": len(g),
                     "a_median": round(float(g.a.median()), 4),
                     "a_q25": round(float(g.a.quantile(.25)), 4),
                     "a_q75": round(float(g.a.quantile(.75)), 4),
                     "a_min": round(float(g.a.min()), 4), "a_max": round(float(g.a.max()), 4),
                     "b_median": round(float(g.b.median()), 4),
                     "fitness_median": round(float(g.fitness.median()), 4),
                     "a_mono_median": round(float(g.a_mono.median()), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "weights_distribution.csv", index=False)
    df.to_csv(TAB / "weights_per_fold.csv", index=False)
    print("\n=== 1. PHÂN BỐ TRỌNG SỐ VTOI (thay Bảng 8 cũ) ===")
    print(out.to_string(index=False))
    return out


# ==================================================================== 2. quét trọng số
def t2_weight_sweep():
    """R3 #3: quét a∈[0,1]; vùng phẳng rộng => phương pháp KHÔNG nhạy với trọng số chính xác."""
    fs = sorted(PAR.glob("sweep_*.csv"))
    if not fs:
        print("[bỏ qua 2] chưa có sweep."); return None
    rows = []
    for f in fs:
        ds = "pronostia" if "pronostia" in f.name else "xjtu_sy"
        s = pd.read_csv(f); s["dataset"] = ds; s["fold"] = f.stem
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    agg = df.groupby(["dataset", "a"]).fitness.agg(["mean", "std", "count"]).reset_index()
    agg.to_csv(TAB / "weight_sweep.csv", index=False)
    print("\n=== 2. QUÉT TRỌNG SỐ (R3 #3) ===")
    for ds, g in agg.groupby("dataset"):
        best = g.loc[g["mean"].idxmax()]
        flat = g[g["mean"] >= best["mean"] - 0.02]          # trong 0.02 của đỉnh = "phẳng"
        print(f"  {ds}: a* = {best.a:.2f} (fitness {best['mean']:.3f}) | "
              f"vùng phẳng a ∈ [{flat.a.min():.2f}, {flat.a.max():.2f}] "
              f"(rộng {flat.a.max()-flat.a.min():.2f})")
    return agg


# ==================================================================== 3. chất lượng HI
def _mono(x):
    d = np.diff(np.asarray(x, float))
    return float(abs((d > 0).sum() - (d < 0).sum()) / len(d)) if len(d) else 0.0


def _prognosability(series):
    xf = np.array([np.asarray(s, float)[-1] for s in series])
    rng = np.array([abs(np.asarray(s, float)[0] - np.asarray(s, float)[-1]) for s in series])
    return float(np.exp(-np.std(xf) / (np.mean(rng) + 1e-12)))


# ------------------------------------------------------------------ Coble & Hines baseline
_HF_CACHE = {}


def _hour_features(proc, name):
    """Cache hour_features.csv — moi bearing xuat hien trong nhieu fold."""
    k = (str(proc), name)
    if k not in _HF_CACHE:
        _HF_CACHE[k] = pd.read_csv(proc / name / "hour_features.csv").sort_values("hour_id")
    return _HF_CACHE[k]


def _trendability(x):
    """Coble & Hines: |corr(x, t)| tren mot lich su."""
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x, np.arange(len(x)))[0, 1]))


def _coble_fitness(series):
    """Fitness cua Coble & Hines (2009) eq. (11): monotonicity + prognosability + trendability.

    Trung binh qua cac lich su cho hai thanh phan per-history; prognosability la dai luong
    quan the nen tinh mot lan tren ca tap.
    """
    if not series:
        return -1e9
    mono = float(np.mean([_mono(s) for s in series]))
    trend = float(np.mean([_trendability(s) for s in series]))
    return mono + _prognosability(series) + trend


class _CobleObjective:
    """Ham muc tieu cua Coble & Hines, dinh nghia o cap module de multiprocessing pickle duoc.

    Cac bearing train duoc gop thanh MOT ma tran; moi lan danh gia chi con mot phep nhan
    ma tran--vector roi cat lat, thay vi mot vong lap Python tren tung bearing.
    """

    def __init__(self, X_train_list):
        self.X = np.concatenate(X_train_list, axis=0)
        self.bounds = np.cumsum([0] + [len(x) for x in X_train_list])
        self.t = [np.arange(len(x), dtype=float) for x in X_train_list]

    def series(self, w):
        proj = self.X @ w
        return [proj[self.bounds[i]:self.bounds[i + 1]] for i in range(len(self.bounds) - 1)]

    def fitness(self, w):
        ss = self.series(w)
        mono = np.mean([_mono(x) for x in ss])
        trend = np.mean([_trendability(x) for x in ss])
        return float(mono + _prognosability(ss) + trend)

    def __call__(self, w):
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            return 1e9
        return -self.fitness(w / nw)


def fit_coble_parameter(X_train_list, seed=42, maxiter=150, popsize=25):
    """Tham so tien luong toi uu cua Coble & Hines: to hop tuyen tinh cac dac trung giam sat,
    trong so toi uu bang tim kiem tien hoa theo fitness hop nhat
    (monotonicity + prognosability + trendability).

    X_train_list : list mang (H_i, D) da chuan hoa bang thong ke cua bearing TRAIN.
    Tra ve vector trong so da chuan hoa (norm 1), huong sao cho fitness cuc dai.

    Ngan sach (150, 25) da kiem hoi tu tren fold dau cua PRONOSTIA: fitness train 1.2652
    so voi 1.2693 o (400, 40), tuc 99.7%, va 1.2338 o (40, 12). Trong so ngau nhien dat
    trung binh 0.6367 va cao nhat 1.1047 tren 300 lan rut, nen buoc toi uu la co tac dung.
    """
    from scipy.optimize import differential_evolution
    obj = _CobleObjective(X_train_list)
    res = differential_evolution(obj, bounds=[(-1.0, 1.0)] * X_train_list[0].shape[1],
                                 seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-3,
                                 polish=True, init="sobol",
                                 workers=-1, updating="deferred")
    w = np.asarray(res.x, float)
    return w / (np.linalg.norm(w) + 1e-12)


def t3_hi_quality():
    """
    R3 #1, #4: định lượng chất lượng chỉ số theo Coble & Hines cho VTOI và các HI cạnh tranh:
    RMS, Kurtosis, Mahalanobis (E), PCA-HI (thành phần chính 1), VTOI.
    Tính theo giao thức FOLD (tham số VTOI từ train), rồi tổng hợp trên các bearing held-out.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from src.lobo_v2 import fit_fold_vtoi

    rows, coble_rows = [], []
    for ds_name, key in DATASETS:
        folds, _ = make_folds(key)
        proc = proc_dir_for(key)
        series = {k: [] for k in ["RMS", "Kurtosis", "Mahalanobis (E)", "PCA-HI",
                                  "Coble optimal parameter", "VTOI (label-free)", "VTOI"]}
        recs = []
        for f in folds:
            ho = f["holdout"]
            cond, params, _, _ = fit_fold_vtoi(key, ho, f["val"], f["train"], seed=42)
            hf = pd.read_csv(proc / ho / "hour_features.csv").sort_values("hour_id")
            deg = np.arange(len(hf)) / max(len(hf) - 1, 1)
            c = cond[ho]

            # PCA-HI: fit trên bearing TRAIN (leakage-free), lấy |PC1|
            Xtr_list = [_hour_features(proc, n)[V.VIB_FEATURES].to_numpy(float)
                        for n in f["train"]]
            Xtr = np.concatenate(Xtr_list, axis=0)
            sc = StandardScaler().fit(Xtr)
            pca = PCA(n_components=1).fit(sc.transform(Xtr))
            Xho = sc.transform(hf[V.VIB_FEATURES].to_numpy(float))
            pc1 = pca.transform(Xho)[:, 0]
            if spearmanr(pc1, deg)[0] < 0:
                pc1 = -pc1                                     # định hướng tăng theo suy thoái

            # Coble & Hines: trong so fit tren bearing TRAIN cua fold, ap dong bang.
            # Ba seed: seed 42 vao bang; ba seed de do on dinh cua nghiem.
            Xs_tr = [sc.transform(X) for X in Xtr_list]
            coble = None
            for sd in (42, 43, 44):
                w_c = fit_coble_parameter(Xs_tr, seed=sd)
                v_c = Xho @ w_c
                coble_rows.append({"dataset": ds_name, "bearing": ho, "seed": sd,
                                   "fitness_train": round(_coble_fitness([X @ w_c for X in Xs_tr]), 4),
                                   "mono_holdout": round(_mono(v_c), 4),
                                   "trend_holdout": round(_trendability(v_c), 4)})
                if sd == 42:
                    coble = v_c

            cand = {"RMS": (hf.RMS_x + hf.RMS_y).to_numpy(float),
                    "Kurtosis": (hf.Kurt_x + hf.Kurt_y).to_numpy(float),
                    "Mahalanobis (E)": c.E_norm.to_numpy(float),
                    "PCA-HI": pc1,
                    "Coble optimal parameter": coble,
                    "VTOI (label-free)": c.VTOI_mono.to_numpy(float),
                    "VTOI": c.VTOI.to_numpy(float)}
            for k, v in cand.items():
                series[k].append(v)
                recs.append({"hi": k, "bearing": ho, "mono": _mono(v),
                             "trend": abs(np.corrcoef(v, np.arange(len(v)))[0, 1]),
                             "rho_deg": spearmanr(v, deg)[0]})

        rec = pd.DataFrame(recs)
        for k in series:
            g = rec[rec.hi == k]
            rows.append({"dataset": ds_name, "indicator": k,
                         "monotonicity_median": round(float(g["mono"].median()), 4),
                         "trendability_min": round(float(g["trend"].min()), 4),
                         "trendability_median": round(float(g["trend"].median()), 4),
                         "prognosability": round(_prognosability(series[k]), 4),
                         "rho_vs_deg_median": round(float(g["rho_deg"].median()), 4),
                         "n_bearings": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "hi_quality.csv", index=False)
    cb = pd.DataFrame(coble_rows)
    cb.to_csv(TAB / "coble_seed_spread.csv", index=False)
    print("\n=== 3b. ON DINH CUA THAM SO COBLE QUA 3 SEED ===")
    print(cb.groupby("dataset").agg(
        fit_train_mean=("fitness_train", "mean"), fit_train_sd=("fitness_train", "std"),
        mono_ho_mean=("mono_holdout", "mean"), mono_ho_sd=("mono_holdout", "std"),
        mono_ho_min=("mono_holdout", "min"), mono_ho_max=("mono_holdout", "max")
    ).round(4).to_string())
    print("\n=== 3. CHẤT LƯỢNG CHỈ SỐ (Coble & Hines) — R3 #1, #4 ===")
    print(out.to_string(index=False))
    return out


# ==================================================================== 4. dải VTOI
def t4_vtoi_range():
    """R2 #15: KHÔNG khẳng định 'giữ nguyên dải động' — báo cáo DẢI THỰC NGHIỆM."""
    from src.lobo_v2 import fit_fold_vtoi
    rows = []
    for ds_name, key in DATASETS:
        for f in make_folds(key)[0]:
            ho = f["holdout"]
            cond, _, _, meta = fit_fold_vtoi(key, ho, f["val"], f["train"], seed=42)
            c = cond[ho]
            for col in ["E_norm", "C_norm", "VTOI"]:
                v = c[col].to_numpy(float)
                rows.append({"dataset": ds_name, "bearing": ho, "quantity": col,
                             "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                             "iqr": round(float(np.percentile(v, 75) - np.percentile(v, 25)), 4),
                             "sat_at_1_frac": round(float(np.mean(v >= 0.999)), 4)})
    df = pd.DataFrame(rows)
    out = (df.groupby(["dataset", "quantity"])
             .agg(iqr_median=("iqr", "median"), range_median=("max", "median"),
                  sat_frac_mean=("sat_at_1_frac", "mean"), n=("bearing", "count"))
             .round(4).reset_index())
    df.to_csv(TAB / "vtoi_range_per_bearing.csv", index=False)
    out.to_csv(TAB / "vtoi_range.csv", index=False)
    print("\n=== 4. DẢI THỰC NGHIỆM CỦA VTOI (R2 #15, R1 #4) ===")
    print(out.to_string(index=False))
    return out


# ==================================================================== 5-6. early warning
def _first_persistent(sig, thr, q=3):
    run = 0
    for i, v in enumerate(sig):
        run = run + 1 if v > thr else 0
        if run >= q:
            return i - q + 1
    return -1


def _sigma3(sig, healthy_frac=0.2, k=3.0):
    n = len(sig); nh = max(5, int(healthy_frac * n))
    return float(sig[:nh].mean() + k * sig[:nh].std() + 1e-9)


def _ew_metrics(first, onset, n):
    """precision/recall ĐỊNH NGHĨA LẠI: không nổ -> precision=0 (bản cũ loại NaN -> thổi phồng)."""
    if first < 0:
        return {"fired": 0, "precision": 0.0, "recall": 0.0, "lead": np.nan}
    pred = np.zeros(n, bool); pred[first:] = True
    true = np.zeros(n, bool); true[onset:] = True
    tp = int((pred & true).sum())
    return {"fired": 1,
            "precision": round(tp / max(int(pred.sum()), 1), 4),
            "recall": round(tp / max(int(true.sum()), 1), 4),
            "lead": int(onset - first)}


def _select_on_train(grid, train_series, first_fn, default, onset_frac=0.6):
    """Chọn một tham số ngưỡng trên các bearing TRAIN bằng F1 trung bình tại onset cố định.

    Dùng CHUNG cho τ của VTOI và cho k của quy tắc k-sigma trên RMS/Kurtosis, để hai họ
    detector nhận cùng một ngân sách tuning (R2 #22).
    """
    best, best_f1 = float(default), -1.0
    for g in grid:
        f1s = []
        for sig in train_series:
            n = len(sig)
            m = _ew_metrics(first_fn(sig, g), max(1, int(onset_frac * n)), n)
            p, r = m["precision"], m["recall"]
            f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        if f1s and np.mean(f1s) > best_f1:
            best_f1, best = float(np.mean(f1s)), float(g)
    return best


def t5_early_warning(onset_fracs=(0.5, 0.6, 0.7, 0.8), tau_grid=np.arange(0.3, 0.95, 0.05),
                     k_grid=np.arange(0.5, 8.01, 0.25)):
    """
    A8  : chỉ dùng 17 bearing ĐÃ DEDUP (bản cũ dùng 28, trùng ổ bi vật lý).
    R2#22: τ được CHỌN TRÊN BEARING TRAIN của fold (không tuning trên test).
    R1#3 : quét onset + onset DỮ LIỆU-DRIVEN (3σ trên RMS, không cần biết H_fail).
    CÔNG BẰNG NGƯỠNG: RMS và Kurtosis cũng được chọn hệ số k của quy tắc k-sigma trên
           bearing TRAIN của fold, bằng ĐÚNG thủ tục F1 dùng cho τ của VTOI, rồi áp đóng băng.
           Không làm vậy thì chỉ chỉ số đề xuất được tuning và so sánh không cùng điều kiện.
    """
    from src.lobo_v2 import fit_fold_vtoi
    rows, sens = [], []
    for ds_name, key in DATASETS:
        proc = proc_dir_for(key)
        for f in make_folds(key)[0]:
            ho = f["holdout"]
            cond, _, _, _ = fit_fold_vtoi(key, ho, f["val"], f["train"], seed=42)

            # --- chọn τ trên BEARING TRAIN (F1 tối đa ở onset 0.6) ---
            best_tau = _select_on_train(
                tau_grid, [cond[n_tr].VTOI.to_numpy(float) for n_tr in f["train"]],
                lambda sig, tau: _first_persistent(sig, tau), default=0.6)

            # --- CÔNG BẰNG: chọn k của quy tắc k-sigma cho RMS/Kurtosis trên CÙNG bearing TRAIN,
            #     bằng CÙNG thủ tục F1. Ngưỡng vẫn tính từ baseline khoẻ của từng bearing nên
            #     việc áp cho bearing held-out không rò rỉ, y như τ của VTOI.
            tr_rms = [(_hour_features(proc, n_tr).RMS_x
                       + _hour_features(proc, n_tr).RMS_y).to_numpy(float) for n_tr in f["train"]]
            tr_kur = [(_hour_features(proc, n_tr).Kurt_x
                       + _hour_features(proc, n_tr).Kurt_y).to_numpy(float) for n_tr in f["train"]]
            k_rms = _select_on_train(k_grid, tr_rms,
                                     lambda sig, k: _first_persistent(sig, _sigma3(sig, k=k)),
                                     default=3.0)
            k_kur = _select_on_train(k_grid, tr_kur,
                                     lambda sig, k: _first_persistent(sig, _sigma3(sig, k=k)),
                                     default=3.0)

            hf = _hour_features(proc, ho)
            rms = (hf.RMS_x + hf.RMS_y).to_numpy(float)
            kur = (hf.Kurt_x + hf.Kurt_y).to_numpy(float)
            v = cond[ho].VTOI.to_numpy(float); n = len(v)

            for of in onset_fracs:
                onset = max(1, int(of * n))
                for det, first in [(f"VTOI (tau={best_tau:.2f}, train-selected)",
                                    _first_persistent(v, best_tau)),
                                   ("VTOI (tau=0.6, a priori)", _first_persistent(v, 0.6)),
                                   ("RMS 3-sigma", _first_persistent(rms, _sigma3(rms))),
                                   ("Kurtosis 3-sigma", _first_persistent(kur, _sigma3(kur))),
                                   ("RMS k-sigma (train-selected)",
                                    _first_persistent(rms, _sigma3(rms, k=k_rms))),
                                   ("Kurtosis k-sigma (train-selected)",
                                    _first_persistent(kur, _sigma3(kur, k=k_kur)))]:
                    m = _ew_metrics(first, onset, n)
                    rec = {"dataset": ds_name, "bearing": ho, "detector": det,
                           "onset_frac": of, "onset": onset, "first_alarm": first, "n": n,
                           "tau_selected": round(best_tau, 3),
                           "k_rms_selected": round(k_rms, 3),
                           "k_kur_selected": round(k_kur, 3), **m}
                    (rows if of == 0.6 else sens).append(rec)

            # --- onset DỮ LIỆU-DRIVEN (không dùng H_fail) — R1 #3, R2 #21 ---
            onset_dd = _first_persistent(rms, _sigma3(rms))
            if onset_dd > 0:
                for det, first in [("VTOI (tau=0.6)", _first_persistent(v, 0.6)),
                                   ("Kurtosis 3-sigma", _first_persistent(kur, _sigma3(kur)))]:
                    rows.append({"dataset": ds_name, "bearing": ho, "detector": det,
                                 "onset_frac": "data-driven(RMS 3sigma)", "onset": onset_dd,
                                 "first_alarm": first, "n": n, "tau_selected": np.nan,
                                 **_ew_metrics(first, onset_dd, n)})

    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(sens)], ignore_index=True)
    df.to_csv(TAB / "early_warning_v2_perbearing.csv", index=False)

    main = df[df.onset_frac == 0.6]
    summ = (main.groupby(["dataset", "detector"])
                .agg(n=("bearing", "count"), fired=("fired", "sum"),
                     precision=("precision", "mean"), recall=("recall", "mean"),
                     lead=("lead", "median")).round(3).reset_index())
    summ["fired_on"] = summ.fired.astype(str) + "/" + summ.n.astype(str)
    summ.to_csv(TAB / "early_warning_v2.csv", index=False)

    sensum = (df.groupby(["dataset", "detector", "onset_frac"])
                .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                     lead=("lead", "median")).round(3).reset_index())
    sensum.to_csv(TAB / "onset_sensitivity.csv", index=False)

    print("\n=== 5. EARLY WARNING v2 (17/15 bearing dedup, precision tính cả ca KHÔNG nổ) ===")
    print(summ[["dataset", "detector", "fired_on", "precision", "recall", "lead"]].to_string(index=False))
    print("\n=== 6. NHẠY CẢM VỚI ONSET (R1 #3, R2 #21) ===")
    print(sensum.to_string(index=False))
    return summ


# ==================================================================== 7. conformal
def t7_conformal(alphas=(0.20, 0.10)):
    """
    R3 #6: khoảng dự đoán CÓ HIỆU CHUẨN bằng split-conformal — KHÔNG cần train lại.
    Hiệu chuẩn trên bearing VAL của từng fold, áp cho bearing TEST.
    """
    fs = sorted(PRED.glob(f"lobo_v2_*_{PROPOSED}.csv"))
    if not fs:
        print("[bỏ qua 7] chưa có file dự đoán v2."); return None
    folds_cache = {k: {f["holdout"]: f["val"][0] for f in make_folds(k)[0]}
                   for _, k in DATASETS}
    rows = []
    for f in fs:
        stem = f.stem.replace("lobo_v2_", "").replace(f"_{PROPOSED}", "")
        key = "pronostia" if stem.startswith("pronostia") else "xjtu_sy"
        rest = stem[len(key) + 1:]
        seed, ho = rest.split("_", 1)
        val_name = folds_cache[key].get(ho)
        vf = f.parent / f"lobo_v2_{key}_{seed}_{val_name}_{PROPOSED}.csv"
        if val_name is None or not vf.exists():
            continue                              # val bearing chưa có dự đoán -> bỏ fold này
        cal = pd.read_csv(vf); te = pd.read_csv(f)
        r_cal = np.abs(cal.rul_pred - cal.rul_true).to_numpy(float)
        pred, true = te.rul_pred.to_numpy(float), te.rul_true.to_numpy(float)
        for al in alphas:
            m = len(r_cal)
            q = float(np.quantile(r_cal, min(np.ceil((m + 1) * (1 - al)) / m, 1.0)))
            lo, hi = pred - q, pred + q
            rows.append({"dataset": key, "seed": seed, "bearing": ho,
                         "nominal": round(1 - al, 2),
                         "picp": round(float(np.mean((true >= lo) & (true <= hi))), 4),
                         "mpiw": round(float(np.mean(hi - lo)), 4),
                         "q": round(q, 4), "n_cal": m})
    if not rows:
        print("[bỏ qua 7] không ghép được cặp val/test."); return None
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "conformal_per_bearing.csv", index=False)
    out = (df.groupby(["dataset", "nominal"])
             .agg(picp_mean=("picp", "mean"), picp_median=("picp", "median"),
                  picp_min=("picp", "min"), mpiw_mean=("mpiw", "mean"),
                  n=("bearing", "count")).round(4).reset_index())
    out.to_csv(TAB / "conformal.csv", index=False)
    print("\n=== 7. SPLIT-CONFORMAL PREDICTION INTERVALS (R3 #6) ===")
    print(out.to_string(index=False))
    return out


# ==================================================================== 8. thang giờ triển khai được
def t8_deployable_hours():
    """
    R2 #7: thang giờ hiện tại dùng H_fail của bearing TEST -> KHÔNG triển khai được.
    Ở đây quy giờ bằng vòng đời TRUNG BÌNH CỦA BEARING TRAIN (có sẵn online) và báo cáo
    mức suy giảm — trung thực về những gì phương pháp làm được trong vận hành.
    """
    fs = sorted(PRED.glob(f"lobo_v2_*_{PROPOSED}.csv"))
    if not fs:
        print("[bỏ qua 8] chưa có file dự đoán."); return None
    life = {}
    for _, key in DATASETS:
        proc = proc_dir_for(key)
        for d in proc.iterdir():
            p = d / "labels_by_hour.csv"
            if p.is_dir() or not p.exists():
                continue
            life[(key, d.name)] = float(pd.read_csv(p).life_hours.iloc[0])
    fold_map = {k: {f["holdout"]: f["train"] for f in make_folds(k)[0]} for _, k in DATASETS}

    rows = []
    for f in fs:
        stem = f.stem.replace("lobo_v2_", "").replace(f"_{PROPOSED}", "")
        key = "pronostia" if stem.startswith("pronostia") else "xjtu_sy"
        seed, ho = stem[len(key) + 1:].split("_", 1)
        p = pd.read_csv(f)
        err = np.abs(p.rul_pred - p.rul_true).to_numpy(float)
        true_scale = float(p.rul_scale_hours.iloc[0]) if "rul_scale_hours" in p else np.nan
        tr = fold_map[key].get(ho, [])
        lt = [life[(key, n)] for n in tr if (key, n) in life]
        dep_scale = float(np.mean(lt)) * 0.6 if lt else np.nan   # 0.6 = (1 - cap_onset)
        rows.append({"dataset": key, "seed": seed, "bearing": ho,
                     "mae_norm": round(float(err.mean()), 4),
                     "mae_h_retrospective": round(float(err.mean() * true_scale), 4),
                     "mae_h_deployable": round(float(err.mean() * dep_scale), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "deployable_hours_per_bearing.csv", index=False)
    out = (df.groupby("dataset")[["mae_norm", "mae_h_retrospective", "mae_h_deployable"]]
             .agg(["mean", "median"]).round(4))
    # .agg([...]) tạo MultiIndex cột -> to_csv ghi ra 2 dòng header + 1 dòng tên index,
    # khiến pd.read_csv đọc lại KHÔNG RA. Làm phẳng trước khi ghi (bảng này vào thẳng paper).
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out.to_csv(TAB / "deployable_hours.csv", index=False)
    print("\n=== 8. THANG GIỜ: retrospective vs TRIỂN KHAI ĐƯỢC (R2 #7) ===")
    print(out.to_string())
    return out


def main():
    print("=" * 78); print(" TIER 3 — PHÂN TÍCH LẠI (CPU) ".center(78, "=")); print("=" * 78)
    for name, fn in [("weights", t1_weight_distribution), ("sweep", t2_weight_sweep),
                     ("hi_quality", t3_hi_quality), ("range", t4_vtoi_range),
                     ("early_warning", t5_early_warning), ("conformal", t7_conformal),
                     ("deployable", t8_deployable_hours)]:
        try:
            fn()
        except Exception as e:
            print(f"\n[LỖI {name}] {type(e).__name__}: {e}")
    print(f"\n[XONG] Bảng ghi tại {TAB}")


if __name__ == "__main__":
    main()
