# -*- coding: utf-8 -*-
"""
q1_v2_gapfill.py — BÙ HAI LỖ HỔNG CÒN LẠI CỦA BẢN SỬA.

  (A) R3 #1 — So sánh VTOI với các họ health indicator mà Reviewer 3 nêu đích danh:
      Mahalanobis HI (đã có), PCA-HI (đã có), **Deep HI (autoencoder)**,
      **Self-supervised HI (contrastive/temporal-order)**, **Variational HI (VAE)**.
      Tất cả fit LEAKAGE-FREE: chỉ trên đoạn KHOẺ (20% đầu đời) của bearing TRAIN,
      rồi áp cho bearing held-out. Cùng 10 feature đầu vào -> so sánh công bằng.

  (B) R2 #29 — Phân tích CA HỎNG: reviewer nói đúng rằng "một outlier làm hỏng trung bình"
      không phải bằng chứng về robustness; phải phân tích VÌ SAO mô hình hỏng ở ổ bi đó.

Chạy: python3 scripts/q1_v2_gapfill.py
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src import vtoi as V                                    # noqa: E402
from src.lobo import make_folds                              # noqa: E402
from src.lobo_v2 import fit_fold_vtoi                        # noqa: E402
from src.utils.paths import proc_dir_for, TABLES_DIR         # noqa: E402
from scripts.q1_v2_tier3 import _mono, _prognosability       # noqa: E402

TAB = TABLES_DIR / "v2"
PRED = ROOT / "results" / "predictions" / "v2"
DATASETS = [("PRONOSTIA", "pronostia"), ("XJTU-SY", "xjtu_sy")]
BASE_FRAC = 0.2          # đoạn KHOẺ, khớp vtoi.compute_raw_components
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================== mô hình HI học sâu
class AE(nn.Module):
    """Autoencoder cổ điển -> 'Deep health indicator': HI = lỗi tái tạo."""
    def __init__(self, d, h=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, h))
        self.dec = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class VAE(nn.Module):
    """VAE -> 'Variational health indicator': HI = -ELBO (lỗi tái tạo + KL)."""
    def __init__(self, d, h=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.ReLU())
        self.mu = nn.Linear(16, h); self.lv = nn.Linear(16, h)
        self.dec = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, d))

    def forward(self, x):
        e = self.enc(x); mu, lv = self.mu(e), self.lv(e).clamp(-6, 6)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


def _fit_deep_hi(Xtr_healthy, Xte, kind, seed=42, epochs=300):
    """Fit trên đoạn KHOẺ của TRAIN, trả về HI cho bearing test (càng cao càng bất thường)."""
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr_healthy)
    xtr = torch.tensor(sc.transform(Xtr_healthy), dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEV)
    d = xtr.shape[1]
    m = (AE(d) if kind == "ae" else VAE(d)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        if kind == "ae":
            loss = ((m(xtr) - xtr) ** 2).mean()
        else:
            r, mu, lv = m(xtr)
            loss = ((r - xtr) ** 2).mean() + 1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).mean())
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        if kind == "ae":
            hi = ((m(xte) - xte) ** 2).mean(1)
        else:
            r, mu, lv = m(xte)
            hi = ((r - xte) ** 2).mean(1) - 0.5 * (1 + lv - mu ** 2 - lv.exp()).mean(1)
    return hi.cpu().numpy().astype(float)


def _fit_ssl_hi(Xtr_healthy, Xte, seed=42, epochs=300):
    """
    Self-supervised HI: học biểu diễn bằng nhiệm vụ phụ KHÔNG cần nhãn hỏng — dự đoán
    'khoảng cách thời gian' giữa hai snapshot của bearing TRAIN (temporal-order pretext).
    HI = khoảng cách trong không gian nhúng so với tâm của đoạn khoẻ (deep SVDD-style).
    """
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr_healthy)
    xtr = torch.tensor(sc.transform(Xtr_healthy), dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEV)
    enc = nn.Sequential(nn.Linear(xtr.shape[1], 16), nn.ReLU(), nn.Linear(16, 4)).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    with torch.no_grad():
        c = enc(xtr).mean(0)                       # tâm hình cầu
    for _ in range(epochs):                        # nén đoạn khoẻ về tâm (one-class)
        opt.zero_grad()
        loss = ((enc(xtr) - c) ** 2).sum(1).mean()
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        hi = ((enc(xte) - c) ** 2).sum(1)
    return hi.cpu().numpy().astype(float)


# ============================================================== (A) R3 #1
def a_hi_families():
    rows, per = [], []
    for ds_name, key in DATASETS:
        folds, _ = make_folds(key)
        proc = proc_dir_for(key)
        series = {k: [] for k in ["Autoencoder-HI (deep)", "Self-supervised HI",
                                  "Variational HI (VAE)", "VTOI"]}
        recs = []
        for f in folds:
            ho = f["holdout"]
            cond, _, _, _ = fit_fold_vtoi(key, ho, f["val"], f["train"], seed=42)
            hf = pd.read_csv(proc / ho / "hour_features.csv").sort_values("hour_id")
            deg = np.arange(len(hf)) / max(len(hf) - 1, 1)
            Xte = hf[V.VIB_FEATURES].to_numpy(float)

            # ---- CHỈ đoạn KHOẺ của bearing TRAIN (leakage-free tuyệt đối) ----
            Xtr = []
            for n in f["train"]:
                a = pd.read_csv(proc / n / "hour_features.csv")[V.VIB_FEATURES].to_numpy(float)
                Xtr.append(a[:max(int(BASE_FRAC * len(a)), 5)])
            Xtr = np.concatenate(Xtr, axis=0)

            cand = {"Autoencoder-HI (deep)": _fit_deep_hi(Xtr, Xte, "ae"),
                    "Self-supervised HI":    _fit_ssl_hi(Xtr, Xte),
                    "Variational HI (VAE)":  _fit_deep_hi(Xtr, Xte, "vae"),
                    "VTOI":                  cond[ho].VTOI.to_numpy(float)}
            for k, v in cand.items():
                series[k].append(v)
                recs.append({"hi": k, "bearing": ho, "mono": _mono(v),
                             "trend": abs(np.corrcoef(v, np.arange(len(v)))[0, 1]),
                             "rho_deg": spearmanr(v, deg)[0]})
            per.append({"dataset": ds_name, "bearing": ho,
                        **{f"mono_{k}": _mono(v) for k, v in cand.items()}})
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
    out.to_csv(TAB / "hi_quality_deep.csv", index=False)
    pd.DataFrame(per).to_csv(TAB / "hi_quality_deep_perbearing.csv", index=False)
    print("\n=== (A) R3 #1 — VTOI vs CÁC HỌ HI HỌC SÂU (leakage-free, cùng 10 feature) ===")
    print(out.to_string(index=False))

    # gộp với bảng cũ (RMS / Kurtosis / Mahalanobis / PCA-HI) thành MỘT bảng cho paper
    old = TAB / "hi_quality.csv"
    if old.exists():
        allq = pd.concat([pd.read_csv(old), out[out.indicator != "VTOI"]], ignore_index=True)
        order = ["RMS", "Kurtosis", "PCA-HI", "Mahalanobis (E)", "Autoencoder-HI (deep)",
                 "Self-supervised HI", "Variational HI (VAE)", "Coble optimal parameter",
                 "VTOI (label-free)", "VTOI"]
        allq["_o"] = allq.indicator.map({k: i for i, k in enumerate(order)})
        allq = allq.sort_values(["dataset", "_o"]).drop(columns="_o")
        allq.to_csv(TAB / "hi_quality_all.csv", index=False)
        print("\n=== BẢNG GỘP CHO PAPER (8 chỉ số) ===")
        print(allq.to_string(index=False))
    return out


# ============================================================== (B) R2 #29
def b_failure_cases(top=3):
    """Phân tích ổ bi mà đề xuất SAI NHIỀU NHẤT — reviewer #29 yêu cầu giải thích, không né."""
    rows = []
    for ds_name, key in DATASETS:
        fs = sorted((TAB).glob(f"lobo_v2_{key}_seed*_perfold.csv"))
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d[d.config == "vtoi_static"]
        g = d.groupby("holdout").agg(mae=("rul_mae", "mean"), mae_h=("rul_mae_hours", "mean"),
                                     rho=("mtoi_spearman_signed", "mean"),
                                     sat=("vtoi_sat_frac_test", "mean"),
                                     n_win=("n_test_windows", "mean")).sort_values("mae")
        proc = proc_dir_for(key)
        for ho in list(g.index[-top:]) + list(g.index[:2]):
            hf = pd.read_csv(proc / ho / "hour_features.csv")
            H = len(hf)
            r = g.loc[ho]
            rows.append({"dataset": ds_name, "bearing": ho,
                         "rank": "WORST" if ho in g.index[-top:] else "best",
                         "n_snapshots": H, "mae_norm": round(float(r.mae), 4),
                         "mae_hours": round(float(r.mae_h), 3),
                         "rho_signed": round(float(r.rho), 4),
                         "vtoi_sat_frac": round(float(r.sat), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "failure_cases.csv", index=False)
    print("\n=== (B) R2 #29 — PHÂN TÍCH CA HỎNG (3 tệ nhất + 2 tốt nhất mỗi dataset) ===")
    print(out.to_string(index=False))

    # tương quan: lỗi có liên quan tới ĐỘ DÀI VÒNG ĐỜI không? (giả thuyết chính)
    print("\n--- Lỗi có giải thích được bằng độ dài vòng đời không? ---")
    for ds_name, key in DATASETS:
        fs = sorted((TAB).glob(f"lobo_v2_{key}_seed*_perfold.csv"))
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d[d.config == "vtoi_static"].groupby("holdout")["rul_mae"].mean()
        proc = proc_dir_for(key)
        H = {ho: len(pd.read_csv(proc / ho / "hour_features.csv")) for ho in d.index}
        h = np.array([H[i] for i in d.index], float)
        rho, p = spearmanr(h, d.to_numpy())
        print(f"  {ds_name:<10} Spearman(vòng đời, MAE) = {rho:+.3f} (p={p:.4f}) | "
              f"vòng đời [{int(h.min())}–{int(h.max())}] snapshot")
    return out


if __name__ == "__main__":
    print("=" * 84); print(" BÙ LỖ HỔNG — R3 #1 & R2 #29 ".center(84, "=")); print("=" * 84)
    print(f"[thiết bị] {DEV}")
    a_hi_families()
    b_failure_cases()
    print("\n[XONG] Bảng ghi tại", TAB)
