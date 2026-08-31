# -*- coding: utf-8 -*-
"""
q1_tier3_physics.py — TIER 3 (reviewer Q1): BẰNG CHỨNG VẬT LÝ cho MTOI.

Chứng minh MTOI tăng ĐỒNG PHA với năng lượng tần số hỏng ổ bi (defect-frequency band energy)
trên XJTU-SY (ổ bi LDK UER204 có thông số hình học công bố rõ + tốc độ trục đã biết).

Phương pháp: với mỗi snapshot (giờ), tính envelope spectrum (Hilbert -> |FFT|) của tín hiệu rung,
cộng năng lượng trong các dải hẹp quanh BPFO/BPFI/BSF/FTF và hài của chúng -> "defect-band energy".
So sánh chuỗi này với MTOI theo vòng đời (Spearman). Vẽ overlay + phổ envelope sớm/muộn.

Sinh:
  results/tables/q1_physics_defect_correlation.csv
  results/figures/q1_physics_mtoi_vs_defect.png
  results/figures/q1_physics_envelope_spectrum.png
Chạy: python scripts/q1_tier3_physics.py
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import spearmanr

# ---- STYLE Q1: chữ to + in đậm, nét dày, độ phân giải cao (dễ đọc khi in) ----
plt.rcParams.update({
    "font.size": 16, "font.weight": "bold",
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 18, "axes.labelweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "figure.titlesize": 20, "figure.titleweight": "bold",
    "axes.linewidth": 1.8, "lines.linewidth": 2.6, "patch.linewidth": 1.5,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "savefig.dpi": 220, "figure.dpi": 220, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
XJTU = ROOT.parent / "data" / "processed" / "xjtu_sy"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FS = 25600.0                      # tần số lấy mẫu (Hz)

# LDK UER204 (XJTU-SY) — thông số hình học công bố:
N_BALL, D_BALL, D_PITCH, THETA = 8, 7.92, 34.55, 0.0
R = D_BALL / D_PITCH * np.cos(np.deg2rad(THETA))
# Hệ số defect-frequency theo BỘI tần quay trục fr:
K_BPFO = (N_BALL / 2) * (1 - R)
K_BPFI = (N_BALL / 2) * (1 + R)
K_BSF  = (D_PITCH / (2 * D_BALL)) * (1 - R ** 2)
K_FTF  = 0.5 * (1 - R)


def shaft_freq(bearing_name):
    m = re.match(r"([0-9.]+)Hz", bearing_name)
    return float(m.group(1)) if m else 35.0


def envelope_spectrum(sig):
    """Envelope spectrum 1 chiều: |FFT(|hilbert(sig)| - mean)|. Trả (freqs, mag)."""
    env = np.abs(hilbert(sig))
    env = env - env.mean()
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / FS)
    return freqs, mag


def defect_band_energy(freqs, mag, fr, harmonics=4, tol_hz=12.5):
    """Tổng năng lượng phổ envelope trong dải hẹp quanh BPFO/BPFI/BSF/FTF và hài."""
    total = {}
    for name, k in [("BPFO", K_BPFO), ("BPFI", K_BPFI), ("BSF", K_BSF), ("FTF", K_FTF)]:
        e = 0.0
        for h in range(1, harmonics + 1):
            fc = k * fr * h
            if fc >= freqs[-1]:
                break
            band = (freqs >= fc - tol_hz) & (freqs <= fc + tol_hz)
            e += float((mag[band] ** 2).sum())
        total[name] = e
    total["ALL"] = total["BPFO"] + total["BPFI"] + total["BSF"] + total["FTF"]
    return total


def per_hour_defect_energy(bearing_dir, fr, max_win_per_hour=6):
    X = np.load(bearing_dir / "X_vib_windows.npy")        # [Nwin, 2, L]
    hid = np.load(bearing_dir / "window_hour_ids.npy")
    hours = np.unique(hid)
    rows = []
    for h in hours:
        idx = np.where(hid == h)[0][:max_win_per_hour]
        acc = None
        for i in idx:
            for ch in range(X.shape[1]):
                f, mag = envelope_spectrum(X[i, ch].astype(float))
                acc = mag if acc is None else acc + mag
        acc = acc / (len(idx) * X.shape[1])
        e = defect_band_energy(f, acc, fr)
        e["hour_id"] = int(h)
        rows.append(e)
    return pd.DataFrame(rows), f


def main():
    bearings = sorted([d for d in XJTU.iterdir() if d.is_dir() and "Bearing" in d.name])
    corr_rows = []
    overlay = []   # (name, hour, defect_norm, mtoi) cho hình
    for bd in bearings:
        fr = shaft_freq(bd.name)
        try:
            de, _ = per_hour_defect_energy(bd, fr)
            mt = pd.read_csv(bd / "mtoi_by_hour.csv")[["hour_id", "MTOI_learnable"]]
            df = de.merge(mt, on="hour_id", how="inner").sort_values("hour_id")
        except Exception as ex:
            print(f"  bỏ {bd.name}: {ex}")
            continue
        if len(df) < 8:
            continue
        rho, p = spearmanr(df["ALL"], df["MTOI_learnable"])
        corr_rows.append({"bearing": bd.name, "fr_Hz": fr, "n_hours": len(df),
                          "spearman_MTOI_vs_defectEnergy": round(float(rho), 3),
                          "p_value": round(float(p), 4)})
        d = df["ALL"].to_numpy(float)
        dn = (d - d.min()) / (d.max() - d.min() + 1e-12)
        overlay.append((bd.name, df["hour_id"].to_numpy(), dn, df["MTOI_learnable"].to_numpy()))
        print(f"  {bd.name}: Spearman(MTOI, defect-energy)={rho:.3f} (p={p:.3g}), {len(df)}h")

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TAB / "q1_physics_defect_correlation.csv", index=False)
    print(f"\n[bảng] q1_physics_defect_correlation.csv ({len(corr)} bearing) | "
          f"Spearman trung bình = {corr['spearman_MTOI_vs_defectEnergy'].mean():.3f}")

    # --- Hình 1: overlay MTOI vs defect-band energy cho 3 bearing tiêu biểu ---
    pick = overlay[:3] if len(overlay) >= 3 else overlay
    fig, axes = plt.subplots(1, len(pick), figsize=(6.2 * len(pick), 5.0))
    if len(pick) == 1:
        axes = [axes]
    for ax, (name, hrs, dn, mt) in zip(axes, pick):
        ax.plot(hrs, dn, label="Defect-band energy (norm.)", color="tab:red", lw=2.8)
        ax.plot(hrs, mt, label="VTOI (learnable)", color="black", lw=3.2)
        ax.set_xlabel("Snapshot index (minute)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(name)
        ax.grid(alpha=0.35, linewidth=1.2)
        ax.legend(loc="upper left")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
    fig.suptitle("VTOI tracks bearing defect-frequency band energy over life (XJTU-SY)")
    fig.tight_layout()
    fig.savefig(FIG / "q1_physics_mtoi_vs_defect.png")
    print("[hình] q1_physics_mtoi_vs_defect.png")

    # --- Hình 2: envelope spectrum sớm vs muộn (1 bearing) với vạch tần số hỏng ---
    bd = bearings[0]
    fr = shaft_freq(bd.name)
    X = np.load(bd / "X_vib_windows.npy"); hid = np.load(bd / "window_hour_ids.npy")
    hours = np.unique(hid)
    h_early, h_late = hours[len(hours) // 10], hours[-2]
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, h, tag in [(axes2[0], h_early, "Early life (healthy)"),
                       (axes2[1], h_late, "Late life (degraded)")]:
        i = np.where(hid == h)[0][0]
        f, mag = envelope_spectrum(X[i, 0].astype(float))
        sel = f <= 900
        ax.plot(f[sel], mag[sel], color="tab:blue", lw=2.0)
        for name, k, col in [("BPFO", K_BPFO, "red"), ("BPFI", K_BPFI, "green"),
                             ("BSF", K_BSF, "orange"), ("FTF", K_FTF, "purple")]:
            ax.axvline(k * fr, color=col, ls="--", lw=2.4, alpha=0.85, label=f"{name} ({k*fr:.0f} Hz)")
        ax.set_title(f"{bd.name} — {tag}")
        ax.set_ylabel("Envelope spectrum |FFT|")
        ax.grid(alpha=0.35, linewidth=1.2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
    axes2[1].set_xlabel("Frequency (Hz)")
    axes2[0].legend(fontsize=13, ncol=4, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(FIG / "q1_physics_envelope_spectrum.png")
    print("[hình] q1_physics_envelope_spectrum.png")

    print(f"\nĐịnh nghĩa tần số hỏng (bội fr): BPFO={K_BPFO:.3f}, BPFI={K_BPFI:.3f}, "
          f"BSF={K_BSF:.3f}, FTF={K_FTF:.3f}")


if __name__ == "__main__":
    main()
