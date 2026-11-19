# -*- coding: utf-8 -*-
r"""
verify_paper_numbers.py — CHỐNG TÁI PHÁT LỖI A3/A9 (số trong .tex ≠ số trong kết quả).

Vấn đề gốc: mọi con số trong bài đang được GÕ TAY từ bảng CSV. Chỉ cần chạy lại thí nghiệm
là bài lệch số mà không ai biết — đó chính là lỗi A3/A9 mà reviewer đã bắt được ở vòng 1.

Script này tấn công vấn đề ở BA lớp, mạnh dần:

  --emit    (MẠNH NHẤT — sửa nguyên nhân gốc)
            Sinh thân bảng LaTeX TRỰC TIẾP từ CSV v2 vào  paper/generated/*.tex.
            Bài chỉ cần \input{} — không còn gõ tay thì không còn chỗ để sai.

  --check   (HÀNG RÀO)
            Đối chiếu từng "claim" đã đăng ký (số cụ thể trong .tex) với ô CSV tương ứng.
            Lệch quá dung sai -> in FAIL và exit code 1 (dùng được trong CI/pre-commit).

  --sweep   (KIỂM TOÁN)
            Quét MỌI số trong .tex, chỉ ra số nào CHƯA được claim nào phủ.
            Trả lời câu hỏi "còn con số nào trong bài chưa ai kiểm chứng?".

  --report  (TIỆN DỤNG)
            In các số v2 quan trọng theo dạng dán thẳng vào RESPONSE_TO_REVIEWERS_v2.md
            (chỗ đang để [[ ]]).

Chạy:
    python3 scripts/verify_paper_numbers.py --report
    python3 scripts/verify_paper_numbers.py --emit
    python3 scripts/verify_paper_numbers.py --check     # exit 1 nếu có số lệch
    python3 scripts/verify_paper_numbers.py --sweep
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # .../MTOI-Bearing
DRIVE = ROOT.parent                                  # .../Bearing_MTOI
TAB = ROOT / "results" / "tables" / "v2"
TEX = DRIVE / "paper" / "ACCESS_latex_template_20240429" / "VTOI_paper.tex"
GEN = DRIVE / "paper" / "generated"

PROPOSED = "vtoi_static"

# Tên hiển thị trong bài (tránh gạch dưới gây lỗi LaTeX, và để bài đọc chuyên nghiệp).
DISPLAY = {
    "vtoi_static":          "VTOI-conditioned (proposed)",
    "vtoi_traj":            "VTOI-conditioned + trajectory head",
    "transformer_vib":      "Transformer",
    "tcn_vib":              "TCN",
    "tcn_transformer_vib":  "TCN--Transformer",
    "cnn_bilstm_attn_vib":  "CNN--BiLSTM--Attention",
    "abl_no_idxhead":       "Ablation: aux.\\ loss only (no conditioning)",
    "abl_cond_noaux":       "Ablation: conditioning only (no aux.\\ loss)",
    "abl_E_only":           "Ablation: abnormality component only",
    "abl_no_smooth":        "Ablation: no smoothness penalty",
    "abl_no_mono":          "Ablation: no monotonicity penalty",
    "ctl_scalar":           "Control: index scalar only",
    "ctl_EC":               "Control: raw components $(E,C)$",
    "ctl_hc10":             "Control: 10 handcrafted features",
    "ctl_random":           "Control: random index",
    "ctl_shuffled":         "Control: shuffled index",
    "ctl_elapsed":          "Control: elapsed-time signal",
    "ctl_lifefrac":         "Control: oracle lifetime fraction",
    "ctl_nodeg":            "Control: monotonicity-only fit",
}
DS_DISPLAY = {"pronostia": "PRONOSTIA", "xjtu_sy": "XJTU-SY", "ims": "IMS"}


def disp(name: str) -> str:
    return DISPLAY.get(name, name.replace("_", "\\_"))


# =================================================================================== CLAIMS
# Mỗi claim = MỘT con số trong .tex phải khớp MỘT ô trong CSV.
#   table/where/col : định vị ô trong CSV
#   tex             : regex có ĐÚNG 1 nhóm bắt (group 1) là con số trong .tex
#   fmt / tol       : định dạng in và dung sai tuyệt đối khi so sánh
# Thêm claim mới = thêm 1 dict. KHÔNG cần sửa code.
CLAIMS = [
    dict(id="lobo_pron_proposed_hours_mean",
         desc="Bảng LOBO — PRONOSTIA / đề xuất / RUL MAE (giờ), trung bình",
         table="main_results.csv", where=dict(dataset="pronostia", config=PROPOSED),
         col="hours_mean", fmt="{:.3f}", tol=5e-4,
         tex=r"gives ([0-9.]+) hours on PRONOSTIA"),
    dict(id="lobo_xjtu_proposed_hours_mean",
         desc="Bảng LOBO — XJTU-SY / đề xuất / RUL MAE (giờ), trung bình",
         table="main_results.csv", where=dict(dataset="xjtu_sy", config=PROPOSED),
         col="hours_mean", fmt="{:.3f}", tol=5e-4,
         tex=r"hours on PRONOSTIA and ([0-9.]+) hours on XJTU-SY"),
    dict(id="n_bearings_pronostia",
         desc="Số ổ bi PRONOSTIA dùng cho LOBO",
         table="main_results.csv", where=dict(dataset="pronostia", config=PROPOSED),
         col="n_bearings", fmt="{:.0f}", tol=0.5,
         tex=r"PRONOSTIA[^\n]*?contributes\s+(seventeen|\d+)\s+accelerated"),
    dict(id="n_bearings_xjtu",
         desc="Số ổ bi XJTU-SY dùng cho LOBO",
         table="main_results.csv", where=dict(dataset="xjtu_sy", config=PROPOSED),
         col="n_bearings", fmt="{:.0f}", tol=0.5,
         tex=r"XJTU-SY[^\n]*?contributes\s+(fifteen|\d+)\s+bearings"),

    # ---- v9: cac con so moi dua vao bai o vong sua nay, dang ky de khong the troi ----
    dict(id="hi_mono_pron_vtoi",
         desc="Table 1 — PRONOSTIA / VTOI / monotonicity trung vi (so dau bang cua abstract)",
         table="hi_quality_all.csv", where=dict(dataset="PRONOSTIA", indicator="VTOI"),
         col="monotonicity_median", fmt="{:.4f}", tol=5e-5,
         tex=r"median monotonicity of ([0-9.]+) against 0\.0067"),
    dict(id="hi_mono_pron_labelfree",
         desc="Table 1 — PRONOSTIA / VTOI label-free / monotonicity",
         table="hi_quality_all.csv",
         where=dict(dataset="PRONOSTIA", indicator="VTOI (label-free)"),
         col="monotonicity_median", fmt="{:.4f}", tol=5e-5,
         tex=r"the label-free variant attains ([0-9]+\.[0-9]+)"),
    dict(id="hi_mono_pron_coble",
         desc="Table 1 — PRONOSTIA / tham so Coble / monotonicity",
         table="hi_quality_all.csv",
         where=dict(dataset="PRONOSTIA", indicator="Coble optimal parameter"),
         col="monotonicity_median", fmt="{:.4f}", tol=5e-5,
         tex=r"Coble optimal parameter & ([0-9.]+) &"),
    dict(id="ew_prec_pron_vtoi",
         desc="Table 3 — PRONOSTIA / VTOI train-selected / precision",
         table="early_warning_v2.csv",
         where={"dataset": "PRONOSTIA", "detector": "VTOI (tau=0.30, train-selected)"},
         col="precision", fmt="{:.3f}", tol=5e-4,
         tex=r"the index attains the highest precision of any detector examined, ([0-9.]+) on PRONOSTIA"),
    dict(id="ew_prec_xjtu_vtoi",
         desc="Table 3 — XJTU-SY / VTOI train-selected / precision",
         table="early_warning_v2.csv",
         where={"dataset": "XJTU-SY", "detector": "VTOI (tau=0.30, train-selected)"},
         col="precision", fmt="{:.3f}", tol=5e-4,
         tex=r"([0-9.]+) on XJTU-SY, against 0\.524"),
    dict(id="ew_prec_pron_rms_ts",
         desc="Table 3 — PRONOSTIA / RMS k-sigma train-selected / precision (so sanh cong bang)",
         table="early_warning_v2.csv",
         where={"dataset": "PRONOSTIA", "detector": "RMS k-sigma (train-selected)"},
         col="precision", fmt="{:.3f}", tol=5e-4,
         tex=r"against ([0-9.]+) and 0\.715 for the train-selected"),
    dict(id="ew_prec_xjtu_rms_ts",
         desc="Table 3 — XJTU-SY / RMS k-sigma train-selected / precision",
         table="early_warning_v2.csv",
         where={"dataset": "XJTU-SY", "detector": "RMS k-sigma (train-selected)"},
         col="precision", fmt="{:.3f}", tol=5e-4,
         tex=r"against 0\.524 and ([0-9.]+) for the train-selected"),

    # ---- v10: so da sua o vong nay (median lead that su, Coble random baseline) ----
    dict(id="ew_lead_pron_vtoi_ts",
         desc="Table 3 - PRONOSTIA / VTOI train-selected / median lead (dau am, snapshot)",
         table="early_warning_v2.csv",
         where={"dataset": "PRONOSTIA", "detector": "VTOI (tau=0.30, train-selected)"},
         col="lead", fmt="{:.0f}", tol=0.5,
         tex=r"at a median lead of \$-([0-9]+)\$ and", sign=-1),
    dict(id="ew_lead_xjtu_vtoi_ts",
         desc="Table 3 - XJTU-SY / VTOI train-selected / median lead",
         table="early_warning_v2.csv",
         where={"dataset": "XJTU-SY", "detector": "VTOI (tau=0.30, train-selected)"},
         col="lead", fmt="{:.0f}", tol=0.5,
         tex=r"median lead of \$-[0-9]+\$ and \$-([0-9]+)\$ snapshots", sign=-1),
    dict(id="ew_lead_pron_rms_ts",
         desc="Table 3 - PRONOSTIA / RMS train-selected / median lead (duong)",
         table="early_warning_v2.csv",
         where={"dataset": "PRONOSTIA", "detector": "RMS k-sigma (train-selected)"},
         col="lead", fmt="{:.0f}", tol=0.5,
         tex=r"alarms early, at \$\+([0-9]+)\$ and"),
    dict(id="ew_lead_xjtu_rms_ts",
         desc="Table 3 - XJTU-SY / RMS train-selected / median lead (duong)",
         table="early_warning_v2.csv",
         where={"dataset": "XJTU-SY", "detector": "RMS k-sigma (train-selected)"},
         col="lead", fmt="{:.0f}", tol=0.5,
         tex=r"at \$\+[0-9]+\$ and \$\+([0-9]+)\$ snapshots"),
    dict(id="coble_random_pron",
         desc="V-A - PRONOSTIA / fitness trung binh cua trong so ngau nhien (300 lan rut/fold)",
         table="coble_random_baseline.csv", where=dict(dataset="PRONOSTIA"),
         col="random_fitness_mean", fmt="{:.3f}", tol=5e-4, agg="mean",
         tex=r"on PRONOSTIA against ([0-9.]+) for weights drawn uniformly"),
]

WORDNUM = {"seventeen": 17, "fifteen": 15, "sixteen": 16, "fourteen": 14,
           "thirteen": 13, "twelve": 12, "eleven": 11, "ten": 10, "nine": 9,
           "eight": 8, "seven": 7, "six": 6, "five": 5, "four": 4, "three": 3, "two": 2}


def to_float(s: str):
    s = s.strip()
    if s.lower() in WORDNUM:
        return float(WORDNUM[s.lower()])
    try:
        return float(s)
    except ValueError:
        return None


def load_table(name: str):
    p = TAB / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def csv_value(claim):
    """Lấy giá trị CSV cho 1 claim. Trả về (value, note)."""
    df = load_table(claim["table"])
    if df is None:
        return None, f"thiếu bảng {claim['table']} (chạy: bash run_v2.sh analyze)"
    m = df
    for k, v in claim.get("where", {}).items():
        if k not in m.columns:
            return None, f"bảng {claim['table']} không có cột '{k}'"
        m = m[m[k].astype(str) == str(v)]
    if len(m) == 0:
        return None, f"không có dòng khớp {claim.get('where')}"
    if claim["col"] not in m.columns:
        return None, f"bảng {claim['table']} không có cột '{claim['col']}'"
    agg = claim.get("agg")
    if len(m) > 1:
        if agg is None:
            return None, f"khớp {len(m)} dòng (điều kiện 'where' chưa đủ chặt)"
        return float(getattr(m[claim["col"]], agg)()), ""
    return float(m.iloc[0][claim["col"]]), ""


# =================================================================================== --check
def cmd_check(verbose=True):
    if not TEX.exists():
        print(f"[LỖI] không thấy .tex: {TEX}")
        return 2
    tex = TEX.read_text(encoding="utf-8", errors="replace")
    n_pass = n_fail = n_skip = 0
    print("=" * 92)
    print(" ĐỐI CHIẾU SỐ TRONG .tex  ↔  CSV KẾT QUẢ v2 ".center(92, "="))
    print("=" * 92)
    for c in CLAIMS:
        want, note = csv_value(c)
        if want is None:
            n_skip += 1
            print(f"  [BỎ QUA] {c['id']:<34} {note}")
            continue
        m = re.search(c["tex"], tex, re.S | re.I)
        if not m:
            n_fail += 1
            print(f"  [KHÔNG THẤY] {c['id']:<30} regex không khớp chỗ nào trong .tex")
            print(f"               CSV nói: {c['fmt'].format(want)}  ({c['desc']})")
            continue
        got = to_float(m.group(1))
        if got is not None:
            got *= c.get("sign", 1)
        if got is None:
            n_fail += 1
            print(f"  [LỖI ĐỌC] {c['id']:<33} không parse được '{m.group(1)}'")
            continue
        line_no = tex[:m.start(1)].count("\n") + 1
        if abs(got - want) <= c["tol"]:
            n_pass += 1
            if verbose:
                print(f"  [OK]   {c['id']:<34} {c['fmt'].format(got)}  (tex dòng {line_no})")
        else:
            n_fail += 1
            print(f"  [LỆCH] {c['id']:<34} tex={m.group(1)}  ≠  csv={c['fmt'].format(want)}"
                  f"   (tex dòng {line_no})")
            print(f"         → {c['desc']}")
    print("-" * 92)
    print(f"  KẾT QUẢ: {n_pass} khớp · {n_fail} LỆCH/không thấy · {n_skip} bỏ qua (thiếu CSV)")
    if n_fail:
        print("  ⚠️  Còn số lệch. Sửa .tex (hoặc dùng --emit để sinh bảng thẳng từ CSV).")
    print("=" * 92)
    return 1 if n_fail else 0


# =================================================================================== --sweep
NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d+|\d{1,4})(?![\w.])")
# Bỏ qua số không phải kết quả: nhãn tham chiếu, năm, số trang, DOI, ORCID...
SKIP_LINE = re.compile(r"\\(cite|ref|label|bibitem|doi|address|author|vol|history)|"
                       r"arXiv|ORCID|e-mail|@|\\includegraphics|^%")


def cmd_sweep():
    if not TEX.exists():
        print(f"[LỖI] không thấy .tex: {TEX}")
        return 2
    lines = TEX.read_text(encoding="utf-8", errors="replace").splitlines()
    covered_lines = set()
    tex_all = "\n".join(lines)
    for c in CLAIMS:
        m = re.search(c["tex"], tex_all, re.S | re.I)
        if m:
            covered_lines.add(tex_all[:m.start(1)].count("\n") + 1)

    in_table = False
    rows = []
    for i, ln in enumerate(lines, 1):
        if re.search(r"\\begin\{table\*?\}|\\begin\{tabular\}", ln):
            in_table = True
        if re.search(r"\\end\{table\*?\}", ln):
            in_table = False
        if SKIP_LINE.search(ln):
            continue
        nums = [n for n in NUM_RE.findall(ln) if not re.fullmatch(r"(19|20)\d{2}", n)]
        if not nums:
            continue
        rows.append(dict(line=i, in_table=in_table, covered=(i in covered_lines),
                         n=len(nums), nums=nums[:12], text=ln.strip()[:96]))

    print("=" * 100)
    print(" KIỂM TOÁN SỐ TRONG .tex — con số nào CHƯA được claim nào kiểm chứng ".center(100, "="))
    print("=" * 100)
    tot = sum(r["n"] for r in rows)
    cov = sum(r["n"] for r in rows if r["covered"])
    in_tab = [r for r in rows if r["in_table"] and not r["covered"]]
    in_txt = [r for r in rows if not r["in_table"] and not r["covered"]]
    print(f"\nTổng số trích được: {tot} · đã có claim phủ: {cov} · CHƯA phủ: {tot - cov}")
    print(f"Đã đăng ký {len(CLAIMS)} claim trong CLAIMS[].\n")

    for title, group in [("TRONG BẢNG (ưu tiên cao — đây là nơi lỗi A3/A9 xảy ra)", in_tab),
                         ("TRONG VĂN BẢN", in_txt)]:
        print("-" * 100)
        print(f" {title}: {sum(r['n'] for r in group)} số / {len(group)} dòng")
        print("-" * 100)
        for r in group[:60]:
            print(f"  d{r['line']:>4}  [{r['n']:>2} số]  {', '.join(r['nums'])}")
            print(f"          {r['text']}")
        if len(group) > 60:
            print(f"  ... còn {len(group) - 60} dòng nữa")
    print("\n👉 Cách xử lý: bảng nào có thể sinh tự động thì dùng --emit rồi \\input{};")
    print("   số lẻ trong văn bản thì thêm 1 dict vào CLAIMS[] để --check canh giữ.")
    print("=" * 100)
    return 0


# =================================================================================== --emit
def _fmt(v, nd=3):
    if pd.isna(v):
        return "--"
    if isinstance(v, str):
        return v.replace("_", "\\_")
    if float(v).is_integer() and abs(float(v)) < 1e6 and nd == 0:
        return f"{int(v)}"
    return f"{float(v):.{nd}f}"


def emit_table(csv_name, out_name, cols, headers, caption, label,
               sort_by=None, bold_min=None, nd=3, star=False, group_by="dataset"):
    """Sinh 1 bảng LaTeX hoàn chỉnh từ 1 CSV. cols/headers khớp thứ tự."""
    df = load_table(csv_name)
    if df is None or df.empty:
        return f"  [bỏ qua] {csv_name}: chưa có"
    keep = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if not keep:
        return f"  [LỖI] {csv_name}: KHÔNG cột nào khớp. Có: {list(df.columns)}"
    # FAIL-LOUD: âm thầm bỏ cột là đúng loại lỗi mà script này sinh ra để CHẶN.
    # Bảng thiếu cột trông vẫn "hợp lệ" nên sẽ lọt vào bài mà không ai biết.
    if missing:
        return (f"  [LỖI] {out_name}: THIẾU cột {missing} trong {csv_name}.\n"
                f"         Cột thật: {list(df.columns)}\n"
                f"         -> Sửa EMIT_SPEC cho khớp. KHÔNG sinh bảng thiếu cột.")
    hdr = [h for c, h in zip(cols, headers) if c in df.columns]

    align = "l" + "r" * (len(keep) - 1)
    L = [f"% === SINH TỰ ĐỘNG bởi scripts/verify_paper_numbers.py --emit — ĐỪNG SỬA TAY ===",
         f"% nguồn: results/tables/v2/{csv_name}",
         "\\begin{table" + ("*" if len(keep) > 5 else "") + "}[!t]",
         "\\centering",
         f"\\caption{{{caption}}}",
         f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{align}}}",
         "\\hline",
         " & ".join(hdr) + " \\\\",
         "\\hline"]

    groups = ([(g, d) for g, d in df.groupby(group_by)] if group_by in df.columns
              else [(None, df)])
    for gname, gdf in groups:
        if gname is not None:
            L.append("\\multicolumn{%d}{l}{\\textbf{%s}} \\\\" %
                     (len(keep), DS_DISPLAY.get(str(gname), str(gname))))
        g = gdf.sort_values(sort_by) if sort_by and sort_by in gdf.columns else gdf
        best = None
        if bold_min and bold_min in g.columns:
            num = pd.to_numeric(g[bold_min], errors="coerce")
            best = num.idxmin() if num.notna().any() else None
        for idx, r in g.iterrows():
            cells = []
            for c in keep:
                v = r[c]
                if c == "config":
                    cells.append(disp(str(v)))
                elif c == "dataset":
                    cells.append(DS_DISPLAY.get(str(v), str(v)))
                else:
                    is_count = (c == "n" or c.startswith("n_") or c == "seeds")
                    s = _fmt(v, 0 if is_count else nd)
                    if best is not None and idx == best and c == bold_min:
                        s = "\\textbf{" + s + "}"
                    cells.append(s)
            L.append(" & ".join(cells) + " \\\\")
        L.append("\\hline")
    L += ["\\end{tabular}", "\\end{table" + ("*" if len(keep) > 5 else "") + "}"]

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / out_name).write_text("\n".join(L) + "\n", encoding="utf-8")
    return f"  [OK] {out_name:<26} ← {csv_name}  ({len(df)} dòng)"


EMIT_SPEC = [
    dict(csv_name="main_results.csv", out_name="tab_main_results.tex",
         cols=["config", "n_bearings", "norm_median", "norm_q25", "norm_q75",
               "norm_worst", "norm_mean", "hours_mean"],
         headers=["Configuration", "$n$", "Median", "Q25", "Q75",
                  "Worst", "Mean", "Mean (h)"],
         caption=("Leave-one-bearing-out performance. The primary metric is the normalised "
                  "lifetime-fraction RUL error, summarised by median and interquartile range "
                  "because held-out lifetimes span two orders of magnitude; the hour scale is "
                  "retained as a secondary retrospective view. Lower is better; the best median "
                  "per dataset is in \\textbf{bold}."),
         label="tab:lobo", sort_by="norm_median", bold_min="norm_median"),
    dict(csv_name="wilcoxon.csv", out_name="tab_wilcoxon.tex",
         cols=["comparison", "n", "mean_a", "mean_b", "median_diff", "p_value",
               "holm_p", "rank_biserial", "prop_wins"],
         headers=["Comparison", "$n$", "Baseline", "Proposed", "$\\Delta$ median",
                  "$p$", "$p_{\\mathrm{Holm}}$", "$r_{rb}$", "Wins"],
         caption=("Wilcoxon signed-rank tests on the seed-averaged normalised RUL error, paired "
                  "by held-out bearing, with Holm correction within each dataset and "
                  "matched-pairs rank-biserial effect size."),
         label="tab:wilcoxon", sort_by="holm_p"),
    dict(csv_name="factorial_2x2.csv", out_name="tab_factorial.tex",
         cols=["conditioning", "aux_loss", "config", "n", "mean", "median"],
         headers=["Conditioning", "Aux.\\ loss", "Configuration", "$n$", "Mean", "Median"],
         caption=("Full $2\\times2$ factorial ablation separating the two mechanisms that were "
                  "confounded in the first submission: conditioning the regressor on the index, "
                  "and the auxiliary index-regression loss. Main effects and their interaction "
                  "are reported on the normalised RUL error."),
         label="tab:factorial", sort_by=None),
    dict(csv_name="controls.csv", out_name="tab_controls.tex",
         cols=["config", "n", "mean", "median", "p_vs_proposed", "proposed_wins"],
         headers=["Control", "$n$", "Mean", "Median", "$p$ vs.\\ proposed", "Wins"],
         caption=("Control battery isolating the source of the accuracy gain. Negative controls "
                  "(random and shuffled index) test whether any extra input channel would "
                  "suffice; the elapsed-time control tests whether a pure time signal explains "
                  "the gain; the oracle lifetime-fraction control gives an upper bound."),
         label="tab:controls", sort_by="mean", bold_min="mean"),
    dict(csv_name="classical.csv", out_name="tab_classical.tex",
         cols=["predictor", "n", "mae_norm_mean", "mae_norm_std", "proposed_wins",
               "p_vs_proposed", "holm_p"],
         headers=["Method", "$n$", "Mean", "Std", "Wins", "$p$",
                  "$p_{\\mathrm{Holm}}$"],
         caption=("Classical and naive baselines evaluated on identical handcrafted features "
                  "under the same leave-one-bearing-out folds."),
         label="tab:classical", sort_by="mae_norm_mean", bold_min="mae_norm_mean"),
    dict(csv_name="seed_variability.csv", out_name="tab_seedvar.tex",
         cols=["config", "n_seeds", "mean_over_seeds", "std_over_seeds", "min_seed", "max_seed"],
         headers=["Configuration", "Seeds", "Mean", "Std", "Min", "Max"],
         caption=("Run-to-run variability across independent random seeds, reported for the core "
                  "configurations to establish that the reported ordering is not a seed artefact."),
         label="tab:seedvar", sort_by="mean_over_seeds", bold_min="mean_over_seeds"),
    dict(csv_name="weights_distribution.csv", out_name="tab_weights.tex",
         cols=["dataset", "seed", "n_folds", "a_median", "a_q25", "a_q75", "a_min", "a_max",
               "b_median", "fitness_median"],
         headers=["Dataset", "Seed", "Folds", "Median $a$", "Q25", "Q75", "Min", "Max",
                  "Median $b$", "Median fitness"],
         caption=("Distribution of the learned component weight $a$ across leave-one-bearing-out "
                  "folds. Each fold fits the weight on its training bearings only; the spread "
                  "therefore measures how stable the index definition is under resampling."),
         label="tab:weights", sort_by=None, group_by="__none__"),
    dict(csv_name="hi_quality.csv", out_name="tab_hi_quality.tex",
         cols=["indicator", "monotonicity_median", "trendability_median", "trendability_min",
               "prognosability", "rho_vs_deg_median", "n_bearings"],
         headers=["Indicator", "Monotonicity", "Trendability", "Trend. (min)",
                  "Prognosability", "$\\rho$ vs.\\ degradation", "$n$"],
         caption=("Health-indicator quality metrics computed on held-out bearings under the "
                  "leakage-free protocol, comparing the proposed index against standard "
                  "vibration indicators."),
         label="tab:hiquality", sort_by=None),
    dict(csv_name="early_warning_v2.csv", out_name="tab_ew.tex",
         cols=["detector", "n", "fired", "precision", "recall", "lead"],
         headers=["Detector", "$n$", "Fired", "Precision", "Recall", "Median lead"],
         caption=("Early-warning performance against an independent detectability onset. The "
                  "alarm threshold is selected on training bearings only; lead time is reported "
                  "in snapshots, with positive values meaning the alarm precedes the onset."),
         label="tab:ew", sort_by=None),
    dict(csv_name="conformal.csv", out_name="tab_conformal.tex",
         cols=["dataset", "nominal", "picp_mean", "picp_median", "picp_min", "mpiw_mean"],
         headers=["Dataset", "Nominal", "PICP (mean)", "PICP (median)", "PICP (worst)",
                  "MPIW"],
         caption=("Split-conformal prediction intervals calibrated on the validation bearings of "
                  "each fold and evaluated on the held-out bearing. PICP is the empirical "
                  "coverage and MPIW the mean interval width."),
         label="tab:conformal", sort_by=None, group_by="__none__"),
]


def cmd_emit():
    print("=" * 92)
    print(" SINH THÂN BẢNG LaTeX TỪ CSV v2 ".center(92, "="))
    print("=" * 92)
    print(f"Đích: {GEN}\n")
    msgs = [emit_table(**spec) for spec in EMIT_SPEC]
    for m in msgs:
        print(m)
    GEN.mkdir(parents=True, exist_ok=True)
    idx = GEN / "README.md"
    idx.write_text(
        "# Bảng sinh tự động\n\n"
        "Mọi file `.tex` trong thư mục này do `scripts/verify_paper_numbers.py --emit` sinh ra\n"
        "TRỰC TIẾP từ `results/tables/v2/*.csv`. **Đừng sửa tay** — sửa sẽ bị ghi đè.\n\n"
        "Cách dùng trong `VTOI_paper.tex`:\n\n"
        "```latex\n\\input{../generated/tab_main_results.tex}\n```\n\n"
        "Lý do: số trong bài không còn được gõ tay, nên không thể lệch với kết quả thí nghiệm\n"
        "(đây là lỗi A3/A9 mà reviewer đã bắt ở vòng 1).\n",
        encoding="utf-8")
    print(f"\n  [OK] README.md")
    print("\n👉 Trong VTOI_paper.tex, thay thân bảng gõ tay bằng:  \\input{../generated/<file>.tex}")
    print("=" * 92)
    return 0


# =================================================================================== --report
def cmd_report():
    print("=" * 92)
    print(" SỐ v2 QUAN TRỌNG — dán vào RESPONSE_TO_REVIEWERS_v2.md (chỗ [[ ]]) ".center(92, "="))
    print("=" * 92)

    mr = load_table("main_results.csv")
    if mr is not None and not mr.empty:
        print("\n### Bảng chính (normalized lifetime-fraction error)")
        c = [x for x in ["dataset", "config", "n_seeds", "n_bearings", "norm_median",
                         "norm_q25", "norm_q75", "norm_worst", "norm_mean",
                         "hours_mean", "mtoi_spearman_signed_mean"] if x in mr.columns]
        print(mr[c].to_string(index=False))
        for ds, g in mr.groupby("dataset"):
            p = g[g.config == PROPOSED]
            if p.empty:
                continue
            best_bl = g[~g.config.isin([PROPOSED, "vtoi_traj"])
                        & ~g.config.str.startswith(("abl_", "ctl_"))]
            print(f"\n  [{DS_DISPLAY.get(ds, ds)}] đề xuất: median={p.iloc[0]['norm_median']:.4f} "
                  f"mean={p.iloc[0]['norm_mean']:.4f} ({p.iloc[0]['hours_mean']:.3f} h)")
            if not best_bl.empty:
                b = best_bl.sort_values("norm_median").iloc[0]
                d = (b["norm_median"] - p.iloc[0]["norm_median"]) / b["norm_median"] * 100
                print(f"      baseline tốt nhất: {b['config']} median={b['norm_median']:.4f} "
                      f"→ cải thiện {d:+.1f}%")
    else:
        print("\n  (chưa có main_results.csv — chạy: bash run_v2.sh analyze)")

    for nm, title in [("wilcoxon.csv", "Wilcoxon + Holm (R2 #11/#27)"),
                      ("controls.csv", "Control battery (R2 #25)"),
                      ("factorial_2x2.csv", "Ablation 2×2 (R2 #2/#3)"),
                      ("hi_quality.csv", "HI quality (R3 #1/#4)"),
                      ("early_warning_v2.csv", "Early warning (R2 #21/#22)"),
                      ("conformal.csv", "Conformal intervals (R3 #6)"),
                      ("weights_distribution.csv", "Phân bố trọng số (R2 #14, R3 #3)")]:
        d = load_table(nm)
        print(f"\n### {title}")
        if d is None or d.empty:
            print("  (chưa có)")
        else:
            print(d.to_string(index=False))

    print("\n" + "=" * 92)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="đối chiếu .tex ↔ CSV (exit 1 nếu lệch)")
    ap.add_argument("--sweep", action="store_true", help="liệt kê số chưa được claim nào phủ")
    ap.add_argument("--emit", action="store_true", help="sinh thân bảng LaTeX từ CSV")
    ap.add_argument("--report", action="store_true", help="in số v2 để điền [[ ]]")
    ap.add_argument("--quiet", action="store_true", help="--check: chỉ in dòng lỗi")
    a = ap.parse_args()
    if not any([a.check, a.sweep, a.emit, a.report]):
        ap.print_help()
        return 0
    rc = 0
    if a.emit:
        rc |= cmd_emit()
    if a.report:
        rc |= cmd_report()
    if a.sweep:
        rc |= cmd_sweep()
    if a.check:
        rc |= cmd_check(verbose=not a.quiet)
    return rc


if __name__ == "__main__":
    sys.exit(main())
