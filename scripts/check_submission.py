# -*- coding: utf-8 -*-
r"""
check_submission.py — CỔNG KIỂM TRA TRƯỚC KHI NỘP (Access-2026-31251)

Vì sao có file này
------------------
Vòng nộp lại 2026-08 suýt hỏng vì **thư phản hồi hứa những thứ bài không có**:
"Appendix A/C/D", "Table 13", "Algorithm 1", bảng onset, bảng giai thừa... đều được
liệt kê trong thư trong khi bài chỉ có 11 bảng và 0 phụ lục. Gốc rễ là
REVIEWER_AUDIT_v2.md đánh ✅ dựa trên "CSV đã có kết quả", không phải "bài đã có bảng".

Reviewer 2 của vòng 1 đã bắt được lỗi lệch số 0,436 vs 0,448. Người đọc kỹ như vậy
CHẮC CHẮN sẽ lật thư đối chiếu bài. Một tham chiếu hụt là "did not address all previous
concerns" — reject không cho nộp lại.

Script này chặn đúng chỗ đó. CHẠY LẠI SAU MỌI LẦN THÊM/BỚT/DI CHUYỂN BẢNG.
Chỉ cần một bảng chèn thêm là mọi số bảng phía sau dịch một nấc và thư sai ngay.

Chạy:
    python3 MTOI-Bearing/scripts/check_submission.py
    python3 MTOI-Bearing/scripts/check_submission.py --quiet   # chỉ in FAIL

Exit code 0 = sạch, 1 = có lỗi (dùng được trong pre-commit / CI).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../Bearing_MTOI
TEX = ROOT / "paper" / "ACCESS_latex_template_20240429" / "VTOI_paper.tex"
AUX = Path("/content/build/VTOI_paper.aux")          # sinh ra khi biên dịch
RESP = ROOT / "paper" / "response_src"
CSVDIR = ROOT / "MTOI-Bearing" / "results" / "tables" / "v2"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

fails: list[str] = []
warns: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


# ----------------------------------------------------------------- đọc bài
def load_paper() -> str:
    if not TEX.exists():
        print(f"❌ không thấy {TEX}")
        sys.exit(1)
    return TEX.read_text(encoding="utf-8")


def numbering(paper: str) -> tuple[dict, dict, dict]:
    r"""Số hiệu THẬT của bảng/hình/algorithm.

    Ưu tiên .aux (chính xác tuyệt đối). Nếu chưa biên dịch thì suy ra từ thứ tự
    xuất hiện trong .tex — LaTeX đánh số float theo thứ tự \caption được thực thi,
    trùng với thứ tự nguồn.
    """
    if AUX.exists():
        aux = AUX.read_text(encoding="utf-8")
        tab = {int(v): k for k, v in re.findall(r"newlabel\{tab:([^}]+)\}\{\{(\d+)\}", aux)}
        fig = {int(v): k for k, v in re.findall(r"newlabel\{fig:([^}]+)\}\{\{(\d+)\}", aux)}
        alg = {int(v): k for k, v in re.findall(r"newlabel\{alg:([^}]+)\}\{\{(\d+)\}", aux)}
        if tab:
            return tab, fig, alg
        warn("`.aux` có nhưng không đọc được nhãn — suy ra từ thứ tự nguồn")
    else:
        warn(f"chưa biên dịch ({AUX} không tồn tại) — suy số hiệu từ thứ tự nguồn")
    tab = {i + 1: k for i, k in enumerate(re.findall(r"\\label\{tab:([^}]+)\}", paper))}
    fig = {i + 1: k for i, k in enumerate(re.findall(r"\\label\{fig:([^}]+)\}", paper))}
    alg = {i + 1: k for i, k in enumerate(re.findall(r"\\label\{alg:([^}]+)\}", paper))}
    return tab, fig, alg


def sections(paper: str) -> dict:
    """Bản đồ 'V-G' -> tên tiểu mục thật."""
    out, order, subs, cur = {}, [], {}, None
    for line in paper.splitlines():
        m = re.match(r"\\section\{(.*)\}", line)
        if m:
            cur = m.group(1)
            order.append(cur)
            subs[cur] = []
            continue
        m = re.match(r"\\subsection\{(.*)\}", line)
        if m and cur:
            subs[cur].append(m.group(1))
    for i, s in enumerate(order):
        if i >= len(ROMAN):
            break
        out[ROMAN[i]] = s
        for j, x in enumerate(subs[s]):
            out[f"{ROMAN[i]}-{chr(65 + j)}"] = x
    return out


# ----------------------------------------------------------------- các phép kiểm
def check_paper_hygiene(paper: str) -> None:
    n_iif = paper.count(r"\subsection{Composite Health Indicators and Their Learned Alternatives}")
    if n_iif != 1:
        fail(f"mục II-F xuất hiện {n_iif} lần (phải đúng 1) — assemble.py mất tính idempotent")

    n_math = len(re.findall(r"\$[0-9][0-9.,]*\$", paper))
    if n_math:
        fail(f"{n_math} số trần còn nằm trong $...$ — sẽ in bằng Computer Modern, lệch font Times")

    n_bad = len(re.findall(r"textbf\{\$", paper))
    if n_bad:
        fail(f"{n_bad} chỗ có math bên trong \\textbf — ieeeaccess.cls VỠ ở đây, dùng \\textless")

    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", paper, re.S)
    if ab:
        w = len(ab.group(1).split())
        if w > 250:
            fail(f"abstract {w} từ (trần IEEE Access là 250)")
        elif w > 240:
            warn(f"abstract {w} từ — sát trần 250, thêm một câu là vi phạm")
    ti = re.search(r"\\title\{(.*?)\}\n", paper, re.S)
    if ti and len(ti.group(1).split()) > 16:
        warn(f"tiêu đề {len(ti.group(1).split())} từ — IEEE Access khuyến nghị 12–15")


def check_all_cited(paper: str) -> None:
    for kind, name in [("tab", "bảng"), ("fig", "hình"), ("alg", "algorithm")]:
        labs = set(re.findall(r"\\label\{(" + kind + r":[^}]+)\}", paper))
        un = [l for l in labs if not re.search(r"\\ref\{" + re.escape(l) + r"\}", paper)]
        if un:
            fail(f"{name} chưa được trích dẫn trong thân bài: {un}")


def check_captions(paper: str) -> None:
    """R2.12: mọi caption bảng phải nêu đơn vị đánh giá."""
    UNIT = r"per-bearing|per bearing|held-out bearing|lifetime-fraction|snapshot|the fold"
    miss = [lab for cap, lab in re.findall(r"\\caption\{(.*?)\}\s*\n\\label\{(tab:[^}]+)\}", paper, re.S)
            if not re.search(UNIT, " ".join(cap.split()))]
    if miss:
        fail(f"caption bảng chưa nêu đơn vị đánh giá (thư đã hứa ở R2.12): {miss}")


def check_refs_doi(paper: str) -> None:
    if "\\begin{thebibliography}" not in paper:
        return
    items = re.split(r"\\bibitem\{", paper.split("\\begin{thebibliography}")[1])[1:]
    nodoi = [i.split("}")[0] for i in items if "doi:" not in i]
    # Các mục dưới đây đã tra Crossref: thực sự KHÔNG có DOI, không phải bỏ sót.
    # Kỷ yếu NeurIPS/ICLR, bản arXiv, và hai bài trước thời DOI.
    known = {"Nectoux2012", "Coble2009", "Coble2010", "Gousseau2016",
             "Mahalanobis1936", "Holm1979", "Vaswani2017", "Loshchilov2019",
             "Bai2018", "Adebayo2018", "Hooker2019"}
    extra = [k for k in nodoi if k not in known]
    if extra:
        warn(f"tài liệu thiếu DOI ngoài 4 mục đã biết là không có DOI: {extra}")


def check_response(paper: str) -> None:
    sys.path.insert(0, str(RESP))
    try:
        import content_r1, content_r2, content_r3  # noqa: E402
    except Exception as ex:                        # pragma: no cover
        fail(f"không nạp được nội dung thư: {ex}")
        return

    tab, fig, alg = numbering(paper)
    sec = sections(paper)
    csvs = "".join(f.read_text(encoding="utf-8", errors="ignore") for f in CSVDIR.glob("*.csv")) \
        if CSVDIR.exists() else ""

    NUM = re.compile(r"\b\d+\.\d{3,4}\b")
    n_items = 0
    for mod, tag in [(content_r1, "R1"), (content_r2, "R2"), (content_r3, "R3")]:
        for item in getattr(mod, tag):
            n_items += 1
            key = f"{tag}.{item[0]}"
            # CHỈ soi phần "Author response" + "Author action"; phần comment là lời
            # reviewer nói về BẢN CŨ, số bảng trong đó KHÔNG được sửa.
            txt = item[2] + "\n" + item[3]

            for t in {int(x) for x in re.findall(r"Table (\d+)", txt)}:
                if t not in tab:
                    fail(f"{key}: nhắc Table {t} — bài chỉ có {len(tab)} bảng")
            for f_ in {int(x) for x in re.findall(r"Figure (\d+)", txt)}:
                if f_ not in fig:
                    fail(f"{key}: nhắc Figure {f_} — bài chỉ có {len(fig)} hình")
            for a in {int(x) for x in re.findall(r"Algorithm (\d+)", txt)}:
                if a not in alg:
                    fail(f"{key}: nhắc Algorithm {a} — bài không có")
            for s in {x for x in re.findall(r"Section ([IVX]+(?:-[A-Z](?:-\d)?)?)", txt)}:
                if s not in sec:
                    fail(f"{key}: nhắc Section {s} — bài không có mục đó")
            for a in set(re.findall(r"Appendix ([A-Z])", txt)):
                fail(f"{key}: nhắc Appendix {a} — bài KHÔNG có phụ lục nào")

            for n in sorted(set(NUM.findall(txt))):
                v = float(n)
                if not any(f"{v:.{d}f}" in paper or f"{v:.{d}f}" in csvs for d in (2, 3, 4)):
                    fail(f"{key}: số {n} không truy nguyên được về bài hay CSV kết quả")

    if n_items != 41:
        fail(f"thư có {n_items} ý, phải là 41 (5 R1 + 30 R2 + 6 R3)")


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="chỉ in FAIL")
    args = ap.parse_args()

    paper = load_paper()
    tab, fig, alg = numbering(paper)
    sec = sections(paper)

    check_paper_hygiene(paper)
    check_all_cited(paper)
    check_captions(paper)
    check_refs_doi(paper)
    check_response(paper)

    if not args.quiet:
        print("=" * 78)
        print("KIỂM TRA TRƯỚC KHI NỘP — Access-2026-31251")
        print("=" * 78)
        print(f"  bài      : {len(tab)} bảng · {len(fig)} hình · {len(alg)} algorithm · "
              f"{len(sec)} mục/tiểu mục")
        print(f"  nguồn số : {'.aux (chính xác)' if AUX.exists() else 'thứ tự nguồn (suy ra)'}")
        print("\n  BẢNG:")
        for i in sorted(tab):
            print(f"    Table {i:2d}  {tab[i]}")
        print("  HÌNH:")
        for i in sorted(fig):
            print(f"    Figure {i}  {fig[i]}")
        print()

    for w in warns:
        print(f"  ⚠️  {w}")
    for f_ in fails:
        print(f"  ❌ {f_}")

    print("-" * 78)
    if fails:
        print(f"  KẾT QUẢ: ❌ {len(fails)} LỖI — KHÔNG ĐƯỢC NỘP cho tới khi sửa xong")
        return 1
    print(f"  KẾT QUẢ: ✅ SẠCH{f' ({len(warns)} cảnh báo)' if warns else ''} — thư khớp bài hoàn toàn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
