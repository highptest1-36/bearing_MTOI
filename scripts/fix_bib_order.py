#!/usr/bin/env python3
r"""Sap lai thu tu \bibitem theo thu tu trich dan dau tien (chuan IEEE).

Chi dong cham vung giua \begin{thebibliography} va \end{thebibliography}.
Khong doi mot ky tu nao cua noi dung tung \bibitem.

    python3 fix_bib_order.py --check    # chi bao cao, khong sua
    python3 fix_bib_order.py --apply    # sua tai cho, tao .bak_biborder
"""
import re, sys, shutil, pathlib

TEX = pathlib.Path("/content/drive/MyDrive/Bearing_MTOI/paper/"
                   "ACCESS_latex_template_20240429/VTOI_paper.tex")
BEG, END = r"\begin{thebibliography}", r"\end{thebibliography}"


def split_tex(src):
    i = src.index(BEG); j = src.index(END)
    head_end = src.index("\n", i) + 1          # giu nguyen dong \begin{...}{00}
    return src[:head_end], src[head_end:j], src[j:]


def parse_items(bib):
    """Tra ve [(key, text_khoi_day_du)] giu nguyen khoang trang giua cac khoi."""
    spans = [(m.group(1), m.start()) for m in re.finditer(r"\\bibitem\{([^}]+)\}", bib)]
    out = []
    for n, (key, start) in enumerate(spans):
        stop = spans[n + 1][1] if n + 1 < len(spans) else len(bib)
        out.append((key, bib[start:stop].rstrip() + "\n"))
    return out


def cite_order(body):
    order = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", body):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k and k not in order:
                order.append(k)
    return order


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    src = TEX.read_text()
    head, bib, tail = split_tex(src)
    items = parse_items(bib)
    by_key = dict(items)
    cur = [k for k, _ in items]
    want = cite_order(src[:src.index(BEG)])

    missing = [k for k in want if k not in by_key]
    uncited = [k for k in cur if k not in want]
    if missing:
        sys.exit(f"LOI: trich dan khong co bibitem: {missing}")
    if uncited:
        sys.exit(f"LOI: bibitem khong duoc trich: {uncited}")

    if cur == want:
        print("OK: danh muc da dung thu tu trich dan dau tien. Khong can sua.")
        return

    print(f"Can sap lai {sum(a != b for a, b in zip(cur, want))}/{len(cur)} vi tri.\n")
    for n, k in enumerate(want, 1):
        old = cur.index(k) + 1
        mark = "   " if old == n else "-> "
        print(f"  {mark}[{n:2d}] {k:18s} (dang la [{old}])")

    if mode != "--apply":
        print("\n(chay lai voi --apply de sua)")
        return

    shutil.copy(TEX, str(TEX) + ".bak_biborder")
    TEX.write_text(head + "\n" + "\n".join(by_key[k] for k in want) + "\n" + tail)
    print(f"\nDa sua. Ban goc: {TEX.name}.bak_biborder")
    print("Bien dich lai 3 lan roi chay check_submission.py + verify_paper_numbers.py")


if __name__ == "__main__":
    main()
