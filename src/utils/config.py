# -*- coding: utf-8 -*-
"""
config.py — ĐỌC FILE CẤU HÌNH YAML.

Mỗi thí nghiệm (dataset chính, ablation, robustness...) có 1 file .yaml trong configs/.
File này giúp đọc YAML thành dict Python, và cho phép truy cập kiểu cfg.train.epochs
cho gọn (qua lớp DotDict).
"""

from pathlib import Path          # thao tác đường dẫn


class DotDict(dict):
    """Cho phép truy cập dict bằng dấu chấm: cfg.train.epochs thay vì cfg['train']['epochs']."""
    def __getattr__(self, key):
        # Khi gọi cfg.key -> lấy giá trị tương ứng trong dict.
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Cấu hình không có khoá: '{key}'")
        # Nếu giá trị lại là dict -> bọc tiếp thành DotDict (đệ quy) để truy cập nhiều tầng.
        if isinstance(val, dict):
            return DotDict(val)
        return val

    # Cho phép gán cfg.key = value.
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path) -> DotDict:
    """Đọc 1 file YAML và trả về DotDict (truy cập bằng dấu chấm)."""
    import yaml                                  # thư viện đọc YAML (PyYAML)
    path = Path(path)                            # ép kiểu Path
    if not path.exists():                        # kiểm tra file tồn tại
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f: # mở file YAML
        data = yaml.safe_load(f)                 # parse YAML -> dict (safe_load an toàn)
    return DotDict(data or {})                   # trả về DotDict (rỗng nếu file trống)


def pretty_print_config(cfg: dict, indent: int = 0):
    """In cấu hình ra màn hình theo dạng cây cho dễ kiểm tra trước khi chạy."""
    for k, v in cfg.items():                     # duyệt từng cặp khoá-giá trị
        if isinstance(v, dict):                  # nếu là dict con -> in tiêu đề rồi đệ quy
            print("  " * indent + f"{k}:")
            pretty_print_config(v, indent + 1)
        else:                                    # nếu là giá trị thường -> in trực tiếp
            print("  " * indent + f"{k}: {v}")
