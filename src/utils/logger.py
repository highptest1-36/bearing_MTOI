# -*- coding: utf-8 -*-
"""
logger.py — IN TIẾN TRÌNH ĐẸP & GHI LOG RA FILE.

Yêu cầu của bạn: khi chạy code phải IN tiến trình + kết quả đầy đủ để theo dõi
"đến đâu rồi và khi nào xong". Module này cung cấp:
  - get_logger(): vừa in ra màn hình, vừa ghi vào results/logs/<tên>.log (để xem lại sau khi disconnect).
  - section(): in tiêu đề một bước cho dễ nhìn.
  - progress_iter(): vòng lặp có thanh tiến trình (tqdm) + ETA (ước lượng thời gian còn lại).
  - Timer: đo thời gian từng đoạn.
"""

import sys                       # ghi ra stdout
import time                      # đo thời gian
import logging                   # thư viện log chuẩn của Python
from datetime import datetime    # tạo timestamp cho tên file log


def get_logger(name: str = "MTOI", to_file: bool = True):
    """
    Tạo (hoặc lấy lại) một logger:
      - In ra màn hình (StreamHandler).
      - Ghi vào file results/logs/<name>_<ngày-giờ>.log nếu to_file=True.
    """
    logger = logging.getLogger(name)               # lấy logger theo tên
    logger.setLevel(logging.INFO)                  # mức log INFO trở lên
    # Nếu logger này đã có handler (đã gọi trước đó) thì không thêm nữa (tránh in lặp).
    if logger.handlers:
        return logger

    # Định dạng dòng log: [giờ:phút:giây] NỘI DUNG
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # Handler 1: in ra màn hình.
    sh = logging.StreamHandler(sys.stdout)         # in ra stdout (Colab hiển thị được)
    sh.setFormatter(fmt)                           # áp định dạng
    logger.addHandler(sh)                          # gắn vào logger

    # Handler 2: ghi ra file log trên Drive (để xem lại nếu rớt mạng).
    if to_file:
        from src.utils.paths import LOGS_DIR       # import muộn để tránh vòng import
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Tên file gồm ngày-giờ để mỗi lần chạy có 1 file riêng, không đè lên nhau.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOGS_DIR / f"{name}_{stamp}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False                       # không đẩy log lên logger gốc (tránh in 2 lần)
    return logger


def section(title: str, logger=None):
    """In một tiêu đề khối lớn cho dễ nhìn (phân tách các bước)."""
    line = "=" * 70                                # đường kẻ ngang
    msg = f"\n{line}\n {title} \n{line}"           # ghép tiêu đề vào giữa 2 đường kẻ
    if logger:                                     # nếu có logger -> dùng logger
        logger.info(msg)
    else:                                          # không thì print thường
        print(msg)


class Timer:
    """Đồng hồ bấm giờ đơn giản: với khối 'with Timer(...)' sẽ tự in thời gian chạy."""
    def __init__(self, name: str = "block", logger=None):
        self.name = name                           # tên đoạn đang đo
        self.logger = logger                       # logger để in (nếu có)

    def __enter__(self):
        self.t0 = time.time()                      # ghi mốc thời gian bắt đầu
        msg = f"[BẮT ĐẦU] {self.name} ..."
        (self.logger.info if self.logger else print)(msg)  # in dòng bắt đầu
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.time() - self.t0                 # tính thời gian đã trôi qua
        # Đổi giây sang dạng dễ đọc h/m/s.
        msg = f"[XONG]    {self.name} trong {format_duration(dt)}"
        (self.logger.info if self.logger else print)(msg)  # in dòng kết thúc


def format_duration(seconds: float) -> str:
    """Đổi số giây thành chuỗi 'Xh Ym Zs' cho dễ đọc."""
    seconds = int(seconds)                         # bỏ phần thập phân
    h = seconds // 3600                            # số giờ
    m = (seconds % 3600) // 60                     # số phút
    s = seconds % 60                               # số giây
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def progress_iter(iterable, total=None, desc="Đang xử lý"):
    """
    Bọc một iterable để hiện thanh tiến trình (tqdm) kèm ETA.
    Nếu không có tqdm thì tự fallback in thủ công mỗi 10% để vẫn theo dõi được.
    """
    try:
        from tqdm.auto import tqdm                 # tqdm tự chọn giao diện phù hợp (notebook/terminal)
        return tqdm(iterable, total=total, desc=desc, ncols=100)
    except Exception:
        # Fallback: không có tqdm -> tự in tiến trình theo phần trăm.
        return _manual_progress(iterable, total=total, desc=desc)


def _manual_progress(iterable, total=None, desc="Đang xử lý"):
    """Bộ đếm tiến trình thủ công khi không có tqdm (in mỗi ~10%)."""
    # Cố gắng lấy tổng số phần tử nếu chưa truyền total.
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = None
    t0 = time.time()                               # mốc bắt đầu để tính ETA
    last_pct = -1                                  # phần trăm in lần trước
    for i, item in enumerate(iterable):            # duyệt từng phần tử
        yield item                                 # trả phần tử ra cho vòng for bên ngoài dùng
        if total:                                  # nếu biết tổng -> tính %
            pct = int((i + 1) * 100 / total)       # phần trăm hoàn thành
            if pct // 10 != last_pct // 10:        # chỉ in khi qua mốc 10% mới
                elapsed = time.time() - t0          # thời gian đã chạy
                eta = elapsed / (i + 1) * (total - i - 1)  # ước lượng thời gian còn lại
                print(f"  {desc}: {pct:3d}% ({i+1}/{total}) | đã chạy {format_duration(elapsed)} | còn ~{format_duration(eta)}")
                last_pct = pct
