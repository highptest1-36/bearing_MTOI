# -*- coding: utf-8 -*-
"""
paths.py — TRUNG TÂM QUẢN LÝ ĐƯỜNG DẪN của toàn bộ dự án MTOI-Bearing.

Triết lý thiết kế (rất quan trọng để sống sót khi Colab bị disconnect):
  - DỮ LIỆU GỐC (zip) nằm trên Google Drive  -> không bao giờ mất.
  - GIẢI NÉN RAW (rất lớn, ~20GB) -> để ở ổ đĩa LOCAL /content (nhanh, nhưng mất khi disconnect).
       => Nếu mất, chỉ cần giải nén lại từ zip trên Drive (Phase 1 tự kiểm tra & làm lại).
  - KẾT QUẢ ĐÃ XỬ LÝ (processed: .npy, .csv features, mtoi, labels, splits) -> ghi lên DRIVE.
       => Các phase nặng (feature, mtoi, train) chỉ chạy 1 lần; lần sau tự load lại từ Drive.
  - CHECKPOINT model, bảng, hình -> ghi lên DRIVE (results/).

Nhờ vậy: mỗi phase đều "idempotent" (chạy lại không hại), và nếu rớt mạng giữa chừng,
bạn chỉ chạy lại phase đang dở, các phase trước tự bỏ qua vì kết quả đã có trên Drive.
"""

import os                                   # thư viện thao tác đường dẫn/biến môi trường
from pathlib import Path                    # Path: làm việc với đường dẫn an toàn, đa nền tảng


# =========================================================================================
# 1) XÁC ĐỊNH GỐC DỰ ÁN TRÊN DRIVE
# =========================================================================================
# Thư mục gốc chứa cả dataset/ và MTOI-Bearing/ trên Google Drive.
# Cho phép ghi đè bằng biến môi trường MTOI_DRIVE_ROOT nếu bạn đổi vị trí.
DRIVE_ROOT = Path(os.environ.get("MTOI_DRIVE_ROOT", "/content/drive/MyDrive/Bearing_MTOI"))

# Thư mục mã nguồn dự án (chứa src/, scripts/, configs/, results/, paper/...).
PROJECT_ROOT = DRIVE_ROOT / "MTOI-Bearing"

# Thư mục chứa 3 file zip dữ liệu GỐC (do bạn đã tải về sẵn).
DATASET_ZIP_DIR = DRIVE_ROOT / "dataset"

# Đường dẫn cụ thể tới từng zip gốc (đã kiểm tra đúng tên file thực tế trên Drive của bạn).
MENDELEY_ZIP   = DATASET_ZIP_DIR / "Mendeley" / "Vibration_Bearing_RuntoFailure.zip"
PRONOSTIA_ZIP  = DATASET_ZIP_DIR / "PRONOSTIA FEMTO-ST" / "10.+FEMTO+Bearing.zip"
# XJTU-SY bị tách làm 3 file zip của Google Drive; bên trong là RAR nhiều phần (part01..part06).
XJTU_ZIPS = [
    DATASET_ZIP_DIR / "XJTU-SY" / "XJTU-SY_Bearing_Datasets-20260618T230340Z-3-001.zip",
    DATASET_ZIP_DIR / "XJTU-SY" / "XJTU-SY_Bearing_Datasets-20260618T230340Z-3-002.zip",
    DATASET_ZIP_DIR / "XJTU-SY" / "XJTU-SY_Bearing_Datasets-20260618T230340Z-3-003.zip",
]


# =========================================================================================
# 2) THƯ MỤC GIẢI NÉN RAW (đặt ở LOCAL cho nhanh; có thể đổi sang Drive nếu muốn bền)
# =========================================================================================
# Mặc định dùng ổ local /content/MTOI_raw để giải nén (tốc độ đọc/ghi nhanh hơn Drive nhiều).
# Nếu bạn muốn giữ raw trên Drive (bền hơn nhưng chậm), đặt biến môi trường MTOI_RAW_DIR.
RAW_DIR = Path(os.environ.get("MTOI_RAW_DIR", "/content/MTOI_raw"))
RAW_MENDELEY  = RAW_DIR / "mendeley"     # nơi chứa 129 file LogFile_*.csv sau giải nén
RAW_PRONOSTIA = RAW_DIR / "pronostia"    # nơi chứa Learning_set/Test_set/... của FEMTO
RAW_XJTU      = RAW_DIR / "xjtu_sy"      # nơi chứa 3 thư mục điều kiện + các bearing folder
RAW_IMS       = RAW_DIR / "ims"          # IMS/NASA: 1st_test/, 2nd_test/, 4th_test/txt/ (R3 #2)


# =========================================================================================
# 3) THƯ MỤC KẾT QUẢ ĐÃ XỬ LÝ (đặt trên DRIVE để KHÔNG MẤT khi disconnect)
# =========================================================================================
DATA_DIR       = DRIVE_ROOT / "data"             # gốc dữ liệu đã xử lý
PROCESSED_DIR  = DATA_DIR / "processed"          # .npy windows, features.csv, mtoi.csv, labels.csv
SPLITS_DIR     = DATA_DIR / "splits"             # file mô tả chia train/val/test theo thời gian

# Tách processed theo từng dataset cho gọn.
PROC_MENDELEY  = PROCESSED_DIR / "mendeley"
PROC_PRONOSTIA = PROCESSED_DIR / "pronostia"
PROC_XJTU      = PROCESSED_DIR / "xjtu_sy"
PROC_IMS       = PROCESSED_DIR / "ims"


# =========================================================================================
# 4) THƯ MỤC KẾT QUẢ THỰC NGHIỆM (results) — cũng trên DRIVE
# =========================================================================================
RESULTS_DIR     = PROJECT_ROOT / "results"
TABLES_DIR      = RESULTS_DIR / "tables"         # các bảng .csv cho paper
FIGURES_DIR     = RESULTS_DIR / "figures"        # các hình .png cho paper
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"    # file model .pt + state.json theo phase
LOGS_DIR        = RESULTS_DIR / "logs"           # log text từng lần chạy
PREDICTIONS_DIR = RESULTS_DIR / "predictions"    # file predictions.csv của model


# =========================================================================================
# 5) HÀM TIỆN ÍCH: TẠO SẴN MỌI THƯ MỤC CẦN THIẾT
# =========================================================================================
def ensure_dirs():
    """Tạo tất cả thư mục cần thiết nếu chưa tồn tại (gọi 1 lần ở đầu mỗi phase)."""
    # Danh sách mọi thư mục cần đảm bảo tồn tại.
    all_dirs = [
        RAW_DIR, RAW_MENDELEY, RAW_PRONOSTIA, RAW_XJTU, RAW_IMS,   # thư mục raw (local)
        DATA_DIR, PROCESSED_DIR, SPLITS_DIR,                       # thư mục processed (drive)
        PROC_MENDELEY, PROC_PRONOSTIA, PROC_XJTU, PROC_IMS,
        RESULTS_DIR, TABLES_DIR, FIGURES_DIR,                      # thư mục results (drive)
        CHECKPOINTS_DIR, LOGS_DIR, PREDICTIONS_DIR,
    ]
    # Duyệt và tạo từng thư mục; parents=True tạo cả cây cha; exist_ok=True không lỗi nếu đã có.
    for d in all_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    # Trả về True để biết đã chạy xong (tiện in log).
    return True


def proc_dir_for(dataset_name: str) -> Path:
    """Trả về thư mục processed tương ứng với tên dataset ('mendeley'|'pronostia'|'xjtu_sy')."""
    mapping = {                                   # bảng tra cứu tên -> thư mục
        "mendeley": PROC_MENDELEY,
        "pronostia": PROC_PRONOSTIA,
        "xjtu_sy": PROC_XJTU,
        "ims": PROC_IMS,
    }
    # Nếu tên không hợp lệ -> báo lỗi rõ ràng để dễ sửa.
    if dataset_name not in mapping:
        raise ValueError(f"Tên dataset không hợp lệ: {dataset_name}. Chọn 1 trong {list(mapping)}")
    return mapping[dataset_name]


def raw_dir_for(dataset_name: str) -> Path:
    """Trả về thư mục raw (đã giải nén) tương ứng với tên dataset."""
    mapping = {
        "mendeley": RAW_MENDELEY,
        "pronostia": RAW_PRONOSTIA,
        "xjtu_sy": RAW_XJTU,
        "ims": RAW_IMS,
    }
    if dataset_name not in mapping:
        raise ValueError(f"Tên dataset không hợp lệ: {dataset_name}. Chọn 1 trong {list(mapping)}")
    return mapping[dataset_name]


# Khi import module này lần đầu, tự động tạo sẵn các thư mục (an toàn, không hại nếu đã có).
ensure_dirs()
