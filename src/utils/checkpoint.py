# -*- coding: utf-8 -*-
"""
checkpoint.py — CƠ CHẾ "RESUME" GIÚP SỐNG SÓT KHI COLAB DISCONNECT.

Ý tưởng cốt lõi:
  - Mỗi PHASE (bước lớn) ghi trạng thái vào 1 file JSON duy nhất trên Drive: state.json.
  - Trước khi chạy 1 phase, ta hỏi: "phase này đã DONE chưa và file output có còn không?"
        + Nếu RỒI  -> bỏ qua (skip), load kết quả cũ -> tiết kiệm thời gian, không chạy lại.
        + Nếu CHƯA -> chạy, rồi đánh dấu DONE + lưu danh sách file output.
  - Nhờ vậy nếu rớt mạng, bạn chỉ cần chạy lại; mọi phase đã xong tự bỏ qua.

state.json có dạng:
{
  "phase1_extract":  {"status": "done", "time": "...", "outputs": [...], "meta": {...}},
  "phase3_features": {"status": "done", ...},
  ...
}
"""

import json                       # đọc/ghi file JSON
import os                         # kiểm tra file tồn tại
import tempfile                   # ghi file tạm để lưu "nguyên tử" (atomic)
from datetime import datetime     # gắn timestamp
from pathlib import Path          # thao tác đường dẫn


class PhaseTracker:
    """Quản lý trạng thái hoàn thành của các phase qua 1 file state.json."""

    def __init__(self, state_path=None):
        # Nếu không truyền đường dẫn -> dùng mặc định results/checkpoints/state.json trên Drive.
        if state_path is None:
            from src.utils.paths import CHECKPOINTS_DIR
            state_path = CHECKPOINTS_DIR / "state.json"
        self.state_path = Path(state_path)                 # ghi nhớ đường dẫn state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)  # đảm bảo thư mục cha tồn tại
        self.state = self._load()                          # nạp trạng thái hiện có (nếu có)

    def _load(self) -> dict:
        """Đọc state.json nếu tồn tại; nếu chưa có thì trả về dict rỗng."""
        if self.state_path.exists():                       # nếu file đã có
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)                    # nạp nội dung JSON
            except Exception:
                # File hỏng (hiếm) -> bắt đầu lại từ rỗng để không crash.
                return {}
        return {}                                          # chưa có file -> rỗng

    def _save(self):
        """Ghi state ra đĩa theo kiểu 'atomic' (ghi file tạm rồi đổi tên) để tránh hỏng file."""
        # Ghi vào file tạm cùng thư mục trước.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.state_path.parent), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)  # ghi đẹp, giữ tiếng Việt
        # Đổi tên file tạm thành file thật -> thao tác nguyên tử, không lo ghi dở.
        os.replace(tmp_path, self.state_path)

    # ----------------------- API CHÍNH -----------------------

    def is_done(self, phase: str, check_outputs: bool = True) -> bool:
        """
        Trả về True nếu phase đã 'done'.
        Nếu check_outputs=True: kiểm tra thêm các file output có còn tồn tại không
        (phòng trường hợp đánh dấu done nhưng file bị xoá).
        """
        info = self.state.get(phase)                       # lấy thông tin phase (nếu có)
        if not info or info.get("status") != "done":       # chưa có hoặc chưa done
            return False
        if check_outputs:                                  # nếu cần kiểm tra file output
            for out in info.get("outputs", []):            # duyệt từng file output đã ghi
                if not Path(out).exists():                 # nếu thiếu 1 file -> coi như CHƯA done
                    return False
        return True                                        # mọi điều kiện ok -> đã done

    def mark_running(self, phase: str):
        """Đánh dấu phase đang chạy (để biết phase nào bị ngắt giữa chừng)."""
        self.state[phase] = {                              # ghi thông tin trạng thái running
            "status": "running",
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()                                       # lưu xuống đĩa ngay

    def mark_done(self, phase: str, outputs=None, meta=None):
        """Đánh dấu phase đã hoàn thành + ghi danh sách file output + metadata tuỳ ý."""
        self.state[phase] = {
            "status": "done",
            "time": datetime.now().isoformat(timespec="seconds"),
            "outputs": [str(o) for o in (outputs or [])],  # lưu đường dẫn dạng chuỗi
            "meta": meta or {},                            # metadata bổ sung (vd: số sample, RMSE...)
        }
        self._save()
        print(f"[CHECKPOINT] Đã đánh dấu DONE phase '{phase}' "
              f"({len(outputs or [])} file output).")

    def get(self, phase: str) -> dict:
        """Lấy toàn bộ thông tin của 1 phase (status/time/outputs/meta)."""
        return self.state.get(phase, {})

    def summary(self):
        """In bảng tóm tắt trạng thái mọi phase đã ghi (tiện theo dõi 'đến đâu rồi')."""
        print("-" * 60)
        print(" TÓM TẮT TIẾN TRÌNH (state.json) ".center(60, "-"))
        if not self.state:                                 # chưa có phase nào
            print(" (chưa có phase nào được ghi nhận)")
        for phase, info in self.state.items():             # duyệt từng phase
            status = info.get("status", "?")               # trạng thái
            t = info.get("time", "?")                      # thời điểm
            n_out = len(info.get("outputs", []))           # số file output
            print(f"  - {phase:28s} | {status:8s} | {t} | {n_out} output")
        print("-" * 60)


def skip_if_exists(*paths) -> bool:
    """
    Tiện ích nhanh: trả về True nếu TẤT CẢ 'paths' đều đã tồn tại.
    Dùng để bỏ qua 1 bước nhỏ khi output của nó đã có sẵn.
    """
    return all(Path(p).exists() for p in paths)
