# -*- coding: utf-8 -*-
"""
samplers.py — BATCH SAMPLER GOM THEO Ổ BI, SNAPSHOT LIÊN TIẾP (trả lời Reviewer 2, ý #17).

VẤN ĐỀ REVIEWER NÊU:
  L_smooth (sai phân bậc 2) và L_mono cần chuỗi dự đoán THEO THỨ TỰ THỜI GIAN của CÙNG một ổ bi.
  Bản cũ dùng DataLoader(shuffle=True) -> mỗi batch chứa một TẬP CON NGẪU NHIÊN các snapshot.

BẢN CŨ ĐÃ ĐÚNG MỘT NỬA:
  `src/losses.py:_traj_penalty()` đã gom theo `bearing_idx` rồi mới tính phạt, nên KHÔNG bao giờ
  tính xuyên biên ổ bi (đây là điểm ta phản bác được). NHƯNG vì batch ngẫu nhiên, sai phân được
  tính trên chuỗi con THƯA, cách quãng không đều -> là XẤP XỈ, không phải sai phân thật.

SAMPLER NÀY khắc phục triệt để: mỗi batch = các cửa sổ của MỘT ổ bi, thuộc các snapshot LIÊN TIẾP.
  -> sai phân bậc 1/2 là ĐÚNG NGHĨA.
  -> vẫn ngẫu nhiên hoá SGD bằng cách xáo trộn THỨ TỰ CÁC BLOCK giữa các epoch.

Dùng: DataLoader(ds, batch_sampler=BearingContiguousBatchSampler(ds, batch_size=256))
"""

import numpy as np
from torch.utils.data import Sampler


class BearingContiguousBatchSampler(Sampler):
    """
    Sinh danh sách batch (mỗi batch là list chỉ số mẫu) sao cho:
      - mọi mẫu trong 1 batch thuộc CÙNG một bearing;
      - các snapshot trong batch LIÊN TIẾP theo hour_id;
      - thứ tự các batch được xáo trộn mỗi epoch (SGD vẫn ngẫu nhiên).

    Tham số:
      dataset    : MTOIWindowDataset (cần .idx, .window_hour_ids, ._row_of, ._bidx)
      batch_size : số CỬA SỔ tối đa mỗi batch (block sẽ được cắt theo ranh giới snapshot)
      seed       : hạt giống cho việc xáo trộn thứ tự block
      drop_last_small : bỏ block có < min_block snapshot (mặc định giữ lại)
      min_snapshots_per_batch : tối thiểu 3 snapshot/batch để sai phân bậc 2 có nghĩa
    """

    def __init__(self, dataset, batch_size=256, seed=42, min_snapshots_per_batch=3,
                 drop_last_small=False):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_snap = int(min_snapshots_per_batch)
        self.drop_last_small = bool(drop_last_small)
        self.epoch = 0
        self._blocks = self._build_blocks(dataset)

    # ------------------------------------------------------------------ #
    def _build_blocks(self, ds):
        """Nhóm chỉ số mẫu theo (bearing, snapshot) rồi cắt thành block liên tiếp."""
        n = len(ds)
        pos = np.arange(n)                                   # vị trí trong dataset (0..len-1)
        hids = ds.window_hour_ids[ds.idx]                    # hour_id TOÀN CỤC của từng mẫu

        # bearing_idx của từng mẫu (tra qua bảng targets đã precompute trong dataset).
        if getattr(ds, "_bidx", None) is not None:
            rows = np.array([ds._row_of[int(h)] for h in hids], dtype=np.int64)
            bids = ds._bidx[rows]
        else:
            bids = np.zeros(n, dtype=np.int64)               # single-run -> coi như 1 bearing

        blocks = []
        for b in np.unique(bids):
            m = bids == b
            p_b, h_b = pos[m], hids[m]
            order = np.argsort(h_b, kind="stable")           # sắp theo thời gian TRONG bearing
            p_b, h_b = p_b[order], h_b[order]

            # Gom theo snapshot: các cửa sổ cùng hour_id phải nằm cùng batch (để gộp đúng).
            uniq, starts = np.unique(h_b, return_index=True)
            starts = np.sort(starts)
            groups = np.split(p_b, starts[1:])               # list mảng chỉ số, mỗi phần tử = 1 snapshot

            cur, cur_n, cur_snaps = [], 0, 0
            for g in groups:
                if cur_n + len(g) > self.batch_size and cur_snaps >= self.min_snap:
                    blocks.append(np.concatenate(cur)); cur, cur_n, cur_snaps = [], 0, 0
                cur.append(g); cur_n += len(g); cur_snaps += 1
            if cur:
                blk = np.concatenate(cur)
                if cur_snaps >= self.min_snap or not self.drop_last_small:
                    blocks.append(blk)
        return [b.tolist() for b in blocks if len(b) > 0]

    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch):
        """Gọi mỗi epoch để đổi thứ tự block (tương tự shuffle của DataLoader)."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for i in rng.permutation(len(self._blocks)):
            yield self._blocks[i]

    def __len__(self):
        return len(self._blocks)

    # ------------------------------------------------------------------ #
    def describe(self):
        """Thống kê để đưa vào bảng implementation của paper."""
        sizes = np.array([len(b) for b in self._blocks])
        return {"n_batches": int(len(sizes)),
                "windows_per_batch_mean": float(sizes.mean()) if len(sizes) else 0.0,
                "windows_per_batch_min": int(sizes.min()) if len(sizes) else 0,
                "windows_per_batch_max": int(sizes.max()) if len(sizes) else 0}
