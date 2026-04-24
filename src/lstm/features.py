from __future__ import annotations

import numpy as np


def bin_scan(scan: np.ndarray, n_bins: int, method: str) -> np.ndarray:
    n_beams = int(scan.shape[0])
    if n_bins >= n_beams:
        return scan.astype(np.float32, copy=True)

    edges = np.linspace(0, n_beams, n_bins + 1, dtype=np.int32)
    binned = np.empty((n_bins,), dtype=np.float32)
    for i in range(n_bins):
        left = int(edges[i])
        right = max(int(edges[i + 1]), left + 1)
        segment = scan[left:right]
        if method == "mean":
            binned[i] = float(np.mean(segment))
        else:
            binned[i] = float(np.min(segment))
    return binned


def process_scan(scan: np.ndarray, clip_min: float, clip_max: float, n_bins: int, method: str) -> np.ndarray:
    clipped = np.clip(np.asarray(scan, dtype=np.float32), clip_min, clip_max).astype(np.float32, copy=False)
    return bin_scan(clipped, n_bins, method)


class PathFeatureLookup:
    def __init__(self, track, lookahead_m: list[float]):
        self.track = track
        self.lookahead_m = np.asarray(lookahead_m, dtype=np.float64)
        self.centerline = track.centerline
        self.centerline_ss = self.centerline.ss.astype(np.float64)
        self.centerline_ks = self.centerline.ks.astype(np.float64)
        self.track_length = float(self.centerline.spline.s[-1])
        self.centerline_size = int(self.centerline_ss.shape[0])

    def get_curvature_lookahead(self, pose_x: float, pose_y: float) -> np.ndarray:
        s_anchor, _ = self.centerline.spline.calc_arclength_inaccurate(pose_x, pose_y)
        s_anchor_wrapped = float(s_anchor) % self.track_length
        s_queries = np.mod(s_anchor_wrapped + self.lookahead_m, self.track_length)
        idxs = np.searchsorted(self.centerline_ss, s_queries, side="left")
        idxs = np.where(idxs >= self.centerline_size, 0, idxs).astype(np.int32)
        return self.centerline_ks[idxs].astype(np.float32, copy=False)
