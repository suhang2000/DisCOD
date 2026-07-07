"""Bounding-box geometry utilities."""
from __future__ import annotations

COORD_MAX = 1000


def is_sane_bbox(bbox, coord_max=COORD_MAX, max_area_frac=0.95):
    x1, y1, x2, y2 = bbox
    if not all(0 <= c <= coord_max for c in bbox):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if ((x2 - x1) * (y2 - y1)) / float(coord_max * coord_max) > max_area_frac:
        return False
    return True


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0
