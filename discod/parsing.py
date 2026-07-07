"""Lenient bounding-box parsing from VLM text output."""
from __future__ import annotations

import re

_TAG_RE = re.compile(r"<(bbox|box)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_TUPLE_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_BARE4_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)


def _classify_and_scale(nums, img_wh):
    absvals = [abs(n) for n in nums]
    if all(a <= 1.0 for a in absvals) and any(a > 0.0 for a in absvals):
        return tuple(int(round(n * 1000)) for n in nums)
    if any(a > 1000 for a in absvals):
        if img_wh is not None:
            w, h = img_wh
            x1, y1, x2, y2 = nums
            return (
                int(round(x1 * 1000 / w)), int(round(y1 * 1000 / h)),
                int(round(x2 * 1000 / w)), int(round(y2 * 1000 / h)),
            )
        return tuple(int(round(n)) for n in nums)
    return tuple(int(round(n)) for n in nums)


def extract_bbox(text: str, img_wh: tuple[int, int] | None = None) -> tuple[int, int, int, int] | None:
    """Extract the first bounding box from `text`, normalized to [0, 1000].

    Prefers coordinates inside a <bbox>/<box> tag, falling back to any bracketed
    4-tuple. Coordinates given in [0, 1] or in absolute pixels (needs `img_wh`)
    are rescaled to the canonical [0, 1000] range. Returns None if nothing parses.
    """
    if not text:
        return None
    for m in _TAG_RE.finditer(text):
        tm = _TUPLE_RE.search(m.group(2)) or _BARE4_RE.search(m.group(2))
        if tm:
            return _classify_and_scale([float(g) for g in tm.groups()], img_wh)
    tm = _TUPLE_RE.search(text)
    if tm:
        return _classify_and_scale([float(g) for g in tm.groups()], img_wh)
    return None
