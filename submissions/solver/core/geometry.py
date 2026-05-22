"""
Center-coordinate geometry helpers.

All functions operate on macro centers in microns.
Touching edges (separation == 0) are NOT considered overlaps.
"""

from typing import Tuple
import torch

# ── placement tolerance ────────────────────────────────────────────────────────

PLACEMENT_GAP: float = 1e-3
"""Minimum gap inserted between shelf-packed macros.

Coordinates are center-based; touching edges (separation == 0) are legal and
are not considered overlaps.  This small gap guards against float32 precision
errors that could cause adjacent-but-touching macros to appear overlapping to
downstream validators.  It is NOT the same as the fixed-macro tolerance used
by official validation.
"""


def rect_edges(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Return (left, right, bottom, top) for a center-defined rectangle."""
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def overlaps_pair(
    cx1: float, cy1: float, w1: float, h1: float,
    cx2: float, cy2: float, w2: float, h2: float,
) -> bool:
    """True iff two center-defined rectangles strictly overlap (touching edges are not an overlap)."""
    return (
        abs(cx1 - cx2) < (w1 + w2) / 2
        and abs(cy1 - cy2) < (h1 + h2) / 2
    )


def in_bounds(cx: float, cy: float, w: float, h: float, canvas_w: float, canvas_h: float) -> bool:
    """True iff a center-defined rectangle is fully within the canvas."""
    left, right, bottom, top = rect_edges(cx, cy, w, h)
    return left >= 0.0 and right <= canvas_w and bottom >= 0.0 and top <= canvas_h


# ── Vectorized tensor operations ──────────────────────────────────────────────


def centers_to_edges(
    positions: torch.Tensor, sizes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert center positions and sizes to axis-aligned box edges.

    Args:
        positions: [N, 2] (cx, cy)
        sizes:     [N, 2] (w, h)

    Returns:
        x_min, x_max, y_min, y_max — each [N]
    """
    half = sizes * 0.5
    x_min = positions[:, 0] - half[:, 0]
    x_max = positions[:, 0] + half[:, 0]
    y_min = positions[:, 1] - half[:, 1]
    y_max = positions[:, 1] + half[:, 1]
    return x_min, x_max, y_min, y_max


_BOUNDS_TOL: float = 1e-3
"""Tolerance for boundary checks (µm). Guards against float32 precision at exact boundaries."""


def bounds_mask(
    positions: torch.Tensor, sizes: torch.Tensor, canvas_w: float, canvas_h: float
) -> torch.Tensor:
    """Return bool tensor [N], True where the macro is fully within canvas bounds.

    A small tolerance (_BOUNDS_TOL) is applied to handle float32 rounding at
    exact canvas edges.
    """
    x_min, x_max, y_min, y_max = centers_to_edges(positions, sizes)
    return (
        (x_min >= -_BOUNDS_TOL)
        & (x_max <= canvas_w + _BOUNDS_TOL)
        & (y_min >= -_BOUNDS_TOL)
        & (y_max <= canvas_h + _BOUNDS_TOL)
    )
