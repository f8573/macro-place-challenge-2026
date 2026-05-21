"""
Deterministic greedy rectangle legalizer.

Algorithm:
  1. Sort movable hard macros by descending area (or area-degree priority).
  2. Pre-seed placed list with fixed macros.
  3. For each macro, try desired position first, then search expanding
     sorted offsets (by L2 distance) until a legal position is found.
  4. If no legal position found after max_attempts, mark result invalid.

All coordinates are macro CENTERS in microns.
Touching edges (separation == 0) are legal and NOT considered overlaps.

Performance notes:
  - Placed arrays are preallocated and maintained as growing numpy arrays.
  - Sorted offsets are computed once per legalize() call.
  - The overlap check is vectorized over all placed macros at once.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class LegalizationResult:
    """Output of a greedy legalization run."""

    positions: torch.Tensor   # [N, 2] center coordinates
    valid: bool
    num_moved: int
    max_move: float
    total_move: float
    runtime_ms: float
    messages: List[str] = field(default_factory=list)


def _make_sorted_offsets(max_rings: int, step: float) -> np.ndarray:
    """Return (K, 2) float64 array of (dx, dy) offsets sorted by L2 distance."""
    r = np.arange(-max_rings, max_rings + 1)
    gx, gy = np.meshgrid(r, r)
    # Exclude (0, 0)
    mask = ~((gx == 0) & (gy == 0))
    dx = (gx * step)[mask]
    dy = (gy * step)[mask]
    dist2 = dx ** 2 + dy ** 2
    order = np.argsort(dist2, kind="stable")
    return np.stack([dx[order], dy[order]], axis=1)


def legalize(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    movable_mask: torch.Tensor = None,
    degrees: Optional[np.ndarray] = None,
    sort_by: str = "area",
    max_rings: int = 60,
) -> LegalizationResult:
    """Greedy deterministic rectangle legalizer.

    Args:
        positions:    [N, 2] center coordinates (desired positions).
        sizes:        [N, 2] (width, height).
        canvas_w:     Canvas width.
        canvas_h:     Canvas height.
        movable_mask: Bool [N] — True for macros to legalize.
        degrees:      Optional [N] connectivity degrees.
        sort_by:      "area" or "area_degree".
        max_rings:    Search radius in rings.

    Returns:
        LegalizationResult with updated center positions.
    """
    t0 = time.perf_counter()
    n = positions.shape[0]

    if movable_mask is None:
        movable_mask = torch.ones(n, dtype=torch.bool)

    out = positions.clone().float()
    ws = sizes[:, 0].numpy().astype(np.float64)
    hs = sizes[:, 1].numpy().astype(np.float64)

    movable_indices = torch.where(movable_mask)[0].numpy().astype(int)
    fixed_indices = torch.where(~movable_mask)[0].numpy().astype(int)

    if len(movable_indices) == 0:
        return LegalizationResult(
            positions=out, valid=True, num_moved=0, max_move=0.0,
            total_move=0.0, runtime_ms=(time.perf_counter() - t0) * 1000,
        )

    # Compute step from smallest movable macro dimension
    min_dim = min(float(ws[movable_indices].min()), float(hs[movable_indices].min()))
    step = max(0.5, min_dim / 2.0)

    # Pre-compute sorted offsets once
    sorted_offsets = _make_sorted_offsets(max_rings, step)  # (K, 2)

    # Sort movable macros by descending priority
    areas = ws * hs
    if sort_by == "area_degree" and degrees is not None:
        deg_arr = np.asarray(degrees, dtype=float)
        if len(deg_arr) >= n:
            priority = areas * np.log1p(deg_arr[:n])
        else:
            priority = areas.copy()
    else:
        priority = areas.copy()

    sorted_movable = sorted_indices = sorted(
        movable_indices.tolist(), key=lambda i: -priority[i]
    )

    # Initialize placed arrays (preallocated, grow as macros are placed)
    cap = n + 1
    placed_cx = np.empty(cap, dtype=np.float64)
    placed_cy = np.empty(cap, dtype=np.float64)
    placed_w = np.empty(cap, dtype=np.float64)
    placed_h = np.empty(cap, dtype=np.float64)
    n_placed = 0

    # Pre-seed with fixed macros
    for idx in fixed_indices:
        placed_cx[n_placed] = float(out[idx, 0].item())
        placed_cy[n_placed] = float(out[idx, 1].item())
        placed_w[n_placed] = ws[idx]
        placed_h[n_placed] = hs[idx]
        n_placed += 1

    msgs: List[str] = []
    num_moved = 0
    max_move = 0.0
    total_move = 0.0
    invalid_count = 0

    # Get views of current placed arrays
    def _check(cx, cy, w_i, h_i):
        if n_placed == 0:
            return True
        pcx = placed_cx[:n_placed]
        pcy = placed_cy[:n_placed]
        pw = placed_w[:n_placed]
        ph = placed_h[:n_placed]
        # Strict overlap: both axes overlap (> 0), not touching (== 0)
        overlap_x = np.abs(cx - pcx) * 2 < w_i + pw
        overlap_y = np.abs(cy - pcy) * 2 < h_i + ph
        return not np.any(overlap_x & overlap_y)

    for i in sorted_movable:
        w_i = ws[i]
        h_i = hs[i]
        desired_cx = float(out[i, 0].item())
        desired_cy = float(out[i, 1].item())

        # Bounds for macro center
        cx_min = w_i / 2.0
        cx_max = canvas_w - w_i / 2.0
        cy_min = h_i / 2.0
        cy_max = canvas_h - h_i / 2.0

        if cx_min > cx_max or cy_min > cy_max:
            invalid_count += 1
            msgs.append(f"Macro {i} ({w_i:.1f}x{h_i:.1f}) does not fit in canvas")
            # Still place at clamped position (best effort)
            cx = max(0.0, min(canvas_w, desired_cx))
            cy = max(0.0, min(canvas_h, desired_cy))
            out[i, 0] = cx
            out[i, 1] = cy
            placed_cx[n_placed] = cx
            placed_cy[n_placed] = cy
            placed_w[n_placed] = w_i
            placed_h[n_placed] = h_i
            n_placed += 1
            continue

        # Clamp desired position
        cx = max(cx_min, min(cx_max, desired_cx))
        cy = max(cy_min, min(cy_max, desired_cy))

        if _check(cx, cy, w_i, h_i):
            final_cx, final_cy = cx, cy
        else:
            # Search expanding offsets
            found = False
            for k in range(len(sorted_offsets)):
                tx = max(cx_min, min(cx_max, desired_cx + sorted_offsets[k, 0]))
                ty = max(cy_min, min(cy_max, desired_cy + sorted_offsets[k, 1]))
                if _check(tx, ty, w_i, h_i):
                    final_cx, final_cy = tx, ty
                    found = True
                    break
            if not found:
                invalid_count += 1
                msgs.append(f"Macro {i} could not be legalized")
                final_cx, final_cy = cx, cy  # keep clamped desired (invalid)
            else:
                final_cx, final_cy = final_cx, final_cy  # already set

        out[i, 0] = final_cx
        out[i, 1] = final_cy
        placed_cx[n_placed] = final_cx
        placed_cy[n_placed] = final_cy
        placed_w[n_placed] = w_i
        placed_h[n_placed] = h_i
        n_placed += 1

        move = ((final_cx - desired_cx) ** 2 + (final_cy - desired_cy) ** 2) ** 0.5
        if move > 1e-9:
            num_moved += 1
            if move > max_move:
                max_move = move
            total_move += move

    runtime_ms = (time.perf_counter() - t0) * 1000
    return LegalizationResult(
        positions=out,
        valid=invalid_count == 0,
        num_moved=num_moved,
        max_move=max_move,
        total_move=total_move,
        runtime_ms=runtime_ms,
        messages=msgs,
    )
