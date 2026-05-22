"""
Deterministic greedy rectangle legalizer.

Algorithm:
  1. Sort movable hard macros by descending area.
  2. Pre-seed placed list with fixed macros.
  3. For each macro, try desired position (clamped to canvas).
     If clear, place immediately.  Otherwise, vectorized-batch-check
     the K nearest candidate offsets against all placed macros and
     take the first legal position found.
  4. If ring search is exhausted, fall back to a full canvas-grid scan
     (using the same step as the ring search) that is guaranteed to find
     a legal slot whenever one exists.
  5. If the canvas-grid scan also fails, mark result invalid.

All coordinates are macro CENTERS in microns.
Touching edges (separation == 0) are legal and NOT considered overlaps.

Performance: the overlap check is fully vectorized (numpy).  For 250 macros
and K=400 candidates, each call to the inner batch-check is a (400, 250)
boolean operation executed in pure numpy — far faster than a Python for loop.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

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


def _sorted_offsets_np(max_rings: int, step: float) -> np.ndarray:
    """Return (K, 2) float64 array of (dx, dy) offsets sorted by L2 distance.

    Excludes (0, 0) since the desired position is always tried first.
    """
    r = np.arange(-max_rings, max_rings + 1, dtype=np.float64) * step
    gx, gy = np.meshgrid(r, r)
    mask = ~((gx == 0) & (gy == 0))
    dx = gx[mask]
    dy = gy[mask]
    dist2 = dx ** 2 + dy ** 2
    order = np.argsort(dist2, kind="stable")
    return np.stack([dx[order], dy[order]], axis=1)  # (K, 2)


def _batch_check_legal(
    bcx: np.ndarray,   # (B,) candidate centers x
    bcy: np.ndarray,   # (B,) candidate centers y
    pcx: np.ndarray,   # (P,) placed centers x
    pcy: np.ndarray,   # (P,) placed centers y
    pw: np.ndarray,    # (P,) placed widths
    ph: np.ndarray,    # (P,) placed heights
    w_i: float,
    h_i: float,
) -> np.ndarray:
    """Return boolean mask (B,): True where position is legal (no overlap)."""
    diff_x = np.abs(bcx[:, None] - pcx[None, :]) * 2   # (B, P)
    diff_y = np.abs(bcy[:, None] - pcy[None, :]) * 2
    overlaps = (diff_x < w_i + pw[None, :]) & (diff_y < h_i + ph[None, :])
    return ~overlaps.any(axis=1)


def _first_legal(
    all_cx: np.ndarray,
    all_cy: np.ndarray,
    pcx: np.ndarray,
    pcy: np.ndarray,
    pw: np.ndarray,
    ph: np.ndarray,
    w_i: float,
    h_i: float,
    chunk: int = 512,
) -> Optional[tuple]:
    """Return (cx, cy) of the first legal position in all_cx/all_cy, or None."""
    K = len(all_cx)
    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        legal = _batch_check_legal(
            all_cx[start:end], all_cy[start:end], pcx, pcy, pw, ph, w_i, h_i
        )
        idx = np.where(legal)[0]
        if idx.size > 0:
            k = idx[0] + start
            return float(all_cx[k]), float(all_cy[k])
    return None


def _canvas_grid_scan(
    w_i: float,
    h_i: float,
    cx_min: float,
    cx_max: float,
    cy_min: float,
    cy_max: float,
    placed_cx: np.ndarray,
    placed_cy: np.ndarray,
    placed_w: np.ndarray,
    placed_h: np.ndarray,
    n_placed: int,
    grid_step: float,
) -> Optional[tuple]:
    """Full canvas-grid scan — guaranteed to find a legal slot if one exists.

    Scans a uniform grid over [cx_min, cx_max] × [cy_min, cy_max] with the
    given step and returns the first legal center, or None if the canvas is
    genuinely full.
    """
    xs = np.arange(cx_min, cx_max + grid_step * 0.5, grid_step)
    ys = np.arange(cy_min, cy_max + grid_step * 0.5, grid_step)
    if xs.size == 0:
        xs = np.array([cx_min])
    if ys.size == 0:
        ys = np.array([cy_min])

    gx, gy = np.meshgrid(xs, ys)
    all_cx = gx.ravel()
    all_cy = gy.ravel()

    if n_placed == 0:
        return float(all_cx[0]), float(all_cy[0])

    pcx = placed_cx[:n_placed]
    pcy = placed_cy[:n_placed]
    pw = placed_w[:n_placed]
    ph = placed_h[:n_placed]

    return _first_legal(all_cx, all_cy, pcx, pcy, pw, ph, w_i, h_i)


def _find_legal_batch(
    desired_cx: float,
    desired_cy: float,
    w_i: float,
    h_i: float,
    cx_min: float,
    cx_max: float,
    cy_min: float,
    cy_max: float,
    sorted_offsets: np.ndarray,
    placed_cx: np.ndarray,
    placed_cy: np.ndarray,
    placed_w: np.ndarray,
    placed_h: np.ndarray,
    n_placed: int,
    ring_step: float,
) -> Optional[tuple]:
    """Vectorized batch search for a legal center position.

    1. Tries the desired (clamped) position first.
    2. Tries all ring-search offsets (sorted by ascending L2 distance).
    3. Falls back to a full canvas-grid scan with the same step.

    Returns (cx, cy) or None.
    """
    # Try desired (clamped) first
    cx0 = max(cx_min, min(cx_max, desired_cx))
    cy0 = max(cy_min, min(cy_max, desired_cy))

    if n_placed == 0:
        return cx0, cy0

    pcx = placed_cx[:n_placed]
    pcy = placed_cy[:n_placed]
    pw = placed_w[:n_placed]
    ph = placed_h[:n_placed]

    def _is_legal_single(cx: float, cy: float) -> bool:
        return not np.any(
            (np.abs(cx - pcx) * 2 < w_i + pw)
            & (np.abs(cy - pcy) * 2 < h_i + ph)
        )

    if _is_legal_single(cx0, cy0):
        return cx0, cy0

    # Ring search
    all_cx = np.clip(desired_cx + sorted_offsets[:, 0], cx_min, cx_max)
    all_cy = np.clip(desired_cy + sorted_offsets[:, 1], cy_min, cy_max)
    result = _first_legal(all_cx, all_cy, pcx, pcy, pw, ph, w_i, h_i)
    if result is not None:
        return result

    # Ring search exhausted — full canvas-grid scan.
    # Step must be ≤ half the macro dimension so no legal slot is missed.
    fallback_step = min(ring_step, w_i / 2.0, h_i / 2.0)
    return _canvas_grid_scan(
        w_i, h_i, cx_min, cx_max, cy_min, cy_max,
        placed_cx, placed_cy, placed_w, placed_h, n_placed,
        grid_step=max(0.05, fallback_step),
    )


def legalize(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    movable_mask: torch.Tensor = None,
    obstacle_mask: torch.Tensor = None,
    degrees: Optional[np.ndarray] = None,
    sort_by: str = "area",
    max_rings: int = 30,
) -> LegalizationResult:
    """Greedy deterministic rectangle legalizer.

    Args:
        positions:     [N, 2] center coordinates (desired positions).
        sizes:         [N, 2] (width, height).
        canvas_w:      Canvas width.
        canvas_h:      Canvas height.
        movable_mask:  Bool [N] — True for hard macros to legalize.
        obstacle_mask: Bool [N] — True for macros that are fixed obstacles
                       (e.g. macro_fixed hard macros).  Soft macros should
                       NOT be in obstacle_mask; they are ignored.  If None,
                       no obstacles are pre-seeded.
        degrees:       Optional [N] per-macro connectivity degrees.
        sort_by:       "area" or "area_degree".
        max_rings:     Search radius in rings (step = min_dim/2, ≥ 0.5µm).
                       After ring search, a canvas-grid fallback with the same
                       step ensures validity on non-full canvases.

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
    # Use explicit obstacle_mask instead of ~movable_mask so soft macros
    # (which are neither movable hard macros nor true fixed obstacles) are
    # not incorrectly seeded as obstacles.
    if obstacle_mask is not None:
        fixed_indices = torch.where(obstacle_mask)[0].numpy().astype(int)
    else:
        fixed_indices = np.array([], dtype=int)

    if len(movable_indices) == 0:
        return LegalizationResult(
            positions=out, valid=True, num_moved=0, max_move=0.0,
            total_move=0.0, runtime_ms=(time.perf_counter() - t0) * 1000,
        )

    # Step size: half the smallest movable dimension, at least 0.5 µm
    min_dim = float(min(ws[movable_indices].min(), hs[movable_indices].min()))
    step = max(0.5, min_dim / 2.0)

    # Pre-compute sorted offsets (once per call)
    sorted_offsets = _sorted_offsets_np(max_rings, step)

    # Sort priority
    areas = ws * hs
    if sort_by == "area_degree" and degrees is not None:
        d = np.asarray(degrees, dtype=float)
        p = areas * np.log1p(d[:n] if len(d) >= n else d)
    else:
        p = areas.copy()

    sorted_movable = sorted(movable_indices.tolist(), key=lambda i: -p[i])

    # Pre-allocate placed arrays
    cap = n + 1
    placed_cx = np.empty(cap, dtype=np.float64)
    placed_cy = np.empty(cap, dtype=np.float64)
    placed_w = np.empty(cap, dtype=np.float64)
    placed_h = np.empty(cap, dtype=np.float64)
    n_placed = 0

    # Seed with fixed macros
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

    for i in sorted_movable:
        w_i = float(ws[i])
        h_i = float(hs[i])
        desired_cx = float(out[i, 0].item())
        desired_cy = float(out[i, 1].item())

        cx_min = w_i / 2.0
        cx_max = canvas_w - w_i / 2.0
        cy_min = h_i / 2.0
        cy_max = canvas_h - h_i / 2.0

        if cx_min > cx_max or cy_min > cy_max:
            # Macro physically larger than the canvas
            invalid_count += 1
            msgs.append(f"Macro {i} ({w_i:.2f}x{h_i:.2f}) exceeds canvas")
            final_cx = max(0.0, min(canvas_w, desired_cx))
            final_cy = max(0.0, min(canvas_h, desired_cy))
        else:
            result = _find_legal_batch(
                desired_cx, desired_cy, w_i, h_i,
                cx_min, cx_max, cy_min, cy_max,
                sorted_offsets,
                placed_cx, placed_cy, placed_w, placed_h, n_placed,
                ring_step=step,
            )
            if result is None:
                # Canvas truly full at the given step resolution
                invalid_count += 1
                msgs.append(f"Macro {i} could not be legalized (canvas full)")
                final_cx = max(cx_min, min(cx_max, desired_cx))
                final_cy = max(cy_min, min(cy_max, desired_cy))
            else:
                final_cx, final_cy = result

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
