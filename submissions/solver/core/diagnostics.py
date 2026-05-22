"""
Structured placement diagnostics for internal validation.

Provides a pure-geometry check that does not require the official evaluator,
useful inside the candidate pipeline to quickly reject bad placements.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch

from submissions.solver.core.geometry import bounds_mask, centers_to_edges


@dataclass
class PlacementDiagnostics:
    """Structured result of a placement geometry check."""

    valid: bool
    num_macros: int
    num_out_of_bounds: int
    num_overlaps: int
    num_nonfinite: int
    overlap_pairs: List[Tuple[int, int]] = field(default_factory=list)
    out_of_bounds_indices: List[int] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


def check_placement(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    mask: torch.Tensor = None,
    overlap_sample: int = 10,
) -> PlacementDiagnostics:
    """Pure geometry check (no official evaluator).

    Args:
        positions: [N, 2] center coordinates.
        sizes:     [N, 2] (width, height).
        canvas_w:  Canvas width.
        canvas_h:  Canvas height.
        mask:      Optional bool [N] mask of macros to check (default: all).
        overlap_sample: Maximum overlap pairs to record.
    """
    n = positions.shape[0]
    msgs: List[str] = []

    if mask is None:
        mask = torch.ones(n, dtype=torch.bool)

    indices = torch.where(mask)[0].tolist()

    # Nonfinite check
    nonfinite = ~torch.isfinite(positions[mask]).all(dim=1)
    num_nonfinite = int(nonfinite.sum().item())
    if num_nonfinite:
        msgs.append(f"{num_nonfinite} macros have NaN/Inf coordinates")

    # Bounds check
    in_b = bounds_mask(positions, sizes, canvas_w, canvas_h)
    oob_indices = [indices[i] for i, v in enumerate((~in_b[mask]).tolist()) if v]
    num_oob = len(oob_indices)
    if num_oob:
        msgs.append(f"{num_oob} macros out of bounds")

    # Fully vectorized overlap check — O(M^2) in numpy, fast for M<=2000
    # Use float64 to avoid false positives from float32 rounding on touching edges.
    pos_m = positions[mask].numpy().astype(np.float64)   # (M, 2)
    sz_m = sizes[mask].numpy().astype(np.float64)        # (M, 2)
    cx = pos_m[:, 0]
    cy = pos_m[:, 1]
    ws = sz_m[:, 0]
    hs = sz_m[:, 1]

    M = len(indices)
    overlap_pairs: List[Tuple[int, int]] = []

    _OV_TOL = 1e-4  # µm tolerance for touching-edge false positives

    # (M, M) pairwise overlap matrices — upper triangle only
    diff_x = np.abs(cx[:, None] - cx[None, :]) * 2   # (M, M)
    diff_y = np.abs(cy[:, None] - cy[None, :]) * 2
    sum_w = ws[:, None] + ws[None, :]
    sum_h = hs[:, None] + hs[None, :]
    # Strict overlap: both separations are negative (touching = 0 is NOT overlap)
    ov = (diff_x < sum_w - _OV_TOL) & (diff_y < sum_h - _OV_TOL)
    np.fill_diagonal(ov, False)  # exclude self-overlap

    ii, jj = np.where(np.triu(ov, k=1))
    for k in range(min(len(ii), overlap_sample)):
        overlap_pairs.append((indices[ii[k]], indices[jj[k]]))

    num_overlaps = len(overlap_pairs)
    if num_overlaps:
        msgs.append(f"{num_overlaps}+ macro overlaps detected")

    valid = num_nonfinite == 0 and num_oob == 0 and num_overlaps == 0

    return PlacementDiagnostics(
        valid=valid,
        num_macros=len(indices),
        num_out_of_bounds=num_oob,
        num_overlaps=num_overlaps,
        num_nonfinite=num_nonfinite,
        overlap_pairs=overlap_pairs,
        out_of_bounds_indices=oob_indices,
        messages=msgs,
    )
