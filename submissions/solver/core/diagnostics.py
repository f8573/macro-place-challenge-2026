"""
Structured placement diagnostics for internal validation.

Provides a pure-geometry check that does not require the official evaluator,
useful inside the candidate pipeline to quickly reject bad placements.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

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

    # Overlap check (O(n^2) but fast for n<=500)
    x_min, x_max, y_min, y_max = centers_to_edges(positions, sizes)
    overlap_pairs: List[Tuple[int, int]] = []
    checked_indices = indices

    for k in range(len(checked_indices)):
        if num_nonfinite and not torch.isfinite(positions[checked_indices[k]]).all():
            continue
        i = checked_indices[k]
        for l in range(k + 1, len(checked_indices)):
            j = checked_indices[l]
            if num_nonfinite and not torch.isfinite(positions[j]).all():
                continue
            # Strict overlap: separation < 0
            ox = x_max[i].item() - x_min[j].item()
            ox2 = x_max[j].item() - x_min[i].item()
            oy = y_max[i].item() - y_min[j].item()
            oy2 = y_max[j].item() - y_min[i].item()
            # Two rects overlap iff both projections have strictly positive overlap
            # Touching (ox == 0) is NOT an overlap
            if ox > 0 and ox2 > 0 and oy > 0 and oy2 > 0:
                overlap_pairs.append((i, j))
                if len(overlap_pairs) >= overlap_sample:
                    break
        if len(overlap_pairs) >= overlap_sample:
            break

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
