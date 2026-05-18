"""
SolverPlacer — Milestone 1 baseline.

Shelf-pack that guarantees zero overlaps and in-bounds placements.
Fixed macros stay; movable hard macros are packed left-to-right in rows
sorted by descending height.  Soft macros remain at initial positions.
"""

import torch
from macro_place.benchmark import Benchmark
#from submissions.solver.core.geometry import PLACEMENT_GAP
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from core.geometry import PLACEMENT_GAP


class SolverPlacer:
    """Thin baseline placer for Milestone 1."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()

        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width

        # Tallest-first shelf-pack heuristic
        indices.sort(key=lambda i: -sizes[i, 1].item())

        gap = PLACEMENT_GAP
        cursor_x = cursor_y = row_h = 0.0

        for idx in indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0

            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2
            cursor_x += w + gap
            row_h = max(row_h, h)

        return placement
