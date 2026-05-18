"""
Lightweight visualization wrapper around macro_place.utils.visualize_placement.

Produces a 3-panel figure: placement / density heatmap / congestion heatmap.
Density and congestion panels require a live PlacementCost object (plc).
"""

from typing import Optional

import torch
from macro_place.benchmark import Benchmark
from macro_place.utils import visualize_placement as _official_visualize


def draw(
    placement: torch.Tensor,
    benchmark: Benchmark,
    save_path: Optional[str] = None,
    plc=None,
) -> None:
    """
    Visualize a placement.

    Args:
        placement:  [num_macros, 2] center positions
        benchmark:  Benchmark object
        save_path:  PNG/SVG output path, or None to display interactively
        plc:        Optional PlacementCost (enables density/congestion panels)
    """
    _official_visualize(placement, benchmark, save_path=save_path, plc=plc)
