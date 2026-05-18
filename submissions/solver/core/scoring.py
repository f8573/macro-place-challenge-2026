"""
Scoring wrapper around macro_place.objective.

Delegates to the official evaluator.  Returns None when plc is unavailable
(e.g., benchmarks loaded from .pt files).

macro_place.objective applies a congestion monkey-patch to PlacementCost at
import time.  Import this module (or call score()) before any direct call to
plc.get_congestion_cost().
"""

from typing import Dict, Optional

import torch
from macro_place.benchmark import Benchmark


def score(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
) -> Optional[Dict]:
    """
    Compute proxy cost if a live PlacementCost object is available.

    Args:
        placement: [num_macros, 2] center positions in microns
        benchmark: Benchmark object
        plc:       PlacementCost object, or None

    Returns:
        Cost dict from compute_proxy_cost, or None if plc is None.
    """
    if plc is None:
        return None
    # Import here so the congestion monkey-patch fires on first use
    from macro_place.objective import compute_proxy_cost
    return compute_proxy_cost(placement, benchmark, plc)
