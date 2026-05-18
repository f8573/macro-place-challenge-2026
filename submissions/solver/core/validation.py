"""
Validation wrapper around macro_place.utils.validate_placement.

Delegates to the official evaluator implementation to avoid evaluator drift.
"""

from typing import List, Tuple

import torch
from macro_place.benchmark import Benchmark
from macro_place.utils import validate_placement as _official_validate


def validate(
    placement: torch.Tensor,
    benchmark: Benchmark,
    check_overlaps: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate placement legality via the official evaluator.

    Checks shape, NaN/Inf, canvas bounds, fixed-macro preservation,
    and (optionally) hard-macro overlaps.

    Returns:
        (is_valid, violations)
    """
    return _official_validate(placement, benchmark, check_overlaps=check_overlaps)
