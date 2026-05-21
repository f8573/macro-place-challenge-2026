"""
Candidate placement data structures for M2B.

A CandidatePlacement is a named tensor of macro positions (centers).
A ScoredCandidate is a CandidatePlacement with legalization and scoring results.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class CandidatePlacement:
    """A named candidate placement (center coordinates)."""

    name: str                   # Unique name, e.g. "spectral_xy"
    family: str                 # Family, e.g. "spectral", "area_degree", "terminal_anchor"
    positions: torch.Tensor     # [N, 2] center coordinates
    seed: int = 0
    notes: str = ""


@dataclass
class ScoredCandidate:
    """A candidate with legalization metadata and optional proxy cost."""

    name: str
    family: str
    positions: torch.Tensor    # [N, 2] legalized center coordinates

    valid: bool
    proxy_cost: Optional[float]    # None if scoring unavailable
    delta_vs_original: Optional[float]  # proxy_cost - original_proxy_cost

    num_overlaps: int
    num_out_of_bounds: int
    num_unplaced: int
    num_moved: int
    max_move: float
    total_move: float
    legalization_ms: float
    scoring_ms: float
    total_ms: float
    notes: str = ""
    messages: List[str] = field(default_factory=list)
