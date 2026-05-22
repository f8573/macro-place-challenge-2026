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
    bypass_legalization: bool = False  # If True, skip legalizer — validate/score raw positions.


@dataclass
class ScoringDiagnostics:
    """Scoring pipeline metadata returned by score_and_select."""

    scoring_available: bool
    scoring_mode: str       # "official" | "local_proxy" | "unavailable"
    score_is_degenerate: bool
    num_unique_scores: int
    selected_due_to: str    # "proxy_cost" | "fallback_original" | "validity_only" | "tie_break"

    # Raw original tracking
    raw_original_valid: bool = False
    raw_original_proxy_cost: Optional[float] = None
    # best_selected_cost - raw_original_proxy_cost; negative means improvement
    delta_vs_raw_original: Optional[float] = None


@dataclass
class ScoredCandidate:
    """A candidate with legalization metadata and optional proxy cost."""

    name: str
    family: str
    positions: torch.Tensor    # [N, 2] (legalized or raw) center coordinates

    valid: bool
    proxy_cost: Optional[float]    # None if scoring unavailable
    delta_vs_original: Optional[float]  # proxy_cost - raw_original proxy_cost

    num_overlaps: int
    num_out_of_bounds: int
    num_unplaced: int
    num_moved: int
    max_move: float
    total_move: float
    legalization_ms: float
    scoring_ms: float
    total_ms: float
    no_op: bool = False      # True when legalizer was skipped or found placement already valid
    notes: str = ""
    messages: List[str] = field(default_factory=list)
