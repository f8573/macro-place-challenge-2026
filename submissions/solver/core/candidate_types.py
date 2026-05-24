"""
Candidate placement data structures for M2B.

A CandidatePlacement is a named tensor of macro positions (centers).
A ScoredCandidate is a CandidatePlacement with legalization and scoring results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Controls deterministic M2B candidate generation."""

    include_transforms: bool = True
    candidate_budget: Optional[int] = None
    neighborhood_macro_limit: int = 20
    neighborhood_step_profile: str = "medium"   # small | medium | large
    disable_global_candidates: bool = False
    only_original_neighborhood: bool = False
    # Refinement pass (second-stage, seeded from winning neighborhood moves)
    refinement_around_winners: bool = False
    refinement_top_k: int = 5
    refinement_combo_size: int = 2  # 2 = combo2 only, 3 = also combo3
    # Seed selection strategy: "conservative" (approx-only) | "diverse" (multi-bucket)
    refinement_seed_strategy: str = "conservative"
    # Number of exploratory seeds in the diverse strategy (distinct macros outside top-approx set)
    refinement_exploration_seeds: int = 1
    # Line-search pass (third-stage, directional search from winning seeds)
    line_search_around_winners: bool = False
    line_search_top_k: int = 3
    line_search_max_scale: float = 4.0
    line_search_stop_after_worse: int = 2  # stop per-macro after N consecutive worse official scores


@dataclass(frozen=True)
class CandidateScoringConfig:
    """Controls legalization, prefiltering, and score deduplication."""

    legalizer_max_rings: int = 25
    enable_hash_cache: bool = True
    prefilter_mode: str = "approx_delta_hpwl"   # off | approx_delta_hpwl
    exploratory_score_count: int = 8
    max_official_scores: Optional[int] = None   # None = unlimited
    # Per-pass scoring budget caps (None = derive from max_official_scores).
    # Default split for max_official_scores=60: seed=32, refinement=10, ls=remaining~18.
    # Unused budget in an earlier bucket flows to downstream buckets.
    seed_discovery_score_budget: Optional[int] = None
    refinement_score_budget: Optional[int] = None
    line_search_score_budget: Optional[int] = None
    # Persistent disk cache for official scores (None = disabled)
    official_score_cache_path: Optional[str] = None
    disable_score_cache: bool = False
    clear_score_cache: bool = False


@dataclass
class CandidatePlacement:
    """A named candidate placement (center coordinates)."""

    name: str                   # Unique name, e.g. "spectral_xy"
    family: str                 # Family, e.g. "spectral", "area_degree", "terminal_anchor"
    positions: torch.Tensor     # [N, 2] center coordinates
    seed: int = 0
    notes: str = ""
    bypass_legalization: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringDiagnostics:
    """Scoring pipeline metadata returned by score_and_select."""

    scoring_available: bool
    scoring_mode: str       # "official" | "local_proxy" | "unavailable"
    score_is_degenerate: bool
    num_unique_scores: int
    selected_due_to: str    # "proxy_cost" | "fallback_original" | "validity_only" | "tie_break"

    raw_original_valid: bool = False
    raw_original_proxy_cost: Optional[float] = None
    delta_vs_raw_original: Optional[float] = None
    best_proxy_cost: Optional[float] = None
    winning_candidate: str = ""
    winning_family: str = ""
    invariant_holds: bool = False
    candidates_generated: int = 0
    candidates_prefiltered: int = 0
    candidates_officially_scored: int = 0
    duplicate_count: int = 0
    prefilter_mode: str = "off"
    # Refinement diagnostics
    refinement_candidates_generated: int = 0
    combo_candidates_generated: int = 0
    best_single_macro_move: str = ""
    best_single_macro_delta: Optional[float] = None
    best_combo_move: str = ""
    best_combo_delta: Optional[float] = None
    # Prefilter detail diagnostics
    prefilter_improving_count: int = 0
    prefilter_best_skipped_hpwl_delta: Optional[float] = None
    exploratory_count: int = 0
    # Line-search diagnostics
    line_search_candidates_generated: int = 0
    best_line_search_move: str = ""
    best_line_search_delta: Optional[float] = None
    # Score cache diagnostics
    cache_hits: int = 0
    cache_misses: int = 0
    # Official scorer timing
    official_scorer_time_ms_total: float = 0.0
    official_scorer_time_ms_avg: float = 0.0
    official_scorer_time_ms_p50: float = 0.0
    official_scorer_time_ms_p95: float = 0.0
    official_scorer_time_ms_max: float = 0.0
    slowest_candidate: str = ""
    candidates_skipped_by_budget: int = 0
    # Fresh-vs-cached breakdown (fresh = actually invoked official scorer; cache hits are free)
    fresh_official_scores: int = 0   # same as candidates_officially_scored, exposed for clarity
    # Candidate-admission audit (counts across all passes)
    admission_prelegal_overlap_candidates: int = 0  # had overlap before legalization
    admission_legalized_successfully: int = 0        # legalizer ran + valid result
    admission_legalization_failed: int = 0           # legalizer ran + invalid result
    # Refinement seed bucket diagnostics (populated when refinement_seed_strategy="diverse")
    refinement_seed_bucket_diagnostics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    """A candidate with legalization metadata and optional proxy cost."""

    name: str
    family: str
    positions: torch.Tensor    # [N, 2] (legalized or raw) center coordinates

    valid: bool
    proxy_cost: Optional[float]
    delta_vs_original: Optional[float]

    num_overlaps: int
    num_out_of_bounds: int
    num_unplaced: int
    num_moved: int
    max_move: float
    total_move: float
    legalization_ms: float
    scoring_ms: float
    total_ms: float
    no_op: bool = False
    notes: str = ""
    was_scored: bool = False
    duplicate_of: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
