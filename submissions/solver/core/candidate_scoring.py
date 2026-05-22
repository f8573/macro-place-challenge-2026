"""
M2B candidate scoring, validation, legalization, and selection.

Pipeline for each candidate:
  1. Legalize (greedy deterministic legalizer).
  2. Validate (check bounds, overlaps, finite coords).
  3. Score valid candidates using proxy cost (or HPWL fallback).
  4. Select the valid candidate with the lowest cost.
  5. Fall back to original if all candidates are invalid or unscored.

The 'original' candidate is always present and always included in scoring.
"""

import time
from typing import List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement, ScoredCandidate
from submissions.solver.core.diagnostics import check_placement
from submissions.solver.legalization.greedy_legalizer import legalize, LegalizationResult


# ---------------------------------------------------------------------------
# HPWL fallback (when no live plc)
# ---------------------------------------------------------------------------


def _compute_hpwl(positions: torch.Tensor, benchmark: Benchmark) -> float:
    """Half-perimeter wirelength over nets using macro positions only."""
    total = 0.0
    num_macros = benchmark.num_macros
    for nodes in benchmark.net_nodes:
        valid = nodes[nodes < num_macros]
        if valid.numel() < 2:
            continue
        xs = positions[valid, 0]
        ys = positions[valid, 1]
        total += float((xs.max() - xs.min() + ys.max() - ys.min()).item())
    return total


def _hpwl_score(positions: torch.Tensor, benchmark: Benchmark) -> float:
    """Normalized HPWL score (lower is better, comparable across candidates)."""
    hpwl = _compute_hpwl(positions, benchmark)
    # Normalize by canvas diagonal to get a dimensionless cost
    diag = (benchmark.canvas_width ** 2 + benchmark.canvas_height ** 2) ** 0.5
    return hpwl / max(diag, 1.0)


# ---------------------------------------------------------------------------
# Score a single legalized placement
# ---------------------------------------------------------------------------


def _score_placement(
    positions: torch.Tensor,
    benchmark: Benchmark,
    plc,
) -> Tuple[Optional[float], float]:
    """Return (proxy_cost_or_None, scoring_ms)."""
    t0 = time.perf_counter()
    cost = None
    if plc is not None:
        try:
            from submissions.solver.core.scoring import score

            result = score(positions, benchmark, plc)
            if result is not None:
                cost = float(result.get("proxy_cost", None) or 0.0)
        except Exception:
            cost = None
    if cost is None:
        # HPWL fallback
        cost = _hpwl_score(positions, benchmark)
    return cost, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Legalize + validate + score one candidate
# ---------------------------------------------------------------------------


def _process_candidate(
    candidate: CandidatePlacement,
    benchmark: Benchmark,
    plc,
    movable_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
    legalizer_max_rings: int = 25,
) -> ScoredCandidate:
    """Legalize, validate, and score a single candidate."""
    t_total_start = time.perf_counter()

    # --- Legalization ---
    leg: LegalizationResult = legalize(
        positions=candidate.positions,
        sizes=benchmark.macro_sizes,
        canvas_w=benchmark.canvas_width,
        canvas_h=benchmark.canvas_height,
        movable_mask=movable_mask,
        obstacle_mask=obstacle_mask,
        max_rings=legalizer_max_rings,
    )

    legal_positions = leg.positions

    # --- Validation ---
    diag = check_placement(
        positions=legal_positions,
        sizes=benchmark.macro_sizes,
        canvas_w=benchmark.canvas_width,
        canvas_h=benchmark.canvas_height,
        mask=movable_mask,
    )

    valid = leg.valid and diag.valid

    # --- Scoring ---
    proxy_cost: Optional[float] = None
    scoring_ms = 0.0
    if valid:
        proxy_cost, scoring_ms = _score_placement(legal_positions, benchmark, plc)

    total_ms = (time.perf_counter() - t_total_start) * 1000

    msgs = list(leg.messages) + list(diag.messages)

    return ScoredCandidate(
        name=candidate.name,
        family=candidate.family,
        positions=legal_positions,
        valid=valid,
        proxy_cost=proxy_cost,
        delta_vs_original=None,   # filled in after all candidates are scored
        num_overlaps=diag.num_overlaps,
        num_out_of_bounds=diag.num_out_of_bounds,
        num_unplaced=len(leg.messages),
        num_moved=leg.num_moved,
        max_move=leg.max_move,
        total_move=leg.total_move,
        legalization_ms=leg.runtime_ms,
        scoring_ms=scoring_ms,
        total_ms=total_ms,
        notes=candidate.notes,
        messages=msgs,
    )


# ---------------------------------------------------------------------------
# Full pipeline: score all candidates + select best
# ---------------------------------------------------------------------------


def score_and_select(
    candidates: List[CandidatePlacement],
    benchmark: Benchmark,
    plc=None,
    legalizer_max_rings: int = 25,
) -> Tuple[ScoredCandidate, List[ScoredCandidate]]:
    """Score all candidates and return (best, ranked_list).

    Selection rule: valid candidate with minimum proxy_cost.
    If no scored valid candidate exists, fall back to 'original'.

    Returns:
        (best_candidate, ranked_list) where ranked_list is sorted
        by proxy_cost ascending (invalid candidates at the end).
    """
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    # Only truly fixed hard macros are obstacles; soft macros are ignored.
    obstacle_mask = benchmark.macro_fixed & benchmark.get_hard_macro_mask()

    scored: List[ScoredCandidate] = []
    for c in candidates:
        sc = _process_candidate(
            c, benchmark, plc, movable_mask, obstacle_mask,
            legalizer_max_rings=legalizer_max_rings,
        )
        scored.append(sc)

    # Compute delta_vs_original
    orig_sc = next((s for s in scored if s.name == "original"), None)
    orig_cost = orig_sc.proxy_cost if orig_sc is not None else None
    for sc in scored:
        if sc.proxy_cost is not None and orig_cost is not None:
            sc.delta_vs_original = sc.proxy_cost - orig_cost

    # Rank: valid first (by cost ascending), then invalid
    valid_scored = [s for s in scored if s.valid and s.proxy_cost is not None]
    invalid_scored = [s for s in scored if not s.valid or s.proxy_cost is None]

    valid_scored.sort(key=lambda s: s.proxy_cost)
    invalid_scored.sort(key=lambda s: s.name)

    ranked = valid_scored + invalid_scored

    # Select best
    if valid_scored:
        best = valid_scored[0]
    elif orig_sc is not None:
        best = orig_sc
    else:
        best = scored[0] if scored else None

    return best, ranked
