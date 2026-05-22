"""
M2B candidate scoring, validation, legalization, and selection.

Pipeline for each candidate:
  1. Legalize (greedy deterministic legalizer) — skipped for bypass_legalization candidates.
  2. Validate (check bounds, overlaps, finite coords).
  3. Score valid candidates using proxy cost (or HPWL fallback).
  4. Select the valid candidate with the lowest cost.
  5. Fall back to original_legalized if original_raw is invalid.

The 'original_raw' candidate always bypasses the legalizer; its raw positions
are validated and scored directly.  The 'original_legalized' candidate passes
through the legalizer but with the no-op shortcut, it is identical to
original_raw whenever the input placement is already valid.

Invariant guaranteed by this module:
  If original_raw is valid, best_selected.proxy_cost <= original_raw.proxy_cost
  (because original_raw is itself a valid selectable candidate).

Scoring modes:
  official      — plc object available; uses compute_proxy_cost
  local_proxy   — plc unavailable; HPWL fallback with real net_nodes
  unavailable   — plc unavailable and net_nodes empty; all scores are 0

When scoring is unavailable or degenerate, 'original_raw' is selected if valid.
"""

import hashlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import (
    CandidatePlacement,
    ScoredCandidate,
    ScoringDiagnostics,
)
from submissions.solver.core.diagnostics import check_placement
from submissions.solver.legalization.greedy_legalizer import legalize, LegalizationResult


# ---------------------------------------------------------------------------
# Scoring mode detection
# ---------------------------------------------------------------------------


def _detect_scoring_mode(plc, benchmark: Benchmark) -> str:
    if plc is not None:
        return "official"
    if not benchmark.net_nodes:
        return "unavailable"
    return "local_proxy"


# ---------------------------------------------------------------------------
# Placement hash (for diversity audits)
# ---------------------------------------------------------------------------


def placement_hash(positions: torch.Tensor) -> str:
    """Return an 8-hex-char MD5 hash of positions rounded to 0.1 µm."""
    arr = np.round(positions.detach().cpu().numpy().astype(np.float32), 1)
    return hashlib.md5(arr.tobytes()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Connectivity audit
# ---------------------------------------------------------------------------


def connectivity_audit(benchmark: Benchmark) -> Dict:
    """Return connectivity statistics for a benchmark.

    Fields:
      num_macros                  — total macro count (hard + soft)
      num_nets                    — total net count
      num_net_edges               — undirected clique edges from nets with ≥2 hard pins
      num_macros_with_degree_gt_0 — hard macros in at least one net
      num_fixed_endpoints         — fixed hard macros + I/O ports
      spectral_available          — True if num_net_edges > 0
      terminal_anchor_available   — True if num_fixed_endpoints > 0
    """
    n_hard = benchmark.num_hard_macros
    num_net_edges = 0
    degrees = np.zeros(n_hard, dtype=np.int32)

    for nodes in benchmark.net_nodes:
        hard_pins = nodes[nodes < n_hard]
        unique_pins = torch.unique(hard_pins)
        k = unique_pins.numel()
        if k >= 2:
            num_net_edges += k * (k - 1) // 2
            for p in unique_pins.tolist():
                degrees[int(p)] += 1

    fixed_mask = benchmark.macro_fixed[:n_hard]
    num_fixed = int(fixed_mask.sum().item())
    num_ports = benchmark.port_positions.shape[0]
    num_fixed_endpoints = num_fixed + num_ports

    return {
        "num_macros": benchmark.num_macros,
        "num_nets": benchmark.num_nets,
        "num_net_edges": num_net_edges,
        "num_macros_with_degree_gt_0": int((degrees > 0).sum()),
        "num_fixed_endpoints": num_fixed_endpoints,
        "spectral_available": num_net_edges > 0,
        "terminal_anchor_available": num_fixed_endpoints > 0,
    }


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
    """Legalize (unless bypass_legalization), validate, and score a single candidate."""
    t_total_start = time.perf_counter()

    if candidate.bypass_legalization:
        # Skip legalization — validate and score raw input positions directly.
        # num_moved=0, no_op=True since no legalizer was run.
        legal_positions = candidate.positions.clone().float()
        leg = LegalizationResult(
            positions=legal_positions,
            valid=True,    # will be overridden by check_placement below
            num_moved=0,
            max_move=0.0,
            total_move=0.0,
            runtime_ms=0.0,
            no_op=True,
        )
    else:
        leg = legalize(
            positions=candidate.positions,
            sizes=benchmark.macro_sizes,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
            movable_mask=movable_mask,
            obstacle_mask=obstacle_mask,
            max_rings=legalizer_max_rings,
        )

    legal_positions = leg.positions

    diag = check_placement(
        positions=legal_positions,
        sizes=benchmark.macro_sizes,
        canvas_w=benchmark.canvas_width,
        canvas_h=benchmark.canvas_height,
        mask=movable_mask,
    )

    valid = diag.valid  # for bypass: leg.valid is a placeholder; use diag only
    if not candidate.bypass_legalization:
        valid = leg.valid and diag.valid

    proxy_cost: Optional[float] = None
    scoring_ms = 0.0
    if valid:
        proxy_cost, scoring_ms = _score_placement(legal_positions, benchmark, plc)

    total_ms = (time.perf_counter() - t_total_start) * 1000
    msgs = ([] if candidate.bypass_legalization else list(leg.messages)) + list(diag.messages)

    return ScoredCandidate(
        name=candidate.name,
        family=candidate.family,
        positions=legal_positions,
        valid=valid,
        proxy_cost=proxy_cost,
        delta_vs_original=None,      # filled in by score_and_select
        num_overlaps=diag.num_overlaps,
        num_out_of_bounds=diag.num_out_of_bounds,
        num_unplaced=0 if candidate.bypass_legalization else len(leg.messages),
        num_moved=leg.num_moved,
        max_move=leg.max_move,
        total_move=leg.total_move,
        legalization_ms=leg.runtime_ms,
        scoring_ms=scoring_ms,
        total_ms=total_ms,
        no_op=leg.no_op,
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
) -> Tuple[ScoredCandidate, List[ScoredCandidate], ScoringDiagnostics]:
    """Score all candidates and return (best, ranked_list, diagnostics).

    Selection rules:
      - original_raw bypasses the legalizer; its raw positions are validated
        and scored.  If valid, it is always in the selectable pool and sets
        the quality baseline.
      - original_legalized runs through the legalizer (no-op when valid).
        It is only selectable when original_raw is invalid (fallback).
      - Among selectable valid candidates, the one with the lowest proxy cost
        wins.
      - Scoring unavailable: select original_raw if valid; rank by validity.
      - Scoring degenerate (all valid candidates tie): prefer original_raw.

    Invariant:
      If original_raw is valid, best_selected.proxy_cost <= raw_original_proxy_cost.

    Prints a warning to stderr when local proxy scoring is unavailable.
    """
    scoring_mode = _detect_scoring_mode(plc, benchmark)

    if scoring_mode == "unavailable":
        print(
            "[M2B] WARNING: Local proxy scoring unavailable: net_nodes empty; "
            "candidate ranking is validity-only.",
            file=sys.stderr,
        )

    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    obstacle_mask = benchmark.macro_fixed & benchmark.get_hard_macro_mask()

    scored: List[ScoredCandidate] = []
    for c in candidates:
        sc = _process_candidate(
            c, benchmark, plc, movable_mask, obstacle_mask,
            legalizer_max_rings=legalizer_max_rings,
        )
        scored.append(sc)

    # Locate raw and legalized original results
    raw_sc = next((s for s in scored if s.name == "original_raw"), None)
    leg_sc = next((s for s in scored if s.name == "original_legalized"), None)

    raw_original_valid = raw_sc.valid if raw_sc is not None else False
    raw_original_proxy_cost = (
        raw_sc.proxy_cost if raw_sc is not None and raw_sc.valid else None
    )

    # Fill delta_vs_original relative to raw original cost
    ref_cost = raw_original_proxy_cost
    for sc in scored:
        if sc.proxy_cost is not None and ref_cost is not None:
            sc.delta_vs_original = sc.proxy_cost - ref_cost

    # Build selectable pool.
    # original_legalized is excluded from selection when original_raw is valid
    # (it is kept in ranked for diagnostics only).
    _diagnostic_only = {"original_legalized"} if raw_original_valid else set()

    valid_scored = [
        s for s in scored
        if s.valid and s.proxy_cost is not None and s.name not in _diagnostic_only
    ]
    invalid_scored = [
        s for s in scored
        if not (s.valid and s.proxy_cost is not None) or s.name in _diagnostic_only
    ]

    # Detect degenerate scoring
    valid_costs = [s.proxy_cost for s in valid_scored]
    unique_costs = {round(c, 9) for c in valid_costs}
    num_unique_scores = len(unique_costs)
    score_is_degenerate = num_unique_scores <= 1
    scoring_available = scoring_mode != "unavailable"

    # Sort valid candidates by cost (stable: earlier candidates win ties).
    valid_sorted = sorted(valid_scored, key=lambda s: s.proxy_cost)

    # Append diagnostics-only candidates at the end of ranked list so they
    # appear in audit output but do not pollute the competitive ranking.
    diagnostic_scored = [s for s in invalid_scored if s.name in _diagnostic_only]
    non_diagnostic_invalid = [s for s in invalid_scored if s.name not in _diagnostic_only]
    non_diagnostic_invalid.sort(key=lambda s: s.name)

    ranked = valid_sorted + diagnostic_scored + non_diagnostic_invalid

    # Select best candidate
    best: Optional[ScoredCandidate]
    if not valid_scored:
        # No valid scored candidates — fall back to original_raw or original_legalized
        best = raw_sc if (raw_sc is not None) else (leg_sc if leg_sc is not None else None)
        if best is None and scored:
            best = scored[0]
        selected_due_to = "fallback_original"
    elif score_is_degenerate:
        # All valid candidates tie — prefer original_raw for determinism
        if raw_sc is not None and raw_sc.valid and raw_sc.name not in _diagnostic_only:
            best = raw_sc
        else:
            best = valid_sorted[0]
        selected_due_to = "validity_only" if scoring_mode == "unavailable" else "tie_break"
    else:
        # Real cost signal: lowest-cost valid selectable candidate wins
        best = valid_sorted[0]
        selected_due_to = "proxy_cost"

    best_cost = best.proxy_cost if best is not None else None
    delta_vs_raw = (
        best_cost - raw_original_proxy_cost
        if best_cost is not None and raw_original_proxy_cost is not None
        else None
    )

    diag = ScoringDiagnostics(
        scoring_available=scoring_available,
        scoring_mode=scoring_mode,
        score_is_degenerate=score_is_degenerate,
        num_unique_scores=num_unique_scores,
        selected_due_to=selected_due_to,
        raw_original_valid=raw_original_valid,
        raw_original_proxy_cost=raw_original_proxy_cost,
        delta_vs_raw_original=delta_vs_raw,
    )

    return best, ranked, diag
