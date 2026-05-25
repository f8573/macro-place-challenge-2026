"""
Directional line-search candidates seeded from winning neighborhood moves.

For each top-K seed (single-macro move with metadata moved_macro_id, dx, dy):
  - Same macro, same direction vector scaled by multipliers
  - Multipliers: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0
    (capped at config.line_search_max_scale)
  - Filters: grossly degraded approx HPWL only (NOT pre-legalization overlap)
  - Family: original_line_search

Pre-legalization overlap is NOT a rejection criterion. Dense benchmarks have
most target positions overlapping; the legalizer resolves overlaps at scoring
time. Candidates carry prelegal_valid/requires_legalization metadata so
diagnostics can show how much legalization work was done.

Scoring-time early stopping (per macro, ascending multiplier order) is
implemented in candidate_scoring.py via line_search_stop_after_worse.

All candidates start from benchmark.macro_positions (original_raw base).
"""

import math
from typing import Any, List, Set

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement, CandidateGenerationConfig
from submissions.solver.core.original_neighborhood import (
    _approx_delta_hpwl,
    _clamp_center,
    _incident_nets,
    _overlaps_any_hard,
)

_LINE_SEARCH_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
_HPWL_HEADROOM_FACTOR = 8.0  # skip candidate if approx delta > this × |seed_approx|

# Scoring priority order: mid-large scales first, then very large, then sub-step.
# 1.5–3.0x first (strong signal), then 1.25x (near-miss refinement), then 4.0x
# (speculative far move), then sub-step scales descending.
_SCORING_PRIORITY_SCALES = [1.5, 2.0, 2.5, 3.0, 1.25, 4.0, 0.75, 0.5, 0.25]
_SCORING_PRIORITY_RANK: dict = {s: i for i, s in enumerate(_SCORING_PRIORITY_SCALES)}


def _safe_mult_name(mult: float) -> str:
    return f"{mult:g}".replace(".", "p")


def generate_original_line_search_candidates(
    benchmark: Benchmark,
    seed_candidates: List[Any],  # CandidatePlacement or ScoredCandidate with metadata
    config: CandidateGenerationConfig,
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate line-search candidates along seed move directions.

    seed_candidates: top-K scored neighborhood candidates with metadata keys
        moved_macro_id, dx, dy.
    existing_names: already-allocated names, mutated in-place.
    """
    if not seed_candidates:
        return []

    base = benchmark.macro_positions.clone().float()
    incident = _incident_nets(benchmark)
    local_names = set(existing_names)
    candidates: List[CandidatePlacement] = []

    max_scale = getattr(config, "line_search_max_scale", 4.0)
    multipliers = [m for m in _LINE_SEARCH_MULTIPLIERS if m <= max_scale + 1e-9]

    canvas_diag = float(benchmark.canvas_width + benchmark.canvas_height)

    for seed in seed_candidates:
        meta = seed.metadata
        macro_id = meta.get("moved_macro_id")
        if macro_id is None:
            continue
        macro_id = int(macro_id)
        dx = float(meta.get("dx", 0.0))
        dy = float(meta.get("dy", 0.0))
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        ox = float(base[macro_id, 0].item())
        oy = float(base[macro_id, 1].item())
        width = float(benchmark.macro_sizes[macro_id, 0].item())
        height = float(benchmark.macro_sizes[macro_id, 1].item())
        incident_ids = incident[macro_id]

        seed_approx = float(meta.get("approx_hpwl_delta", 0.0))
        if seed_approx < -1e-9:
            hpwl_cutoff = abs(seed_approx) * _HPWL_HEADROOM_FACTOR
        else:
            hpwl_cutoff = max(abs(seed_approx) * _HPWL_HEADROOM_FACTOR, canvas_diag * 0.05)

        for mult in multipliers:
            new_x = ox + dx * mult
            new_y = oy + dy * mult
            cx, cy = _clamp_center(
                new_x, new_y, width, height, benchmark.canvas_width, benchmark.canvas_height
            )

            if abs(cx - ox) < 1e-6 and abs(cy - oy) < 1e-6:
                continue

            # Pre-legalization overlap is recorded as metadata but NOT used to reject.
            # Dense benchmarks have most target positions overlapping; the legalizer
            # resolves overlaps at scoring time.
            prelegal_valid = not _overlaps_any_hard(
                macro_id, cx, cy, base, benchmark.macro_sizes, benchmark.num_hard_macros
            )

            positions = base.clone()
            positions[macro_id, 0] = cx
            positions[macro_id, 1] = cy
            approx = _approx_delta_hpwl(benchmark, base, positions, incident_ids)

            # Skip the HPWL cutoff only for overlap-free candidates.
            # When prelegal_valid=False the intended position is inside another macro,
            # so the approx HPWL is computed at an unreliable intermediate location.
            # The legalizer will move the macro to its actual final position — do not
            # pre-reject it based on the overlapping-position signal.
            if prelegal_valid and approx > hpwl_cutoff:
                continue

            name = f"original_line_search_m{macro_id}_scale{_safe_mult_name(mult)}x"
            if name in local_names:
                continue
            local_names.add(name)

            intended_dx = dx * mult
            intended_dy = dy * mult
            candidates.append(
                CandidatePlacement(
                    name=name,
                    family="original_line_search",
                    positions=positions,
                    metadata={
                        "moved_macro_id": macro_id,
                        "dx": float(cx - ox),
                        "dy": float(cy - oy),
                        "scale_multiplier": float(mult),
                        "approx_hpwl_delta": float(approx),
                        "seed_name": seed.name,
                        "refinement_type": f"line_search_scale{mult:g}x",
                        "prelegal_valid": prelegal_valid,
                        "requires_legalization": not prelegal_valid,
                        "intended_dx": float(intended_dx),
                        "intended_dy": float(intended_dy),
                        "intended_move_norm": float(math.sqrt(intended_dx ** 2 + intended_dy ** 2)),
                    },
                )
            )

    existing_names.update(local_names)
    return candidates
