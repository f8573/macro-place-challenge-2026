"""
Targeted refinement candidates around winning original_neighborhood moves.

Seeds are the top-K scored improving (or near-miss) single-macro neighborhood
candidates.  Generates:
  - finer/coarser steps along the winning direction (0.125x–2.0x)
  - tiny absolute cardinal moves (0.05–0.50 um)
  - two-axis local grid around the winning position
  - combo2 combinations of the top non-conflicting single-macro seeds
  - combo3 combinations (if refinement_combo_size >= 3)

All candidates start from benchmark.macro_positions (original_raw base).
"""

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement, CandidateGenerationConfig
from submissions.solver.core.original_neighborhood import (
    _approx_delta_hpwl,
    _clamp_center,
    _incident_nets,
    _overlaps_any_hard,
)

_STEP_MULTIPLIERS = [0.125, 0.25, 0.5, 1.5, 2.0]  # 1.0x already done by neighborhood
_TINY_STEPS_UM = [0.05, 0.10, 0.25, 0.50]
_CARDINAL_DIRS: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_GRID_DIRS: List[Tuple[int, int]] = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]


def _safe_name(suffix: str) -> str:
    return suffix.replace(".", "p").replace("-", "m").replace("+", "p")


def _make_single_candidate(
    benchmark: Benchmark,
    base: torch.Tensor,
    macro_id: int,
    new_x: float,
    new_y: float,
    suffix: str,
    incident_ids: List[int],
    local_names: Set[str],
) -> Optional[CandidatePlacement]:
    width = float(benchmark.macro_sizes[macro_id, 0].item())
    height = float(benchmark.macro_sizes[macro_id, 1].item())
    ox = float(base[macro_id, 0].item())
    oy = float(base[macro_id, 1].item())

    cx, cy = _clamp_center(new_x, new_y, width, height, benchmark.canvas_width, benchmark.canvas_height)
    if abs(cx - ox) < 1e-6 and abs(cy - oy) < 1e-6:
        return None
    # Pre-legalization overlap is NOT a rejection criterion. Dense benchmarks have
    # most target positions overlapping; the legalizer resolves overlaps at scoring time.
    prelegal_valid = not _overlaps_any_hard(
        macro_id, cx, cy, base, benchmark.macro_sizes, benchmark.num_hard_macros
    )
    intended_dx = new_x - ox
    intended_dy = new_y - oy

    name = f"original_refinement_m{macro_id}_{_safe_name(suffix)}"
    # Resolve name collisions with a counter suffix
    base_name = name
    counter = 0
    while name in local_names:
        counter += 1
        name = f"{base_name}_{counter}"

    positions = base.clone()
    positions[macro_id, 0] = cx
    positions[macro_id, 1] = cy
    approx = _approx_delta_hpwl(benchmark, base, positions, incident_ids)
    local_names.add(name)
    return CandidatePlacement(
        name=name,
        family="original_refinement",
        positions=positions,
        metadata={
            "moved_macro_id": macro_id,
            "dx": float(cx - ox),
            "dy": float(cy - oy),
            "approx_hpwl_delta": approx,
            "refinement_type": suffix,
            "prelegal_valid": prelegal_valid,
            "requires_legalization": not prelegal_valid,
            "intended_dx": float(intended_dx),
            "intended_dy": float(intended_dy),
            "intended_move_norm": float(math.sqrt(intended_dx ** 2 + intended_dy ** 2)),
        },
    )


def _finer_step_candidates(
    benchmark: Benchmark,
    base: torch.Tensor,
    macro_id: int,
    dx: float,
    dy: float,
    incident_ids: List[int],
    local_names: Set[str],
) -> List[CandidatePlacement]:
    ox = float(base[macro_id, 0].item())
    oy = float(base[macro_id, 1].item())
    out = []
    for mult in _STEP_MULTIPLIERS:
        c = _make_single_candidate(
            benchmark, base, macro_id,
            ox + dx * mult, oy + dy * mult,
            f"scale{mult:g}x",
            incident_ids, local_names,
        )
        if c:
            out.append(c)
    # Opposite direction sanity check
    c = _make_single_candidate(
        benchmark, base, macro_id,
        ox - dx, oy - dy,
        "opposite",
        incident_ids, local_names,
    )
    if c:
        out.append(c)
    return out


def _grid_candidates(
    benchmark: Benchmark,
    base: torch.Tensor,
    macro_id: int,
    dx: float,
    dy: float,
    incident_ids: List[int],
    local_names: Set[str],
) -> List[CandidatePlacement]:
    ox = float(base[macro_id, 0].item())
    oy = float(base[macro_id, 1].item())
    half_dx = abs(dx) / 2.0 if abs(dx) > 1e-9 else abs(dy) / 2.0
    half_dy = abs(dy) / 2.0 if abs(dy) > 1e-9 else abs(dx) / 2.0
    if half_dx < 1e-9:
        half_dx = float(benchmark.canvas_width) * 0.005
    if half_dy < 1e-9:
        half_dy = float(benchmark.canvas_height) * 0.005
    out = []
    for gx, gy in _GRID_DIRS:
        c = _make_single_candidate(
            benchmark, base, macro_id,
            ox + gx * half_dx, oy + gy * half_dy,
            f"grid_{gx:+d}_{gy:+d}",
            incident_ids, local_names,
        )
        if c:
            out.append(c)
    return out


def _tiny_absolute_candidates(
    benchmark: Benchmark,
    base: torch.Tensor,
    macro_id: int,
    incident_ids: List[int],
    local_names: Set[str],
) -> List[CandidatePlacement]:
    ox = float(base[macro_id, 0].item())
    oy = float(base[macro_id, 1].item())
    out = []
    for step in _TINY_STEPS_UM:
        for ddx, ddy in _CARDINAL_DIRS:
            c = _make_single_candidate(
                benchmark, base, macro_id,
                ox + ddx * step, oy + ddy * step,
                f"tiny{step:g}um_{ddx:+d}_{ddy:+d}",
                incident_ids, local_names,
            )
            if c:
                out.append(c)
    return out


def _combo_candidates(
    benchmark: Benchmark,
    base: torch.Tensor,
    seeds: List[Any],   # CandidatePlacement or ScoredCandidate with metadata
    combo_size: int,
    incident: List[List[int]],
    local_names: Set[str],
) -> List[CandidatePlacement]:
    # Deduplicate seeds by macro_id (keep first per macro)
    seen_macros: Dict[int, Any] = {}
    for seed in seeds:
        mid = seed.metadata.get("moved_macro_id")
        if mid is None:
            continue
        mid = int(mid)
        if mid not in seen_macros:
            seen_macros[mid] = seed

    unique_seeds = list(seen_macros.values())
    if len(unique_seeds) < combo_size:
        return []

    out = []
    # Limit combinations to avoid combinatorial explosion
    pool = unique_seeds[:min(len(unique_seeds), 8)]
    for combo in combinations(pool, combo_size):
        positions = base.clone()
        macro_ids = []
        move_dxs = []
        move_dys = []

        for seed in combo:
            macro_id = int(seed.metadata["moved_macro_id"])
            dx = float(seed.metadata.get("dx", 0.0))
            dy = float(seed.metadata.get("dy", 0.0))
            ox = float(base[macro_id, 0].item())
            oy = float(base[macro_id, 1].item())
            width = float(benchmark.macro_sizes[macro_id, 0].item())
            height = float(benchmark.macro_sizes[macro_id, 1].item())
            cx, cy = _clamp_center(ox + dx, oy + dy, width, height, benchmark.canvas_width, benchmark.canvas_height)
            positions[macro_id, 0] = cx
            positions[macro_id, 1] = cy
            macro_ids.append(macro_id)
            move_dxs.append(float(cx - ox))
            move_dys.append(float(cy - oy))

        # Pre-legalization overlap is NOT a rejection criterion; record it as metadata.
        # The legalizer resolves overlaps at scoring time.
        prelegal_valid = all(
            not _overlaps_any_hard(
                macro_id,
                float(positions[macro_id, 0].item()),
                float(positions[macro_id, 1].item()),
                positions,
                benchmark.macro_sizes,
                benchmark.num_hard_macros,
            )
            for macro_id in macro_ids
        )

        # Approx HPWL for all incident nets of moved macros
        all_net_ids: List[int] = []
        seen_nets: Set[int] = set()
        for macro_id in macro_ids:
            for net_id in incident[macro_id]:
                if net_id not in seen_nets:
                    seen_nets.add(net_id)
                    all_net_ids.append(net_id)
        approx = _approx_delta_hpwl(benchmark, base, positions, all_net_ids)

        ids_str = "_".join(f"m{mid}" for mid in sorted(macro_ids))
        name = f"original_refinement_combo{combo_size}_{ids_str}"
        if name in local_names:
            continue
        local_names.add(name)

        out.append(CandidatePlacement(
            name=name,
            family="original_refinement",
            positions=positions,
            metadata={
                "moved_macro_ids": macro_ids,
                "combo_size": combo_size,
                "approx_hpwl_delta": approx,
                "refinement_type": f"combo{combo_size}",
                "prelegal_valid": prelegal_valid,
                "requires_legalization": not prelegal_valid,
            },
        ))

    return out


def generate_original_refinement_candidates(
    benchmark: Benchmark,
    seed_candidates: List[Any],  # CandidatePlacement or ScoredCandidate with metadata
    config: CandidateGenerationConfig,
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate refinement candidates seeded by winning neighborhood moves.

    seed_candidates: top-K scored neighborhood candidates (CandidatePlacement or
        ScoredCandidate); must have metadata keys moved_macro_id, dx, dy.
    existing_names: names already allocated (modified in-place).
    """
    if not seed_candidates:
        return []

    base = benchmark.macro_positions.clone().float()
    incident = _incident_nets(benchmark)
    local_names = set(existing_names)
    candidates: List[CandidatePlacement] = []

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
        incident_ids = incident[macro_id]

        candidates.extend(_finer_step_candidates(benchmark, base, macro_id, dx, dy, incident_ids, local_names))
        candidates.extend(_grid_candidates(benchmark, base, macro_id, dx, dy, incident_ids, local_names))
        candidates.extend(_tiny_absolute_candidates(benchmark, base, macro_id, incident_ids, local_names))

    # Combo candidates
    if config.refinement_combo_size >= 2 and len(seed_candidates) >= 2:
        candidates.extend(_combo_candidates(benchmark, base, seed_candidates, 2, incident, local_names))

    if config.refinement_combo_size >= 3 and len(seed_candidates) >= 3:
        candidates.extend(_combo_candidates(benchmark, base, seed_candidates, 3, incident, local_names))

    # Propagate local_names back to existing_names so callers see allocations
    existing_names.update(local_names)
    return candidates
