"""M4B legalization-aware regional repair candidate generation."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_scoring import _compute_hpwl, placement_hash
from submissions.solver.core.candidate_types import ScoredCandidate
from submissions.solver.core.diagnostics import PlacementDiagnostics, check_placement
from submissions.solver.legalization.greedy_legalizer import LegalizationResult, legalize


FAMILY = "m4b_region_repair"
SOURCE_STAGE = 6
MOVE_TYPES = ("centroid_shift", "spread")


@dataclass(frozen=True)
class Region:
    region_id: int
    row: int
    col: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def centroid(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def signature(self) -> str:
        payload = f"{self.row},{self.col},{self.x0:.6f},{self.x1:.6f},{self.y0:.6f},{self.y1:.6f}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def partition_regions(canvas_w: float, canvas_h: float, grid_dims: Tuple[int, int]) -> List[Region]:
    rows, cols = grid_dims
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_dims must be positive, got {grid_dims!r}")
    regions: List[Region] = []
    for row in range(rows):
        for col in range(cols):
            rid = row * cols + col
            x0 = canvas_w * col / cols
            x1 = canvas_w * (col + 1) / cols
            y0 = canvas_h * row / rows
            y1 = canvas_h * (row + 1) / rows
            regions.append(Region(rid, row, col, x0, x1, y0, y1))
    return regions


def assign_macros_to_regions(
    positions: torch.Tensor,
    movable_indices: Sequence[int],
    regions: Sequence[Region],
    grid_dims: Tuple[int, int],
    canvas_w: float,
    canvas_h: float,
) -> Dict[int, List[int]]:
    rows, cols = grid_dims
    by_region: Dict[int, List[int]] = {r.region_id: [] for r in regions}
    for macro_id in movable_indices:
        x = float(positions[macro_id, 0].item())
        y = float(positions[macro_id, 1].item())
        col = min(cols - 1, max(0, int(math.floor(x / max(canvas_w, 1e-9) * cols))))
        row = min(rows - 1, max(0, int(math.floor(y / max(canvas_h, 1e-9) * rows))))
        by_region[row * cols + col].append(int(macro_id))
    for ids in by_region.values():
        ids.sort()
    return by_region


def _candidate_name(region_id: int, macro_ids: Sequence[int], move_type: str) -> str:
    macro_part = "_".join(f"m{mid}" for mid in macro_ids)
    return f"m4b_r{region_id}_{macro_part}_{move_type}"


def _clamp_positions_to_canvas(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    macro_ids: Iterable[int],
    canvas_w: float,
    canvas_h: float,
) -> torch.Tensor:
    out = positions.clone().float()
    for mid in macro_ids:
        half_w = float(sizes[mid, 0].item()) / 2.0
        half_h = float(sizes[mid, 1].item()) / 2.0
        out[mid, 0] = max(half_w, min(canvas_w - half_w, float(out[mid, 0].item())))
        out[mid, 1] = max(half_h, min(canvas_h - half_h, float(out[mid, 1].item())))
    return out


def _perturb_positions(
    base_positions: torch.Tensor,
    sizes: torch.Tensor,
    macro_ids: Sequence[int],
    region: Region,
    move_type: str,
    perturbation_fraction: float,
    canvas_w: float,
    canvas_h: float,
) -> torch.Tensor:
    out = base_positions.clone().float()
    if move_type == "centroid_shift":
        target_x, target_y = region.centroid
        for mid in macro_ids:
            dx = target_x - float(base_positions[mid, 0].item())
            dy = target_y - float(base_positions[mid, 1].item())
            norm = math.hypot(dx, dy)
            if norm <= 1e-12:
                dx, dy, norm = 1.0, 0.0, 1.0
            step = perturbation_fraction * float(sizes[mid, 0].item()) / 2.0
            out[mid, 0] = float(base_positions[mid, 0].item()) + step * dx / norm
            out[mid, 1] = float(base_positions[mid, 1].item()) + step * dy / norm
    elif move_type == "spread":
        group = base_positions[list(macro_ids)].float()
        centroid = group.mean(dim=0)
        for local_idx, mid in enumerate(macro_ids):
            dx = float(group[local_idx, 0].item() - centroid[0].item())
            dy = float(group[local_idx, 1].item() - centroid[1].item())
            norm = math.hypot(dx, dy)
            if norm <= 1e-12:
                angle = 2.0 * math.pi * local_idx / max(1, len(macro_ids))
                dx, dy, norm = math.cos(angle), math.sin(angle), 1.0
            step = perturbation_fraction * float(sizes[mid, 0].item()) / 2.0
            out[mid, 0] = float(base_positions[mid, 0].item()) + step * dx / norm
            out[mid, 1] = float(base_positions[mid, 1].item()) + step * dy / norm
    else:
        raise ValueError(f"Unsupported M4B move type: {move_type}")
    return _clamp_positions_to_canvas(out, sizes, macro_ids, canvas_w, canvas_h)


def compute_approx_delta(
    benchmark: Benchmark,
    base_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    incident_net_ids: Optional[Sequence[int]] = None,
) -> float:
    if incident_net_ids is not None:
        from submissions.solver.core.original_neighborhood import _approx_delta_hpwl

        return _approx_delta_hpwl(
            benchmark,
            base_positions,
            candidate_positions,
            list(incident_net_ids),
        )
    return float(_compute_hpwl(candidate_positions, benchmark) - _compute_hpwl(base_positions, benchmark))


def _incident_net_ids_by_macro(benchmark: Benchmark) -> Dict[int, List[int]]:
    incident: Dict[int, List[int]] = {i: [] for i in range(benchmark.num_hard_macros)}
    for net_id, nodes in enumerate(benchmark.net_nodes):
        for node in nodes.tolist():
            if 0 <= int(node) < benchmark.num_hard_macros:
                incident.setdefault(int(node), []).append(net_id)
    return incident


def classify_legalization_failure(
    *,
    diagnostics: PlacementDiagnostics,
    legalization_result: LegalizationResult,
    max_displacement_um: float,
    duplicate: bool,
) -> Optional[str]:
    if diagnostics.num_out_of_bounds > 0:
        return "out_of_bounds"
    if not legalization_result.valid or diagnostics.num_overlaps > 0 or diagnostics.num_nonfinite > 0:
        return "overlap_unresolved"
    if legalization_result.max_move > max_displacement_um:
        return "displacement_too_large"
    if duplicate:
        return "duplicate_after_legalization"
    return None


def _make_scored_candidate(
    *,
    name: str,
    positions: torch.Tensor,
    valid: bool,
    duplicate_of: Optional[str],
    legalizer_result: LegalizationResult,
    diagnostics: PlacementDiagnostics,
    metadata: Dict,
    messages: List[str],
    total_ms: float,
) -> ScoredCandidate:
    return ScoredCandidate(
        name=name,
        family=FAMILY,
        positions=positions,
        valid=valid,
        proxy_cost=None,
        delta_vs_original=None,
        num_overlaps=diagnostics.num_overlaps,
        num_out_of_bounds=diagnostics.num_out_of_bounds,
        num_unplaced=len(legalizer_result.messages),
        num_moved=legalizer_result.num_moved,
        max_move=legalizer_result.max_move,
        total_move=legalizer_result.total_move,
        legalization_ms=legalizer_result.runtime_ms,
        scoring_ms=0.0,
        total_ms=total_ms,
        no_op=legalizer_result.no_op,
        notes="m4b_region_repair",
        was_scored=False,
        duplicate_of=duplicate_of,
        metadata=metadata,
        messages=messages,
    )


def generate_m4b_region_repair_candidates(
    *,
    benchmark: Benchmark,
    base_positions: torch.Tensor,
    existing_hashes: Optional[Dict[str, str]] = None,
    grid_dims: Tuple[int, int] = (3, 3),
    min_macros_per_region: int = 2,
    max_combos_per_region: int = 16,
    legalization_max_displacement_um: float = 200.0,
    perturbation_fraction: float = 0.5,
    legalizer_max_rings: int = 25,
) -> Tuple[List[ScoredCandidate], Dict[str, str]]:
    """Generate M4B audit-pool rows and admit only legal non-duplicates as valid.

    Returns the full audit list plus an updated placement-hash owner map.
    """
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    obstacle_mask = benchmark.macro_fixed & benchmark.get_hard_macro_mask()
    movable_indices = torch.where(movable_mask)[0].tolist()
    regions = partition_regions(benchmark.canvas_width, benchmark.canvas_height, grid_dims)
    by_region = assign_macros_to_regions(
        base_positions,
        movable_indices,
        regions,
        grid_dims,
        benchmark.canvas_width,
        benchmark.canvas_height,
    )
    owner_by_hash: Dict[str, str] = dict(existing_hashes) if existing_hashes else {}
    audit_rows: List[ScoredCandidate] = []
    incident_by_macro = _incident_net_ids_by_macro(benchmark)

    for region in regions:
        region_macros = by_region.get(region.region_id, [])
        if len(region_macros) < min_macros_per_region:
            continue

        emitted_for_region = 0
        combos = itertools.chain(
            itertools.combinations(region_macros, 2),
            itertools.combinations(region_macros, 3),
        )
        for macro_ids_tuple in combos:
            macro_ids = tuple(sorted(int(mid) for mid in macro_ids_tuple))
            for move_type in MOVE_TYPES:
                if emitted_for_region >= max_combos_per_region:
                    break
                emitted_for_region += 1
                name = _candidate_name(region.region_id, macro_ids, move_type)
                t0 = time.perf_counter()
                pre_positions = _perturb_positions(
                    base_positions,
                    benchmark.macro_sizes,
                    macro_ids,
                    region,
                    move_type,
                    perturbation_fraction,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                )
                incident_ids = sorted(
                    {
                        net_id
                        for mid in macro_ids
                        for net_id in incident_by_macro.get(int(mid), [])
                    }
                )
                pre_delta = compute_approx_delta(
                    benchmark,
                    base_positions,
                    pre_positions,
                    incident_ids,
                )
                repair_mask = torch.zeros_like(movable_mask)
                for mid in macro_ids:
                    repair_mask[mid] = True
                repair_obstacles = benchmark.get_hard_macro_mask() & ~repair_mask
                leg = legalize(
                    positions=pre_positions,
                    sizes=benchmark.macro_sizes,
                    canvas_w=benchmark.canvas_width,
                    canvas_h=benchmark.canvas_height,
                    movable_mask=repair_mask,
                    obstacle_mask=repair_obstacles,
                    max_rings=legalizer_max_rings,
                )
                diag = check_placement(
                    positions=leg.positions,
                    sizes=benchmark.macro_sizes,
                    canvas_w=benchmark.canvas_width,
                    canvas_h=benchmark.canvas_height,
                    mask=movable_mask,
                    obstacle_mask=obstacle_mask,
                )
                h: Optional[str] = None
                duplicate_owner: Optional[str] = None
                if leg.valid and diag.valid:
                    h = placement_hash(leg.positions)
                    duplicate_owner = owner_by_hash.get(h)
                reason = classify_legalization_failure(
                    diagnostics=diag,
                    legalization_result=leg,
                    max_displacement_um=legalization_max_displacement_um,
                    duplicate=duplicate_owner is not None,
                )
                admitted_to_score_pool = reason is None
                post_delta = (
                    compute_approx_delta(
                        benchmark,
                        base_positions,
                        leg.positions,
                        incident_ids,
                    )
                    if admitted_to_score_pool
                    else None
                )
                if admitted_to_score_pool and h is not None:
                    owner_by_hash[h] = name

                displacement_mean = (
                    float(leg.total_move) / int(leg.num_moved)
                    if leg.num_moved > 0
                    else 0.0
                )
                metadata = {
                    "generated": True,
                    "pass_id": SOURCE_STAGE,
                    "source_stage": SOURCE_STAGE,
                    "region_id": region.region_id,
                    "region_signature": region.signature,
                    "moved_macro_ids": list(macro_ids),
                    "move_type": move_type,
                    "legalization_status": "legalized" if admitted_to_score_pool else "failed",
                    "legalization_failure_reason": reason,
                    "pre_legalization_approx_delta": pre_delta,
                    "post_legalization_approx_delta": post_delta,
                    "legalization_displacement_max": float(leg.max_move),
                    "legalization_displacement_mean": displacement_mean,
                    "placement_hash": h if (admitted_to_score_pool or reason == "duplicate_after_legalization") else None,
                    "duplicate_after_legalization": reason == "duplicate_after_legalization",
                }
                if admitted_to_score_pool:
                    metadata["approx_hpwl_delta"] = post_delta
                    if post_delta is not None and post_delta >= 0:
                        metadata["m4b_soft_label"] = "no_improvement_candidate"
                else:
                    metadata["skip_reason"] = reason or "legalization_failed"

                audit_rows.append(
                    _make_scored_candidate(
                        name=name,
                        positions=leg.positions,
                        valid=admitted_to_score_pool,
                        duplicate_of=duplicate_owner if reason == "duplicate_after_legalization" else None,
                        legalizer_result=leg,
                        diagnostics=diag,
                        metadata=metadata,
                        messages=list(leg.messages) + list(diag.messages),
                        total_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                )
            if emitted_for_region >= max_combos_per_region:
                break

    return audit_rows, owner_by_hash


def summarize_m4b_audit_rows(rows: Sequence[ScoredCandidate]) -> Dict[str, float | int]:
    generated = len(rows)
    legalized = sum(1 for row in rows if row.valid and row.duplicate_of is None)
    duplicates = sum(1 for row in rows if row.metadata.get("duplicate_after_legalization"))
    failed = generated - legalized
    denom_adjusted = generated - duplicates
    return {
        "generated_count": generated,
        "legalized_count": legalized,
        "legalization_failed_count": failed,
        "duplicate_after_legalization_count": duplicates,
        "raw_legalized_rate": legalized / generated if generated else 0.0,
        "adjusted_legalized_rate": legalized / denom_adjusted if denom_adjusted else 0.0,
    }
