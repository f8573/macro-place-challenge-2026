"""
M3B candidate generation: up to 3 joint moves per net-coupled 3-macro cluster.

For each cluster (a, b, c) where a < b < c:

Generated move types (at most one CandidatePlacement each):
  1. cyclic rotation       — A takes B's position, B takes C's position,
                             C takes A's position.
  2. reverse cyclic        — A takes C's position, C takes B's position,
                             B takes A's position.
  3. centroid_shift        — translate all three macros by exactly one GRID_STEP
                             toward the cluster's shared-net centroid, preserving
                             their relative offsets.  Axis chosen by larger absolute
                             centroid delta; tie-break x before y.

Candidate 4 (compact-to-centroid preserving x/y rank order) is intentionally
omitted.  Maintaining rank-order constraints while moving three macros toward a
centroid requires checking pairwise crossing conditions — that is structurally
equivalent to spatial legalization and would violate the no-clamping/no-repair
invariant.  Omitting it leaves 3 candidates per cluster, which is within the
spec-allowed maximum of 4.

Coordinates are snapped to GRID_STEP (0.05 µm) after generation.  No clamping
is applied: out-of-bounds coordinates are produced as-is and rejected by the
existing _prepare_candidate / check_placement validation path before scoring.
Fixed-hard macro positions are asserted unchanged in every candidate.
All candidates use bypass_legalization=True so the validator runs on raw
proposed coordinates.
"""

import math
from typing import List, Optional, Set, Tuple

import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement
from submissions.solver.core.m3b_cluster_enumeration import MacroTriple

GRID_STEP: float = 0.05  # µm — must match placement_hash grid tolerance


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------


def snap_to_grid(value: float, step: float = GRID_STEP) -> float:
    """Round value to nearest grid step."""
    return round(value / step) * step


# ---------------------------------------------------------------------------
# Shared-net centroid for a 3-macro cluster
# ---------------------------------------------------------------------------


def _compute_cluster_net_centroid(
    benchmark: Benchmark,
    positions: torch.Tensor,
    macro_a: int,
    macro_b: int,
    macro_c: int,
) -> Optional[Tuple[float, float]]:
    """Weighted centroid of macro positions across nets shared by ≥2 cluster members.

    Returns None when no qualifying net exists or total weight is zero.
    """
    cluster_set = {macro_a, macro_b, macro_c}
    n_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros

    total_wx = 0.0
    total_wy = 0.0
    total_weight = 0.0

    for ni, net_nodes in enumerate(benchmark.net_nodes):
        node_list = [int(n) for n in net_nodes.tolist()]
        hard_ids_in_net = {n for n in node_list if n < n_hard}

        # Net must contain at least 2 of the three cluster macros.
        if len(hard_ids_in_net & cluster_set) < 2:
            continue

        net_weight = float(benchmark.net_weights[ni].item())
        macro_ids_in_net = [n for n in node_list if n < num_macros]
        if not macro_ids_in_net:
            continue

        for n in macro_ids_in_net:
            total_wx += net_weight * float(positions[n, 0].item())
            total_wy += net_weight * float(positions[n, 1].item())
        total_weight += net_weight * len(macro_ids_in_net)

    if total_weight < 1e-10:
        return None
    return total_wx / total_weight, total_wy / total_weight


# ---------------------------------------------------------------------------
# Per-cluster candidate generation
# ---------------------------------------------------------------------------


def generate_cluster_candidates(
    benchmark: Benchmark,
    winner_positions: torch.Tensor,
    macro_a: int,
    macro_b: int,
    macro_c: int,
    cluster_idx: int,
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate up to 3 M3B candidates for the cluster (macro_a, macro_b, macro_c).

    macro_a < macro_b < macro_c (canonical order guaranteed by enumeration).
    cluster_idx is the 0-based rank of this cluster (used in candidate naming).

    Candidates whose name is already in existing_names are skipped silently.
    Fixed-hard macro positions are checked by assertion in each generated placement.
    """
    n_hard = benchmark.num_hard_macros
    fixed_mask = benchmark.macro_fixed

    # Invariant: no macro in the cluster is fixed-hard.
    for mid, label in [(macro_a, "macro_a"), (macro_b, "macro_b"), (macro_c, "macro_c")]:
        if int(mid) < n_hard and bool(fixed_mask[mid].item()):
            raise ValueError(
                f"{label}={mid} is a fixed-hard macro and must not appear in a cluster"
            )

    cx_a = float(winner_positions[macro_a, 0].item())
    cy_a = float(winner_positions[macro_a, 1].item())
    cx_b = float(winner_positions[macro_b, 0].item())
    cy_b = float(winner_positions[macro_b, 1].item())
    cx_c = float(winner_positions[macro_c, 0].item())
    cy_c = float(winner_positions[macro_c, 1].item())

    prefix = f"m3b_c{cluster_idx}_{macro_a}_{macro_b}_{macro_c}"
    candidates: List[CandidatePlacement] = []

    def _build(
        name: str,
        new_a: Tuple[float, float],
        new_b: Tuple[float, float],
        new_c: Tuple[float, float],
        move_type: str,
    ) -> Optional[CandidatePlacement]:
        if name in existing_names:
            return None
        pos = winner_positions.clone().float()
        pos[macro_a, 0] = new_a[0]
        pos[macro_a, 1] = new_a[1]
        pos[macro_b, 0] = new_b[0]
        pos[macro_b, 1] = new_b[1]
        pos[macro_c, 0] = new_c[0]
        pos[macro_c, 1] = new_c[1]

        # Assert fixed-hard macros are unchanged.
        for fi in range(n_hard):
            if bool(fixed_mask[fi].item()):
                assert float(pos[fi, 0].item()) == float(winner_positions[fi, 0].item()), (
                    f"Fixed-hard macro {fi} x-coord changed in M3B candidate {name}"
                )
                assert float(pos[fi, 1].item()) == float(winner_positions[fi, 1].item()), (
                    f"Fixed-hard macro {fi} y-coord changed in M3B candidate {name}"
                )

        return CandidatePlacement(
            name=name,
            family="m3b_cluster_refinement",
            positions=pos,
            bypass_legalization=True,
            metadata={
                "cluster_idx": cluster_idx,
                "macro_a": macro_a,
                "macro_b": macro_b,
                "macro_c": macro_c,
                "move_type": move_type,
            },
        )

    def _snap(x: float, y: float) -> Tuple[float, float]:
        return snap_to_grid(x), snap_to_grid(y)

    # 1. Cyclic rotation: A→B's pos, B→C's pos, C→A's pos.
    c = _build(
        f"{prefix}_cyclic",
        _snap(cx_b, cy_b),
        _snap(cx_c, cy_c),
        _snap(cx_a, cy_a),
        "cyclic",
    )
    if c:
        candidates.append(c)

    # 2. Reverse cyclic: A→C's pos, C→B's pos, B→A's pos.
    c = _build(
        f"{prefix}_rcyclic",
        _snap(cx_c, cy_c),
        _snap(cx_a, cy_a),
        _snap(cx_b, cy_b),
        "rcyclic",
    )
    if c:
        candidates.append(c)

    # 3. Centroid shift: all three move by exactly one GRID_STEP toward the
    #    cluster's shared-net centroid, preserving their relative offsets.
    #    No clamping — OOB results are rejected by validation before scoring.
    centroid = _compute_cluster_net_centroid(
        benchmark, winner_positions, macro_a, macro_b, macro_c
    )
    if centroid is not None:
        mid_x = (cx_a + cx_b + cx_c) / 3.0
        mid_y = (cy_a + cy_b + cy_c) / 3.0
        delta_x = centroid[0] - mid_x
        delta_y = centroid[1] - mid_y

        # Axis with larger absolute delta; tie-break x before y.
        if abs(delta_x) >= abs(delta_y):
            step_x = math.copysign(GRID_STEP, delta_x) if abs(delta_x) > 1e-12 else GRID_STEP
            step_y = 0.0
        else:
            step_x = 0.0
            step_y = math.copysign(GRID_STEP, delta_y) if abs(delta_y) > 1e-12 else GRID_STEP

        c = _build(
            f"{prefix}_centroid_shift",
            _snap(cx_a + step_x, cy_a + step_y),
            _snap(cx_b + step_x, cy_b + step_y),
            _snap(cx_c + step_x, cy_c + step_y),
            "centroid_shift",
        )
        if c:
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def generate_m3b_candidates_for_clusters(
    benchmark: Benchmark,
    winner_positions: torch.Tensor,
    triples: List[MacroTriple],
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate M3B candidates for all clusters.

    Maintains a running set of names to avoid duplicates across clusters.
    Each cluster contributes at most 3 candidates.
    """
    all_candidates: List[CandidatePlacement] = []
    live_names: Set[str] = set(existing_names)

    for cluster_idx, (macro_a, macro_b, macro_c, _score) in enumerate(triples):
        cluster_cands = generate_cluster_candidates(
            benchmark,
            winner_positions,
            macro_a,
            macro_b,
            macro_c,
            cluster_idx,
            live_names,
        )
        for c in cluster_cands:
            live_names.add(c.name)
        all_candidates.extend(cluster_cands)

    return all_candidates
