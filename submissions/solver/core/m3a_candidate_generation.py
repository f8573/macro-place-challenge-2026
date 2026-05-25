"""
M3A candidate generation: up to 6 joint moves per net-coupled macro pair.

For each pair (a, b) where a < b:
  a  = moved macro  (orientation is stable: smaller macro-id is always 'a')
  b  = anchor macro

Generated move types (at most one CandidatePlacement each):
  1. swap              — swap center positions of a and b
  2. left              — align a immediately left  of b  (a.right  = b.left)
  3. right             — align a immediately right of b  (a.left   = b.right)
  4. below             — align a immediately below b     (a.top    = b.bottom)
  5. above             — align a immediately above b     (a.bottom = b.top)
  6. centroid_shift    — translate both macros by exactly one GRID_STEP toward
                         the shared-net centroid, preserving their relative offset.
                         Axis chosen by larger absolute centroid delta; tie-break x.

Coordinates are snapped to GRID_STEP (0.05 µm) after generation.  No clamping
is applied: out-of-bounds coordinates are produced as-is and rejected by the
existing _prepare_candidate / check_placement validation path before scoring.
Fixed-hard macro positions are asserted unchanged in every candidate.
All candidates use bypass_legalization=True so the validator runs on the raw
proposed coordinates.
"""

import math
from typing import List, Optional, Set, Tuple

import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement
from submissions.solver.core.m3a_pair_enumeration import MacroPair

GRID_STEP: float = 0.05  # µm — must match placement_hash grid tolerance


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def snap_to_grid(value: float, step: float = GRID_STEP) -> float:
    """Round value to nearest grid step."""
    return round(value / step) * step


def _snap2(cx: float, cy: float) -> Tuple[float, float]:
    return snap_to_grid(cx), snap_to_grid(cy)


def _snap_to_m3a_grid(cx: float, cy: float) -> Tuple[float, float]:
    """Snap (cx, cy) to the M3A 0.05 µm grid.  No clamping — OOB coords are
    rejected by the _prepare_candidate / check_placement validation path."""
    return snap_to_grid(cx), snap_to_grid(cy)


# ---------------------------------------------------------------------------
# Shared-net centroid
# ---------------------------------------------------------------------------


def _compute_shared_net_centroid(
    benchmark: Benchmark,
    positions: torch.Tensor,
    macro_a: int,
    macro_b: int,
) -> Optional[Tuple[float, float]]:
    """Weighted centroid of all macro positions across nets shared by a and b.

    Returns None when there are no shared nets or total weight is zero.
    """
    n_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros

    total_wx = 0.0
    total_wy = 0.0
    total_weight = 0.0

    for ni, net_nodes in enumerate(benchmark.net_nodes):
        node_list = [int(n) for n in net_nodes.tolist()]
        hard_ids = [n for n in node_list if n < n_hard]

        if macro_a not in hard_ids or macro_b not in hard_ids:
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
# Per-pair candidate generation
# ---------------------------------------------------------------------------


def generate_pair_candidates(
    benchmark: Benchmark,
    winner_positions: torch.Tensor,
    macro_a: int,
    macro_b: int,
    pair_idx: int,
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate up to 6 M3A candidates for the pair (macro_a, macro_b).

    macro_a is the 'moved' macro; macro_b is the anchor.
    pair_idx is the 0-based rank of this pair (used in candidate naming).

    Candidates whose name is already in existing_names are skipped silently.
    Fixed-hard macro positions are checked by assertion in each generated placement.
    """
    sizes = benchmark.macro_sizes
    n_hard = benchmark.num_hard_macros
    fixed_mask = benchmark.macro_fixed

    # Invariant: neither macro in the pair is fixed-hard.
    if int(macro_a) < n_hard and bool(fixed_mask[macro_a].item()):
        raise ValueError(f"macro_a={macro_a} is a fixed-hard macro and must not appear in a pair")
    if int(macro_b) < n_hard and bool(fixed_mask[macro_b].item()):
        raise ValueError(f"macro_b={macro_b} is a fixed-hard macro and must not appear in a pair")

    w_a = float(sizes[macro_a, 0].item())
    h_a = float(sizes[macro_a, 1].item())
    w_b = float(sizes[macro_b, 0].item())
    h_b = float(sizes[macro_b, 1].item())

    cx_a = float(winner_positions[macro_a, 0].item())
    cy_a = float(winner_positions[macro_a, 1].item())
    cx_b = float(winner_positions[macro_b, 0].item())
    cy_b = float(winner_positions[macro_b, 1].item())

    prefix = f"m3a_p{pair_idx}_{macro_a}_{macro_b}"
    candidates: List[CandidatePlacement] = []

    def _build(
        name: str,
        new_cx_a: float,
        new_cy_a: float,
        move_type: str,
        new_cx_b: Optional[float] = None,
        new_cy_b: Optional[float] = None,
    ) -> Optional[CandidatePlacement]:
        if name in existing_names:
            return None
        pos = winner_positions.clone().float()
        pos[macro_a, 0] = new_cx_a
        pos[macro_a, 1] = new_cy_a
        if new_cx_b is not None:
            pos[macro_b, 0] = new_cx_b
        if new_cy_b is not None:
            pos[macro_b, 1] = new_cy_b

        # Assert fixed-hard macros are unchanged.
        for fi in range(n_hard):
            if bool(fixed_mask[fi].item()):
                assert float(pos[fi, 0].item()) == float(winner_positions[fi, 0].item()), (
                    f"Fixed-hard macro {fi} x-coord changed in M3A candidate {name}"
                )
                assert float(pos[fi, 1].item()) == float(winner_positions[fi, 1].item()), (
                    f"Fixed-hard macro {fi} y-coord changed in M3A candidate {name}"
                )

        return CandidatePlacement(
            name=name,
            family="m3a_pair_refinement",
            positions=pos,
            bypass_legalization=True,
            metadata={
                "pair_idx": pair_idx,
                "macro_a": macro_a,
                "macro_b": macro_b,
                "move_type": move_type,
            },
        )

    # 1. Swap: a takes b's position, b takes a's position.
    new_cx_a_sw, new_cy_a_sw = _snap_to_m3a_grid(cx_b, cy_b)
    new_cx_b_sw, new_cy_b_sw = _snap_to_m3a_grid(cx_a, cy_a)
    c = _build(f"{prefix}_swap", new_cx_a_sw, new_cy_a_sw, "swap", new_cx_b_sw, new_cy_b_sw)
    if c:
        candidates.append(c)

    # 2. Left: a's right edge flush with b's left edge; a's y unchanged.
    raw_cx = cx_b - w_b / 2.0 - w_a / 2.0
    new_cx_a_l, new_cy_a_l = _snap_to_m3a_grid(raw_cx, cy_a)
    c = _build(f"{prefix}_left", new_cx_a_l, new_cy_a_l, "left")
    if c:
        candidates.append(c)

    # 3. Right: a's left edge flush with b's right edge; a's y unchanged.
    raw_cx = cx_b + w_b / 2.0 + w_a / 2.0
    new_cx_a_r, new_cy_a_r = _snap_to_m3a_grid(raw_cx, cy_a)
    c = _build(f"{prefix}_right", new_cx_a_r, new_cy_a_r, "right")
    if c:
        candidates.append(c)

    # 4. Below: a's top edge flush with b's bottom edge; a's x unchanged.
    raw_cy = cy_b - h_b / 2.0 - h_a / 2.0
    new_cx_a_bl, new_cy_a_bl = _snap_to_m3a_grid(cx_a, raw_cy)
    c = _build(f"{prefix}_below", new_cx_a_bl, new_cy_a_bl, "below")
    if c:
        candidates.append(c)

    # 5. Above: a's bottom edge flush with b's top edge; a's x unchanged.
    raw_cy = cy_b + h_b / 2.0 + h_a / 2.0
    new_cx_a_ab, new_cy_a_ab = _snap_to_m3a_grid(cx_a, raw_cy)
    c = _build(f"{prefix}_above", new_cx_a_ab, new_cy_a_ab, "above")
    if c:
        candidates.append(c)

    # 6. Centroid shift: both macros move by exactly one GRID_STEP toward the
    #    shared-net centroid, preserving their relative offset.
    centroid = _compute_shared_net_centroid(benchmark, winner_positions, macro_a, macro_b)
    if centroid is not None:
        mid_x = (cx_a + cx_b) / 2.0
        mid_y = (cy_a + cy_b) / 2.0
        delta_x = centroid[0] - mid_x
        delta_y = centroid[1] - mid_y

        # Choose axis with larger absolute delta; tie-break: x before y.
        if abs(delta_x) >= abs(delta_y):
            step_x = math.copysign(GRID_STEP, delta_x) if abs(delta_x) > 1e-12 else GRID_STEP
            step_y = 0.0
        else:
            step_x = 0.0
            step_y = math.copysign(GRID_STEP, delta_y) if abs(delta_y) > 1e-12 else GRID_STEP

        # Move both macros by one grid step and snap.  No clamping: OOB results
        # are rejected by validation before scoring.
        cs_cx_a = snap_to_grid(cx_a + step_x)
        cs_cy_a = snap_to_grid(cy_a + step_y)
        cs_cx_b = snap_to_grid(cx_b + step_x)
        cs_cy_b = snap_to_grid(cy_b + step_y)

        c = _build(
            f"{prefix}_centroid_shift",
            cs_cx_a, cs_cy_a, "centroid_shift",
            cs_cx_b, cs_cy_b,
        )
        if c:
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def generate_m3a_candidates_for_pairs(
    benchmark: Benchmark,
    winner_positions: torch.Tensor,
    pairs: List[MacroPair],
    existing_names: Set[str],
) -> List[CandidatePlacement]:
    """Generate M3A candidates for all pairs.

    Maintains a running set of names to avoid duplicates across pairs.
    Each pair contributes at most 6 candidates.
    """
    all_candidates: List[CandidatePlacement] = []
    live_names: Set[str] = set(existing_names)

    for pair_idx, (macro_a, macro_b, _shared_nets) in enumerate(pairs):
        pair_cands = generate_pair_candidates(
            benchmark,
            winner_positions,
            macro_a,
            macro_b,
            pair_idx,
            live_names,
        )
        for c in pair_cands:
            live_names.add(c.name)
        all_candidates.extend(pair_cands)

    return all_candidates
