"""
Original-anchored deterministic neighborhood candidates.

All candidates start from original_raw and move only one selected hard macro.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement, CandidateGenerationConfig


_STEP_PROFILES = {
    "small": {
        "scale_fracs": [0.25],
        "die_fracs": [0.01],
    },
    "medium": {
        "scale_fracs": [0.25, 0.50],
        "die_fracs": [0.01, 0.02],
    },
    "large": {
        "scale_fracs": [0.25, 0.50, 1.00],
        "die_fracs": [0.01, 0.02, 0.03],
    },
}


def _clamp_center(x: float, y: float, width: float, height: float, canvas_w: float, canvas_h: float) -> Tuple[float, float]:
    return (
        max(width / 2.0, min(canvas_w - width / 2.0, x)),
        max(height / 2.0, min(canvas_h - height / 2.0, y)),
    )


def _overlaps_any_hard(
    macro_id: int,
    x: float,
    y: float,
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard_macros: int,
) -> bool:
    width = float(sizes[macro_id, 0].item())
    height = float(sizes[macro_id, 1].item())
    left = x - width / 2.0
    right = x + width / 2.0
    bottom = y - height / 2.0
    top = y + height / 2.0

    for other in range(num_hard_macros):
        if other == macro_id:
            continue
        ow = float(sizes[other, 0].item())
        oh = float(sizes[other, 1].item())
        ox = float(positions[other, 0].item())
        oy = float(positions[other, 1].item())
        oleft = ox - ow / 2.0
        oright = ox + ow / 2.0
        obottom = oy - oh / 2.0
        otop = oy + oh / 2.0
        if left < oright and right > oleft and bottom < otop and top > obottom:
            return True
    return False


def _incident_nets(benchmark: Benchmark) -> List[List[int]]:
    incident: List[List[int]] = [[] for _ in range(benchmark.num_hard_macros)]
    for net_id, nodes in enumerate(benchmark.net_nodes):
        for node in torch.unique(nodes[nodes < benchmark.num_hard_macros]).tolist():
            incident[int(node)].append(net_id)
    return incident


def _endpoint_positions_for_net(
    benchmark: Benchmark,
    positions: torch.Tensor,
    net_id: int,
) -> List[Tuple[float, float]]:
    endpoints: List[Tuple[float, float]] = []
    if benchmark.net_pin_nodes:
        for owner, _pin_idx in benchmark.net_pin_nodes[net_id].tolist():
            if owner < benchmark.num_macros:
                endpoints.append((float(positions[owner, 0].item()), float(positions[owner, 1].item())))
            else:
                port_idx = owner - benchmark.num_macros
                if port_idx < benchmark.port_positions.shape[0]:
                    endpoints.append(
                        (
                            float(benchmark.port_positions[port_idx, 0].item()),
                            float(benchmark.port_positions[port_idx, 1].item()),
                        )
                    )
    else:
        for owner in benchmark.net_nodes[net_id].tolist():
            if owner < benchmark.num_macros:
                endpoints.append((float(positions[owner, 0].item()), float(positions[owner, 1].item())))
    return endpoints


def _net_hpwl_from_endpoints(endpoints: List[Tuple[float, float]]) -> float:
    if len(endpoints) < 2:
        return 0.0
    xs = [p[0] for p in endpoints]
    ys = [p[1] for p in endpoints]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _approx_delta_hpwl(
    benchmark: Benchmark,
    original_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    incident_net_ids: List[int],
) -> float:
    delta = 0.0
    for net_id in incident_net_ids:
        net_w = float(benchmark.net_weights[net_id].item()) if benchmark.net_weights.numel() else 1.0
        before = _net_hpwl_from_endpoints(_endpoint_positions_for_net(benchmark, original_positions, net_id))
        after = _net_hpwl_from_endpoints(_endpoint_positions_for_net(benchmark, candidate_positions, net_id))
        delta += net_w * (after - before)
    return float(delta)


def _macro_centroid(
    benchmark: Benchmark,
    macro_id: int,
    positions: torch.Tensor,
    incident_net_ids: List[int],
) -> Tuple[float, float]:
    weight_sum = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for net_id in incident_net_ids:
        net_w = float(benchmark.net_weights[net_id].item()) if benchmark.net_weights.numel() else 1.0
        if benchmark.net_pin_nodes:
            for owner, _pin_idx in benchmark.net_pin_nodes[net_id].tolist():
                if owner == macro_id:
                    continue
                if owner < benchmark.num_macros:
                    x_sum += net_w * float(positions[owner, 0].item())
                    y_sum += net_w * float(positions[owner, 1].item())
                    weight_sum += net_w
                else:
                    port_idx = owner - benchmark.num_macros
                    if port_idx < benchmark.port_positions.shape[0]:
                        x_sum += net_w * float(benchmark.port_positions[port_idx, 0].item())
                        y_sum += net_w * float(benchmark.port_positions[port_idx, 1].item())
                        weight_sum += net_w
        else:
            for owner in benchmark.net_nodes[net_id].tolist():
                if owner == macro_id or owner >= benchmark.num_macros:
                    continue
                x_sum += net_w * float(positions[owner, 0].item())
                y_sum += net_w * float(positions[owner, 1].item())
                weight_sum += net_w

    if weight_sum <= 1e-9:
        return float(positions[macro_id, 0].item()), float(positions[macro_id, 1].item())
    return x_sum / weight_sum, y_sum / weight_sum


def _fixed_endpoint_strength(benchmark: Benchmark, macro_id: int, incident_net_ids: List[int]) -> float:
    total = 0.0
    for net_id in incident_net_ids:
        net_w = float(benchmark.net_weights[net_id].item()) if benchmark.net_weights.numel() else 1.0
        if benchmark.net_pin_nodes:
            for owner, _pin_idx in benchmark.net_pin_nodes[net_id].tolist():
                if owner == macro_id:
                    continue
                if owner >= benchmark.num_macros:
                    total += net_w
                elif owner < benchmark.num_hard_macros and bool(benchmark.macro_fixed[owner].item()):
                    total += net_w
        else:
            for owner in benchmark.net_nodes[net_id].tolist():
                if owner == macro_id:
                    continue
                if owner < benchmark.num_hard_macros and bool(benchmark.macro_fixed[owner].item()):
                    total += net_w
    return total


def _macro_hpwl_contribution(
    benchmark: Benchmark,
    positions: torch.Tensor,
    incident_net_ids: List[int],
) -> float:
    total = 0.0
    for net_id in incident_net_ids:
        net_w = float(benchmark.net_weights[net_id].item()) if benchmark.net_weights.numel() else 1.0
        total += net_w * _net_hpwl_from_endpoints(_endpoint_positions_for_net(benchmark, positions, net_id))
    return total


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmax = float(values.max())
    vmin = float(values.min())
    if vmax <= vmin + 1e-12:
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)


def _candidate_from_move(
    benchmark: Benchmark,
    base_positions: torch.Tensor,
    macro_id: int,
    new_x: float,
    new_y: float,
    move_type: str,
    incident_net_ids: List[int],
    selection_summary: Dict[str, object],
    selection_score: float,
) -> CandidatePlacement:
    positions = base_positions.clone()
    old_x = float(base_positions[macro_id, 0].item())
    old_y = float(base_positions[macro_id, 1].item())
    positions[macro_id, 0] = new_x
    positions[macro_id, 1] = new_y
    approx_delta = _approx_delta_hpwl(benchmark, base_positions, positions, incident_net_ids)
    suffix = move_type.replace(".", "p").replace("-", "m")
    return CandidatePlacement(
        name=f"original_neighborhood_m{macro_id}_{suffix}",
        family="original_neighborhood",
        positions=positions,
        metadata={
            "moved_macro_id": macro_id,
            "dx": float(new_x - old_x),
            "dy": float(new_y - old_y),
            "move_type": move_type,
            "approx_hpwl_delta": approx_delta,
            "selection_score": float(selection_score),
            "selected_macro_count": int(selection_summary["selected_macro_count"]),
            "macro_selection_reason": selection_summary["macro_selection_reason"],
            "top_selected_macros": selection_summary["top_selected_macros"],
        },
    )


def _nearest_legal_slot(
    benchmark: Benchmark,
    base_positions: torch.Tensor,
    macro_id: int,
    step_x: float,
    step_y: float,
) -> Tuple[float, float] | None:
    width = float(benchmark.macro_sizes[macro_id, 0].item())
    height = float(benchmark.macro_sizes[macro_id, 1].item())
    ox = float(base_positions[macro_id, 0].item())
    oy = float(base_positions[macro_id, 1].item())
    best = None
    best_dist = None
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]
    for mx, my in offsets:
        cx, cy = _clamp_center(
            ox + mx * step_x,
            oy + my * step_y,
            width,
            height,
            benchmark.canvas_width,
            benchmark.canvas_height,
        )
        if _overlaps_any_hard(macro_id, cx, cy, base_positions, benchmark.macro_sizes, benchmark.num_hard_macros):
            continue
        dist = math.hypot(cx - ox, cy - oy)
        if best is None or dist < best_dist - 1e-9:
            best = (cx, cy)
            best_dist = dist
    return best


def generate_original_neighborhood_candidates(
    benchmark: Benchmark,
    config: CandidateGenerationConfig,
) -> List[CandidatePlacement]:
    """Generate conservative original-anchored candidates."""
    if config.neighborhood_step_profile not in _STEP_PROFILES:
        raise ValueError(f"Unknown neighborhood_step_profile: {config.neighborhood_step_profile}")

    base = benchmark.macro_positions.clone().float()
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    movable_ids = torch.where(movable_mask)[0].tolist()
    if not movable_ids:
        return []

    incident = _incident_nets(benchmark)
    area_term = []
    degree_term = []
    weighted_degree_term = []
    fixed_strength_term = []
    hpwl_term = []
    for macro_id in movable_ids:
        incident_ids = incident[macro_id]
        degree = len(incident_ids)
        weighted_degree = sum(
            float(benchmark.net_weights[nid].item()) if benchmark.net_weights.numel() else 1.0
            for nid in incident_ids
        )
        area = float(benchmark.macro_sizes[macro_id, 0].item() * benchmark.macro_sizes[macro_id, 1].item())
        area_term.append(area * math.log1p(degree))
        degree_term.append(float(degree))
        weighted_degree_term.append(weighted_degree)
        fixed_strength_term.append(_fixed_endpoint_strength(benchmark, macro_id, incident_ids))
        hpwl_term.append(_macro_hpwl_contribution(benchmark, base, incident_ids))

    degree_n = _normalize(np.asarray(degree_term, dtype=np.float64))
    weighted_degree_n = _normalize(np.asarray(weighted_degree_term, dtype=np.float64))
    fixed_strength_n = _normalize(np.asarray(fixed_strength_term, dtype=np.float64))
    hpwl_n = _normalize(np.asarray(hpwl_term, dtype=np.float64))
    area_n = _normalize(np.asarray(area_term, dtype=np.float64))
    selection_scores = (
        1.50 * weighted_degree_n
        + 1.25 * hpwl_n
        + 1.00 * degree_n
        + 1.00 * area_n
        + 1.25 * fixed_strength_n
    )

    ranked = sorted(
        (
            {
                "macro_id": macro_id,
                "score": float(selection_scores[idx]),
                "weighted_degree": float(weighted_degree_term[idx]),
                "incident_nets": int(degree_term[idx]),
                "hpwl_contribution": float(hpwl_term[idx]),
                "fixed_endpoint_strength": float(fixed_strength_term[idx]),
            }
            for idx, macro_id in enumerate(movable_ids)
        ),
        key=lambda item: (-item["score"], item["macro_id"]),
    )
    selected = ranked[: max(0, min(config.neighborhood_macro_limit, len(ranked)))]
    selection_summary = {
        "selected_macro_count": len(selected),
        "macro_selection_reason": (
            "score = 1.5*weighted_degree + 1.25*hpwl_contribution + "
            "degree + area*log(1+degree) + 1.25*fixed_endpoint_strength"
        ),
        "top_selected_macros": selected[: min(10, len(selected))],
    }
    if not selected:
        return []

    profile = _STEP_PROFILES[config.neighborhood_step_profile]
    budget = None
    if config.candidate_budget is not None:
        budget = max(0, config.candidate_budget - 2)

    candidates: List[CandidatePlacement] = []
    for item in selected:
        macro_id = int(item["macro_id"])
        incident_ids = incident[macro_id]
        width = float(benchmark.macro_sizes[macro_id, 0].item())
        height = float(benchmark.macro_sizes[macro_id, 1].item())
        ox = float(base[macro_id, 0].item())
        oy = float(base[macro_id, 1].item())
        scale_frac = profile["scale_fracs"][0]
        die_frac = profile["die_fracs"][0]
        step_x = max(scale_frac * width, die_frac * benchmark.canvas_width)
        step_y = max(scale_frac * height, die_frac * benchmark.canvas_height)

        move_specs: List[Tuple[str, float, float]] = []
        for label, dx, dy in [
            ("left", -step_x, 0.0),
            ("right", step_x, 0.0),
            ("down", 0.0, -step_y),
            ("up", 0.0, step_y),
        ]:
            move_specs.append((f"{label}_s", ox + dx, oy + dy))

        centroid_x, centroid_y = _macro_centroid(benchmark, macro_id, base, incident_ids)
        vec_x = centroid_x - ox
        vec_y = centroid_y - oy
        norm = math.hypot(vec_x, vec_y)
        if norm > 1e-9:
            move_len = max(step_x, step_y)
            move_specs.append(
                (
                    "toward_centroid",
                    ox + (vec_x / norm) * move_len,
                    oy + (vec_y / norm) * move_len,
                )
            )

        slot = _nearest_legal_slot(benchmark, base, macro_id, step_x, step_y)
        if slot is not None:
            move_specs.append(("nearest_whitespace", slot[0], slot[1]))

        seen_targets = set()
        for move_type, tx, ty in move_specs:
            cx, cy = _clamp_center(tx, ty, width, height, benchmark.canvas_width, benchmark.canvas_height)
            target_key = (round(cx, 4), round(cy, 4))
            if target_key == (round(ox, 4), round(oy, 4)) or target_key in seen_targets:
                continue
            seen_targets.add(target_key)
            candidates.append(
                _candidate_from_move(
                    benchmark=benchmark,
                    base_positions=base,
                    macro_id=macro_id,
                    new_x=cx,
                    new_y=cy,
                    move_type=move_type,
                    incident_net_ids=incident_ids,
                    selection_summary=selection_summary,
                    selection_score=float(item["score"]),
                )
            )
            if budget is not None and len(candidates) >= budget:
                return candidates[:budget]

    return candidates
