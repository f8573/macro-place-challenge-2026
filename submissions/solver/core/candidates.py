"""
M2B candidate generation pipeline.

Generates all candidate families:
  A. original          — benchmark initial positions (always present)
  B. area_degree_*     — sorted packing heuristics
  C. spectral_*        — spectral projection + cheap transforms
  D. terminal_anchor_* — fixed-object anchor positions

Each candidate is a CandidatePlacement with [N, 2] center positions.
All candidate names are unique within a single call to generate_candidates().
"""

import math
from typing import List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import CandidatePlacement


# ---------------------------------------------------------------------------
# Helper: macro connectivity degrees
# ---------------------------------------------------------------------------


def _compute_degrees(benchmark: Benchmark) -> np.ndarray:
    """Return degree (number of nets touching) for each hard macro."""
    n_hard = benchmark.num_hard_macros
    degrees = np.zeros(n_hard, dtype=np.float64)
    for nodes in benchmark.net_nodes:
        hard_pins = nodes[nodes < n_hard]
        unique_pins = torch.unique(hard_pins)
        for p in unique_pins.tolist():
            degrees[int(p)] += 1
    return degrees


# ---------------------------------------------------------------------------
# Helper: apply cheap transform to positions
# ---------------------------------------------------------------------------


def _apply_transform(
    positions: torch.Tensor,
    transform: str,
    canvas_w: float,
    canvas_h: float,
    sizes: torch.Tensor,
    movable_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a named geometric transform to movable macro positions."""
    out = positions.clone()
    idx = torch.where(movable_mask)[0]

    cx_center = canvas_w / 2.0
    cy_center = canvas_h / 2.0

    if transform == "identity":
        pass
    elif transform == "flip_x":
        out[idx, 0] = canvas_w - positions[idx, 0]
    elif transform == "flip_y":
        out[idx, 1] = canvas_h - positions[idx, 1]
    elif transform == "flip_xy":
        out[idx, 0] = canvas_w - positions[idx, 0]
        out[idx, 1] = canvas_h - positions[idx, 1]
    elif transform == "swap_xy":
        # Swap x and y (only valid when canvas is square-ish; clamp afterward)
        out[idx, 0] = positions[idx, 1]
        out[idx, 1] = positions[idx, 0]
    elif transform == "center_scale_085":
        out[idx, 0] = cx_center + (positions[idx, 0] - cx_center) * 0.85
        out[idx, 1] = cy_center + (positions[idx, 1] - cy_center) * 0.85
    elif transform == "center_scale_070":
        out[idx, 0] = cx_center + (positions[idx, 0] - cx_center) * 0.70
        out[idx, 1] = cy_center + (positions[idx, 1] - cy_center) * 0.70

    # Clamp to keep centers inside canvas
    half_w = sizes[idx, 0] / 2.0
    half_h = sizes[idx, 1] / 2.0
    out[idx, 0] = torch.clamp(out[idx, 0], half_w, canvas_w - half_w)
    out[idx, 1] = torch.clamp(out[idx, 1], half_h, canvas_h - half_h)

    return out


# ---------------------------------------------------------------------------
# A. Original candidate
# ---------------------------------------------------------------------------


def _original_candidate(benchmark: Benchmark) -> CandidatePlacement:
    return CandidatePlacement(
        name="original",
        family="original",
        positions=benchmark.macro_positions.clone().float(),
    )


# ---------------------------------------------------------------------------
# B. Area/degree packing candidates
# ---------------------------------------------------------------------------


def _grid_placement(
    sorted_indices: List[int],
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    base_positions: torch.Tensor,
) -> torch.Tensor:
    """Place macros in a sorted grid filling the canvas from the center outward."""
    n_movable = len(sorted_indices)
    if n_movable == 0:
        return base_positions.clone()

    out = base_positions.clone().float()

    # Determine grid dimensions
    # Use a grid with approximately equal rows and columns
    n_cols = max(1, int(math.ceil(math.sqrt(n_movable))))
    n_rows = max(1, int(math.ceil(n_movable / n_cols)))

    # Cell size based on canvas
    cell_w = canvas_w / n_cols
    cell_h = canvas_h / n_rows

    # Place center of each macro at center of its grid cell
    for k, i in enumerate(sorted_indices):
        row = k // n_cols
        col = k % n_cols
        # Grid center
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        # Clamp to keep macro inside canvas
        w_i = float(sizes[i, 0].item())
        h_i = float(sizes[i, 1].item())
        cx = max(w_i / 2.0, min(canvas_w - w_i / 2.0, cx))
        cy = max(h_i / 2.0, min(canvas_h - h_i / 2.0, cy))
        out[i, 0] = cx
        out[i, 1] = cy

    return out


def _center_outward_placement(
    sorted_indices: List[int],
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    base_positions: torch.Tensor,
) -> torch.Tensor:
    """Place macros in spiral order from center outward (highest priority → center)."""
    out = base_positions.clone().float()
    cx_center = canvas_w / 2.0
    cy_center = canvas_h / 2.0
    n = len(sorted_indices)
    if n == 0:
        return out

    # Generate spiral positions
    positions_list = [(cx_center, cy_center)]
    ring = 1
    while len(positions_list) < n:
        step = max(
            float(sizes[sorted_indices[0], 0].item()),
            float(sizes[sorted_indices[0], 1].item()),
        )
        for angle_steps in range(8 * ring):
            angle = 2.0 * math.pi * angle_steps / (8 * ring)
            r = ring * step
            x = cx_center + r * math.cos(angle)
            y = cy_center + r * math.sin(angle)
            positions_list.append((x, y))
            if len(positions_list) >= n:
                break
        ring += 1
        if ring > 200:
            break

    for k, i in enumerate(sorted_indices):
        w_i = float(sizes[i, 0].item())
        h_i = float(sizes[i, 1].item())
        if k < len(positions_list):
            cx, cy = positions_list[k]
        else:
            cx, cy = cx_center, cy_center
        cx = max(w_i / 2.0, min(canvas_w - w_i / 2.0, cx))
        cy = max(h_i / 2.0, min(canvas_h - h_i / 2.0, cy))
        out[i, 0] = cx
        out[i, 1] = cy

    return out


def _quadrant_placement(
    sorted_indices: List[int],
    priorities: np.ndarray,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    base_positions: torch.Tensor,
) -> torch.Tensor:
    """Distribute macros across 4 quadrants weighted by priority."""
    out = base_positions.clone().float()
    n = len(sorted_indices)
    if n == 0:
        return out

    # Assign to quadrants in round-robin by priority rank
    quadrant_centers = [
        (canvas_w * 0.25, canvas_h * 0.25),
        (canvas_w * 0.75, canvas_h * 0.25),
        (canvas_w * 0.25, canvas_h * 0.75),
        (canvas_w * 0.75, canvas_h * 0.75),
    ]
    quadrant_items = [[], [], [], []]
    for k, i in enumerate(sorted_indices):
        quadrant_items[k % 4].append(i)

    for q_idx, items in enumerate(quadrant_items):
        if not items:
            continue
        qcx, qcy = quadrant_centers[q_idx]
        half_w_q = canvas_w * 0.22
        half_h_q = canvas_h * 0.22
        n_q = len(items)
        n_cols_q = max(1, int(math.ceil(math.sqrt(n_q))))
        n_rows_q = max(1, int(math.ceil(n_q / n_cols_q)))
        cell_w = 2 * half_w_q / max(n_cols_q, 1)
        cell_h = 2 * half_h_q / max(n_rows_q, 1)
        for k, i in enumerate(items):
            row = k // n_cols_q
            col = k % n_cols_q
            cx = (qcx - half_w_q) + (col + 0.5) * cell_w
            cy = (qcy - half_h_q) + (row + 0.5) * cell_h
            w_i = float(sizes[i, 0].item())
            h_i = float(sizes[i, 1].item())
            cx = max(w_i / 2.0, min(canvas_w - w_i / 2.0, cx))
            cy = max(h_i / 2.0, min(canvas_h - h_i / 2.0, cy))
            out[i, 0] = cx
            out[i, 1] = cy

    return out


def _area_degree_candidates(
    benchmark: Benchmark,
    degrees: np.ndarray,
    movable_indices: List[int],
) -> List[CandidatePlacement]:
    """Generate area/degree packing candidates."""
    sizes = benchmark.macro_sizes
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    base = benchmark.macro_positions.clone().float()

    areas = np.array([sizes[i, 0].item() * sizes[i, 1].item() for i in movable_indices])
    degs = np.array([degrees[i] for i in movable_indices])
    priorities_ad = areas * np.log1p(degs)

    # Sort indices by different criteria
    by_area = sorted(range(len(movable_indices)), key=lambda k: -areas[k])
    by_degree = sorted(range(len(movable_indices)), key=lambda k: -degs[k])
    by_ad = sorted(range(len(movable_indices)), key=lambda k: -priorities_ad[k])

    idx_by_area = [movable_indices[k] for k in by_area]
    idx_by_degree = [movable_indices[k] for k in by_degree]
    idx_by_ad = [movable_indices[k] for k in by_ad]

    candidates = []

    # area_degree_center_first: high area*degree → center
    pos = _center_outward_placement(idx_by_ad, sizes, canvas_w, canvas_h, base)
    candidates.append(CandidatePlacement("area_degree_center_first", "area_degree", pos))

    # largest_center_first: largest area → center
    pos = _center_outward_placement(idx_by_area, sizes, canvas_w, canvas_h, base)
    candidates.append(CandidatePlacement("largest_center_first", "area_degree", pos))

    # highest_degree_center_first: most connected → center
    pos = _center_outward_placement(idx_by_degree, sizes, canvas_w, canvas_h, base)
    candidates.append(CandidatePlacement("highest_degree_center_first", "area_degree", pos))

    # area_degree_quadrants: quadrant distribution
    priorities_full = np.zeros(benchmark.num_macros)
    for k, i in enumerate(movable_indices):
        priorities_full[i] = priorities_ad[k]
    pos = _quadrant_placement(idx_by_ad, priorities_full, sizes, canvas_w, canvas_h, base)
    candidates.append(CandidatePlacement("area_degree_quadrants", "area_degree", pos))

    # area_degree_grid: sorted grid
    pos = _grid_placement(idx_by_ad, sizes, canvas_w, canvas_h, base)
    candidates.append(CandidatePlacement("area_degree_grid", "area_degree", pos))

    return candidates


# ---------------------------------------------------------------------------
# C. Spectral candidates (delegated to spectral_projection.py)
# ---------------------------------------------------------------------------


def _spectral_candidates(benchmark: Benchmark) -> List[CandidatePlacement]:
    try:
        from submissions.solver.core.spectral_projection import generate_spectral_candidates

        return generate_spectral_candidates(benchmark)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# D. Terminal-anchor candidates
# ---------------------------------------------------------------------------


def _compute_terminal_anchors(
    benchmark: Benchmark,
    movable_indices: List[int],
) -> np.ndarray:
    """Compute per-macro anchor positions from fixed endpoints.

    Returns [N, 2] array of anchor center positions (canvas coordinates).
    Macros with no fixed connections default to die center.
    """
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    num_macros = benchmark.num_macros
    num_hard = benchmark.num_hard_macros

    # Fixed macro positions
    fixed_mask = benchmark.macro_fixed.numpy().astype(bool)
    macro_pos = benchmark.macro_positions.numpy()

    # Port positions (if available)
    port_positions = benchmark.port_positions.numpy()  # [P, 2]
    has_ports = port_positions.shape[0] > 0

    anchors = np.full((num_macros, 2), [canvas_w / 2.0, canvas_h / 2.0])

    for i in movable_indices:
        total_weight = 0.0
        wx = 0.0
        wy = 0.0

        for ni, nodes in enumerate(benchmark.net_nodes):
            node_list = nodes.tolist()
            if i not in node_list:
                continue
            net_w = float(benchmark.net_weights[ni].item())
            k = len(node_list)
            contribution = net_w / max(1, k - 1)

            for j in node_list:
                if j == i:
                    continue
                if j < num_macros and fixed_mask[j]:
                    # Connected to a fixed macro
                    wx += contribution * macro_pos[j, 0]
                    wy += contribution * macro_pos[j, 1]
                    total_weight += contribution

        # Also use port positions from net_pin_nodes if available
        if has_ports and benchmark.net_pin_nodes:
            for ni, pin_nodes in enumerate(benchmark.net_pin_nodes):
                if pin_nodes.numel() == 0:
                    continue
                # Check if macro i is in this net (owner column)
                owners = pin_nodes[:, 0].tolist()
                if i not in owners:
                    continue
                net_w = float(benchmark.net_weights[ni].item())
                k = pin_nodes.shape[0]
                contribution = net_w / max(1, k - 1)
                for row in range(pin_nodes.shape[0]):
                    owner = int(pin_nodes[row, 0].item())
                    if owner >= num_macros:
                        # I/O port
                        port_idx = owner - num_macros
                        if port_idx < port_positions.shape[0]:
                            wx += contribution * port_positions[port_idx, 0]
                            wy += contribution * port_positions[port_idx, 1]
                            total_weight += contribution

        if total_weight > 1e-10:
            anchors[i, 0] = wx / total_weight
            anchors[i, 1] = wy / total_weight

    return anchors


def _terminal_anchor_candidates(
    benchmark: Benchmark,
    degrees: np.ndarray,
    movable_indices: List[int],
) -> List[CandidatePlacement]:
    """Generate terminal-anchor candidates."""
    anchors = _compute_terminal_anchors(benchmark, movable_indices)
    sizes = benchmark.macro_sizes
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    base = benchmark.macro_positions.clone().float()

    # Clamp anchors to canvas
    out_base = base.clone()
    for i in movable_indices:
        w_i = float(sizes[i, 0].item())
        h_i = float(sizes[i, 1].item())
        cx = max(w_i / 2.0, min(canvas_w - w_i / 2.0, anchors[i, 0]))
        cy = max(h_i / 2.0, min(canvas_h - h_i / 2.0, anchors[i, 1]))
        out_base[i, 0] = cx
        out_base[i, 1] = cy

    candidates = [CandidatePlacement("terminal_anchor", "terminal_anchor", out_base.clone())]

    # terminal_anchor_area_sorted: same positions, different legalization order hint
    # (the candidate positions are identical; the legalizer sort is configured externally)
    candidates.append(
        CandidatePlacement("terminal_anchor_area_sorted", "terminal_anchor", out_base.clone(),
                           notes="area_sort")
    )

    # terminal_anchor_degree_sorted
    candidates.append(
        CandidatePlacement("terminal_anchor_degree_sorted", "terminal_anchor", out_base.clone(),
                           notes="degree_sort")
    )

    # terminal_anchor_center_scale_085: pull toward center by 15%
    movable_t = torch.zeros(benchmark.num_macros, dtype=torch.bool)
    for i in movable_indices:
        movable_t[i] = True

    pos_scaled = _apply_transform(out_base, "center_scale_085", canvas_w, canvas_h, sizes, movable_t)
    candidates.append(
        CandidatePlacement("terminal_anchor_center_scale_085", "terminal_anchor", pos_scaled)
    )

    return candidates


# ---------------------------------------------------------------------------
# E. Transforms applied to spectral and area-degree bases
# ---------------------------------------------------------------------------


def _transform_variants(
    base: CandidatePlacement,
    transforms: List[str],
    canvas_w: float,
    canvas_h: float,
    sizes: torch.Tensor,
    movable_mask: torch.Tensor,
) -> List[CandidatePlacement]:
    """Apply transforms to a base candidate, returning new candidates."""
    variants = []
    for t in transforms:
        if t == "identity":
            continue  # base is already the identity
        pos = _apply_transform(base.positions, t, canvas_w, canvas_h, sizes, movable_mask)
        name = f"{base.name}_{t}"
        variants.append(CandidatePlacement(name, base.family, pos))
    return variants


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_candidates(
    benchmark: Benchmark,
    include_transforms: bool = True,
) -> List[CandidatePlacement]:
    """Generate all M2B candidates.

    Always includes 'original' as the first candidate.
    Never raises; failures in individual families produce empty sublists.

    Returns:
        List of CandidatePlacement with unique names.
    """
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    sizes = benchmark.macro_sizes

    movable_hard = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    movable_indices = torch.where(movable_hard)[0].tolist()

    degrees = _compute_degrees(benchmark)

    candidates: List[CandidatePlacement] = []

    # A. Original (always first)
    candidates.append(_original_candidate(benchmark))

    # B. Area/degree
    try:
        candidates.extend(_area_degree_candidates(benchmark, degrees, movable_indices))
    except Exception:
        pass

    # C. Spectral
    try:
        candidates.extend(_spectral_candidates(benchmark))
    except Exception:
        pass

    # D. Terminal-anchor
    try:
        candidates.extend(_terminal_anchor_candidates(benchmark, degrees, movable_indices))
    except Exception:
        pass

    # E. Cheap transforms on spectral base (if spectral succeeded)
    if include_transforms:
        transforms = ["flip_x", "flip_y", "flip_xy", "center_scale_085", "center_scale_070"]
        spectral_base = next((c for c in candidates if c.name == "spectral_xy"), None)
        if spectral_base is not None:
            try:
                variants = _transform_variants(
                    spectral_base, transforms, canvas_w, canvas_h, sizes, movable_hard
                )
                candidates.extend(variants)
            except Exception:
                pass

        # Transforms on area_degree_center_first
        ad_base = next((c for c in candidates if c.name == "area_degree_center_first"), None)
        if ad_base is not None:
            try:
                variants = _transform_variants(
                    ad_base, transforms, canvas_w, canvas_h, sizes, movable_hard
                )
                candidates.extend(variants)
            except Exception:
                pass

    # Deduplicate names (keep first occurrence)
    seen = set()
    unique: List[CandidatePlacement] = []
    for c in candidates:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)

    return unique
