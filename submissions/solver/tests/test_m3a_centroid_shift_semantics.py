"""test_m3a_centroid_shift_semantics — centroid shift moves both macros by exactly one grid step."""

import math
import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.m3a_candidate_generation import (
    generate_pair_candidates,
    GRID_STEP,
    snap_to_grid,
    _compute_shared_net_centroid,
)


def _bm_with_x_bias():
    """Pair (0,1) with shared net whose centroid is clearly to the right (+x)."""
    pos = torch.tensor([
        [20.0, 50.0],   # macro 0 (a) — left side
        [40.0, 50.0],   # macro 1 (b) — slightly right of a
        [90.0, 50.0],   # macro 2 — far right, pulls centroid toward +x
    ])
    return make_benchmark(
        n_hard=3, canvas=100.0, macro_size=8.0,
        net_nodes=[[0, 1, 2]],
        positions=pos,
    )


def _bm_with_y_bias():
    """Pair (0,1) with shared net centroid clearly above (+y)."""
    pos = torch.tensor([
        [50.0, 20.0],   # macro 0 (a)
        [50.0, 40.0],   # macro 1 (b)
        [50.0, 90.0],   # macro 2 — far up
    ])
    return make_benchmark(
        n_hard=3, canvas=100.0, macro_size=8.0,
        net_nodes=[[0, 1, 2]],
        positions=pos,
    )


def _get_centroid_shift(bm):
    wp = bm.macro_positions.clone().float()
    cands = generate_pair_candidates(bm, wp, 0, 1, 0, existing_names=set())
    return next((c for c in cands if c.metadata.get("move_type") == "centroid_shift"), None)


def test_centroid_shift_candidate_exists():
    bm = _bm_with_x_bias()
    c = _get_centroid_shift(bm)
    assert c is not None, "centroid_shift candidate not generated"


def test_x_dominant_delta_moves_along_x():
    bm = _bm_with_x_bias()
    wp = bm.macro_positions.clone().float()
    c = _get_centroid_shift(bm)
    assert c is not None
    # Both macros should have moved in x by exactly GRID_STEP.
    for mid in (0, 1):
        orig_x = float(wp[mid, 0].item())
        orig_y = float(wp[mid, 1].item())
        new_x = float(c.positions[mid, 0].item())
        new_y = float(c.positions[mid, 1].item())
        dx = abs(new_x - orig_x)
        dy = abs(new_y - orig_y)
        assert abs(dx - GRID_STEP) < 1e-5 or dx < 1e-5, (
            f"macro {mid}: x-move should be ~GRID_STEP or 0 (clamped), got dx={dx}"
        )
        assert dy < 1e-5, f"macro {mid}: y should not move when x dominates, got dy={dy}"


def test_y_dominant_delta_moves_along_y():
    bm = _bm_with_y_bias()
    wp = bm.macro_positions.clone().float()
    c = _get_centroid_shift(bm)
    assert c is not None
    for mid in (0, 1):
        orig_x = float(wp[mid, 0].item())
        orig_y = float(wp[mid, 1].item())
        new_x = float(c.positions[mid, 0].item())
        new_y = float(c.positions[mid, 1].item())
        dx = abs(new_x - orig_x)
        dy = abs(new_y - orig_y)
        assert dx < 1e-5, f"macro {mid}: x should not move when y dominates, got dx={dx}"
        assert abs(dy - GRID_STEP) < 1e-5 or dy < 1e-5, (
            f"macro {mid}: y-move should be ~GRID_STEP or 0 (clamped), got dy={dy}"
        )


def test_tie_break_x_before_y():
    """When |delta_x| == |delta_y|, x axis is chosen."""
    pos = torch.tensor([
        [30.0, 30.0],  # macro 0
        [50.0, 50.0],  # macro 1
        [80.0, 80.0],  # macro 2 — equal push in x and y
    ])
    bm = make_benchmark(n_hard=3, canvas=100.0, macro_size=5.0,
                        net_nodes=[[0, 1, 2]], positions=pos)
    wp = bm.macro_positions.clone().float()
    c = _get_centroid_shift(bm)
    assert c is not None
    # x-delta and y-delta are equal; x must be chosen.
    orig_x0 = float(wp[0, 0].item())
    orig_y0 = float(wp[0, 1].item())
    new_x0 = float(c.positions[0, 0].item())
    new_y0 = float(c.positions[0, 1].item())
    assert abs(new_x0 - orig_x0) > 1e-5, "x should have moved (tie-break x before y)"
    assert abs(new_y0 - orig_y0) < 1e-5, "y should not have moved (tie-break x before y)"


def test_both_macros_move_same_step():
    """Both macros in the pair move by the same (dx, dy) vector."""
    bm = _bm_with_x_bias()
    wp = bm.macro_positions.clone().float()
    c = _get_centroid_shift(bm)
    assert c is not None
    dx0 = float(c.positions[0, 0].item()) - float(wp[0, 0].item())
    dy0 = float(c.positions[0, 1].item()) - float(wp[0, 1].item())
    dx1 = float(c.positions[1, 0].item()) - float(wp[1, 0].item())
    dy1 = float(c.positions[1, 1].item()) - float(wp[1, 1].item())
    assert abs(dx0 - dx1) < 1e-5, f"x-step differs between a ({dx0}) and b ({dx1})"
    assert abs(dy0 - dy1) < 1e-5, f"y-step differs between a ({dy0}) and b ({dy1})"


def test_centroid_result_snapped_to_grid():
    bm = _bm_with_x_bias()
    c = _get_centroid_shift(bm)
    assert c is not None
    for i in range(c.positions.shape[0]):
        x = float(c.positions[i, 0].item())
        y = float(c.positions[i, 1].item())
        assert abs(x - snap_to_grid(x)) < 1e-6
        assert abs(y - snap_to_grid(y)) < 1e-6


def test_no_centroid_shift_when_no_shared_nets():
    """If macros share no net, centroid_shift should not be generated."""
    pos = torch.tensor([
        [20.0, 50.0],
        [80.0, 50.0],
    ])
    # Two macros on separate nets — no shared net.
    bm = make_benchmark(n_hard=2, canvas=100.0, macro_size=5.0,
                        net_nodes=[[0], [1]], positions=pos)
    wp = bm.macro_positions.clone().float()
    cands = generate_pair_candidates(bm, wp, 0, 1, 0, existing_names=set())
    shift = next((c for c in cands if c.metadata.get("move_type") == "centroid_shift"), None)
    assert shift is None, "centroid_shift should not be generated when macros share no net"
