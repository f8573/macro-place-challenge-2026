"""test_m3a_edge_alignment_semantics — left/right/above/below place a relative to b."""

import math
import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.m3a_candidate_generation import generate_pair_candidates, GRID_STEP


def _bm_center():
    """Two macros of size 10 at canvas center area; canvas 200x200."""
    pos = torch.tensor([
        [60.0, 100.0],   # macro 0 (a)
        [100.0, 100.0],  # macro 1 (b)
        [40.0, 40.0],    # macro 2
        [160.0, 160.0],  # macro 3
    ])
    return make_benchmark(
        n_hard=4, canvas=200.0, macro_size=10.0,
        net_nodes=[[0, 1]],
        positions=pos,
    )


def _get_cand(bm, move_type):
    wp = bm.macro_positions.clone().float()
    cands = generate_pair_candidates(bm, wp, macro_a=0, macro_b=1, pair_idx=0, existing_names=set())
    return next((c for c in cands if c.metadata.get("move_type") == move_type), None)


def test_left_places_a_left_of_b():
    bm = _bm_center()
    c = _get_cand(bm, "left")
    assert c is not None, "left candidate missing"
    # a.right = cx_a + w_a/2; b.left = cx_b - w_b/2
    cx_a = float(c.positions[0, 0].item())
    cx_b = float(c.positions[1, 0].item())
    w_a = float(bm.macro_sizes[0, 0].item())
    w_b = float(bm.macro_sizes[1, 0].item())
    # a's right edge should be <= b's left edge (touching is ok)
    assert cx_a + w_a / 2.0 <= cx_b - w_b / 2.0 + 1e-4


def test_right_places_a_right_of_b():
    bm = _bm_center()
    c = _get_cand(bm, "right")
    assert c is not None, "right candidate missing"
    cx_a = float(c.positions[0, 0].item())
    cx_b = float(c.positions[1, 0].item())
    w_a = float(bm.macro_sizes[0, 0].item())
    w_b = float(bm.macro_sizes[1, 0].item())
    assert cx_a - w_a / 2.0 >= cx_b + w_b / 2.0 - 1e-4


def test_below_places_a_below_b():
    bm = _bm_center()
    c = _get_cand(bm, "below")
    assert c is not None, "below candidate missing"
    cy_a = float(c.positions[0, 1].item())
    cy_b = float(c.positions[1, 1].item())
    h_a = float(bm.macro_sizes[0, 1].item())
    h_b = float(bm.macro_sizes[1, 1].item())
    # a's top edge <= b's bottom edge
    assert cy_a + h_a / 2.0 <= cy_b - h_b / 2.0 + 1e-4


def test_above_places_a_above_b():
    bm = _bm_center()
    c = _get_cand(bm, "above")
    assert c is not None, "above candidate missing"
    cy_a = float(c.positions[0, 1].item())
    cy_b = float(c.positions[1, 1].item())
    h_a = float(bm.macro_sizes[0, 1].item())
    h_b = float(bm.macro_sizes[1, 1].item())
    # a's bottom edge >= b's top edge
    assert cy_a - h_a / 2.0 >= cy_b + h_b / 2.0 - 1e-4


def test_left_right_preserves_a_y():
    bm = _bm_center()
    wp = bm.macro_positions.clone().float()
    orig_cy_a = float(wp[0, 1].item())
    for move_type in ("left", "right"):
        c = _get_cand(bm, move_type)
        assert c is not None
        cy_a = float(c.positions[0, 1].item())
        # y should be unchanged (or clamped to the same grid value)
        assert abs(cy_a - orig_cy_a) < GRID_STEP + 1e-6, (
            f"{move_type}: a's y changed from {orig_cy_a} to {cy_a}"
        )


def test_above_below_preserves_a_x():
    bm = _bm_center()
    wp = bm.macro_positions.clone().float()
    orig_cx_a = float(wp[0, 0].item())
    for move_type in ("above", "below"):
        c = _get_cand(bm, move_type)
        assert c is not None
        cx_a = float(c.positions[0, 0].item())
        assert abs(cx_a - orig_cx_a) < GRID_STEP + 1e-6, (
            f"{move_type}: a's x changed from {orig_cx_a} to {cx_a}"
        )


def test_swap_exchanges_centers():
    bm = _bm_center()
    wp = bm.macro_positions.clone().float()
    c = _get_cand(bm, "swap")
    assert c is not None, "swap candidate missing"
    orig_cx_a = float(wp[0, 0].item())
    orig_cy_a = float(wp[0, 1].item())
    orig_cx_b = float(wp[1, 0].item())
    orig_cy_b = float(wp[1, 1].item())
    # After snap+clamp the positions should be near the swapped values.
    new_cx_a = float(c.positions[0, 0].item())
    new_cy_a = float(c.positions[0, 1].item())
    new_cx_b = float(c.positions[1, 0].item())
    new_cy_b = float(c.positions[1, 1].item())
    assert abs(new_cx_a - orig_cx_b) < GRID_STEP + 1e-4
    assert abs(new_cy_a - orig_cy_b) < GRID_STEP + 1e-4
    assert abs(new_cx_b - orig_cx_a) < GRID_STEP + 1e-4
    assert abs(new_cy_b - orig_cy_a) < GRID_STEP + 1e-4


def test_anchor_b_unchanged_in_edge_alignments():
    """For left/right/above/below, b's position should not change."""
    bm = _bm_center()
    wp = bm.macro_positions.clone().float()
    orig_cx_b = float(wp[1, 0].item())
    orig_cy_b = float(wp[1, 1].item())
    for move_type in ("left", "right", "above", "below"):
        c = _get_cand(bm, move_type)
        assert c is not None
        assert float(c.positions[1, 0].item()) == orig_cx_b, (
            f"{move_type}: b's x changed"
        )
        assert float(c.positions[1, 1].item()) == orig_cy_b, (
            f"{move_type}: b's y changed"
        )
