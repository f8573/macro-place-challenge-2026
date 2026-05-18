"""Unit tests for core/geometry.py — center-coordinate rectangle math."""

import torch
import pytest

from submissions.solver.core.geometry import (
    rect_edges,
    overlaps_pair,
    in_bounds,
    centers_to_edges,
    bounds_mask,
)


# ── rect_edges ────────────────────────────────────────────────────────────────


def test_rect_edges_basic():
    l, r, b, t = rect_edges(5.0, 3.0, 4.0, 2.0)
    assert l == pytest.approx(3.0)
    assert r == pytest.approx(7.0)
    assert b == pytest.approx(2.0)
    assert t == pytest.approx(4.0)


def test_rect_edges_at_origin():
    l, r, b, t = rect_edges(1.0, 1.0, 2.0, 2.0)
    assert l == pytest.approx(0.0)
    assert r == pytest.approx(2.0)
    assert b == pytest.approx(0.0)
    assert t == pytest.approx(2.0)


def test_rect_edges_non_square():
    l, r, b, t = rect_edges(10.0, 5.0, 6.0, 2.0)
    assert l == pytest.approx(7.0)
    assert r == pytest.approx(13.0)
    assert b == pytest.approx(4.0)
    assert t == pytest.approx(6.0)


# ── overlaps_pair ─────────────────────────────────────────────────────────────


def test_overlaps_pair_clear_overlap():
    # Two 2×2 macros at (0,0) and (1,1) — clearly overlapping
    assert overlaps_pair(0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0) is True


def test_overlaps_pair_clear_separation():
    # 2×2 at origin, 2×2 ten units away — no overlap
    assert overlaps_pair(0.0, 0.0, 2.0, 2.0, 10.0, 0.0, 2.0, 2.0) is False


def test_overlaps_pair_touching_x_not_overlap():
    # Right edge of A == Left edge of B: exactly touching, not an overlap
    # A: center (1,0), width 2 → right=2
    # B: center (3,0), width 2 → left=2
    assert overlaps_pair(1.0, 0.0, 2.0, 2.0, 3.0, 0.0, 2.0, 2.0) is False


def test_overlaps_pair_touching_y_not_overlap():
    # Top edge of A == Bottom edge of B
    assert overlaps_pair(0.0, 1.0, 2.0, 2.0, 0.0, 3.0, 2.0, 2.0) is False


def test_overlaps_pair_partial_x_no_y():
    # Overlap in x but not in y — no 2D overlap
    # A: cx=0, cy=0, 2×2  →  x:[−1,1], y:[−1,1]
    # B: cx=0, cy=5, 2×2  →  x:[−1,1], y:[4,6]
    assert overlaps_pair(0.0, 0.0, 2.0, 2.0, 0.0, 5.0, 2.0, 2.0) is False


def test_overlaps_pair_identical():
    # Same position → maximum overlap
    assert overlaps_pair(5.0, 5.0, 3.0, 3.0, 5.0, 5.0, 3.0, 3.0) is True


# ── in_bounds ─────────────────────────────────────────────────────────────────


def test_in_bounds_center():
    assert in_bounds(5.0, 5.0, 2.0, 2.0, 10.0, 10.0) is True


def test_in_bounds_flush_left():
    # Left edge exactly at 0
    assert in_bounds(1.0, 5.0, 2.0, 2.0, 10.0, 10.0) is True


def test_in_bounds_flush_right():
    # Right edge exactly at canvas_w
    assert in_bounds(9.0, 5.0, 2.0, 2.0, 10.0, 10.0) is True


def test_in_bounds_out_left():
    assert in_bounds(0.0, 5.0, 2.0, 2.0, 10.0, 10.0) is False


def test_in_bounds_out_right():
    assert in_bounds(10.0, 5.0, 2.0, 2.0, 10.0, 10.0) is False


def test_in_bounds_out_bottom():
    assert in_bounds(5.0, 0.0, 2.0, 2.0, 10.0, 10.0) is False


def test_in_bounds_out_top():
    assert in_bounds(5.0, 10.0, 2.0, 2.0, 10.0, 10.0) is False


# ── centers_to_edges ──────────────────────────────────────────────────────────


def test_centers_to_edges_shape():
    positions = torch.tensor([[2.0, 3.0], [5.0, 5.0]])
    sizes = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
    x_min, x_max, y_min, y_max = centers_to_edges(positions, sizes)
    assert x_min.shape == (2,)
    assert x_max.shape == (2,)
    assert y_min.shape == (2,)
    assert y_max.shape == (2,)


def test_centers_to_edges_values():
    positions = torch.tensor([[2.0, 3.0]])
    sizes = torch.tensor([[4.0, 6.0]])
    x_min, x_max, y_min, y_max = centers_to_edges(positions, sizes)
    assert x_min[0].item() == pytest.approx(0.0)
    assert x_max[0].item() == pytest.approx(4.0)
    assert y_min[0].item() == pytest.approx(0.0)
    assert y_max[0].item() == pytest.approx(6.0)


def test_centers_to_edges_symmetry():
    positions = torch.tensor([[5.0, 5.0]])
    sizes = torch.tensor([[2.0, 2.0]])
    x_min, x_max, y_min, y_max = centers_to_edges(positions, sizes)
    assert x_max[0] - x_min[0] == pytest.approx(2.0)
    assert y_max[0] - y_min[0] == pytest.approx(2.0)


# ── bounds_mask ───────────────────────────────────────────────────────────────


def test_bounds_mask_all_in():
    positions = torch.tensor([[2.0, 2.0], [5.0, 5.0]])
    sizes = torch.ones(2, 2) * 2.0
    mask = bounds_mask(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert mask.all()


def test_bounds_mask_one_out():
    positions = torch.tensor([[2.0, 2.0], [0.0, 0.0]])  # second sticks out
    sizes = torch.ones(2, 2) * 2.0
    mask = bounds_mask(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert mask[0].item() is True
    assert mask[1].item() is False


def test_bounds_mask_flush_edge_in_bounds():
    # Left edge exactly at 0, right at canvas_w — should be in bounds
    positions = torch.tensor([[5.0, 5.0]])  # center of 10×10 canvas with 10×10 macro
    sizes = torch.tensor([[10.0, 10.0]])
    mask = bounds_mask(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert mask[0].item() is True
