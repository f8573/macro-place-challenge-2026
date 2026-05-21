"""Tests for legalization/greedy_legalizer.py."""

import torch
import pytest
import numpy as np

from submissions.solver.legalization.greedy_legalizer import legalize, LegalizationResult


def _sizes(n, w=2.0, h=2.0):
    return torch.full((n, 2), float("nan")).float().fill_(0).add_(
        torch.tensor([[w, h]] * n, dtype=torch.float32)
    )


def _movable(n):
    return torch.ones(n, dtype=torch.bool)


# ---------------------------------------------------------------------------
# test_greedy_legalizer_repairs_simple_overlap
# ---------------------------------------------------------------------------


def test_greedy_legalizer_repairs_simple_overlap():
    # Two 2x2 macros placed at the same center — clearly overlapping
    positions = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    result = legalize(positions, sizes, canvas_w=20.0, canvas_h=20.0)
    assert result.valid, f"Expected valid after legalization: {result.messages}"
    # Verify no overlap in result
    cx0, cy0 = result.positions[0].tolist()
    cx1, cy1 = result.positions[1].tolist()
    w, h = 2.0, 2.0
    sep_x = abs(cx0 - cx1) * 2
    sep_y = abs(cy0 - cy1) * 2
    # At least one axis must not overlap (separation >= w or h)
    assert sep_x >= w or sep_y >= h, f"Macros still overlap: ({cx0},{cy0}) ({cx1},{cy1})"


# ---------------------------------------------------------------------------
# test_greedy_legalizer_repairs_out_of_bounds
# ---------------------------------------------------------------------------


def test_greedy_legalizer_repairs_out_of_bounds():
    # Place macro outside canvas
    positions = torch.tensor([[25.0, 25.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    result = legalize(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert result.valid
    cx, cy = result.positions[0].tolist()
    assert 1.0 <= cx <= 9.0, f"cx={cx} out of bounds"
    assert 1.0 <= cy <= 9.0, f"cy={cy} out of bounds"


# ---------------------------------------------------------------------------
# test_greedy_legalizer_is_deterministic
# ---------------------------------------------------------------------------


def test_greedy_legalizer_is_deterministic():
    torch.manual_seed(42)
    n = 6
    positions = torch.rand(n, 2) * 18.0 + 1.0
    sizes = torch.full((n, 2), 2.0, dtype=torch.float32)
    r1 = legalize(positions.clone(), sizes.clone(), canvas_w=20.0, canvas_h=20.0)
    r2 = legalize(positions.clone(), sizes.clone(), canvas_w=20.0, canvas_h=20.0)
    assert torch.allclose(r1.positions, r2.positions), "Legalizer not deterministic"


# ---------------------------------------------------------------------------
# test_unlegalizable_case_returns_failure
# ---------------------------------------------------------------------------


def test_unlegalizable_case_returns_failure():
    # Canvas is 1x1 but macros are 1x1 each — 3 macros can't fit without overlap
    # Actually with touching allowed, 4 macros of size 0.5x1 fit exactly in 1x1 canvas
    # Use macros larger than canvas to guarantee failure
    positions = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    sizes = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    # Canvas is 1x1 — three 1x1 macros cannot fit
    result = legalize(positions, sizes, canvas_w=1.0, canvas_h=1.0, max_rings=5)
    # At least some macros should fail to legalize
    assert not result.valid, "Expected failure for overcrowded canvas"


# ---------------------------------------------------------------------------
# test_touching_edges_after_legalization_are_allowed
# ---------------------------------------------------------------------------


def test_touching_edges_after_legalization_are_allowed():
    # Two 2x2 macros placed so they just touch: centers at (1,1) and (3,1)
    # Right edge of first = left edge of second at x=2 — touching, not overlapping
    positions = torch.tensor([[1.0, 1.0], [3.0, 1.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    result = legalize(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert result.valid, f"Touching edges should be legal: {result.messages}"


# ---------------------------------------------------------------------------
# test_fixed_macros_not_moved
# ---------------------------------------------------------------------------


def test_fixed_macros_not_moved():
    # First macro is fixed, second is movable
    positions = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    movable = torch.tensor([False, True], dtype=torch.bool)
    result = legalize(positions, sizes, canvas_w=20.0, canvas_h=20.0, movable_mask=movable)
    # Fixed macro must not move
    assert result.positions[0, 0].item() == pytest.approx(2.0), "Fixed macro was moved"
    assert result.positions[0, 1].item() == pytest.approx(2.0), "Fixed macro was moved"


# ---------------------------------------------------------------------------
# test_single_macro_placed_correctly
# ---------------------------------------------------------------------------


def test_single_macro_placed_in_bounds():
    positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    result = legalize(positions, sizes, canvas_w=10.0, canvas_h=10.0)
    assert result.valid
    cx, cy = result.positions[0].tolist()
    assert 1.0 <= cx <= 9.0
    assert 1.0 <= cy <= 9.0


# ---------------------------------------------------------------------------
# test_many_macros_all_legalized
# ---------------------------------------------------------------------------


def test_many_macros_all_legalized():
    n = 20
    positions = torch.rand(n, 2) * 40.0 + 5.0
    sizes = torch.full((n, 2), 3.0, dtype=torch.float32)
    result = legalize(positions, sizes, canvas_w=100.0, canvas_h=100.0)
    assert result.valid, f"Should legalize 20 3x3 macros in 100x100 canvas: {result.messages}"
