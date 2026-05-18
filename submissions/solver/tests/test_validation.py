"""Unit tests for core/validation.py — wraps official validate_placement."""

import torch
import pytest

from submissions.solver.core.validation import validate
from macro_place.benchmark import Benchmark


def _make_benchmark(n_hard: int = 4, canvas: float = 20.0, macro_size: float = 2.0):
    """Synthetic Benchmark with no fixed macros and no nets."""
    positions = torch.zeros(n_hard, 2)
    sizes = torch.full((n_hard, 2), macro_size)
    fixed = torch.zeros(n_hard, dtype=torch.bool)
    return Benchmark(
        name="test",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=4,
        grid_cols=4,
    )


def _valid_placement(benchmark: Benchmark) -> torch.Tensor:
    """A trivially valid shelf-packed placement for the synthetic benchmark."""
    n = benchmark.num_macros
    placement = torch.zeros(n, 2)
    w = benchmark.macro_sizes[0, 0].item()
    h = benchmark.macro_sizes[0, 1].item()
    for i in range(n):
        placement[i, 0] = w / 2 + i * (w + 0.5)
        placement[i, 1] = h / 2
    return placement


# ── shape ────────────────────────────────────────────────────────────────────


def test_validate_correct_shape():
    bm = _make_benchmark()
    placement = _valid_placement(bm)
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert is_valid, violations


def test_validate_wrong_shape_fails():
    bm = _make_benchmark()
    bad = torch.zeros(bm.num_macros + 1, 2)
    is_valid, violations = validate(bad, bm, check_overlaps=False)
    assert not is_valid


# ── finite coordinates ────────────────────────────────────────────────────────


def test_validate_nan_fails():
    bm = _make_benchmark()
    placement = _valid_placement(bm)
    placement[0, 0] = float("nan")
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert not is_valid


def test_validate_inf_fails():
    bm = _make_benchmark()
    placement = _valid_placement(bm)
    placement[0, 1] = float("inf")
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert not is_valid


# ── bounds ────────────────────────────────────────────────────────────────────


def test_validate_in_bounds_passes():
    bm = _make_benchmark()
    placement = _valid_placement(bm)
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert is_valid, violations


def test_validate_out_of_bounds_fails():
    bm = _make_benchmark()
    placement = _valid_placement(bm)
    placement[0, 0] = bm.canvas_width + 10.0
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert not is_valid


# ── fixed macros ──────────────────────────────────────────────────────────────


def _make_benchmark_with_fixed():
    n_hard = 4
    canvas = 20.0
    macro_size = 2.0
    positions = torch.zeros(n_hard, 2)
    sizes = torch.full((n_hard, 2), macro_size)
    fixed = torch.zeros(n_hard, dtype=torch.bool)
    fixed[0] = True
    positions[0] = torch.tensor([5.0, 5.0])
    return Benchmark(
        name="test_fixed",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=4,
        grid_cols=4,
    )


def test_validate_fixed_macro_unchanged_passes():
    bm = _make_benchmark_with_fixed()
    placement = bm.macro_positions.clone()
    w = bm.macro_sizes[0, 0].item()
    h = bm.macro_sizes[0, 1].item()
    for i in range(1, bm.num_macros):
        placement[i, 0] = w / 2 + i * (w + 0.5)
        placement[i, 1] = h / 2
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert is_valid, violations


def test_validate_fixed_macro_moved_fails():
    bm = _make_benchmark_with_fixed()
    placement = bm.macro_positions.clone()
    placement[0, 0] += 2.0
    is_valid, violations = validate(placement, bm, check_overlaps=False)
    assert not is_valid
    assert any("fixed" in v.lower() or "Fixed" in v for v in violations)


# ── overlaps ──────────────────────────────────────────────────────────────────


def test_validate_overlapping_macros_fails():
    bm = _make_benchmark(n_hard=2)
    placement = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
    is_valid, violations = validate(placement, bm, check_overlaps=True)
    assert not is_valid


def test_validate_touching_macros_not_overlap():
    # Right edge of macro 0 == left edge of macro 1: touching, not overlapping
    # macro size = 2x2; centers at x=1 and x=3 → edges at x=2 exactly
    bm = _make_benchmark(n_hard=2, macro_size=2.0)
    placement = torch.tensor([[1.0, 1.0], [3.0, 1.0]])
    is_valid, violations = validate(placement, bm, check_overlaps=True)
    assert is_valid, f"Touching edges should not be an overlap: {violations}"


def test_validate_separated_macros_passes():
    bm = _make_benchmark(n_hard=2, macro_size=2.0)
    placement = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    is_valid, violations = validate(placement, bm, check_overlaps=True)
    assert is_valid, violations
