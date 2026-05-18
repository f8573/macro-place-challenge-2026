"""
Fixed-macro-aware placer tests.

Verifies that SolverPlacer:
  - preserves fixed macro positions
  - does not collide with fixed macros when a valid shelf placement exists
  - returns unchanged positions for all-fixed benchmarks
  - leaves soft macro positions unchanged
"""

import torch
import pytest

from macro_place.benchmark import Benchmark
from submissions.solver.placer import SolverPlacer
from submissions.solver.core.geometry import overlaps_pair


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_benchmark(
    n_hard: int,
    n_soft: int = 0,
    canvas: float = 200.0,
    macro_size: float = 4.0,
    fixed_hard_indices: list | None = None,
    fixed_hard_positions: list | None = None,
    soft_positions: list | None = None,
) -> Benchmark:
    """
    Build a synthetic benchmark.  Hard macros come first (indices 0..n_hard-1),
    soft macros follow (indices n_hard..n_hard+n_soft-1).
    """
    n_total = n_hard + n_soft
    positions = torch.zeros(n_total, 2)
    sizes = torch.full((n_total, 2), macro_size)
    fixed = torch.zeros(n_total, dtype=torch.bool)

    if fixed_hard_indices and fixed_hard_positions:
        for idx, pos in zip(fixed_hard_indices, fixed_hard_positions):
            fixed[idx] = True
            positions[idx] = torch.tensor(pos, dtype=torch.float32)

    if soft_positions:
        for i, pos in enumerate(soft_positions):
            positions[n_hard + i] = torch.tensor(pos, dtype=torch.float32)

    return Benchmark(
        name="test",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_total,
        num_hard_macros=n_hard,
        num_soft_macros=n_soft,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_total)],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
    )


# ── fixed macro preservation ──────────────────────────────────────────────────


def test_placer_preserves_fixed_macro_positions():
    """Fixed hard macros must remain at their original positions after placement."""
    bm = _make_benchmark(
        n_hard=4,
        fixed_hard_indices=[0, 1],
        fixed_hard_positions=[[150.0, 150.0], [160.0, 160.0]],
    )
    placer = SolverPlacer()
    result = placer.place(bm)

    assert result[0, 0].item() == pytest.approx(150.0)
    assert result[0, 1].item() == pytest.approx(150.0)
    assert result[1, 0].item() == pytest.approx(160.0)
    assert result[1, 1].item() == pytest.approx(160.0)


# ── no collision with fixed macros ────────────────────────────────────────────


def test_placer_no_collision_with_fixed_macros():
    """
    Shelf-packed movable macros must not overlap fixed macros when the canvas
    provides enough room.  Fixed macros are placed far in the top-right corner;
    movable macros are small enough that shelf packing from (0,0) stays in the
    bottom-left.
    """
    macro_size = 4.0
    canvas = 200.0
    # Fixed macro in the top-right corner, well clear of the packing origin.
    fixed_cx, fixed_cy = 180.0, 180.0

    bm = _make_benchmark(
        n_hard=5,
        canvas=canvas,
        macro_size=macro_size,
        fixed_hard_indices=[0],
        fixed_hard_positions=[[fixed_cx, fixed_cy]],
    )
    placer = SolverPlacer()
    result = placer.place(bm)

    fw, fh = macro_size, macro_size
    for i in range(1, 5):
        cx = result[i, 0].item()
        cy = result[i, 1].item()
        w = bm.macro_sizes[i, 0].item()
        h = bm.macro_sizes[i, 1].item()
        assert not overlaps_pair(cx, cy, w, h, fixed_cx, fixed_cy, fw, fh), (
            f"Movable macro {i} at ({cx:.2f}, {cy:.2f}) overlaps fixed macro "
            f"at ({fixed_cx}, {fixed_cy})"
        )


# ── all-fixed benchmark ───────────────────────────────────────────────────────


def test_placer_all_fixed_returns_unchanged():
    """When every macro is fixed, place() must return the original positions."""
    bm = _make_benchmark(
        n_hard=3,
        fixed_hard_indices=[0, 1, 2],
        fixed_hard_positions=[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
    )
    placer = SolverPlacer()
    result = placer.place(bm)

    assert torch.allclose(result, bm.macro_positions)


# ── soft macros unchanged ─────────────────────────────────────────────────────


def test_placer_soft_macros_remain_unchanged():
    """Soft macros must not be moved by the M1 placer."""
    soft_pos = [[70.0, 70.0], [80.0, 80.0]]
    bm = _make_benchmark(
        n_hard=3,
        n_soft=2,
        soft_positions=soft_pos,
    )
    placer = SolverPlacer()
    result = placer.place(bm)

    n_hard = bm.num_hard_macros
    for i, pos in enumerate(soft_pos):
        assert result[n_hard + i, 0].item() == pytest.approx(pos[0])
        assert result[n_hard + i, 1].item() == pytest.approx(pos[1])


# ── result shape and finite values ───────────────────────────────────────────


def test_placer_output_shape():
    bm = _make_benchmark(n_hard=4, n_soft=2)
    result = SolverPlacer().place(bm)
    assert result.shape == (6, 2)


def test_placer_output_finite():
    bm = _make_benchmark(n_hard=4, n_soft=2)
    result = SolverPlacer().place(bm)
    assert result.isfinite().all()
