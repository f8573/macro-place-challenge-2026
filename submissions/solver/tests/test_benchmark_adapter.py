"""Unit tests for core/benchmark_adapter.py."""

import torch
import pytest

from submissions.solver.core.benchmark_adapter import canvas_area, hard_macro_area, utilization, inspect
from macro_place.benchmark import Benchmark


def _make_benchmark(
    n_hard: int = 4,
    macro_size: float = 2.0,
    canvas: float = 20.0,
    n_fixed: int = 1,
):
    positions = torch.zeros(n_hard, 2)
    sizes = torch.full((n_hard, 2), macro_size)
    fixed = torch.zeros(n_hard, dtype=torch.bool)
    for i in range(n_fixed):
        fixed[i] = True
        positions[i] = torch.tensor([float(i + 1), float(i + 1)])
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
        num_nets=3,
        net_nodes=[torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2, 3])],
        net_weights=torch.ones(3),
        grid_rows=8,
        grid_cols=8,
    )


def test_canvas_area():
    bm = _make_benchmark(canvas=10.0)
    assert canvas_area(bm) == pytest.approx(100.0)


def test_canvas_area_rect():
    positions = torch.zeros(1, 2)
    sizes = torch.ones(1, 2)
    bm = Benchmark(
        name="rect",
        canvas_width=5.0,
        canvas_height=8.0,
        num_macros=1,
        num_hard_macros=1,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=torch.zeros(1, dtype=torch.bool),
        macro_names=["m0"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=4,
        grid_cols=4,
    )
    assert canvas_area(bm) == pytest.approx(40.0)


def test_hard_macro_area():
    # 4 hard macros, each 2×2 = 4 μm² → total 16
    bm = _make_benchmark(n_hard=4, macro_size=2.0)
    assert hard_macro_area(bm) == pytest.approx(16.0)


def test_utilization():
    # 4 macros × 4 μm² each = 16 μm²; canvas = 400 μm²
    bm = _make_benchmark(n_hard=4, macro_size=2.0, canvas=20.0)
    expected = 16.0 / 400.0
    assert utilization(bm) == pytest.approx(expected)


def test_inspect_keys():
    bm = _make_benchmark()
    stats = inspect(bm)
    expected_keys = {
        "name",
        "canvas_width",
        "canvas_height",
        "canvas_area_um2",
        "num_macros",
        "num_hard_macros",
        "num_soft_macros",
        "num_fixed",
        "num_movable_hard",
        "num_nets",
        "hard_macro_area_um2",
        "utilization",
        "grid_rows",
        "grid_cols",
    }
    assert expected_keys.issubset(set(stats.keys()))


def test_inspect_counts():
    bm = _make_benchmark(n_hard=4, n_fixed=1)
    stats = inspect(bm)
    assert stats["num_macros"] == 4
    assert stats["num_hard_macros"] == 4
    assert stats["num_soft_macros"] == 0
    assert stats["num_fixed"] == 1
    assert stats["num_movable_hard"] == 3
    assert stats["num_nets"] == 3


def test_inspect_name():
    bm = _make_benchmark()
    assert inspect(bm)["name"] == "test"


def test_inspect_utilization_positive():
    bm = _make_benchmark()
    assert inspect(bm)["utilization"] > 0.0
