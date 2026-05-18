"""Unit tests for core/scoring.py."""

import torch
import pytest

from submissions.solver.core.scoring import score
from macro_place.benchmark import Benchmark


def _make_benchmark():
    n = 2
    positions = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    sizes = torch.ones(n, 2) * 2.0
    return Benchmark(
        name="test",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=n,
        num_hard_macros=n,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=torch.zeros(n, dtype=torch.bool),
        macro_names=["m0", "m1"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=4,
        grid_cols=4,
    )


def test_score_none_plc_returns_none():
    bm = _make_benchmark()
    placement = bm.macro_positions.clone()
    result = score(placement, bm, plc=None)
    assert result is None


def test_score_none_plc_does_not_raise():
    bm = _make_benchmark()
    placement = bm.macro_positions.clone()
    # Should complete without error even when plc is unavailable
    try:
        score(placement, bm, plc=None)
    except Exception as e:
        pytest.fail(f"score() raised unexpectedly with plc=None: {e}")
