"""Tests for core/spectral_projection.py."""

import torch
import pytest

pytest.importorskip("scipy")

from macro_place.benchmark import Benchmark
from submissions.solver.core.spectral_projection import generate_spectral_candidates


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _make_benchmark_with_nets(
    n_hard: int = 6,
    canvas: float = 100.0,
    macro_size: float = 5.0,
    net_nodes=None,
) -> Benchmark:
    positions = torch.zeros(n_hard, 2, dtype=torch.float32)
    for i in range(n_hard):
        positions[i, 0] = (i % 3) * 30.0 + 15.0
        positions[i, 1] = (i // 3) * 50.0 + 25.0
    sizes = torch.full((n_hard, 2), macro_size, dtype=torch.float32)
    fixed = torch.zeros(n_hard, dtype=torch.bool)

    if net_nodes is None:
        # Default: chain net
        nn = [torch.tensor([i, i + 1], dtype=torch.long) for i in range(n_hard - 1)]
    else:
        nn = [torch.tensor(ns, dtype=torch.long) for ns in net_nodes]

    return Benchmark(
        name="test_spectral",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=len(nn),
        net_nodes=nn,
        net_weights=torch.ones(len(nn)),
        grid_rows=8,
        grid_cols=8,
    )


# ---------------------------------------------------------------------------
# test_spectral_projection_returns_finite_coordinates
# ---------------------------------------------------------------------------


def test_spectral_projection_returns_finite_coordinates():
    bm = _make_benchmark_with_nets(n_hard=6)
    candidates = generate_spectral_candidates(bm)
    # May return empty list if spectral fails, but if it returns, coords must be finite
    for c in candidates:
        assert torch.isfinite(c.positions).all(), \
            f"Non-finite coordinates in spectral candidate '{c.name}'"


# ---------------------------------------------------------------------------
# test_spectral_projection_handles_disconnected_graph
# ---------------------------------------------------------------------------


def test_spectral_projection_handles_disconnected_graph():
    # Disconnected: two isolated pairs, no connection between them
    bm = _make_benchmark_with_nets(
        n_hard=6, net_nodes=[[0, 1], [2, 3]]  # nodes 4, 5 isolated
    )
    # Should not crash; may return fewer candidates or empty list
    try:
        candidates = generate_spectral_candidates(bm)
        # If it returns candidates, all must have finite coords
        for c in candidates:
            assert torch.isfinite(c.positions).all()
    except Exception as exc:
        pytest.fail(f"generate_spectral_candidates raised on disconnected graph: {exc}")


# ---------------------------------------------------------------------------
# test_spectral_projection_fallback_does_not_crash
# ---------------------------------------------------------------------------


def test_spectral_projection_fallback_does_not_crash():
    """Spectral failure (no nets) should return empty list, not crash."""
    bm = _make_benchmark_with_nets(n_hard=5, net_nodes=[])  # No connectivity
    try:
        candidates = generate_spectral_candidates(bm)
        # Either empty list or valid candidates
        for c in candidates:
            assert torch.isfinite(c.positions).all()
    except Exception as exc:
        pytest.fail(f"Spectral fallback should not crash: {exc}")


# ---------------------------------------------------------------------------
# test_spectral_produces_required_variants
# ---------------------------------------------------------------------------


def test_spectral_produces_required_variants():
    bm = _make_benchmark_with_nets(n_hard=8)
    candidates = generate_spectral_candidates(bm)
    if not candidates:
        pytest.skip("Spectral embedding failed — variants cannot be checked")
    names = {c.name for c in candidates}
    expected = {
        "spectral_xy", "spectral_flip_x", "spectral_flip_y", "spectral_flip_xy",
        "spectral_swap_xy", "spectral_swap_flip_x",
        "spectral_center_scale_085", "spectral_center_scale_070",
    }
    assert names == expected, f"Missing spectral variants. Got: {names}"


# ---------------------------------------------------------------------------
# test_spectral_positions_in_canvas
# ---------------------------------------------------------------------------


def test_spectral_positions_in_canvas():
    bm = _make_benchmark_with_nets(n_hard=6)
    candidates = generate_spectral_candidates(bm)
    if not candidates:
        pytest.skip("No spectral candidates")
    n_hard = bm.num_hard_macros
    canvas_w = bm.canvas_width
    canvas_h = bm.canvas_height
    for c in candidates:
        for i in range(n_hard):
            cx = c.positions[i, 0].item()
            cy = c.positions[i, 1].item()
            w_i = bm.macro_sizes[i, 0].item()
            h_i = bm.macro_sizes[i, 1].item()
            assert cx >= w_i / 2 - 1e-3, f"{c.name}: cx={cx} below left bound"
            assert cx <= canvas_w - w_i / 2 + 1e-3, f"{c.name}: cx={cx} above right bound"
            assert cy >= h_i / 2 - 1e-3, f"{c.name}: cy={cy} below bottom bound"
            assert cy <= canvas_h - h_i / 2 + 1e-3, f"{c.name}: cy={cy} above top bound"


# ---------------------------------------------------------------------------
# test_spectral_is_deterministic
# ---------------------------------------------------------------------------


def test_spectral_is_deterministic():
    bm = _make_benchmark_with_nets(n_hard=6)
    c1 = generate_spectral_candidates(bm)
    c2 = generate_spectral_candidates(bm)
    assert len(c1) == len(c2), "Spectral must return same number of candidates"
    for a, b in zip(c1, c2):
        assert torch.allclose(a.positions, b.positions, atol=1e-5), \
            f"Spectral not deterministic for '{a.name}'"
