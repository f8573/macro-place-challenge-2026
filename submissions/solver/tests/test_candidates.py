"""Tests for M2B candidate generation and selection."""

import torch
import pytest
import numpy as np

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.legalization.greedy_legalizer import legalize


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _make_benchmark(
    n_hard: int = 4,
    canvas: float = 100.0,
    macro_size: float = 10.0,
    n_nets: int = 0,
    net_nodes=None,
    fixed_mask=None,
    n_soft: int = 0,
    positions: torch.Tensor = None,
) -> Benchmark:
    """Create a minimal synthetic benchmark."""
    n_total = n_hard + n_soft
    if positions is None:
        base_positions = torch.zeros(n_total, 2, dtype=torch.float32)
        for i in range(n_hard):
            base_positions[i, 0] = (i % 4) * 20.0 + 10.0
            base_positions[i, 1] = (i // 4) * 20.0 + 10.0
        for i in range(n_hard, n_total):
            base_positions[i, 0] = 50.0
            base_positions[i, 1] = 50.0
    else:
        base_positions = positions

    sizes = torch.full((n_total, 2), macro_size, dtype=torch.float32)

    if fixed_mask is None:
        fixed = torch.zeros(n_total, dtype=torch.bool)
    else:
        fixed = torch.tensor(fixed_mask, dtype=torch.bool)

    if net_nodes is None:
        nn = []
        nw = torch.zeros(0)
    else:
        nn = [torch.tensor(ns, dtype=torch.long) for ns in net_nodes]
        nw = torch.ones(len(nn))

    return Benchmark(
        name="test",
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_total,
        num_hard_macros=n_hard,
        num_soft_macros=n_soft,
        macro_positions=base_positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_total)],
        num_nets=len(nn),
        net_nodes=nn,
        net_weights=nw,
        grid_rows=8,
        grid_cols=8,
    )


# ---------------------------------------------------------------------------
# Existing tests (updated for original_raw / original_legalized naming)
# ---------------------------------------------------------------------------


def test_original_candidate_is_always_present():
    bm = _make_benchmark()
    candidates = generate_candidates(bm)
    names = [c.name for c in candidates]
    assert "original_raw" in names, f"'original_raw' not in candidates: {names}"
    assert candidates[0].name == "original_raw", "original_raw must be first"


def test_candidate_generation_is_deterministic():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1, 2], [1, 3], [4, 5]])
    c1 = generate_candidates(bm)
    c2 = generate_candidates(bm)
    names1 = [c.name for c in c1]
    names2 = [c.name for c in c2]
    assert names1 == names2, "Candidate names are not deterministic"
    for a, b in zip(c1, c2):
        assert torch.allclose(a.positions, b.positions), f"Positions differ for {a.name}"


def test_candidate_names_are_unique():
    bm = _make_benchmark(n_hard=5, n_nets=2, net_nodes=[[0, 1], [2, 3]])
    candidates = generate_candidates(bm)
    names = [c.name for c in candidates]
    assert len(names) == len(set(names)), f"Duplicate candidate names: {names}"


def test_invalid_candidate_cannot_be_selected():
    bm = _make_benchmark(n_hard=4)
    _, ranked, _ = score_and_select(generate_candidates(bm), bm, plc=None)
    found_valid = False
    for sc in ranked:
        if sc.valid:
            found_valid = True
        else:
            assert found_valid or not any(s.valid for s in ranked), \
                "Invalid candidate ranked before valid candidates"


def test_best_candidate_is_lowest_proxy_cost():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1], [2, 3], [4, 5]])
    best, ranked, _ = score_and_select(generate_candidates(bm), bm, plc=None)
    # Selectable valid candidates (exclude diagnostic-only original_legalized when raw is valid)
    _, _, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    raw_valid = diag.raw_original_valid
    selectable = [
        s for s in ranked
        if s.valid and s.proxy_cost is not None
        and not (s.name == "original_legalized" and raw_valid)
    ]
    if not selectable:
        pytest.skip("No valid selectable candidates with costs")
    min_cost = min(s.proxy_cost for s in selectable)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - min_cost) < 1e-9, \
        f"Best cost {best.proxy_cost} != min selectable cost {min_cost}"


def test_original_candidate_used_as_fallback():
    bm = _make_benchmark(n_hard=2, canvas=20.0)
    best, ranked, _ = score_and_select(generate_candidates(bm), bm, plc=None)
    assert best is not None
    assert best.valid, "Best candidate should be valid (original_raw is always valid fallback)"


def test_at_least_three_non_original_families():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1, 2], [3, 4], [1, 5]])
    candidates = generate_candidates(bm)
    families = {c.family for c in candidates if c.family != "original"}
    assert len(families) >= 3, f"Expected 3+ non-original families, got: {families}"


def test_all_candidate_positions_finite():
    bm = _make_benchmark(n_hard=5)
    candidates = generate_candidates(bm)
    for c in candidates:
        assert torch.isfinite(c.positions).all(), \
            f"Candidate '{c.name}' has non-finite positions"


def test_legalized_candidate_is_in_bounds():
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    _, ranked, _ = score_and_select(generate_candidates(bm), bm, plc=None)
    for sc in ranked:
        if not sc.valid:
            continue
        assert sc.num_out_of_bounds == 0, \
            f"Candidate '{sc.name}' has {sc.num_out_of_bounds} out-of-bounds macros"


def test_touching_edges_are_not_overlap():
    """Two touching macros should produce a valid placement."""
    bm = _make_benchmark(n_hard=2, canvas=20.0, macro_size=2.0)
    bm.macro_positions[0] = torch.tensor([1.0, 5.0])
    bm.macro_positions[1] = torch.tensor([3.0, 5.0])
    _, ranked, _ = score_and_select(generate_candidates(bm), bm, plc=None)
    orig = next((s for s in ranked if s.name == "original_raw"), None)
    assert orig is not None
    assert orig.valid, f"Touching edges should be legal: {orig.messages}"


def test_overlapping_macros_are_repaired():
    """Overlapping input positions should be repaired by legalizer."""
    bm = _make_benchmark(n_hard=3, canvas=100.0, macro_size=5.0)
    bm.macro_positions[:] = torch.tensor([[50.0, 50.0]] * 3)
    candidates = generate_candidates(bm)
    _, ranked, _ = score_and_select(candidates, bm, plc=None)
    valid_ones = [s for s in ranked if s.valid]
    assert len(valid_ones) > 0, "Should have at least one valid legalized candidate"


# ---------------------------------------------------------------------------
# New regression tests — legalizer no-op
# ---------------------------------------------------------------------------


def test_legalizer_noops_on_valid_placement():
    """Legalizer returns positions unchanged when input has no overlaps or OOB."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    movable = bm.get_movable_mask() & bm.get_hard_macro_mask()
    result = legalize(
        positions=bm.macro_positions,
        sizes=bm.macro_sizes,
        canvas_w=bm.canvas_width,
        canvas_h=bm.canvas_height,
        movable_mask=movable,
        obstacle_mask=bm.macro_fixed & bm.get_hard_macro_mask(),
    )
    assert result.no_op, "Legalizer should be a no-op on an already-valid placement"
    assert torch.allclose(result.positions.float(), bm.macro_positions.float()), \
        "No-op legalizer must return original positions unchanged"


def test_legalizer_reports_zero_movement_on_noop():
    """No-op legalizer reports num_moved=0, max_move=0, total_move=0."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    movable = bm.get_movable_mask() & bm.get_hard_macro_mask()
    result = legalize(
        positions=bm.macro_positions,
        sizes=bm.macro_sizes,
        canvas_w=bm.canvas_width,
        canvas_h=bm.canvas_height,
        movable_mask=movable,
        obstacle_mask=bm.macro_fixed & bm.get_hard_macro_mask(),
    )
    assert result.num_moved == 0, f"Expected num_moved=0, got {result.num_moved}"
    assert result.max_move == 0.0, f"Expected max_move=0, got {result.max_move}"
    assert result.total_move == 0.0, f"Expected total_move=0, got {result.total_move}"


def test_valid_raw_original_is_selectable():
    """original_raw must appear as a valid scored candidate when its positions are valid."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    assert diag.raw_original_valid, "original_raw should be valid for a clean benchmark"
    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None, "original_raw must appear in ranked list"
    assert raw.valid, "original_raw must be valid"
    assert raw.proxy_cost is not None, "original_raw must have a proxy cost when valid"


def test_best_cost_never_exceeds_valid_raw_original_cost():
    """Invariant: best_selected.proxy_cost <= raw_original_proxy_cost when raw is valid."""
    bm = _make_benchmark(
        n_hard=6, n_nets=3, net_nodes=[[0, 1], [2, 3], [4, 5]], canvas=100.0
    )
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated: best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )


def test_original_raw_and_original_legalized_are_distinct_diagnostics():
    """Both original_raw and original_legalized must appear in the output."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    names = [s.name for s in ranked]
    assert "original_raw" in names, "original_raw must be in ranked list"
    assert "original_legalized" in names, "original_legalized must be in ranked list"
    raw = next(s for s in ranked if s.name == "original_raw")
    leg = next(s for s in ranked if s.name == "original_legalized")
    # They are distinct objects with different names
    assert raw.name != leg.name
    # raw bypasses legalization; legalized does not
    assert raw.no_op, "original_raw should have no_op=True (bypass_legalization)"


def test_invalid_raw_original_falls_back_to_legalized_original():
    """When original_raw is invalid (overlapping positions), original_legalized is selectable."""
    # All macros at same center — raw positions are invalid (overlaps)
    positions = torch.tensor([[50.0, 50.0]] * 4, dtype=torch.float32)
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, positions=positions)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)

    # original_raw should be invalid (overlaps detected in check_placement)
    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None
    assert not raw.valid, "original_raw should be invalid when positions overlap"
    assert not diag.raw_original_valid

    # original_legalized should be valid (legalizer repairs overlaps)
    leg = next((s for s in ranked if s.name == "original_legalized"), None)
    assert leg is not None
    assert leg.valid, "original_legalized should be valid after legalization"

    # Best should not be the invalid original_raw
    assert best.valid, "Best candidate must be valid"
    assert best.name != "original_raw", "Invalid original_raw must not be selected as best"


def test_soft_macros_are_not_seeded_as_obstacles():
    """Soft macros must not appear in the obstacle_mask passed to the legalizer.

    With soft macros as obstacles the legalizer would waste time avoiding clusters
    that are free to move; this also caused incorrect 'no legal slot' failures.
    """
    bm = _make_benchmark(n_hard=3, canvas=60.0, macro_size=8.0, n_soft=5)
    # Soft macros at canvas center — would block hard macros if treated as obstacles
    for i in range(3, 8):
        bm.macro_positions[i] = torch.tensor([30.0, 30.0])

    movable = bm.get_movable_mask() & bm.get_hard_macro_mask()
    obstacle = bm.macro_fixed & bm.get_hard_macro_mask()

    # obstacle_mask should contain only fixed hard macros (none here)
    assert not obstacle.any(), "No fixed hard macros in this benchmark"

    # Legalizer must succeed without considering soft macros as obstacles
    result = legalize(
        positions=bm.macro_positions,
        sizes=bm.macro_sizes,
        canvas_w=bm.canvas_width,
        canvas_h=bm.canvas_height,
        movable_mask=movable,
        obstacle_mask=obstacle,
    )
    assert result.valid, "Legalizer should find valid placement ignoring soft macros"


def test_touching_edges_with_float_tolerance_do_not_trigger_legalization():
    """Macros touching at exactly 0-separation must NOT trigger the legalizer."""
    # Two macros of size 10 touching at x-boundary: centers at x=5 and x=15
    canvas = 30.0
    size = 10.0
    positions = torch.tensor([[5.0, 15.0], [15.0, 15.0]], dtype=torch.float32)
    bm = _make_benchmark(n_hard=2, canvas=canvas, macro_size=size, positions=positions)

    movable = bm.get_movable_mask() & bm.get_hard_macro_mask()
    result = legalize(
        positions=bm.macro_positions,
        sizes=bm.macro_sizes,
        canvas_w=bm.canvas_width,
        canvas_h=bm.canvas_height,
        movable_mask=movable,
        obstacle_mask=bm.macro_fixed & bm.get_hard_macro_mask(),
    )
    assert result.no_op, (
        "Touching-edge placement should be detected as already valid "
        "and trigger the legalizer no-op shortcut"
    )
    assert result.num_moved == 0
