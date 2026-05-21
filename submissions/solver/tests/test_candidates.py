"""Tests for M2B candidate generation and selection."""

import torch
import pytest

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


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
) -> Benchmark:
    """Create a minimal synthetic benchmark."""
    positions = torch.zeros(n_hard, 2, dtype=torch.float32)
    # Place macros in a grid-like pattern
    for i in range(n_hard):
        positions[i, 0] = (i % 4) * 20.0 + 10.0
        positions[i, 1] = (i // 4) * 20.0 + 10.0

    sizes = torch.full((n_hard, 2), macro_size, dtype=torch.float32)

    if fixed_mask is None:
        fixed = torch.zeros(n_hard, dtype=torch.bool)
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
        num_macros=n_hard,
        num_hard_macros=n_hard,
        num_soft_macros=0,
        macro_positions=positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_hard)],
        num_nets=len(nn),
        net_nodes=nn,
        net_weights=nw,
        grid_rows=8,
        grid_cols=8,
    )


# ---------------------------------------------------------------------------
# test_original_candidate_is_always_present
# ---------------------------------------------------------------------------


def test_original_candidate_is_always_present():
    bm = _make_benchmark()
    candidates = generate_candidates(bm)
    names = [c.name for c in candidates]
    assert "original" in names, f"'original' not in candidates: {names}"
    assert candidates[0].name == "original", "original must be first"


# ---------------------------------------------------------------------------
# test_candidate_generation_is_deterministic
# ---------------------------------------------------------------------------


def test_candidate_generation_is_deterministic():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1, 2], [1, 3], [4, 5]])
    c1 = generate_candidates(bm)
    c2 = generate_candidates(bm)
    names1 = [c.name for c in c1]
    names2 = [c.name for c in c2]
    assert names1 == names2, "Candidate names are not deterministic"
    for a, b in zip(c1, c2):
        assert torch.allclose(a.positions, b.positions), f"Positions differ for {a.name}"


# ---------------------------------------------------------------------------
# test_candidate_names_are_unique
# ---------------------------------------------------------------------------


def test_candidate_names_are_unique():
    bm = _make_benchmark(n_hard=5, n_nets=2, net_nodes=[[0, 1], [2, 3]])
    candidates = generate_candidates(bm)
    names = [c.name for c in candidates]
    assert len(names) == len(set(names)), f"Duplicate candidate names: {names}"


# ---------------------------------------------------------------------------
# test_invalid_candidate_cannot_be_selected
# ---------------------------------------------------------------------------


def test_invalid_candidate_cannot_be_selected():
    bm = _make_benchmark(n_hard=4)
    _, ranked = score_and_select(generate_candidates(bm), bm, plc=None)
    # Best is always the first valid one; no invalid should appear before valid ones
    found_valid = False
    for sc in ranked:
        if sc.valid:
            found_valid = True
        else:
            # All invalid candidates should come after valid ones in ranked list
            assert found_valid or not any(s.valid for s in ranked), \
                "Invalid candidate ranked before valid candidates"


# ---------------------------------------------------------------------------
# test_best_candidate_is_lowest_proxy_cost
# ---------------------------------------------------------------------------


def test_best_candidate_is_lowest_proxy_cost():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1], [2, 3], [4, 5]])
    best, ranked = score_and_select(generate_candidates(bm), bm, plc=None)
    valid_costs = [s.proxy_cost for s in ranked if s.valid and s.proxy_cost is not None]
    if not valid_costs:
        pytest.skip("No valid candidates with costs")
    min_cost = min(valid_costs)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - min_cost) < 1e-9, \
        f"Best cost {best.proxy_cost} != min cost {min_cost}"


# ---------------------------------------------------------------------------
# test_original_candidate_used_as_fallback
# ---------------------------------------------------------------------------


def test_original_candidate_used_as_fallback():
    # A benchmark so constrained no generated candidate is better — but original valid
    bm = _make_benchmark(n_hard=2, canvas=20.0)
    best, ranked = score_and_select(generate_candidates(bm), bm, plc=None)
    # Since original is always valid, best should always be some valid candidate
    assert best is not None
    assert best.valid, "Best candidate should be valid (original is always valid fallback)"


# ---------------------------------------------------------------------------
# test_at_least_three_non_original_families
# ---------------------------------------------------------------------------


def test_at_least_three_non_original_families():
    bm = _make_benchmark(n_hard=6, n_nets=3, net_nodes=[[0, 1, 2], [3, 4], [1, 5]])
    candidates = generate_candidates(bm)
    families = {c.family for c in candidates if c.name != "original"}
    assert len(families) >= 3, f"Expected 3+ non-original families, got: {families}"


# ---------------------------------------------------------------------------
# test_all_candidate_positions_finite
# ---------------------------------------------------------------------------


def test_all_candidate_positions_finite():
    bm = _make_benchmark(n_hard=5)
    candidates = generate_candidates(bm)
    for c in candidates:
        assert torch.isfinite(c.positions).all(), \
            f"Candidate '{c.name}' has non-finite positions"


# ---------------------------------------------------------------------------
# test_legalized_candidate_is_in_bounds
# ---------------------------------------------------------------------------


def test_legalized_candidate_is_in_bounds():
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    _, ranked = score_and_select(generate_candidates(bm), bm, plc=None)
    for sc in ranked:
        if not sc.valid:
            continue
        # All legalized valid candidates must be in bounds
        assert sc.num_out_of_bounds == 0, \
            f"Candidate '{sc.name}' has {sc.num_out_of_bounds} out-of-bounds macros"


# ---------------------------------------------------------------------------
# test_touching_edges_are_not_overlap (M2B boundary)
# ---------------------------------------------------------------------------


def test_touching_edges_are_not_overlap():
    """Two touching macros should produce a valid placement."""
    bm = _make_benchmark(n_hard=2, canvas=20.0, macro_size=2.0)
    # Position them touching at x=2: centers at (1,5) and (3,5)
    bm.macro_positions[0] = torch.tensor([1.0, 5.0])
    bm.macro_positions[1] = torch.tensor([3.0, 5.0])
    _, ranked = score_and_select(generate_candidates(bm), bm, plc=None)
    # The original candidate should be valid (touching edges allowed)
    orig = next((s for s in ranked if s.name == "original"), None)
    assert orig is not None
    assert orig.valid, f"Touching edges should be legal: {orig.messages}"


# ---------------------------------------------------------------------------
# test_overlapping_macros_are_repaired
# ---------------------------------------------------------------------------


def test_overlapping_macros_are_repaired():
    """Overlapping input positions should be repaired by legalizer."""
    bm = _make_benchmark(n_hard=3, canvas=100.0, macro_size=5.0)
    # All macros at same center — clearly overlapping
    bm.macro_positions[:] = torch.tensor([[50.0, 50.0]] * 3)
    candidates = generate_candidates(bm)
    _, ranked = score_and_select(candidates, bm, plc=None)
    valid_ones = [s for s in ranked if s.valid]
    assert len(valid_ones) > 0, "Should have at least one valid legalized candidate"
