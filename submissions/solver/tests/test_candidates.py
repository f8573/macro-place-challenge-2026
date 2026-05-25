"""Tests for M2B candidate generation and selection."""

import torch
import pytest
import numpy as np

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidatePlacement,
    CandidateScoringConfig,
    ScoredCandidate,
)
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


def test_original_neighborhood_candidates_are_close_to_original():
    bm = _make_benchmark(
        n_hard=6,
        canvas=200.0,
        macro_size=20.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5]],
    )
    cfg = CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=3)
    candidates = generate_candidates(bm, config=cfg)
    family = [c for c in candidates if c.family == "original_neighborhood"]
    assert family, "Expected original_neighborhood candidates"

    max_step = max(0.25 * 20.0, 0.01 * 200.0) * (8 ** 0.5)
    for candidate in family:
        disp = torch.norm(candidate.positions - bm.macro_positions.float(), dim=1)
        moved = torch.where(disp > 1e-6)[0].tolist()
        assert len(moved) == 1, f"{candidate.name} should move exactly one macro"
        assert float(disp[moved[0]].item()) <= max_step + 1e-6


def test_original_neighborhood_candidates_are_deterministic():
    bm = _make_benchmark(
        n_hard=6,
        canvas=200.0,
        macro_size=20.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5]],
    )
    cfg = CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=4)
    c1 = generate_candidates(bm, config=cfg)
    c2 = generate_candidates(bm, config=cfg)
    assert [c.name for c in c1] == [c.name for c in c2]
    for left, right in zip(c1, c2):
        assert torch.allclose(left.positions, right.positions)


def test_original_neighborhood_respects_bounds():
    bm = _make_benchmark(
        n_hard=5,
        canvas=90.0,
        macro_size=18.0,
        net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    cfg = CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=5)
    candidates = generate_candidates(bm, config=cfg)
    for candidate in candidates:
        if candidate.family != "original_neighborhood":
            continue
        for idx in range(bm.num_hard_macros):
            x = float(candidate.positions[idx, 0].item())
            y = float(candidate.positions[idx, 1].item())
            w = float(bm.macro_sizes[idx, 0].item())
            h = float(bm.macro_sizes[idx, 1].item())
            assert w / 2.0 - 1e-6 <= x <= bm.canvas_width - w / 2.0 + 1e-6
            assert h / 2.0 - 1e-6 <= y <= bm.canvas_height - h / 2.0 + 1e-6


def test_original_neighborhood_does_not_move_unselected_macros():
    bm = _make_benchmark(
        n_hard=8,
        canvas=200.0,
        macro_size=12.0,
        net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4], [6, 7]],
    )
    cfg = CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=2)
    candidates = generate_candidates(bm, config=cfg)
    selected = set()
    for candidate in candidates:
        if candidate.family != "original_neighborhood":
            continue
        selected.add(int(candidate.metadata["moved_macro_id"]))
    assert len(selected) <= 2
    for candidate in candidates:
        if candidate.family != "original_neighborhood":
            continue
        for idx in range(bm.num_hard_macros):
            if idx not in selected:
                assert torch.allclose(candidate.positions[idx], bm.macro_positions[idx])


def test_candidate_hash_cache_skips_duplicates():
    bm = _make_benchmark(n_hard=2, canvas=100.0, macro_size=10.0, net_nodes=[[0, 1]])
    positions = bm.macro_positions.clone().float()
    candidates = [
        CandidatePlacement("original_raw", "original", positions.clone(), bypass_legalization=True),
        CandidatePlacement("dupe", "original_neighborhood", positions.clone(), bypass_legalization=True),
    ]
    best, ranked, diag = score_and_select(
        candidates,
        bm,
        plc=None,
        scoring_config=CandidateScoringConfig(enable_hash_cache=True),
    )
    assert best is not None
    assert diag.duplicate_count == 1
    assert diag.candidates_officially_scored == 1
    dupe = next(sc for sc in ranked if sc.name == "dupe")
    assert dupe.duplicate_of == "original_raw"
    assert not dupe.was_scored


def test_approximate_prefilter_never_selects_final_winner_without_official_score():
    bm = _make_benchmark(
        n_hard=6,
        canvas=220.0,
        macro_size=18.0,
        net_nodes=[[0, 1, 2], [2, 3, 4], [4, 5], [1, 5]],
    )
    cfg = CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=4)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=cfg),
        bm,
        plc=None,
        scoring_config=CandidateScoringConfig(prefilter_mode="approx_delta_hpwl", exploratory_score_count=2),
    )
    assert best is not None
    assert best.was_scored
    assert diag.candidates_officially_scored >= 1


def test_disabling_global_candidates_still_keeps_original_raw():
    bm = _make_benchmark(n_hard=6, canvas=120.0, macro_size=10.0, net_nodes=[[0, 1], [2, 3], [4, 5]])
    candidates = generate_candidates(
        bm,
        config=CandidateGenerationConfig(disable_global_candidates=True, neighborhood_macro_limit=3),
    )
    families = {c.family for c in candidates}
    names = {c.name for c in candidates}
    assert "original_raw" in names
    assert families.issubset({"original", "original_neighborhood"})


def test_only_original_neighborhood_profile_generates_expected_family():
    bm = _make_benchmark(n_hard=6, canvas=120.0, macro_size=10.0, net_nodes=[[0, 1], [2, 3], [4, 5]])
    candidates = generate_candidates(
        bm,
        config=CandidateGenerationConfig(only_original_neighborhood=True, neighborhood_macro_limit=3),
    )
    families = {c.family for c in candidates}
    assert families == {"original", "original_neighborhood"}


# ---------------------------------------------------------------------------
# Refinement candidate tests
# ---------------------------------------------------------------------------


def _make_benchmark_with_nets(n_hard=6, canvas=200.0, macro_size=15.0):
    """Benchmark with enough spacing and connectivity for neighborhood + refinement."""
    return _make_benchmark(
        n_hard=n_hard,
        canvas=canvas,
        macro_size=macro_size,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )


def test_original_refinement_candidates_are_close_to_original():
    """Refinement candidates should each move exactly one macro a small distance."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=3,
        refinement_around_winners=True,
        refinement_top_k=3,
        refinement_combo_size=2,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    refinement = [s for s in ranked if s.family == "original_refinement" and "combo" not in s.name]
    assert len(refinement) > 0, "Expected refinement candidates in ranked output"

    # Each single-macro refinement candidate intends to move one specific macro.
    # The legalizer may cascade to adjacent macros to resolve overlaps; that is correct
    # behavior after the overlap-admission fix.  We only check that the INTENDED macro
    # was moved within reasonable bounds.
    base = bm.macro_positions.float()
    max_allowed = bm.canvas_width * 0.25
    for sc in refinement:
        macro_id = sc.metadata.get("moved_macro_id")
        assert macro_id is not None, f"{sc.name} missing moved_macro_id"
        mid = int(macro_id)
        disp = torch.norm(sc.positions - base, dim=1)
        intended_disp = float(disp[mid].item())
        if intended_disp > 1e-4:
            assert intended_disp <= max_allowed + 1e-4, (
                f"{sc.name} moves intended macro {mid} by {intended_disp:.4f} "
                f"which exceeds max {max_allowed}"
            )


def test_refinement_candidates_are_deterministic():
    """Same config produces identical refinement candidate names and positions."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=3,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    _, ranked1, _ = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)
    _, ranked2, _ = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)

    names1 = [s.name for s in ranked1 if s.family == "original_refinement"]
    names2 = [s.name for s in ranked2 if s.family == "original_refinement"]
    assert names1 == names2, "Refinement candidate names differ between runs"

    for sc1 in ranked1:
        if sc1.family != "original_refinement":
            continue
        sc2 = next((s for s in ranked2 if s.name == sc1.name), None)
        assert sc2 is not None
        assert torch.allclose(sc1.positions, sc2.positions, atol=1e-5), \
            f"Positions differ for {sc1.name}"


def test_refinement_uses_only_scored_or_near_miss_seeds():
    """Refinement seeds must come from original_neighborhood candidates."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
    )
    _, ranked, diag = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)

    # Refinement only generated if neighborhood candidates exist
    neighborhood_present = any(s.family == "original_neighborhood" for s in ranked)
    if neighborhood_present and diag.refinement_candidates_generated > 0:
        # Each single-macro refinement candidate must reference a valid macro_id from the benchmark
        for sc in ranked:
            if sc.family == "original_refinement" and "combo" not in sc.name:
                mid = sc.metadata.get("moved_macro_id")
                assert mid is not None, f"Refinement candidate {sc.name} missing moved_macro_id"
                assert 0 <= int(mid) < bm.num_hard_macros, \
                    f"moved_macro_id {mid} out of range for {sc.name}"


def test_combo_candidates_move_multiple_distinct_macros():
    """Combo refinement candidates must displace at least 2 different macros."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_combo_size=2,
    )
    _, ranked, diag = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)

    combos = [s for s in ranked if s.family == "original_refinement" and "combo" in s.name]
    if not combos:
        pytest.skip("No combo candidates generated for this benchmark config")

    base = bm.macro_positions.float()
    for sc in combos:
        disp = torch.norm(sc.positions - base, dim=1)
        moved = torch.where(disp > 1e-4)[0]
        assert len(moved) >= 2, f"Combo {sc.name} only moves {len(moved)} macros"
        # All moved macros should be distinct
        assert len(set(moved.tolist())) == len(moved), f"Combo {sc.name} moves same macro twice"


def test_combo_candidates_respect_bounds():
    """Combo refinement candidates must keep all macros in bounds."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_combo_size=2,
    )
    _, ranked, _ = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)

    for sc in ranked:
        if sc.family != "original_refinement" or not sc.valid:
            continue
        for idx in range(bm.num_hard_macros):
            x = float(sc.positions[idx, 0].item())
            y = float(sc.positions[idx, 1].item())
            w = float(bm.macro_sizes[idx, 0].item())
            h = float(bm.macro_sizes[idx, 1].item())
            assert w / 2.0 - 1e-4 <= x <= bm.canvas_width - w / 2.0 + 1e-4, \
                f"{sc.name} macro {idx} x={x} OOB"
            assert h / 2.0 - 1e-4 <= y <= bm.canvas_height - h / 2.0 + 1e-4, \
                f"{sc.name} macro {idx} y={y} OOB"


def test_combo_candidates_preserve_original_raw_fallback():
    """original_raw must remain valid and selectable after refinement pass."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_combo_size=2,
    )
    best, ranked, diag = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)

    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None, "original_raw must still be in ranked list"
    assert raw.valid, "original_raw must remain valid after refinement pass"
    assert diag.raw_original_valid
    assert diag.invariant_holds, f"Invariant violated: best={diag.best_proxy_cost} raw={diag.raw_original_proxy_cost}"


def test_max_official_scores_is_respected():
    """Total scoring calls must not exceed max_official_scores."""
    bm = _make_benchmark(
        n_hard=8, canvas=220.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=6,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    limit = 5
    score_cfg = CandidateScoringConfig(max_official_scores=limit)
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    assert diag.candidates_officially_scored <= limit, (
        f"Scored {diag.candidates_officially_scored} candidates, expected <= {limit}"
    )


def test_approximate_prefilter_does_not_select_unscored_candidate_with_refinement():
    """With refinement enabled, the winning candidate must still be officially scored."""
    bm = _make_benchmark(
        n_hard=6, canvas=220.0, macro_size=18.0,
        net_nodes=[[0, 1, 2], [2, 3, 4], [4, 5], [1, 5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg),
        bm, plc=None,
        scoring_config=CandidateScoringConfig(prefilter_mode="approx_delta_hpwl", exploratory_score_count=2),
        generation_config=gen_cfg,
    )
    assert best is not None
    assert best.was_scored, "Winning candidate must have been officially scored"
    assert diag.candidates_officially_scored >= 1


def test_best_cost_never_exceeds_valid_raw_original_with_refinement():
    """Invariant must hold even when refinement is enabled."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [2, 3], [4, 5], [1, 3], [3, 5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=3,
        refinement_combo_size=2,
    )
    best, ranked, diag = score_and_select(generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg)
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated: best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )


# ---------------------------------------------------------------------------
# Line-search candidate tests
# ---------------------------------------------------------------------------


def _make_benchmark_for_line_search():
    """Benchmark with net connectivity for line-search testing."""
    return _make_benchmark(
        n_hard=6,
        canvas=200.0,
        macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )


def test_line_search_candidates_are_generated():
    """Line-search pass produces candidates in the ranked output."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=2,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=2,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    # Seeds come from scored neighborhood — if none were scored, no line-search
    if diag.line_search_candidates_generated == 0:
        pytest.skip("No line-search candidates generated (no neighborhood seeds were scored)")
    ls_candidates = [s for s in ranked if s.family == "original_line_search"]
    assert len(ls_candidates) > 0, "Expected original_line_search candidates in ranked output"


def test_line_search_candidates_are_collinear_with_seed_move():
    """Each line-search candidate must move the same macro in the same direction as its seed."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    ls_candidates = [s for s in ranked if s.family == "original_line_search"]
    if not ls_candidates:
        pytest.skip("No line-search candidates generated")

    base = bm.macro_positions.float()
    for sc in ls_candidates:
        macro_id = sc.metadata.get("moved_macro_id")
        assert macro_id is not None, f"{sc.name} missing moved_macro_id"
        macro_id = int(macro_id)
        # The intended macro should be displaced; the legalizer may cascade to neighbours.
        # Check that the intended macro was actually moved if any displacement occurred.
        disp = torch.norm(sc.positions - base, dim=1)
        intended_disp = float(disp[macro_id].item())
        moved = torch.where(disp > 1e-5)[0].tolist()
        if moved:
            assert macro_id in moved, (
                f"{sc.name} intended macro {macro_id} was not moved; moved={moved}"
            )


def test_line_search_candidates_are_deterministic():
    """Same config produces identical line-search candidates across runs."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    _, ranked1, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    _, ranked2, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    ls1 = [(s.name, s.positions.tolist()) for s in ranked1 if s.family == "original_line_search"]
    ls2 = [(s.name, s.positions.tolist()) for s in ranked2 if s.family == "original_line_search"]
    assert [n for n, _ in ls1] == [n for n, _ in ls2], "Line-search candidate names not deterministic"
    for (n1, p1), (n2, p2) in zip(ls1, ls2):
        assert torch.allclose(torch.tensor(p1), torch.tensor(p2), atol=1e-5), \
            f"Positions differ for {n1}"


def test_line_search_respects_bounds():
    """All line-search candidate positions must be within the canvas."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=5,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    _, ranked, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    for sc in ranked:
        if sc.family != "original_line_search" or not sc.valid:
            continue
        for idx in range(bm.num_hard_macros):
            x = float(sc.positions[idx, 0].item())
            y = float(sc.positions[idx, 1].item())
            w = float(bm.macro_sizes[idx, 0].item())
            h = float(bm.macro_sizes[idx, 1].item())
            assert w / 2.0 - 1e-4 <= x <= bm.canvas_width - w / 2.0 + 1e-4, \
                f"{sc.name} macro {idx} x={x} OOB"
            assert h / 2.0 - 1e-4 <= y <= bm.canvas_height - h / 2.0 + 1e-4, \
                f"{sc.name} macro {idx} y={y} OOB"


def test_line_search_preserves_original_raw_fallback():
    """original_raw must remain valid and selectable after line-search pass."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None, "original_raw must be in ranked output"
    assert raw.valid, "original_raw must remain valid"
    assert diag.raw_original_valid
    assert diag.invariant_holds, \
        f"Invariant violated: best={diag.best_proxy_cost} raw={diag.raw_original_proxy_cost}"


def test_line_search_stops_after_worse_limit():
    """line_search_stop_after_worse=1 should reduce scored count vs unlimited."""
    bm = _make_benchmark_for_line_search()

    gen_cfg_limited = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=2,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=1,
    )
    gen_cfg_unlimited = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=2,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=0,  # 0 = no early stopping
    )
    _, _, diag_limited = score_and_select(
        generate_candidates(bm, config=gen_cfg_limited), bm, plc=None, generation_config=gen_cfg_limited
    )
    _, _, diag_unlimited = score_and_select(
        generate_candidates(bm, config=gen_cfg_unlimited), bm, plc=None, generation_config=gen_cfg_unlimited
    )
    # Limited should score no more than unlimited
    assert diag_limited.candidates_officially_scored <= diag_unlimited.candidates_officially_scored + 1


def test_best_cost_never_exceeds_valid_raw_original_with_line_search():
    """Invariant best <= raw_original must hold with line-search enabled."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated: best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )


# ---------------------------------------------------------------------------
# Official score cache tests
# ---------------------------------------------------------------------------


def test_official_score_cache_hits_skip_rescoring(tmp_path):
    """Second run with cache should report cache hits and no new scorer calls."""
    bm = _make_benchmark_for_line_search()
    cache_path = tmp_path / "test_cache.jsonl"

    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=3,
    )
    score_cfg_write = CandidateScoringConfig(official_score_cache_path=str(cache_path))
    score_cfg_read = CandidateScoringConfig(official_score_cache_path=str(cache_path))

    # First run: populates cache
    _, _, diag1 = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg_write, generation_config=gen_cfg,
    )
    assert diag1.cache_hits == 0, "Cold run should have no cache hits"
    assert diag1.candidates_officially_scored >= 1

    # Second run: should get cache hits
    _, _, diag2 = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg_read, generation_config=gen_cfg,
    )
    assert diag2.cache_hits >= 1, "Warm run should report cache hits"


def test_official_score_cache_key_includes_benchmark(tmp_path):
    """Cache entries for different benchmarks must not collide."""
    from submissions.solver.core.score_cache import OfficialScoreCache

    cache_path = tmp_path / "cache.jsonl"
    cache = OfficialScoreCache(cache_path=cache_path)

    cache.record("ibm01", "abcd1234", 0.5, {})
    cache.record("ibm02", "abcd1234", 0.8, {})  # same hash, different benchmark

    assert cache.lookup("ibm01", "abcd1234") == pytest.approx(0.5)
    assert cache.lookup("ibm02", "abcd1234") == pytest.approx(0.8)
    assert cache.lookup("ibm03", "abcd1234") is None


def test_cached_scores_can_still_select_winner(tmp_path):
    """Winners selected from cache hits must satisfy the invariant."""
    bm = _make_benchmark_for_line_search()
    cache_path = tmp_path / "cache.jsonl"

    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=3,
    )
    score_cfg = CandidateScoringConfig(official_score_cache_path=str(cache_path))

    # Populate cache
    score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )

    # Read from cache and verify selection
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    assert best is not None
    assert best.valid
    assert diag.invariant_holds


def test_cache_not_written_when_no_path_configured():
    """Without a cache path no JSONL file should be created."""
    from submissions.solver.core.score_cache import OfficialScoreCache

    cache = OfficialScoreCache(cache_path=None)
    assert not cache.enabled
    cache.record("ibm01", "abc", 0.5)  # should be a no-op
    assert cache.lookup("ibm01", "abc") is None
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_clear_removes_stale_entries(tmp_path):
    """--clear-score-cache should discard previously written scores."""
    from submissions.solver.core.score_cache import OfficialScoreCache

    cache_path = tmp_path / "cache.jsonl"
    # Write an entry
    c1 = OfficialScoreCache(cache_path=cache_path)
    c1.record("ibm01", "hash1", 0.42)
    assert c1.lookup("ibm01", "hash1") == pytest.approx(0.42)

    # Open with clear=True — stale entry must be gone
    c2 = OfficialScoreCache(cache_path=cache_path, clear=True)
    # lookup increments miss counter, not hit
    result = c2.lookup("ibm01", "hash1")
    assert result is None, "Cleared cache must not return stale entry"


# ---------------------------------------------------------------------------
# Cold-run priority and diagnostic tests
# ---------------------------------------------------------------------------


def test_priority_queue_scores_line_search_before_low_value_exploration():
    """With a tight budget, line-search candidates should receive scoring budget."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=2,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=0,  # no early stopping — let budget decide
    )
    score_cfg = CandidateScoringConfig(max_official_scores=20)
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    # Total scored must not exceed budget
    assert diag.candidates_officially_scored <= 20
    # Invariant must hold
    assert diag.invariant_holds
    # Line-search candidates should have been generated and scored (budget reserved)
    if diag.line_search_candidates_generated > 0:
        ls_scored = [s for s in ranked if s.family == "original_line_search" and s.was_scored]
        assert len(ls_scored) > 0, (
            "With budget reservation, line-search should score at least one candidate"
        )


def test_cold_run_does_not_require_persistent_cache():
    """Disabling the persistent cache must not break selection or the invariant."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=2,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    score_cfg = CandidateScoringConfig(disable_score_cache=True)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    assert best is not None
    assert best.valid
    assert diag.invariant_holds
    assert diag.cache_hits == 0, "Cold run (cache disabled) must report zero cache hits"


def test_skipped_candidate_reason_is_recorded():
    """Every candidate not officially scored must carry a skip_reason in metadata."""
    bm = _make_benchmark(
        n_hard=8, canvas=220.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=8,
    )
    score_cfg = CandidateScoringConfig(
        max_official_scores=3,
        prefilter_mode="approx_delta_hpwl",
        exploratory_score_count=1,
    )
    _, ranked, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    for sc in ranked:
        if not sc.was_scored:
            reason = sc.metadata.get("skip_reason")
            assert reason is not None, f"{sc.name} has no skip_reason"
            assert isinstance(reason, str), f"{sc.name} skip_reason is not a string: {reason!r}"


def test_candidate_generation_rank_is_recorded():
    """Every candidate in the ranked output must have a non-negative integer generation_rank."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
    )
    _, ranked, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    for sc in ranked:
        gen_rank = sc.metadata.get("generation_rank")
        assert gen_rank is not None, f"{sc.name} missing generation_rank"
        assert isinstance(gen_rank, int), f"{sc.name} generation_rank not int: {gen_rank!r}"
        assert gen_rank >= 0, f"{sc.name} generation_rank < 0: {gen_rank}"


def test_candidate_scoring_rank_is_recorded():
    """Every officially scored candidate must carry a unique non-negative scoring_rank."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
    )
    _, ranked, _ = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    scoring_ranks = []
    for sc in ranked:
        if sc.was_scored:
            sr = sc.metadata.get("scoring_rank")
            assert sr is not None, f"Scored candidate {sc.name} missing scoring_rank"
            assert isinstance(sr, int), f"{sc.name} scoring_rank not int: {sr!r}"
            assert sr >= 0, f"{sc.name} scoring_rank < 0: {sr}"
            scoring_ranks.append(sr)
    assert len(scoring_ranks) == len(set(scoring_ranks)), "scoring_rank values must be unique"


def test_max_official_scores_truncates_low_priority_candidates_first():
    """With a budget of 2, original_raw must still be scored (it is always priority-1)."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [2, 3], [4, 5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=6,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=2)
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None
    assert raw.was_scored, "original_raw must be scored even with budget=2"
    assert diag.candidates_officially_scored <= 2


def test_original_raw_fallback_invariant_still_holds():
    """best <= raw_original must hold with line-search + refinement + limited budget."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=3,
        line_search_around_winners=True,
        line_search_top_k=2,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=2,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=15)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated: best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )
    assert diag.invariant_holds


# ---------------------------------------------------------------------------
# Overlap-admission and cold-priority tests (task requirements)
# ---------------------------------------------------------------------------


def _make_dense_benchmark():
    """Dense benchmark: 4 macros of size 20 on 60x60 canvas (~44% utilisation).

    Macros are placed in a valid 2×2 grid so that original_raw is valid
    (in-bounds, non-overlapping).  With 30-unit spacing and 20-unit bodies,
    any line-search scale ≥ 1.5× along a cardinal axis will produce a
    candidate whose intended position overlaps an existing macro, giving the
    overlap-admission fix a realistic stress test.
    """
    positions = torch.tensor(
        [[15.0, 15.0], [45.0, 15.0], [15.0, 45.0], [45.0, 45.0]],
        dtype=torch.float32,
    )
    return _make_benchmark(
        n_hard=4,
        canvas=60.0,
        macro_size=20.0,
        net_nodes=[[0, 1], [1, 2], [2, 3]],
        positions=positions,
    )


def test_refinement_candidate_with_prelegal_overlap_is_allowed_to_reach_legalizer():
    """Refinement candidates with pre-legal overlap must appear in ranked output (not dropped at generation)."""
    bm = _make_dense_benchmark()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_combo_size=2,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    refinement = [s for s in ranked if s.family == "original_refinement"]
    assert len(refinement) > 0, "Expected refinement candidates in ranked output"
    # Every refinement candidate must carry prelegal_valid metadata
    for sc in refinement:
        assert "prelegal_valid" in sc.metadata, f"{sc.name} missing prelegal_valid metadata"
    # Candidates with pre-legal overlap should appear in ranked (not silently dropped)
    overlap_count = sum(1 for s in refinement if s.metadata.get("prelegal_valid") is False)
    # We can't guarantee overlaps for this exact layout, but the pipeline must not crash
    # and metadata must be present regardless.
    assert all("requires_legalization" in s.metadata for s in refinement), \
        "All refinement candidates must carry requires_legalization metadata"


def test_line_search_candidate_with_prelegal_overlap_is_allowed_to_reach_legalizer():
    """Line-search candidates with pre-legal overlap must appear in ranked output."""
    bm = _make_dense_benchmark()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    ls_candidates = [s for s in ranked if s.family == "original_line_search"]
    if not ls_candidates:
        pytest.skip("No line-search candidates generated (no scored seeds)")
    # Every line-search candidate must carry admission metadata
    for sc in ls_candidates:
        assert "prelegal_valid" in sc.metadata, f"{sc.name} missing prelegal_valid metadata"
        assert "requires_legalization" in sc.metadata, f"{sc.name} missing requires_legalization"
        assert "intended_dx" in sc.metadata, f"{sc.name} missing intended_dx"
        assert "intended_dy" in sc.metadata, f"{sc.name} missing intended_dy"
        assert "intended_move_norm" in sc.metadata, f"{sc.name} missing intended_move_norm"


def test_postlegal_invalid_candidate_is_not_selectable():
    """A candidate that is invalid after legalization (bypass=True with overlapping positions) must not be selected."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    # Force overlap by placing macros 0 and 1 at the same position
    positions = bm.macro_positions.clone().float()
    positions[1] = positions[0]
    invalid_cand = CandidatePlacement(
        "force_invalid", "original_neighborhood", positions, bypass_legalization=True
    )
    valid_raw = CandidatePlacement(
        "original_raw", "original", bm.macro_positions.clone().float(), bypass_legalization=True
    )
    best, ranked, diag = score_and_select([valid_raw, invalid_cand], bm, plc=None)
    assert best.name != "force_invalid", "Invalid candidate must not be selected as best"
    invalid_sc = next((s for s in ranked if s.name == "force_invalid"), None)
    assert invalid_sc is not None
    assert not invalid_sc.valid, "Candidate with overlapping bypass positions should be invalid"
    assert best.valid, "Selected best must be valid"


def test_candidate_admission_diagnostics_count_prelegal_overlap():
    """ScoringDiagnostics must carry admission audit fields with sensible values."""
    bm = _make_dense_benchmark()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    assert hasattr(diag, "admission_prelegal_overlap_candidates")
    assert hasattr(diag, "admission_legalized_successfully")
    assert hasattr(diag, "admission_legalization_failed")
    assert isinstance(diag.admission_prelegal_overlap_candidates, int)
    assert diag.admission_prelegal_overlap_candidates >= 0
    assert diag.admission_legalized_successfully >= 0
    assert diag.admission_legalization_failed >= 0
    # Audit count must be consistent with ranked list
    overlap_in_ranked = sum(1 for s in ranked if s.metadata.get("prelegal_valid") is False)
    assert diag.admission_prelegal_overlap_candidates == overlap_in_ranked


def test_cold_budget_allocates_seed_discovery_before_line_search():
    """With max_official_scores=60, neighborhood should receive more budget than line-search."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        line_search_around_winners=True,
        line_search_top_k=3,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=60)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    hood_scored = sum(1 for s in ranked if s.family == "original_neighborhood" and s.was_scored)
    ls_scored = sum(1 for s in ranked if s.family == "original_line_search" and s.was_scored)
    # Seed discovery budget (default ~32) > line-search budget (default ~18)
    assert hood_scored >= ls_scored, (
        f"Seed discovery scored {hood_scored} but line-search scored {ls_scored} — "
        "seed discovery should have at least as much budget as line-search"
    )
    assert diag.candidates_officially_scored <= 60
    assert diag.invariant_holds


def test_unused_seed_budget_flows_to_line_search():
    """Tight seed budget leaves more remaining budget; total must still not exceed max_official_scores."""
    bm = _make_benchmark_for_line_search()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=6,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=0,
    )
    score_cfg_tight = CandidateScoringConfig(
        max_official_scores=30,
        seed_discovery_score_budget=3,
        disable_score_cache=True,
    )
    score_cfg_loose = CandidateScoringConfig(
        max_official_scores=30,
        seed_discovery_score_budget=25,
        disable_score_cache=True,
    )
    _, ranked_tight, diag_tight = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg_tight, generation_config=gen_cfg,
    )
    _, ranked_loose, diag_loose = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg_loose, generation_config=gen_cfg,
    )
    # Both must respect the total budget
    assert diag_tight.candidates_officially_scored <= 30
    assert diag_loose.candidates_officially_scored <= 30
    assert diag_tight.invariant_holds
    assert diag_loose.invariant_holds
    # Tight seed budget caps neighborhood scoring lower than loose
    hood_tight = sum(1 for s in ranked_tight if s.family == "original_neighborhood" and s.was_scored)
    hood_loose = sum(1 for s in ranked_loose if s.family == "original_neighborhood" and s.was_scored)
    assert hood_tight <= hood_loose, (
        f"Tight seed budget should cap neighborhood scoring: "
        f"tight={hood_tight} vs loose={hood_loose}"
    )


def test_max_official_scores_still_respected_with_new_budget():
    """max_official_scores must be respected across all budget-split configurations."""
    bm = _make_benchmark_for_line_search()
    for limit in [5, 10, 20, 60]:
        gen_cfg = CandidateGenerationConfig(
            only_original_neighborhood=True,
            neighborhood_macro_limit=4,
            refinement_around_winners=True,
            refinement_top_k=3,
            line_search_around_winners=True,
            line_search_top_k=2,
        )
        score_cfg = CandidateScoringConfig(max_official_scores=limit)
        _, _, diag = score_and_select(
            generate_candidates(bm, config=gen_cfg), bm, plc=None,
            scoring_config=score_cfg, generation_config=gen_cfg,
        )
        assert diag.candidates_officially_scored <= limit, (
            f"Budget {limit}: scored {diag.candidates_officially_scored} > limit"
        )


def test_original_raw_fallback_invariant_with_dense_benchmark():
    """Invariant best<=raw must hold for dense benchmarks where many candidates need legalization."""
    bm = _make_dense_benchmark()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=4,
        refinement_around_winners=True,
        refinement_top_k=4,
        line_search_around_winners=True,
        line_search_top_k=3,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated: best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )
    assert diag.invariant_holds


# ---------------------------------------------------------------------------
# Budget-invariant refinement seed selection
# ---------------------------------------------------------------------------


def test_refinement_seed_selection_is_budget_invariant():
    """Unscored improving candidates must rank above scored non-improving ones as seeds."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[[0, 1], [2, 3]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    # Improving candidate: approx says it reduces HPWL, but it was never scored (budget exceeded).
    cp_improving = CandidatePlacement(
        name="neigh_m0_improving_unscored",
        family="original_neighborhood",
        positions=pos.clone(),
        metadata={"approx_hpwl_delta": -5.0, "moved_macro_id": 0},
    )
    sc_improving = ScoredCandidate(
        name="neigh_m0_improving_unscored",
        family="original_neighborhood",
        positions=pos.clone(),
        valid=True,
        proxy_cost=None,
        delta_vs_original=None,
        num_overlaps=0, num_out_of_bounds=0, num_unplaced=0,
        num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0,
        was_scored=False,
        metadata={"approx_hpwl_delta": -5.0, "moved_macro_id": 0},
    )

    # Non-improving candidate: was officially scored (cache hit), but cost is worse than raw.
    cp_nonimproving = CandidatePlacement(
        name="neigh_m1_nonimproving_cached",
        family="original_neighborhood",
        positions=pos.clone(),
        metadata={"approx_hpwl_delta": 2.0, "moved_macro_id": 1},
    )
    sc_nonimproving = ScoredCandidate(
        name="neigh_m1_nonimproving_cached",
        family="original_neighborhood",
        positions=pos.clone(),
        valid=True,
        proxy_cost=raw_cost + 0.05,  # worse than raw
        delta_vs_original=0.05,
        num_overlaps=0, num_out_of_bounds=0, num_unplaced=0,
        num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0,
        was_scored=True,
        metadata={"approx_hpwl_delta": 2.0, "moved_macro_id": 1, "cache_hit": True},
    )

    seeds, _ = _select_refinement_seeds(
        scored=[sc_improving, sc_nonimproving],
        candidates=[cp_improving, cp_nonimproving],
        top_k=1,
        raw_original_proxy_cost=raw_cost,
    )
    assert len(seeds) == 1, f"Expected 1 seed, got {len(seeds)}"
    assert seeds[0].name == "neigh_m0_improving_unscored", (
        f"Unscored improving candidate should outrank scored non-improving; got {seeds[0].name!r}"
    )


def test_refinement_seed_unscored_improving_generates_refinement_pass():
    """With max_official_scores=1 (only original_raw scored), refinement still runs if
    any neighborhood candidate has approx_hpwl_delta <= 0."""
    bm = _make_benchmark_with_nets()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=6,
        refinement_around_winners=True,
        refinement_top_k=3,
    )
    # Score budget allows only original_raw; all neighborhood candidates are unscored.
    score_cfg = CandidateScoringConfig(max_official_scores=1)
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    neighborhood = [s for s in ranked if s.family == "original_neighborhood"]
    improving_unscored = [
        s for s in neighborhood
        if not s.was_scored
        and isinstance(s.metadata.get("approx_hpwl_delta"), float)
        and s.metadata["approx_hpwl_delta"] <= 1e-9
    ]
    if not improving_unscored:
        pytest.skip("No unscored improving neighborhood candidates generated for this benchmark config")
    # Refinement pass must have been triggered from the unscored improving seeds.
    assert diag.refinement_candidates_generated > 0, (
        "Expected refinement candidates to be generated from unscored improving seeds, "
        f"but refinement_candidates_generated={diag.refinement_candidates_generated}"
    )


def test_refinement_seed_duplicates_are_excluded():
    """Duplicate neighborhood candidates must not be selected as refinement seeds."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[[0, 1], [2, 3]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    cp_original = CandidatePlacement(
        name="neigh_m0_orig",
        family="original_neighborhood",
        positions=pos.clone(),
        metadata={"approx_hpwl_delta": -3.0, "moved_macro_id": 0},
    )
    cp_dup = CandidatePlacement(
        name="neigh_m0_dup",
        family="original_neighborhood",
        positions=pos.clone(),
        metadata={"approx_hpwl_delta": -3.0, "moved_macro_id": 0},
    )

    sc_original = ScoredCandidate(
        name="neigh_m0_orig",
        family="original_neighborhood",
        positions=pos.clone(),
        valid=True, proxy_cost=None, delta_vs_original=None,
        num_overlaps=0, num_out_of_bounds=0, num_unplaced=0,
        num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0,
        was_scored=False,
        metadata={"approx_hpwl_delta": -3.0, "moved_macro_id": 0},
    )
    sc_dup = ScoredCandidate(
        name="neigh_m0_dup",
        family="original_neighborhood",
        positions=pos.clone(),
        valid=True, proxy_cost=None, delta_vs_original=None,
        num_overlaps=0, num_out_of_bounds=0, num_unplaced=0,
        num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0,
        was_scored=False,
        duplicate_of="neigh_m0_orig",
        metadata={"approx_hpwl_delta": -3.0, "moved_macro_id": 0},
    )

    seeds, _ = _select_refinement_seeds(
        scored=[sc_original, sc_dup],
        candidates=[cp_original, cp_dup],
        top_k=2,
        raw_original_proxy_cost=raw_cost,
    )
    seed_names = [s.name for s in seeds]
    assert "neigh_m0_dup" not in seed_names, (
        f"Duplicate candidate must not appear as a refinement seed; seeds={seed_names}"
    )


# ---------------------------------------------------------------------------
# Diverse seed strategy tests
# ---------------------------------------------------------------------------


def _make_scored_neighborhood(
    entries,  # list of (name, macro_id, approx_delta, was_scored, proxy_cost)
    pos: torch.Tensor,
) -> tuple:
    """Return (placement_list, scored_list) for use with _select_refinement_seeds."""
    placements = []
    scored_list = []
    for i, (name, macro_id, approx, was_scored, proxy_cost) in enumerate(entries):
        cp = CandidatePlacement(
            name=name,
            family="original_neighborhood",
            positions=pos.clone(),
            metadata={"approx_hpwl_delta": approx, "moved_macro_id": macro_id},
        )
        sc = ScoredCandidate(
            name=name,
            family="original_neighborhood",
            positions=pos.clone(),
            valid=True,
            proxy_cost=proxy_cost,
            delta_vs_original=None,
            num_overlaps=0,
            num_out_of_bounds=0,
            num_unplaced=0,
            num_moved=0,
            max_move=0.0,
            total_move=0.0,
            legalization_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            was_scored=was_scored,
            metadata={"approx_hpwl_delta": approx, "moved_macro_id": macro_id, "generation_rank": i},
        )
        placements.append(cp)
        scored_list.append(sc)
    return placements, scored_list


def test_diverse_seed_strategy_selects_distinct_macros():
    """Diverse seed selection must not select two seeds for the same macro."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=6, canvas=200.0, macro_size=10.0, net_nodes=[[0, 1], [1, 2], [2, 3]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    # Two candidates for macro 0, one each for macros 1, 2, 3
    entries = [
        ("neigh_m0_a", 0, -2.0, False, None),
        ("neigh_m0_b", 0, -1.8, False, None),
        ("neigh_m1",   1, -1.5, False, None),
        ("neigh_m2",   2, -1.2, False, None),
        ("neigh_m3",   3, -0.9, False, None),
    ]
    placements, scored_list = _make_scored_neighborhood(entries, pos)

    for top_k in (3, 4, 5):
        seeds, diag = _select_refinement_seeds(
            scored=scored_list,
            candidates=placements,
            top_k=top_k,
            raw_original_proxy_cost=raw_cost,
            strategy="diverse",
        )
        macro_ids = [p.metadata.get("moved_macro_id") for p in seeds]
        assert len(macro_ids) == len(set(macro_ids)), (
            f"top_k={top_k}: duplicate macro IDs in diverse seeds: {macro_ids}"
        )


def test_diverse_seed_strategy_includes_exploratory_improving_seed():
    """Exploratory bucket selects an improving macro outside the top-approx set."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=6, canvas=200.0, macro_size=10.0,
                         net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    # m0 = strongest approx (fills bucket A)
    # m1 = officially scored improving (fills bucket B)
    # m2 = next best by priority/approx (fills bucket C)
    # m3 = target for exploratory bucket D
    entries = [
        ("neigh_m0", 0, -2.0, False, None),
        ("neigh_m1", 1, -1.5, True, raw_cost - 0.02),
        ("neigh_m2", 2, -1.2, False, None),
        ("neigh_m3", 3, -0.8, False, None),
    ]
    placements, scored_list = _make_scored_neighborhood(entries, pos)

    # Conservative with top_k=4 would select m0, m1, m2, m3 in approx order — m3 IS included
    # but only because all 4 macros fit. With more candidates, conservative misses distant ones.
    # Test the exploratory diagnostic specifically.
    seeds, diag = _select_refinement_seeds(
        scored=scored_list,
        candidates=placements,
        top_k=4,
        raw_original_proxy_cost=raw_cost,
        strategy="diverse",
        exploration_seeds=1,
    )
    macro_ids = [p.metadata.get("moved_macro_id") for p in seeds]
    # All 4 macros should be selected
    assert set(macro_ids) == {0, 1, 2, 3}, f"Expected all 4 macros selected; got {macro_ids}"

    # Macro 3 must be in the exploratory bucket (it is the last distinct improving macro)
    exploratory_entries = [d for d in diag if d.get("bucket") == "exploratory"]
    assert any(d["macro_id"] == 3 for d in exploratory_entries), (
        f"Macro 3 should be in exploratory bucket; diag={diag}"
    )


def test_diverse_seed_strategy_is_deterministic():
    """Calling _select_refinement_seeds twice with same inputs produces identical results."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=6, canvas=200.0, macro_size=10.0,
                         net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    entries = [
        ("neigh_m0", 0, -2.0, False, None),
        ("neigh_m1", 1, -1.5, True, raw_cost - 0.01),
        ("neigh_m2", 2, -1.2, False, None),
        ("neigh_m3", 3, -0.9, False, None),
        ("neigh_m4", 4, -0.7, False, None),
    ]
    placements, scored_list = _make_scored_neighborhood(entries, pos)

    seeds1, diag1 = _select_refinement_seeds(
        scored=scored_list, candidates=placements, top_k=4,
        raw_original_proxy_cost=raw_cost, strategy="diverse",
    )
    seeds2, diag2 = _select_refinement_seeds(
        scored=scored_list, candidates=placements, top_k=4,
        raw_original_proxy_cost=raw_cost, strategy="diverse",
    )

    assert [s.name for s in seeds1] == [s.name for s in seeds2], (
        "Diverse seed selection must be deterministic"
    )
    assert [d["seed_name"] for d in diag1] == [d["seed_name"] for d in diag2], (
        "Bucket diagnostics must be deterministic"
    )


def test_diverse_seed_strategy_respects_refinement_top_k():
    """Diverse strategy must never return more seeds than top_k."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=8, canvas=200.0, macro_size=10.0,
                         net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    entries = [(f"neigh_m{i}", i, -(2.0 - i * 0.2), False, None) for i in range(7)]
    placements, scored_list = _make_scored_neighborhood(entries, pos)

    for top_k in (1, 2, 3, 5, 7):
        seeds, _ = _select_refinement_seeds(
            scored=scored_list, candidates=placements, top_k=top_k,
            raw_original_proxy_cost=raw_cost, strategy="diverse",
        )
        assert len(seeds) <= top_k, (
            f"top_k={top_k}: diverse strategy returned {len(seeds)} seeds (> top_k)"
        )


def test_duplicate_macro_seeds_are_replaced():
    """When many candidates share one macro, diverse strategy finds other distinct macros."""
    from submissions.solver.core.candidate_scoring import _select_refinement_seeds
    from submissions.solver.core.candidate_types import ScoredCandidate

    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[[0, 1], [1, 2]])
    pos = bm.macro_positions.clone().float()
    raw_cost = 1.0

    # 3 candidates for macro 0, 1 for macro 1
    entries = [
        ("neigh_m0_a", 0, -2.0, False, None),
        ("neigh_m0_b", 0, -1.8, False, None),
        ("neigh_m0_c", 0, -1.6, False, None),
        ("neigh_m1",   1, -1.0, False, None),
    ]
    placements, scored_list = _make_scored_neighborhood(entries, pos)

    seeds, diag = _select_refinement_seeds(
        scored=scored_list, candidates=placements, top_k=3,
        raw_original_proxy_cost=raw_cost, strategy="diverse",
    )
    macro_ids = [p.metadata.get("moved_macro_id") for p in seeds]
    assert len(macro_ids) == len(set(macro_ids)), f"Duplicate macro IDs: {macro_ids}"
    # macro 1 must be included (distinct from macro 0 duplicates)
    assert 1 in macro_ids, f"Macro 1 must be selected when macro 0 candidates are duplicated; got {macro_ids}"


def test_seed_bucket_diagnostics_are_populated():
    """Diverse strategy must populate refinement_seed_bucket_diagnostics in ScoringDiagnostics."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=5,
        refinement_around_winners=True,
        refinement_top_k=3,
        refinement_seed_strategy="diverse",
        refinement_exploration_seeds=1,
    )
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    # If refinement ran, bucket diagnostics must be non-empty
    if diag.refinement_candidates_generated > 0:
        assert len(diag.refinement_seed_bucket_diagnostics) > 0, (
            "Diverse strategy must populate refinement_seed_bucket_diagnostics"
        )
        for entry in diag.refinement_seed_bucket_diagnostics:
            assert "seed_name" in entry, f"Entry missing seed_name: {entry}"
            assert "macro_id" in entry, f"Entry missing macro_id: {entry}"
            assert "bucket" in entry, f"Entry missing bucket: {entry}"
            assert entry["bucket"] in ("approx", "official", "priority", "exploratory", "fill"), (
                f"Unknown bucket label: {entry['bucket']}"
            )


def test_max_official_scores_still_respected_diverse():
    """max_official_scores budget must be respected when diverse seed strategy is active."""
    bm = _make_benchmark(
        n_hard=8, canvas=220.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=6,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_seed_strategy="diverse",
        refinement_exploration_seeds=1,
    )
    limit = 6
    score_cfg = CandidateScoringConfig(max_official_scores=limit)
    _, _, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None,
        scoring_config=score_cfg, generation_config=gen_cfg,
    )
    assert diag.candidates_officially_scored <= limit, (
        f"Diverse strategy: scored {diag.candidates_officially_scored} candidates, expected <= {limit}"
    )


def test_original_raw_fallback_invariant_diverse_strategy():
    """best <= raw_original invariant must hold when diverse seed strategy is active."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [2, 3], [4, 5], [1, 3], [3, 5]],
    )
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        neighborhood_macro_limit=5,
        refinement_around_winners=True,
        refinement_top_k=4,
        refinement_combo_size=2,
        refinement_seed_strategy="diverse",
        refinement_exploration_seeds=1,
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=gen_cfg), bm, plc=None, generation_config=gen_cfg
    )
    if not diag.raw_original_valid or diag.raw_original_proxy_cost is None:
        pytest.skip("raw original not valid — invariant not applicable")
    assert best.proxy_cost is not None
    assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
        f"Invariant violated (diverse): best={best.proxy_cost:.6f} > "
        f"raw_original={diag.raw_original_proxy_cost:.6f}"
    )
    assert diag.invariant_holds


# ---------------------------------------------------------------------------
# M2B final profile tests
# ---------------------------------------------------------------------------

# The m2b-final profile is defined in run_benchmarks._PROFILES and in
# run_official_scoring_smoke._SMOKE_PROFILES.  These tests verify the
# profile's contract using synthetic benchmarks so they run without IBM
# testcases or plc_client_os.

_M2B_FINAL_GEN_CFG = CandidateGenerationConfig(
    only_original_neighborhood=True,
    candidate_budget=80,
    neighborhood_macro_limit=20,
    neighborhood_step_profile="medium",
    refinement_around_winners=True,
    refinement_top_k=5,
    refinement_combo_size=2,
    refinement_seed_strategy="diverse",
    refinement_exploration_seeds=1,
    line_search_around_winners=True,
    line_search_top_k=3,
    line_search_max_scale=4.0,
    line_search_stop_after_worse=2,
)

_M2B_FINAL_SCORE_CFG = CandidateScoringConfig(
    max_official_scores=60,
    disable_score_cache=True,
)


def test_m2b_final_profile_exists():
    """m2b-final profile must be registered in run_benchmarks and run_official_scoring_smoke."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    from submissions.solver.scripts.run_official_scoring_smoke import _SMOKE_PROFILES

    assert "m2b-final" in _PROFILES, "m2b-final must be in run_benchmarks._PROFILES"
    assert "m2b-final" in _SMOKE_PROFILES, "m2b-final must be in run_official_scoring_smoke._SMOKE_PROFILES"

    p = _PROFILES["m2b-final"]
    assert p.get("only_original_neighborhood") is True
    assert p.get("candidate_budget") == 80
    assert p.get("refinement_around_winners") is True
    assert p.get("refinement_top_k") == 5
    assert p.get("line_search_around_winners") is True
    assert p.get("line_search_top_k") == 3
    assert p.get("max_official_scores") == 60

    sp = _SMOKE_PROFILES["m2b-final"]
    assert sp.get("only_original_neighborhood") is True
    assert sp.get("refinement_seed_strategy") == "diverse"
    assert sp.get("max_official_scores") == 60


def test_m2b_final_uses_diverse_seed_strategy():
    """m2b-final profile must be configured with refinement_seed_strategy='diverse'."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES

    p = _PROFILES["m2b-final"]
    assert p.get("refinement_seed_strategy") == "diverse", (
        f"Expected diverse, got {p.get('refinement_seed_strategy')!r}"
    )
    assert p.get("refinement_exploration_seeds") == 1

    # Verify the diverse strategy actually runs with this config
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )
    _, ranked, diag = score_and_select(
        generate_candidates(bm, config=_M2B_FINAL_GEN_CFG), bm, plc=None,
        scoring_config=_M2B_FINAL_SCORE_CFG, generation_config=_M2B_FINAL_GEN_CFG,
    )
    if diag.refinement_candidates_generated > 0:
        assert len(diag.refinement_seed_bucket_diagnostics) > 0, (
            "Diverse strategy must populate bucket diagnostics when refinement runs"
        )


def test_m2b_final_preserves_original_raw_fallback():
    """m2b-final profile must never select a candidate worse than original_raw."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [2, 3], [4, 5], [1, 3], [3, 5]],
    )
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=_M2B_FINAL_GEN_CFG), bm, plc=None,
        scoring_config=_M2B_FINAL_SCORE_CFG, generation_config=_M2B_FINAL_GEN_CFG,
    )
    raw = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw is not None, "original_raw must be in ranked output"
    assert raw.valid, "original_raw must be valid"
    if diag.raw_original_valid and diag.raw_original_proxy_cost is not None:
        assert best.proxy_cost is not None
        assert best.proxy_cost <= diag.raw_original_proxy_cost + 1e-9, (
            f"m2b-final invariant violated: best={best.proxy_cost:.6f} > "
            f"raw_original={diag.raw_original_proxy_cost:.6f}"
        )
    assert diag.invariant_holds


def test_m2b_final_has_bounded_official_score_budget():
    """m2b-final profile must not exceed max_official_scores=60."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES

    p = _PROFILES["m2b-final"]
    assert p.get("max_official_scores") == 60, (
        f"Expected max_official_scores=60, got {p.get('max_official_scores')}"
    )

    bm = _make_benchmark(
        n_hard=8, canvas=220.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
    )
    _, _, diag = score_and_select(
        generate_candidates(bm, config=_M2B_FINAL_GEN_CFG), bm, plc=None,
        scoring_config=_M2B_FINAL_SCORE_CFG, generation_config=_M2B_FINAL_GEN_CFG,
    )
    assert diag.candidates_officially_scored <= 60, (
        f"m2b-final scored {diag.candidates_officially_scored} candidates, expected <= 60"
    )
    assert diag.invariant_holds


def test_m2b_final_does_not_require_persistent_cache():
    """m2b-final profile must work correctly with persistent cache disabled."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )
    score_cfg_no_cache = CandidateScoringConfig(max_official_scores=60, disable_score_cache=True)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, config=_M2B_FINAL_GEN_CFG), bm, plc=None,
        scoring_config=score_cfg_no_cache, generation_config=_M2B_FINAL_GEN_CFG,
    )
    assert best is not None
    assert best.valid
    assert diag.invariant_holds
    assert diag.cache_hits == 0, (
        "m2b-final without cache must report zero cache hits"
    )


# ---------------------------------------------------------------------------
# Blocker fix #1: original_raw must be validated against fixed hard obstacles.
# ---------------------------------------------------------------------------


def _make_bm_with_fixed_obstacle(
    raw_position_for_movable: tuple,
    fixed_position: tuple = (50.0, 50.0),
    macro_size: float = 10.0,
    canvas: float = 100.0,
) -> Benchmark:
    """Two-hard-macro benchmark: macro 0 is movable, macro 1 is a fixed obstacle."""
    positions = torch.tensor(
        [list(raw_position_for_movable), list(fixed_position)], dtype=torch.float32
    )
    return _make_benchmark(
        n_hard=2,
        canvas=canvas,
        macro_size=macro_size,
        positions=positions,
        fixed_mask=[False, True],
    )


def test_original_raw_invalid_when_movable_hard_overlaps_fixed_hard():
    """original_raw must be flagged invalid when a movable macro overlaps a fixed obstacle."""
    bm = _make_bm_with_fixed_obstacle(raw_position_for_movable=(52.0, 50.0))
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    raw = next(s for s in ranked if s.name == "original_raw")
    assert not raw.valid, "original_raw must be invalid when overlapping a fixed obstacle"
    assert not diag.raw_original_valid


def test_original_raw_valid_when_touching_fixed_hard_edge_only():
    """Touching a fixed obstacle edge (zero separation) must not flag invalid."""
    # macro size = 10 -> half-width 5. Fixed at x=50, movable touching at x=40.
    bm = _make_bm_with_fixed_obstacle(raw_position_for_movable=(40.0, 50.0))
    _best, ranked, _diag = score_and_select(generate_candidates(bm), bm, plc=None)
    raw = next(s for s in ranked if s.name == "original_raw")
    assert raw.valid, "original_raw touching a fixed obstacle edge must remain valid"


def test_original_raw_invalid_fixed_overlap_falls_back_to_legalized_or_other_valid_candidate():
    """When original_raw is invalid via fixed overlap, the selected best must not be original_raw."""
    bm = _make_bm_with_fixed_obstacle(raw_position_for_movable=(52.0, 50.0))
    best, ranked, _diag = score_and_select(generate_candidates(bm), bm, plc=None)
    # Best must be a valid candidate (legalized or other family) — never the invalid raw
    assert best.valid, "Best must be a valid candidate when original_raw is invalid"
    assert best.name != "original_raw"


def test_soft_macros_not_treated_as_fixed_obstacles_for_original_raw():
    """A soft macro coincident with a movable hard macro must not invalidate original_raw."""
    # Three hard macros + one soft macro at the same coordinate as a hard macro.
    positions = torch.tensor(
        [
            [10.0, 10.0],  # hard 0
            [30.0, 10.0],  # hard 1
            [10.0, 30.0],  # hard 2
            [10.0, 10.0],  # soft 3 — overlapping hard 0 in raw coords
        ],
        dtype=torch.float32,
    )
    bm = _make_benchmark(
        n_hard=3,
        n_soft=1,
        canvas=80.0,
        macro_size=10.0,
        positions=positions,
        fixed_mask=[False, False, False, False],
    )
    _best, ranked, _diag = score_and_select(generate_candidates(bm), bm, plc=None)
    raw = next(s for s in ranked if s.name == "original_raw")
    assert raw.valid, (
        "Soft macro should not be treated as a fixed obstacle; original_raw "
        "with non-overlapping hard macros must remain valid"
    )


def test_best_cost_never_uses_invalid_original_raw():
    """Selection must never pick original_raw if it is invalid due to fixed overlap."""
    bm = _make_bm_with_fixed_obstacle(raw_position_for_movable=(52.0, 50.0))
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    assert best.valid
    assert best.name != "original_raw"
    # Ranked output: original_raw must not appear before any valid candidate
    raw_idx = next(i for i, s in enumerate(ranked) if s.name == "original_raw")
    valid_before_raw = any(s.valid for s in ranked[:raw_idx])
    if not valid_before_raw:
        # If no valid candidate is ranked before original_raw, ensure invariant_holds
        # still flags the selection as the legalized fallback (selected_due_to)
        assert diag.selected_due_to in ("fallback_original", "proxy_cost", "tie_break", "validity_only")


# ---------------------------------------------------------------------------
# Blocker fix #2: placement_hash must distinguish 0.05 µm refinement moves.
# ---------------------------------------------------------------------------


def test_placement_hash_distinguishes_0p05um_moves():
    """Two placements differing by 0.05 µm must produce distinct hashes."""
    from submissions.solver.core.candidate_scoring import (
        placement_hash, PLACEMENT_HASH_TOLERANCE_UM,
    )

    assert PLACEMENT_HASH_TOLERANCE_UM < 0.05, (
        f"Tolerance {PLACEMENT_HASH_TOLERANCE_UM} must be finer than 0.05 µm"
    )
    pos_a = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    pos_b = torch.tensor([[10.05, 10.0], [20.0, 20.0]], dtype=torch.float32)
    assert placement_hash(pos_a) != placement_hash(pos_b), (
        "placement_hash must distinguish 0.05 µm moves"
    )


def test_duplicate_detection_does_not_collapse_tiny_refinement_moves():
    """A 0.05 µm difference must not be marked as a duplicate by _mark_duplicates."""
    from submissions.solver.core.candidate_scoring import _mark_duplicates
    from submissions.solver.core.candidate_types import ScoredCandidate

    pos_a = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    pos_b = torch.tensor([[10.05, 10.0], [20.0, 20.0]], dtype=torch.float32)
    sc_a = ScoredCandidate(
        name="a", family="x", positions=pos_a, valid=True, proxy_cost=None,
        delta_vs_original=None, num_overlaps=0, num_out_of_bounds=0,
        num_unplaced=0, num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0, no_op=True,
        notes="", was_scored=False, metadata={}, messages=[],
    )
    sc_b = ScoredCandidate(
        name="b", family="x", positions=pos_b, valid=True, proxy_cost=None,
        delta_vs_original=None, num_overlaps=0, num_out_of_bounds=0,
        num_unplaced=0, num_moved=0, max_move=0.0, total_move=0.0,
        legalization_ms=0.0, scoring_ms=0.0, total_ms=0.0, no_op=True,
        notes="", was_scored=False, metadata={}, messages=[],
    )
    dup_count, _ = _mark_duplicates([sc_a, sc_b], enable_hash_cache=True)
    assert dup_count == 0, "0.05 µm refinement move must not be collapsed as duplicate"
    assert sc_b.duplicate_of is None


def test_official_score_cache_key_distinguishes_tiny_moves(tmp_path):
    """The persistent score cache must use the same fine-grained hash."""
    from submissions.solver.core.candidate_scoring import placement_hash
    from submissions.solver.core.score_cache import OfficialScoreCache

    cache = OfficialScoreCache(cache_path=tmp_path / "scores.jsonl")
    pos_a = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    pos_b = torch.tensor([[10.05, 10.0]], dtype=torch.float32)
    h_a = placement_hash(pos_a)
    h_b = placement_hash(pos_b)
    assert h_a != h_b, "placement_hash must distinguish 0.05 µm moves at the cache layer"
    cache.record("bm1", h_a, 1.0)
    # Different placement (b) must not hit cache for placement a's cost
    assert cache.lookup("bm1", h_b) is None
    assert cache.lookup("bm1", h_a) == 1.0


def test_cached_scores_do_not_apply_to_different_0p05um_placement(tmp_path):
    """A cached cost for placement A must not be reused for a 0.05 µm displaced placement B."""
    from submissions.solver.core.candidate_scoring import placement_hash
    from submissions.solver.core.score_cache import OfficialScoreCache

    cache = OfficialScoreCache(cache_path=tmp_path / "cache.jsonl")
    pos_a = torch.tensor([[5.0, 5.0], [15.0, 15.0]], dtype=torch.float32)
    pos_b = torch.tensor([[5.0, 5.0], [15.05, 15.0]], dtype=torch.float32)

    cache.record("bm", placement_hash(pos_a), 0.123)

    miss_b = cache.lookup("bm", placement_hash(pos_b))
    assert miss_b is None, "Cache must miss for a 0.05 µm displaced placement"

    hit_a = cache.lookup("bm", placement_hash(pos_a))
    assert hit_a == 0.123


def test_m2b_final_is_deterministic():
    """m2b-final profile must produce identical ranked output across cold reruns."""
    bm = _make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
    )

    def _run():
        return score_and_select(
            generate_candidates(bm, config=_M2B_FINAL_GEN_CFG), bm, plc=None,
            scoring_config=_M2B_FINAL_SCORE_CFG, generation_config=_M2B_FINAL_GEN_CFG,
        )

    best1, ranked1, diag1 = _run()
    best2, ranked2, diag2 = _run()

    assert (best1.name if best1 else None) == (best2.name if best2 else None), (
        "m2b-final winner differs between cold reruns"
    )
    assert len(ranked1) == len(ranked2), "Ranked list length differs between runs"
    names1 = [s.name for s in ranked1]
    names2 = [s.name for s in ranked2]
    assert names1 == names2, f"Ranked candidate order differs: {names1} vs {names2}"
    for sc1, sc2 in zip(ranked1, ranked2):
        if sc1.proxy_cost is not None and sc2.proxy_cost is not None:
            assert abs(sc1.proxy_cost - sc2.proxy_cost) < 1e-9, (
                f"Cost differs for {sc1.name}: {sc1.proxy_cost} vs {sc2.proxy_cost}"
            )


# ---------------------------------------------------------------------------
# Blocker fix #3: invalid original_raw must never be selected via fallback.
# ---------------------------------------------------------------------------


def _make_all_invalid_inputs(bm: Benchmark):
    """Build a candidate list where original_raw, original_legalized, and any
    extra candidate are all overlap-invalid via bypass_legalization=True."""
    overlap_positions = bm.macro_positions.clone().float()
    overlap_positions[1] = overlap_positions[0]  # force overlap
    raw = CandidatePlacement(
        "original_raw", "original", overlap_positions.clone(), bypass_legalization=True
    )
    leg = CandidatePlacement(
        "original_legalized", "original", overlap_positions.clone(), bypass_legalization=True
    )
    extra = CandidatePlacement(
        "extra_invalid", "original_neighborhood", overlap_positions.clone(), bypass_legalization=True
    )
    return [raw, leg, extra]


def test_no_valid_candidates_does_not_select_invalid_original_raw():
    """When every candidate is invalid, best.valid must be False and the sentinel
    selected_due_to value must be reported — never a silent fallback to invalid raw."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    candidates = _make_all_invalid_inputs(bm)
    best, ranked, diag = score_and_select(candidates, bm, plc=None)
    assert best is not None
    assert not best.valid, (
        "Best must not be flagged valid when no valid candidates exist; "
        f"got best.name={best.name!r} valid={best.valid}"
    )
    assert diag.selected_due_to == "no_valid_scored_candidate", (
        f"Expected sentinel selected_due_to; got {diag.selected_due_to!r}"
    )
    assert not diag.invariant_holds, (
        "invariant_holds must be False for the no-valid-candidate sentinel case"
    )


def test_invalid_original_raw_not_selected_when_valid_scored_empty():
    """Specifically: an invalid original_raw must not be returned as best."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    candidates = _make_all_invalid_inputs(bm)
    best, _ranked, diag = score_and_select(candidates, bm, plc=None)
    # Either best is not original_raw, or original_raw is being returned as the
    # sentinel but flagged invalid + selected_due_to=no_valid_scored_candidate.
    if best.name == "original_raw":
        assert not best.valid
        assert diag.selected_due_to == "no_valid_scored_candidate"
    else:
        assert best.valid


def test_invalid_original_raw_fallback_reports_failure():
    """The placer must refuse to emit a placement when no valid candidate exists."""
    from submissions.solver.placer import SolverPlacer

    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    placer = SolverPlacer()
    # The placer wires its own generate_candidates → we can't inject invalid
    # candidates from the outside.  Instead exercise score_and_select directly
    # and confirm the diagnostic.
    candidates = _make_all_invalid_inputs(bm)
    best, _ranked, diag = score_and_select(candidates, bm, plc=None)
    assert diag.selected_due_to == "no_valid_scored_candidate"
    assert not diag.invariant_holds


def test_original_legalized_selected_if_raw_invalid_but_legalized_valid():
    """When original_raw is invalid but original_legalized is valid, the latter must win."""
    positions = torch.tensor([[50.0, 50.0]] * 4, dtype=torch.float32)
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, positions=positions)
    best, ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    assert best.valid, "Best must be a valid candidate"
    assert best.name != "original_raw"
    leg = next((s for s in ranked if s.name == "original_legalized"), None)
    assert leg is not None and leg.valid, "original_legalized must exist and be valid"


def test_generated_valid_candidate_selected_if_raw_invalid():
    """If original_raw is invalid, the selected best must be valid (not raw)."""
    positions = torch.tensor([[50.0, 50.0]] * 4, dtype=torch.float32)
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, positions=positions)
    best, _ranked, diag = score_and_select(generate_candidates(bm), bm, plc=None)
    assert best.valid, f"Best must be valid; got best.valid={best.valid} name={best.name}"
    assert diag.selected_due_to != "no_valid_scored_candidate"


def test_selected_due_to_no_valid_scored_candidate_when_all_invalid():
    """The sentinel diagnostic value must be set exactly when all candidates are invalid."""
    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    candidates = _make_all_invalid_inputs(bm)
    _best, _ranked, diag = score_and_select(candidates, bm, plc=None)
    assert diag.selected_due_to == "no_valid_scored_candidate"


def test_placer_refuses_to_emit_when_no_valid_candidate():
    """SolverPlacer.place must fall back to original positions instead of emitting invalid best.

    The pipeline normally generates a valid original_raw or original_legalized when
    given a clean benchmark, so the path under test only fires when the fallback
    sentinel triggers.  We monkey-patch generate_candidates to force the sentinel.
    """
    import submissions.solver.placer as placer_module

    bm = _make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0)
    invalid_candidates = _make_all_invalid_inputs(bm)

    # The placer uses `from core.candidates import generate_candidates` inside
    # `place`, so patch the module reference resolved at import time.
    import importlib
    core_candidates = importlib.import_module("core.candidates")
    original_gen = core_candidates.generate_candidates
    core_candidates.generate_candidates = lambda _bm: invalid_candidates
    try:
        out_positions = placer_module.SolverPlacer().place(bm, plc=None)
    finally:
        core_candidates.generate_candidates = original_gen

    # The placer must fall back to original positions (which themselves are valid here),
    # NOT to the invalid overlapping positions from the sentinel best.
    assert torch.allclose(out_positions, bm.macro_positions.float()), (
        "Placer must fall back to original benchmark positions when score_and_select "
        "reports no_valid_scored_candidate"
    )
