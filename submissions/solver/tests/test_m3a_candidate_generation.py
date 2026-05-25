"""test_m3a_candidate_generation — each pair yields at most 6 candidates."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import (
    generate_pair_candidates,
    generate_m3a_candidates_for_pairs,
    GRID_STEP,
    snap_to_grid,
)


def _bm_well_spaced():
    """4 macros spaced far apart with room to move."""
    pos = torch.tensor([
        [15.0, 15.0],
        [85.0, 15.0],
        [15.0, 85.0],
        [85.0, 85.0],
    ])
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=8.0,
        net_nodes=[[0, 1], [0, 2]],
        positions=pos,
    )


def test_at_most_six_candidates_per_pair():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pi, existing_names=set())
        assert len(cands) <= 6, f"pair ({a},{b}) generated {len(cands)} > 6 candidates"


def test_at_least_one_candidate_per_coupled_pair():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pi, existing_names=set())
        assert len(cands) >= 1, f"pair ({a},{b}) generated 0 candidates"


def test_candidate_names_are_unique_within_pair():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pi, existing_names=set())
        names = [c.name for c in cands]
        assert len(names) == len(set(names)), f"duplicate names in pair ({a},{b}): {names}"


def test_candidate_names_unique_across_all_pairs():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    all_cands = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    names = [c.name for c in all_cands]
    assert len(names) == len(set(names)), "duplicate candidate names across pairs"


def test_all_candidates_have_correct_family():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        for c in generate_pair_candidates(bm, wp, a, b, pi, set()):
            assert c.family == "m3a_pair_refinement", f"wrong family: {c.family}"


def test_all_candidates_bypass_legalization():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        for c in generate_pair_candidates(bm, wp, a, b, pi, set()):
            assert c.bypass_legalization, f"candidate {c.name} must bypass legalization"


def test_existing_names_are_skipped():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=1)
    a, b, _ = pairs[0]
    all_cands = generate_pair_candidates(bm, wp, a, b, 0, set())
    assert len(all_cands) >= 1
    # Block the first candidate's name.
    blocked = {all_cands[0].name}
    with_block = generate_pair_candidates(bm, wp, a, b, 0, blocked)
    names_with_block = {c.name for c in with_block}
    assert all_cands[0].name not in names_with_block


def test_coordinates_snapped_to_grid():
    bm = _bm_well_spaced()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=5)
    for pi, (a, b, _) in enumerate(pairs):
        for c in generate_pair_candidates(bm, wp, a, b, pi, set()):
            pos = c.positions
            for i in range(pos.shape[0]):
                x = float(pos[i, 0].item())
                y = float(pos[i, 1].item())
                # Use float64 round-trip (placement_hash precision = 3 decimals).
                # float32 stores 85.05 as 85.05000305...; after rounding to 3dp it is 85.050.
                import numpy as np
                x64 = round(float(np.float64(x)), 3)
                y64 = round(float(np.float64(y)), 3)
                assert abs(x64 - snap_to_grid(x64)) < 1e-9, f"{c.name}: x={x} not on grid"
                assert abs(y64 - snap_to_grid(y64)) < 1e-9, f"{c.name}: y={y} not on grid"


def test_fixed_hard_macros_unchanged_in_candidates():
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=8.0,
        net_nodes=[[1, 2], [1, 3]],
        fixed_mask=[True, False, False, False],
        positions=torch.tensor([[50.0, 50.0], [15.0, 15.0], [85.0, 15.0], [50.0, 85.0]]),
    )
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    for pi, (a, b, _) in enumerate(pairs):
        for c in generate_pair_candidates(bm, wp, a, b, pi, set()):
            # Fixed macro 0 must be unchanged.
            orig_x = float(wp[0, 0].item())
            orig_y = float(wp[0, 1].item())
            assert float(c.positions[0, 0].item()) == orig_x
            assert float(c.positions[0, 1].item()) == orig_y
