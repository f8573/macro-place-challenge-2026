"""test_m3a_pair_enumeration — deterministic pair list under fixed benchmark."""

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs


def _bm_two_coupled():
    """4 movable macros, nets coupling (0,1) twice and (2,3) once."""
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 1], [2, 3]],
    )


def test_pair_list_is_stable_across_calls():
    bm = _bm_two_coupled()
    p1 = enumerate_net_coupled_pairs(bm, top_k=10)
    p2 = enumerate_net_coupled_pairs(bm, top_k=10)
    assert p1 == p2, "pair list must be identical on repeated calls"


def test_pair_sorted_by_shared_net_count_descending():
    bm = _bm_two_coupled()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    assert len(pairs) >= 2
    # (0,1) has 2 shared nets; (2,3) has 1 shared net
    assert pairs[0][:2] == (0, 1), f"top pair should be (0,1), got {pairs[0]}"
    assert pairs[0][2] == 2
    assert pairs[1][:2] == (2, 3)
    assert pairs[1][2] == 1


def test_pair_orientation_always_smaller_first():
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                        net_nodes=[[3, 0], [1, 2]])
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    for a, b, _ in pairs:
        assert a < b, f"pair must be (smaller, larger), got ({a}, {b})"


def test_top_k_caps_result_length():
    bm = make_benchmark(n_hard=6, canvas=200.0, macro_size=10.0,
                        net_nodes=[[0, 1], [0, 2], [1, 2], [3, 4], [4, 5]])
    pairs = enumerate_net_coupled_pairs(bm, top_k=2)
    assert len(pairs) <= 2


def test_top_k_zero_returns_empty():
    bm = _bm_two_coupled()
    assert enumerate_net_coupled_pairs(bm, top_k=0) == []


def test_fixed_macros_excluded_from_pairs():
    """Fixed-hard macro 0 must not appear in any pair."""
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        fixed_mask=[True, False, False, False],
    )
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    for a, b, _ in pairs:
        assert a != 0 and b != 0, "fixed macro 0 must not appear in pairs"


def test_no_pairs_when_all_fixed():
    bm = make_benchmark(
        n_hard=3, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2]],
        fixed_mask=[True, True, True],
    )
    assert enumerate_net_coupled_pairs(bm, top_k=10) == []


def test_no_pairs_when_no_nets():
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[])
    assert enumerate_net_coupled_pairs(bm, top_k=10) == []


def test_tie_broken_by_macro_ids():
    """Two pairs with equal shared-net count — lower (a,b) comes first."""
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [2, 3]],
    )
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    assert len(pairs) == 2
    assert pairs[0][:2] == (0, 1)
    assert pairs[1][:2] == (2, 3)


def test_self_connections_not_counted():
    """A net with a single macro id repeated should not form a self-pair."""
    bm = make_benchmark(
        n_hard=2, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 0, 1]],
    )
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    assert len(pairs) == 1
    assert pairs[0][:2] == (0, 1)
    assert pairs[0][2] == 1, "duplicate node in net should not inflate count"
