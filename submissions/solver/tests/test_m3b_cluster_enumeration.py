"""test_m3b_cluster_enumeration — deterministic cluster list under fixed benchmark."""

import pytest

from conftest import make_benchmark
from submissions.solver.core.m3b_cluster_enumeration import enumerate_net_coupled_triples


def _bm_triangle():
    """3 movable macros fully connected: (0,1), (0,2), (1,2) each with 1 net."""
    return make_benchmark(
        n_hard=3, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
    )


def _bm_two_triangles():
    """6 macros: triangle (0,1,2) strongly connected + triangle (3,4,5) weakly connected."""
    return make_benchmark(
        n_hard=6, canvas=200.0, macro_size=10.0,
        net_nodes=[
            [0, 1], [0, 1],   # pair (0,1) count=2
            [0, 2], [0, 2],   # pair (0,2) count=2
            [1, 2], [1, 2],   # pair (1,2) count=2
            [3, 4],           # pair (3,4) count=1
            [3, 5],           # pair (3,5) count=1
            [4, 5],           # pair (4,5) count=1
        ],
    )


def test_cluster_list_stable_across_calls():
    bm = _bm_triangle()
    t1 = enumerate_net_coupled_triples(bm, top_k=10)
    t2 = enumerate_net_coupled_triples(bm, top_k=10)
    assert t1 == t2, "cluster list must be identical on repeated calls"


def test_single_triangle_yields_one_triple():
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    assert len(triples) == 1
    a, b, c, score = triples[0]
    assert (a, b, c) == (0, 1, 2)


def test_triple_score_is_aggregate_pair_coupling():
    bm = _bm_two_triangles()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    # Strong triangle (0,1,2): score = 2+2+2 = 6
    # Weak triangle (3,4,5): score = 1+1+1 = 3
    assert len(triples) == 2
    assert triples[0][:3] == (0, 1, 2)
    assert triples[0][3] == 6
    assert triples[1][:3] == (3, 4, 5)
    assert triples[1][3] == 3


def test_canonical_ordering_a_lt_b_lt_c():
    bm = _bm_triangle()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    for a, b, c, _ in triples:
        assert a < b < c, f"canonical order violated: ({a},{b},{c})"


def test_top_k_caps_result_length():
    bm = _bm_two_triangles()
    triples = enumerate_net_coupled_triples(bm, top_k=1)
    assert len(triples) <= 1


def test_top_k_zero_returns_empty():
    bm = _bm_triangle()
    assert enumerate_net_coupled_triples(bm, top_k=0) == []


def test_fixed_macros_excluded_from_clusters():
    """Fixed-hard macro 0 must not appear in any cluster."""
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]],
        fixed_mask=[True, False, False, False],
    )
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    for a, b, c, _ in triples:
        assert a != 0 and b != 0 and c != 0, "fixed macro 0 must not appear in clusters"


def test_no_clusters_when_all_macros_fixed():
    bm = make_benchmark(
        n_hard=3, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
        fixed_mask=[True, True, True],
    )
    assert enumerate_net_coupled_triples(bm, top_k=10) == []


def test_no_clusters_when_fewer_than_three_macros():
    bm = make_benchmark(
        n_hard=2, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1]],
    )
    assert enumerate_net_coupled_triples(bm, top_k=10) == []


def test_no_clusters_when_no_nets():
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0, net_nodes=[])
    assert enumerate_net_coupled_triples(bm, top_k=10) == []


def test_no_cluster_when_no_full_triangle():
    """A chain 0—1—2 has pairs (0,1) and (1,2) but NOT (0,2): no triple."""
    bm = make_benchmark(
        n_hard=3, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2]],
    )
    assert enumerate_net_coupled_triples(bm, top_k=10) == []


def test_tie_broken_by_macro_ids():
    """Two triangles with equal aggregate coupling — lower (a,b,c) comes first."""
    bm = make_benchmark(
        n_hard=6, canvas=200.0, macro_size=10.0,
        net_nodes=[
            [0, 1], [0, 2], [1, 2],  # triangle (0,1,2) score=3
            [3, 4], [3, 5], [4, 5],  # triangle (3,4,5) score=3
        ],
    )
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    assert len(triples) == 2
    assert triples[0][:3] == (0, 1, 2)
    assert triples[1][:3] == (3, 4, 5)


def test_ranks_by_aggregate_net_coupling():
    """Denser triangle should rank before sparser one."""
    bm = _bm_two_triangles()
    triples = enumerate_net_coupled_triples(bm, top_k=10)
    # (0,1,2) has score 6; (3,4,5) has score 3
    assert triples[0][3] >= triples[1][3]
    assert triples[0][:3] == (0, 1, 2)
