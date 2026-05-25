"""test_m3a_pair_orientation — pair ordering is stable macro-id; a < b always."""

import torch

from conftest import make_benchmark
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import generate_pair_candidates


def _winner(bm):
    return bm.macro_positions.clone().float()


def test_pair_a_is_always_smaller_id():
    bm = make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                        net_nodes=[[3, 0], [2, 1], [0, 3]])
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    for a, b, _ in pairs:
        assert a < b


def test_candidate_name_uses_stable_a_b_order():
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[3, 0]],
    )
    pairs = enumerate_net_coupled_pairs(bm, top_k=1)
    assert len(pairs) == 1
    a, b, _ = pairs[0]
    assert a < b, "pair orientation must place smaller id first"
    wp = _winner(bm)
    cands = generate_pair_candidates(bm, wp, a, b, pair_idx=0, existing_names=set())
    for c in cands:
        # Name should encode a then b.
        assert f"_{a}_{b}_" in c.name, f"Candidate name {c.name!r} does not encode (a={a}, b={b})"


def test_slice1_does_not_generate_symmetric_variants():
    """Slice 1 must not generate (b, a) variants — a is always the moved macro."""
    bm = make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 2], [1, 3]],
    )
    pairs = enumerate_net_coupled_pairs(bm, top_k=2)
    wp = _winner(bm)
    existing: set = set()
    for pi, (a, b, _) in enumerate(pairs):
        cands = generate_pair_candidates(bm, wp, a, b, pi, existing)
        for c in cands:
            existing.add(c.name)
            # The name must encode (a, b), not (b, a).
            assert f"_{a}_{b}_" in c.name
            reverse_token = f"_{b}_{a}_"
            assert reverse_token not in c.name, (
                f"Symmetric (b, a) variant found: {c.name!r}"
            )
