"""test_m3a_determinism — two runs with same benchmark produce identical candidate list and winner."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
from submissions.solver.core.m3a_candidate_generation import generate_m3a_candidates_for_pairs


def _bm():
    return make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3]],
    )


def test_pair_enumeration_deterministic():
    bm = _bm()
    p1 = enumerate_net_coupled_pairs(bm, top_k=20)
    p2 = enumerate_net_coupled_pairs(bm, top_k=20)
    assert p1 == p2


def test_candidate_generation_deterministic():
    bm = _bm()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    c1 = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    c2 = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    assert [c.name for c in c1] == [c.name for c in c2]
    for a, b in zip(c1, c2):
        assert torch.allclose(a.positions, b.positions, atol=1e-7)


def test_full_pipeline_deterministic():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=50)

    def _run():
        cands = generate_candidates(bm, gen_cfg)
        return score_and_select(cands, bm, plc=None,
                                scoring_config=score_cfg,
                                generation_config=gen_cfg)

    best1, ranked1, diag1 = _run()
    best2, ranked2, diag2 = _run()

    assert best1.name == best2.name, "Winner name differs across runs"
    assert torch.allclose(best1.positions, best2.positions, atol=1e-7)

    names1 = [s.name for s in ranked1]
    names2 = [s.name for s in ranked2]
    assert names1 == names2, "Ranked list order differs across runs"

    assert diag1.m3a_pairs_considered == diag2.m3a_pairs_considered
    assert diag1.m3a_candidates_generated == diag2.m3a_candidates_generated


def test_m3a_candidate_names_deterministic():
    bm = _bm()
    wp = bm.macro_positions.clone().float()
    pairs = enumerate_net_coupled_pairs(bm, top_k=10)
    run1 = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    run2 = generate_m3a_candidates_for_pairs(bm, wp, pairs, set())
    assert [c.name for c in run1] == [c.name for c in run2]
