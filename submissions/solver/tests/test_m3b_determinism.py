"""test_m3b_determinism — two runs with same seed produce identical cluster list, candidates, winner."""

import pytest

from conftest import make_benchmark
from submissions.solver.core.m3b_cluster_enumeration import enumerate_net_coupled_triples
from submissions.solver.core.m3b_candidate_generation import generate_m3b_candidates_for_clusters
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=5, canvas=100.0, macro_size=10.0,
        net_nodes=[
            [0, 1], [0, 2], [1, 2],
            [1, 3], [2, 3],
            [2, 4], [3, 4],
        ],
    )


def test_cluster_list_deterministic():
    bm = _bm()
    t1 = enumerate_net_coupled_triples(bm, top_k=20)
    t2 = enumerate_net_coupled_triples(bm, top_k=20)
    assert t1 == t2


def test_candidate_list_deterministic():
    bm = _bm()
    pos = bm.macro_positions.clone().float()
    triples = enumerate_net_coupled_triples(bm, top_k=20)

    cands1 = generate_m3b_candidates_for_clusters(bm, pos, triples, set())
    cands2 = generate_m3b_candidates_for_clusters(bm, pos, triples, set())

    assert [c.name for c in cands1] == [c.name for c in cands2], (
        "Candidate name lists differ between calls"
    )
    for c1, c2 in zip(cands1, cands2):
        diff = (c1.positions - c2.positions).abs().max().item()
        assert diff < 1e-7, f"Candidate {c1.name} positions differ: max_diff={diff}"


def test_winner_deterministic():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=20)

    cands1 = generate_candidates(bm, gen_cfg)
    best1, _, _ = score_and_select(cands1, bm, plc=None,
                                   scoring_config=score_cfg,
                                   generation_config=gen_cfg)

    cands2 = generate_candidates(bm, gen_cfg)
    best2, _, _ = score_and_select(cands2, bm, plc=None,
                                   scoring_config=score_cfg,
                                   generation_config=gen_cfg)

    assert best1 is not None and best2 is not None
    assert best1.name == best2.name, (
        f"Winner differs between runs: {best1.name!r} vs {best2.name!r}"
    )


def test_cluster_ordering_stable():
    """Cluster ordering must be stable regardless of dict iteration order."""
    bm = _bm()
    for _ in range(5):
        t = enumerate_net_coupled_triples(bm, top_k=20)
        assert t == enumerate_net_coupled_triples(bm, top_k=20)
