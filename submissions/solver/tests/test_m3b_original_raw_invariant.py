"""test_m3b_original_raw_invariant — original_raw remains in the candidate pool unconditionally."""

import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2]],
    )


def test_original_raw_present_when_m3b_enabled():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=20)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    names = [s.name for s in ranked]
    assert "original_raw" in names, "original_raw must be present with M3B enabled"


def test_original_raw_present_when_m3b_disabled():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=False,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=20)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    names = [s.name for s in ranked]
    assert "original_raw" in names, "original_raw must be present with M3B disabled"


def test_original_raw_present_with_zero_budget():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    names = [s.name for s in ranked]
    assert "original_raw" in names, "original_raw must be present even with zero budget"
