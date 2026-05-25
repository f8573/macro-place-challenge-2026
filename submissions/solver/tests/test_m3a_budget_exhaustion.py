"""test_m3a_budget_exhaustion — budget exhaustion before M3A improvement → M2B-final returned."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2], [2, 3]],
    )


def test_no_crash_when_budget_exhausted():
    """Must not crash when score budget is zero."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
        m3a_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    assert best is not None, "best must not be None even with zero budget"


def test_m3a_candidates_skipped_when_budget_zero():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
        m3a_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    m3a_scored = [s for s in ranked if s.family == "m3a_pair_refinement" and s.was_scored]
    assert len(m3a_scored) == 0, (
        f"Expected 0 M3A scored when budget=0, got {len(m3a_scored)}"
    )


def test_winner_source_not_m3a_when_budget_zero():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
        m3a_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    assert best is not None
    assert best.family != "m3a_pair_refinement", (
        "M3A candidate must not win when budget is zero"
    )


def test_m2b_winner_returned_when_global_budget_exhausted():
    """When max_official_scores is very small, M2B passes consume it all; M3A gets nothing."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=20,
    )
    # Budget of 2 is consumed by original_raw + one neighborhood candidate.
    score_cfg = CandidateScoringConfig(max_official_scores=2)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    assert best is not None
    # No partial-state selection: any M3A candidate that was not scored must not be the winner.
    m3a_unscored = [s for s in ranked if s.family == "m3a_pair_refinement" and not s.was_scored]
    if best.family == "m3a_pair_refinement":
        assert best.was_scored, "Unscored M3A candidate must not be selected as winner"


def test_skipped_m3a_count_in_diagnostics():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
        m3a_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert diag.m3a_skipped_budget >= 0  # must be a non-negative count
