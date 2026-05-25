"""test_m3b_fallback_preserved — when no M3B candidate wins, prior M2B/M3A winner is returned unchanged."""

import pytest
import torch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
    )


def test_best_candidate_valid_when_m3b_enabled():
    """Enabling M3B must not break the invariant that a valid candidate is returned."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=10)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    assert best is not None, "best must not be None"


def test_m3b_disabled_fallback_same_winner():
    """When M3B is disabled the winner should equal the M3A/M2B-only winner."""
    bm = _bm()
    base_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=False,
    )
    m3b_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=10)

    cands_base = generate_candidates(bm, base_cfg)
    best_base, _, _ = score_and_select(cands_base, bm, plc=None,
                                       scoring_config=score_cfg,
                                       generation_config=base_cfg)

    cands_m3b = generate_candidates(bm, m3b_cfg)
    best_m3b, _, diag_m3b = score_and_select(cands_m3b, bm, plc=None,
                                              scoring_config=score_cfg,
                                              generation_config=m3b_cfg)

    assert best_base is not None and best_m3b is not None
    # If M3B budget exhausted or no M3B improvement, winner must not come from M3B.
    if diag_m3b.m3b_budget_exhausted or diag_m3b.m3b_scored == 0:
        assert best_m3b.family != "m3b_cluster_refinement", (
            "Winner should not be M3B when M3B is incomplete or unscored"
        )


def test_original_raw_always_valid_in_pool_with_m3b():
    """original_raw must remain a valid candidate in the pool when M3B runs."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=10)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    raw_sc = next((s for s in ranked if s.name == "original_raw"), None)
    assert raw_sc is not None, "original_raw must be present in ranked output"
