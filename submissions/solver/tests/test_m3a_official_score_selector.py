"""test_m3a_official_score_selector — selector picks best proxy_cost across all candidates."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
    CandidatePlacement,
    ScoredCandidate,
)
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                          net_nodes=[[0, 1], [1, 2]])


def test_selector_picks_lowest_proxy_cost():
    """The returned best candidate must have the lowest proxy_cost among valid scored."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    valid_scored = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if not valid_scored:
        pytest.skip("No valid scored candidates")
    min_cost = min(s.proxy_cost for s in valid_scored)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - min_cost) < 1e-9, (
        f"best.proxy_cost={best.proxy_cost} != min_cost={min_cost}"
    )


def test_m3a_candidate_can_win_when_it_has_lower_cost():
    """If an M3A candidate is scored with a lower cost, it wins."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
        m3a_score_budget=50,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    # Check invariant: best is valid and has minimum proxy_cost.
    valid_scored = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if not valid_scored:
        pytest.skip("No valid scored candidates")
    min_cost = min(s.proxy_cost for s in valid_scored)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - min_cost) < 1e-9


def test_proxy_cost_is_sole_selector_not_heuristic():
    """Winner is determined by proxy_cost, not by M3A heuristic order."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    valid_scored = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if len(valid_scored) < 2:
        pytest.skip("Need at least 2 valid scored candidates")
    costs = sorted(s.proxy_cost for s in valid_scored)
    assert abs(best.proxy_cost - costs[0]) < 1e-9
