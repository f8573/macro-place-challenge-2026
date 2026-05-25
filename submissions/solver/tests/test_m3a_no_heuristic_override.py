"""test_m3a_no_heuristic_override — M3A heuristic ranking cannot override official proxy score."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(n_hard=4, canvas=100.0, macro_size=10.0,
                          net_nodes=[[0, 1], [0, 2], [1, 3]])


def test_winner_selected_by_proxy_cost_not_pair_rank():
    """Even if the top-ranked pair yields worse candidates, the actual winner is decided by cost."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)

    valid_scored = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if not valid_scored:
        pytest.skip("No valid scored candidates")

    best_cost = min(s.proxy_cost for s in valid_scored)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - best_cost) < 1e-9, (
        "Winner was selected by something other than minimum proxy_cost"
    )


def test_m3a_family_tag_does_not_grant_priority():
    """Having family='m3a_pair_refinement' must not grant selection priority over cost."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)

    valid_scored = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if not valid_scored:
        pytest.skip("No valid scored candidates")

    min_cost = min(s.proxy_cost for s in valid_scored)
    # If an M3A candidate wins, it must genuinely have the lowest cost.
    if best.family == "m3a_pair_refinement":
        assert abs(best.proxy_cost - min_cost) < 1e-9, (
            f"M3A winner has cost {best.proxy_cost} but min is {min_cost}"
        )
    else:
        # Non-M3A won — verify no M3A candidate has a strictly lower cost.
        m3a_costs = [s.proxy_cost for s in valid_scored if s.family == "m3a_pair_refinement"]
        if m3a_costs:
            assert min(m3a_costs) >= best.proxy_cost - 1e-9
