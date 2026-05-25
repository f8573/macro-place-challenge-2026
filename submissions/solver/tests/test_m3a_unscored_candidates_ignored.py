"""test_m3a_unscored_candidates_ignored — unscored M3A candidates cannot be selected."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [1, 2]],
    )


def test_unscored_candidate_never_wins():
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
    # The winner must have been scored.
    assert best is not None
    if best.proxy_cost is not None:
        assert best.was_scored, f"Winner {best.name} has proxy_cost but was_scored=False"


def test_unscored_candidates_have_skip_reason():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=8,
        m3a_score_budget=3,  # limit to force some unscored
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    for sc in ranked:
        if not sc.was_scored:
            reason = sc.metadata.get("skip_reason", "")
            assert reason != "", (
                f"Unscored candidate {sc.name} has no skip_reason"
            )
            assert reason != "scored", (
                f"Unscored candidate {sc.name} has skip_reason='scored'"
            )


def test_proxy_cost_none_implies_not_selected():
    """A candidate with proxy_cost=None should never be returned as best when scored ones exist."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=4,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    valid_with_cost = [s for s in ranked if s.valid and s.proxy_cost is not None and s.was_scored]
    if valid_with_cost:
        assert best.proxy_cost is not None, "Best must have a proxy_cost when scored candidates exist"
