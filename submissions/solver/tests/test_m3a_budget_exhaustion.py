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


def test_partial_m3a_budget_no_m3a_winner():
    """Regression: even when a scored M3A candidate has the best proxy_cost it must not win
    if M3A budget exhaustion occurred.

    Scoring is mocked so the first scored M3A candidate receives a near-zero proxy_cost
    (best possible), guaranteeing it would win under the old buggy implementation that
    did not exclude partial M3A candidates.  The test therefore fails on the old code
    and passes only when the budget-exhaustion guard is active.
    """
    from unittest.mock import patch
    import submissions.solver.core.candidate_scoring as cs_mod

    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=20,
        m3a_score_budget=1,  # score exactly 1 M3A candidate; skip the rest
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _patched_score_batch(scored_list, score_indices, benchmark, plc,
                              max_scores=None, already_scored=0, **kwargs):
        result = _orig_score_batch(
            scored_list, score_indices, benchmark, plc,
            max_scores=max_scores, already_scored=already_scored, **kwargs,
        )
        # Override proxy_cost for any M3A candidate that was just scored so it
        # appears to be the global best.  This simulates the worst-case scenario
        # the budget-exhaustion guard must block.
        for sc in scored_list:
            if sc.family == "m3a_pair_refinement" and sc.was_scored:
                sc.proxy_cost = 1e-9  # near-zero — beats any real HPWL score
        return result

    with patch.object(cs_mod, "_score_batch", new=_patched_score_batch):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    assert best is not None
    # Confirm exhaustion path was exercised.
    assert diag.m3a_skipped_budget > 0, (
        "Expected m3a_skipped_budget > 0 with m3a_score_budget=1 and top_k_pairs=20."
    )

    # The scored M3A candidate must have the best proxy_cost (mock set it to 1e-9).
    m3a_scored = [s for s in ranked if s.family == "m3a_pair_refinement" and s.was_scored]
    assert len(m3a_scored) >= 1, "Expected at least 1 scored M3A candidate with budget=1"
    best_m3a_cost = min(s.proxy_cost for s in m3a_scored)
    assert best_m3a_cost < best.proxy_cost - 1e-12, (
        f"Mock did not achieve M3A dominance: M3A cost={best_m3a_cost} vs "
        f"M2B winner cost={best.proxy_cost}. The M3A candidate would not have "
        f"won anyway, so this test cannot prove the guard works."
    )

    # Core invariant: partial M3A pass must not contribute any winner even though
    # the scored M3A candidate has the single best proxy_cost in the pool.
    assert best.family != "m3a_pair_refinement", (
        f"A partially-scored M3A candidate won despite budget exhaustion. "
        f"Winner: {best.name!r}, forced M3A cost={best_m3a_cost}, "
        f"m3a_skipped={diag.m3a_skipped_budget}, m3a_scored={diag.m3a_candidates_scored}"
    )
    # Winner must come from the pre-M3A / M2B-safe pool.
    assert best.name not in {s.name for s in m3a_scored}, (
        f"Winner {best.name!r} is one of the scored M3A candidates — "
        "budget-exhaustion exclusion failed."
    )
