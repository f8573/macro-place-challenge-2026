"""test_m3b_budget_exhaustion — partial M3B candidate must not win on budget exhaustion."""

import pytest
from unittest.mock import patch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
import submissions.solver.core.candidate_scoring as cs_mod
from submissions.solver.core.candidate_scoring import score_and_select


def _bm():
    return make_benchmark(
        n_hard=4, canvas=100.0, macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
    )


def test_no_crash_when_m3b_budget_zero():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert best is not None


def test_m3b_candidates_not_scored_when_budget_zero():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    m3b_scored = [s for s in ranked if s.family == "m3b_cluster_refinement" and s.was_scored]
    assert len(m3b_scored) == 0, f"Expected 0 M3B scored with budget=0, got {len(m3b_scored)}"


def test_winner_not_m3b_when_budget_zero():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=0)
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, _diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert best is not None
    assert best.family != "m3b_cluster_refinement", "M3B candidate must not win when budget=0"


def test_budget_exhausted_flag_set():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    # m3b_score_budget=0 means any valid M3B candidate is skipped → exhausted.
    if diag.m3b_candidates_generated > 0:
        m3b_valid_exists = diag.m3b_valid > 0
        if m3b_valid_exists:
            assert diag.m3b_budget_exhausted, (
                "Expected m3b_budget_exhausted=True when budget=0 and valid M3B candidates exist"
            )


def test_partial_m3b_budget_no_m3b_winner():
    """Regression: even when a scored M3B candidate has the best proxy_cost it must not win
    if M3B budget exhaustion occurred.

    Scoring is mocked so the first scored M3B candidate receives a near-zero proxy_cost
    (best possible), guaranteeing it would win under an implementation that does not
    exclude partial M3B candidates.  The test fails on broken code and passes only when
    the budget-exhaustion guard is active.
    """
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=20,
        m3b_score_budget=1,  # score exactly 1 M3B candidate; skip the rest
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
        # Override proxy_cost for any M3B candidate that was scored to make it
        # appear to be the global best — this is the worst-case the guard must block.
        for sc in scored_list:
            if sc.family == "m3b_cluster_refinement" and sc.was_scored:
                sc.proxy_cost = 1e-9
        return result

    with patch.object(cs_mod, "_score_batch", new=_patched_score_batch):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    assert best is not None

    # Confirm exhaustion path was exercised.
    assert diag.m3b_skipped_budget > 0, (
        "Expected m3b_skipped_budget > 0 with m3b_score_budget=1 and many clusters."
    )

    # The scored M3B candidate must have the best proxy_cost (mock set it to 1e-9).
    m3b_scored = [s for s in ranked if s.family == "m3b_cluster_refinement" and s.was_scored]
    assert len(m3b_scored) >= 1, "Expected ≥1 scored M3B candidate with budget=1"
    best_m3b_cost = min(s.proxy_cost for s in m3b_scored)
    assert best_m3b_cost < best.proxy_cost - 1e-12, (
        f"Mock did not achieve M3B dominance: M3B cost={best_m3b_cost} vs "
        f"winner cost={best.proxy_cost}. Cannot prove the guard works."
    )

    # Core invariant: partial M3B pass must not contribute any winner.
    assert best.family != "m3b_cluster_refinement", (
        f"A partially-scored M3B candidate won despite budget exhaustion. "
        f"Winner: {best.name!r}, M3B cost={best_m3b_cost}, "
        f"m3b_skipped={diag.m3b_skipped_budget}"
    )
    assert best.name not in {s.name for s in m3b_scored}, (
        f"Winner {best.name!r} is one of the scored M3B candidates — "
        "budget-exhaustion exclusion failed."
    )


def test_m3a_candidates_unaffected_by_m3b_exhaustion():
    """M3B budget exhaustion must not block M3A candidates from winning."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3b_score_budget=0,  # M3B exhausted
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _force_m3a_best(scored_list, score_indices, benchmark, plc, **kwargs):
        result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)
        for sc in scored_list:
            if sc.family == "m3a_pair_refinement" and sc.valid and sc.was_scored:
                sc.proxy_cost = 1e-9
                break
        return result

    with patch.object(cs_mod, "_score_batch", new=_force_m3a_best):
        best, _ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    # M3A must still be selectable even when M3B is exhausted.
    # (M3A exhaustion is a separate guard — here M3A itself is not exhausted.)
    assert best is not None
    if not diag.m3a_skipped_budget:
        # M3A is complete — if it achieved best cost it should win.
        m3a_scored = [s for s in _ranked if s.family == "m3a_pair_refinement" and s.was_scored and s.valid]
        if m3a_scored:
            best_m3a = min(s.proxy_cost for s in m3a_scored)
            if abs(best_m3a - 1e-9) < 1e-12:
                assert best.family in ("m3a_pair_refinement",), (
                    "M3A should win when M3B is exhausted and M3A achieved best cost"
                )
