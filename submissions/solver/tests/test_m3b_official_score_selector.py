"""test_m3b_official_score_selector — selector picks by official proxy score across all families."""

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


def test_selector_uses_proxy_cost_not_heuristic():
    """Winner is determined solely by proxy_cost; no heuristic override is allowed."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=30)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)

    if not diag.scoring_available or diag.score_is_degenerate:
        return  # can't verify ordering without discriminating scores

    valid_scored = [s for s in ranked if s.valid and s.was_scored and s.proxy_cost is not None]
    if not valid_scored:
        return

    best_cost = min(s.proxy_cost for s in valid_scored)
    assert best.proxy_cost is not None
    assert abs(best.proxy_cost - best_cost) < 1e-9 or diag.selected_due_to in (
        "tie_break", "validity_only", "fallback_original",
        "fallback_legalized_original", "fallback_other_valid",
        "no_valid_scored_candidate",
    ), (
        f"Winner cost {best.proxy_cost} does not equal global min {best_cost}; "
        f"selected_due_to={diag.selected_due_to!r}"
    )


def test_m3b_can_win_when_it_scores_best():
    """When an M3B candidate receives the dominant proxy_cost it must win."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=100)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _patched(scored_list, score_indices, benchmark, plc, **kwargs):
        result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)
        # Force the first valid M3B candidate to have the best possible cost.
        for sc in scored_list:
            if sc.family == "m3b_cluster_refinement" and sc.valid and sc.was_scored:
                sc.proxy_cost = 1e-9
                break
        return result

    with patch.object(cs_mod, "_score_batch", new=_patched):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    if diag.m3b_budget_exhausted:
        pytest.skip("M3B budget exhausted — cannot verify M3B win in this run")

    m3b_scored = [s for s in ranked if s.family == "m3b_cluster_refinement" and s.was_scored and s.valid]
    if not m3b_scored:
        pytest.skip("no scored M3B candidates")

    # If M3B achieved the minimum cost, the winner must be M3B.
    m3b_best_cost = min(s.proxy_cost for s in m3b_scored)
    all_valid_scored = [s for s in ranked if s.valid and s.was_scored and s.proxy_cost is not None
                        and s.name != "original_legalized"]
    global_min = min(s.proxy_cost for s in all_valid_scored)
    if abs(m3b_best_cost - global_min) < 1e-12:
        assert best.family == "m3b_cluster_refinement", (
            f"M3B has the best cost ({m3b_best_cost}) but winner is {best.family!r}"
        )


def test_no_heuristic_overrides_official_score():
    """Official score determines winner — heuristic metadata must not override it."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=30)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)

    assert best is not None
    if best.valid and best.proxy_cost is not None and best.was_scored:
        # No other valid scored candidate (excluding original_legalized) should have lower cost.
        valid_scored = [
            s for s in ranked
            if s.valid and s.was_scored and s.proxy_cost is not None
            and s.name != "original_legalized"
        ]
        for s in valid_scored:
            assert s.proxy_cost >= best.proxy_cost - 1e-9, (
                f"Candidate {s.name!r} (cost={s.proxy_cost}) has lower cost than winner "
                f"{best.name!r} (cost={best.proxy_cost})"
            )
