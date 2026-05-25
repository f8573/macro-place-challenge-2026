"""test_m3a_smoke_profile — m3a-smoke pipeline completes without crash and respects invariants."""

import torch
import pytest

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.candidate_scoring import score_and_select


def _smoke_configs():
    """Config matching the m3a-smoke profile (16 pairs, same budget as m2b-final)."""
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        candidate_budget=80,
        neighborhood_macro_limit=20,
        neighborhood_step_profile="medium",
        refinement_around_winners=True,
        refinement_top_k=5,
        refinement_combo_size=2,
        refinement_seed_strategy="diverse",
        refinement_exploration_seeds=1,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=2,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=16,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=60)
    return gen_cfg, score_cfg


def _bm_smoke():
    return make_benchmark(
        n_hard=6, canvas=200.0, macro_size=15.0,
        net_nodes=[[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [1, 3]],
    )


def test_smoke_completes_without_error():
    bm = _bm_smoke()
    gen_cfg, score_cfg = _smoke_configs()
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                          scoring_config=score_cfg,
                                          generation_config=gen_cfg)
    assert best is not None


def test_smoke_original_raw_in_pool():
    bm = _bm_smoke()
    gen_cfg, score_cfg = _smoke_configs()
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, _diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert any(s.name == "original_raw" for s in ranked)


def test_smoke_does_not_exceed_budget():
    bm = _bm_smoke()
    gen_cfg, score_cfg = _smoke_configs()
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    max_budget = score_cfg.max_official_scores or 999999
    assert diag.candidates_officially_scored <= max_budget, (
        f"scored {diag.candidates_officially_scored} > budget {max_budget}"
    )


def test_smoke_winner_is_valid():
    bm = _bm_smoke()
    gen_cfg, score_cfg = _smoke_configs()
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, _diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert best.valid, f"Winner {best.name} is invalid"


def test_smoke_m3a_diagnostics_populated():
    bm = _bm_smoke()
    gen_cfg, score_cfg = _smoke_configs()
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert diag.m3a_top_k_pairs == 16
    # Pairs are either 0 (no nets enough) or > 0.
    assert diag.m3a_pairs_considered >= 0
    assert diag.m3a_candidates_generated >= 0


def test_smoke_m3a_ties_or_beats_m2b():
    """Smoke run must not regress below m2b-final on same benchmark (local-proxy only)."""
    bm = _bm_smoke()

    # M2B result (no M3A)
    gen_m2b = CandidateGenerationConfig(
        only_original_neighborhood=True,
        candidate_budget=80,
        neighborhood_macro_limit=20,
        neighborhood_step_profile="medium",
        refinement_around_winners=True,
        refinement_top_k=5,
        refinement_combo_size=2,
        refinement_seed_strategy="diverse",
        refinement_exploration_seeds=1,
        line_search_around_winners=True,
        line_search_top_k=3,
        line_search_max_scale=4.0,
        line_search_stop_after_worse=2,
    )
    score_cfg = CandidateScoringConfig(max_official_scores=60)
    cands_m2b = generate_candidates(bm, gen_m2b)
    best_m2b, _, _ = score_and_select(cands_m2b, bm, plc=None,
                                      scoring_config=score_cfg,
                                      generation_config=gen_m2b)

    gen_cfg, score_cfg2 = _smoke_configs()
    cands_m3a = generate_candidates(bm, gen_cfg)
    best_m3a, _, _ = score_and_select(cands_m3a, bm, plc=None,
                                      scoring_config=score_cfg2,
                                      generation_config=gen_cfg)

    if best_m2b.proxy_cost is None or best_m3a.proxy_cost is None:
        pytest.skip("Scoring unavailable for comparison")

    assert best_m3a.proxy_cost <= best_m2b.proxy_cost + 1e-9, (
        f"m3a-smoke regressed: m3a={best_m3a.proxy_cost} > m2b={best_m2b.proxy_cost}"
    )
