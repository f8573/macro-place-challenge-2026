"""test_m3b_profile_wiring — M3B profiles registered; old profiles unchanged; M3B disabled leaves M3A/M2B untouched."""

import pytest

from submissions.solver.scripts.run_benchmarks import _PROFILES
from submissions.solver.core.candidate_types import CandidateGenerationConfig


def test_m3b_smoke_profile_registered():
    assert "m3b-smoke" in _PROFILES, "m3b-smoke profile must be registered"


def test_m3b_default_profile_registered():
    assert "m3b-default" in _PROFILES, "m3b-default profile must be registered"


def test_m3b_budget_stress_profile_registered():
    assert "m3b-budget-stress" in _PROFILES, "m3b-budget-stress profile must be registered"


def test_m2b_final_profile_unchanged():
    """m2b-final profile must not have m3b_cluster_refinement enabled."""
    p = _PROFILES["m2b-final"]
    assert not p.get("m3b_cluster_refinement", False), (
        "m2b-final must not enable M3B"
    )


def test_m3a_default_profile_unchanged():
    """m3a-default profile must not have m3b_cluster_refinement enabled."""
    p = _PROFILES["m3a-default"]
    assert not p.get("m3b_cluster_refinement", False), (
        "m3a-default must not enable M3B"
    )


def test_m3b_smoke_enables_m3b():
    p = _PROFILES["m3b-smoke"]
    assert p.get("m3b_cluster_refinement", False), "m3b-smoke must enable m3b_cluster_refinement"


def test_m3b_default_enables_m3b():
    p = _PROFILES["m3b-default"]
    assert p.get("m3b_cluster_refinement", False), "m3b-default must enable m3b_cluster_refinement"


def test_m3b_budget_stress_enables_m3b():
    p = _PROFILES["m3b-budget-stress"]
    assert p.get("m3b_cluster_refinement", False), "m3b-budget-stress must enable m3b_cluster_refinement"


def test_m3b_smoke_small_top_k():
    p = _PROFILES["m3b-smoke"]
    assert p.get("m3b_top_k_clusters", 999) <= 16, (
        "m3b-smoke must use a small m3b_top_k_clusters (≤16)"
    )


def test_m3b_budget_stress_tight_budget():
    """m3b-budget-stress must use a very small max_official_scores."""
    p = _PROFILES["m3b-budget-stress"]
    assert p.get("max_official_scores", 9999) <= 10, (
        "m3b-budget-stress must have max_official_scores ≤ 10"
    )


def test_candidate_generation_config_has_m3b_fields():
    """CandidateGenerationConfig must accept and hold M3B fields."""
    cfg = CandidateGenerationConfig(
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=16,
        m3b_score_budget=8,
    )
    assert cfg.m3b_cluster_refinement is True
    assert cfg.m3b_top_k_clusters == 16
    assert cfg.m3b_score_budget == 8


def test_candidate_generation_config_m3b_disabled_by_default():
    cfg = CandidateGenerationConfig()
    assert cfg.m3b_cluster_refinement is False


def test_m3b_disabled_leaves_m2b_config_untouched():
    """With M3B disabled, config must equal m2b-final config fields."""
    p = _PROFILES["m2b-final"]
    cfg = CandidateGenerationConfig(
        candidate_budget=p.get("candidate_budget"),
        neighborhood_macro_limit=p.get("neighborhood_macro_limit", 20),
        neighborhood_step_profile=p.get("neighborhood_step_profile", "medium"),
        only_original_neighborhood=p.get("only_original_neighborhood", False),
        refinement_around_winners=p.get("refinement_around_winners", False),
        refinement_top_k=p.get("refinement_top_k", 5),
        line_search_around_winners=p.get("line_search_around_winners", False),
        m3b_cluster_refinement=False,
    )
    assert cfg.m3b_cluster_refinement is False
    assert cfg.m3a_pair_refinement is False  # m2b-final doesn't enable M3A either


def test_m3b_profiles_include_m3a_enabled():
    """m3b-default and m3b-smoke should also enable M3A for a full pipeline test."""
    for profile_name in ("m3b-smoke", "m3b-default"):
        p = _PROFILES[profile_name]
        assert p.get("m3a_pair_refinement", False), (
            f"{profile_name} should enable m3a_pair_refinement for full pipeline"
        )
