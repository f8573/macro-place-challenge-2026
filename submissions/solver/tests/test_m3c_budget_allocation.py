"""test_m3c_budget_allocation — M3C slice-1: deterministic budget-aware score allocation."""

import pytest  # noqa: F401 — used for pytest.skip
from unittest.mock import patch

from conftest import make_benchmark
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig
from submissions.solver.core.candidates import generate_candidates
import submissions.solver.core.candidate_scoring as cs_mod
from submissions.solver.core.candidate_scoring import score_and_select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bm(n_hard=4):
    return make_benchmark(
        n_hard=n_hard,
        canvas=100.0,
        macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
    )


def _m3c_gen_cfg(**kwargs) -> CandidateGenerationConfig:
    defaults = dict(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3c_budget_allocation=True,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
        m3c_rollover_unused_budget=True,
    )
    defaults.update(kwargs)
    return CandidateGenerationConfig(**defaults)


def _score_cfg(max_scores=60) -> CandidateScoringConfig:
    return CandidateScoringConfig(max_official_scores=max_scores)


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------

def test_m3c_disabled_by_default():
    cfg = CandidateGenerationConfig()
    assert cfg.m3c_budget_allocation is False


def test_m3c_fields_exist():
    cfg = CandidateGenerationConfig(
        m3c_budget_allocation=True,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
        m3c_rollover_unused_budget=True,
    )
    assert cfg.m3c_budget_allocation is True
    assert cfg.m3c_pre_m3_budget == 50
    assert cfg.m3c_m3a_reserved_budget == 5
    assert cfg.m3c_m3b_reserved_budget == 5
    assert cfg.m3c_rollover_unused_budget is True


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------

def test_m3c_diagnostics_report_allocations():
    bm = _bm()
    gen_cfg = _m3c_gen_cfg()
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert diag.m3c_enabled is True
    assert diag.m3c_pre_m3_budget_alloc == 50
    assert diag.m3c_m3a_budget_alloc == 5
    assert diag.m3c_m3b_budget_alloc == 5


def test_m3c_disabled_leaves_diagnostics_false():
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(only_original_neighborhood=True)
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert diag.m3c_enabled is False
    assert diag.m3c_pre_m3_budget_alloc is None
    assert diag.m3c_m3a_budget_alloc is None
    assert diag.m3c_m3b_budget_alloc is None


def test_m3c_allocation_defaults_when_fields_none():
    """When m3c_pre_m3_budget/m3a/m3b are None, defaults (5/5/derived) are used."""
    bm = _bm()
    gen_cfg = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3c_budget_allocation=True,
        m3c_pre_m3_budget=None,
        m3c_m3a_reserved_budget=None,
        m3c_m3b_reserved_budget=None,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert diag.m3c_enabled is True
    assert diag.m3c_m3a_budget_alloc == 5
    assert diag.m3c_m3b_budget_alloc == 5
    assert diag.m3c_pre_m3_budget_alloc == 50  # 60 - 5 - 5


# ---------------------------------------------------------------------------
# No budget expansion
# ---------------------------------------------------------------------------

def _count_fresh_scores(ranked):
    return sum(1 for s in ranked if s.was_scored and not s.metadata.get("cache_hit"))


def test_m3c_total_fresh_scores_within_budget():
    """Total fresh official scores must never exceed max_official_scores."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert diag.fresh_official_scores <= 60, (
        f"fresh_official_scores={diag.fresh_official_scores} exceeds max_official_scores=60"
    )


def test_m3c_total_never_exceeds_with_tight_budget():
    """With total alloc = max_official_scores, fresh scores never expand beyond cap."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=8,
        m3c_m3a_reserved_budget=1,
        m3c_m3b_reserved_budget=1,
    )
    score_cfg = _score_cfg(10)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert diag.fresh_official_scores <= 10, (
        f"fresh={diag.fresh_official_scores} > max=10"
    )


def test_m3c_guard_clamps_overallocated_profile():
    """If pre_m3 + m3a + m3b > max_official_scores, the guard clamps it down."""
    bm = _bm()
    # Intentionally over-allocate: 50 + 5 + 5 = 60 but max is only 20
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
    )
    score_cfg = _score_cfg(20)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert diag.fresh_official_scores <= 20, (
        f"fresh={diag.fresh_official_scores} exceeded clamped max=20"
    )


# ---------------------------------------------------------------------------
# M3A partial exclusion
# ---------------------------------------------------------------------------

def test_m3c_m3a_receives_reserved_budget():
    """With tight pre-M3 budget but reserved M3A budget, M3A candidates get scored."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=2,   # very tight pre-M3 pool
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=0,
    )
    score_cfg = _score_cfg(7)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    m3a_scored = [s for s in ranked if s.family == "m3a_pair_refinement" and s.was_scored]
    assert len(m3a_scored) > 0, (
        "M3A candidates must receive scoring opportunities from the reserved budget "
        "even when pre-M3 pool is exhausted"
    )


def test_m3c_m3a_excluded_when_budget_exhausted():
    """M3A excluded when the admitted frontier itself is partially scored (true budget exhaustion).

    Uses a mock that limits scoring to 1 fresh score within a frontier of ≥2 candidates,
    simulating mid-frontier budget exhaustion. Outside-frontier candidates (m3c_not_admitted)
    do NOT trigger this exclusion — only within-frontier budget_exceeded does.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3a_top_k_pairs=20,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=3,  # frontier = 3; mock limits to 1 → true exhaustion
        m3c_m3b_reserved_budget=0,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _throttle_m3a_frontier(scored_list, score_indices, benchmark, plc, **kwargs):
        indices = score_indices if isinstance(score_indices, list) else sorted(score_indices)
        is_m3a = bool(indices) and scored_list[indices[0]].family == "m3a_pair_refinement"
        if is_m3a and len(indices) > 1:
            # Limit to 1 fresh score within the frontier → admitted-frontier exhaustion
            kw = dict(kwargs)
            kw["max_scores"] = 1
            result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kw)
            for sc in scored_list:
                if sc.family == "m3a_pair_refinement" and sc.was_scored:
                    sc.proxy_cost = 1e-9  # make scored M3A appear best
            return result
        return _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)

    with patch.object(cs_mod, "_score_batch", new=_throttle_m3a_frontier):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    assert best is not None
    if diag.m3a_skipped_budget > 0:
        # Admitted-frontier exhaustion — M3A must be excluded from selection
        assert best.family != "m3a_pair_refinement", (
            f"Partial M3A frontier won despite admitted-frontier exhaustion. "
            f"m3a_skipped={diag.m3a_skipped_budget}"
        )


# ---------------------------------------------------------------------------
# M3B partial exclusion
# ---------------------------------------------------------------------------

def test_m3c_m3b_receives_reserved_budget():
    """With tight pre-M3 budget but reserved M3B budget, M3B candidates get scored."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=2,   # very tight pre-M3 pool
        m3c_m3a_reserved_budget=0,
        m3c_m3b_reserved_budget=5,
        m3c_rollover_unused_budget=False,  # no rollover to simplify
    )
    score_cfg = _score_cfg(7)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    m3b_scored = [s for s in ranked if s.family == "m3b_cluster_refinement" and s.was_scored]
    assert len(m3b_scored) > 0, (
        "M3B candidates must receive scoring opportunities from the reserved budget "
        "even when pre-M3 pool is exhausted"
    )


def test_m3c_m3b_excluded_when_budget_exhausted():
    """M3B excluded when the admitted frontier itself is partially scored (true budget exhaustion).

    Uses a mock that limits scoring to 1 fresh score within a frontier of ≥2 candidates,
    simulating mid-frontier budget exhaustion. Outside-frontier candidates (m3c_not_admitted)
    do NOT trigger this exclusion — only within-frontier budget_exceeded does.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3b_top_k_clusters=20,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=3,  # frontier = 3; mock limits to 1 → true exhaustion
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _throttle_m3b_frontier(scored_list, score_indices, benchmark, plc, **kwargs):
        indices = score_indices if isinstance(score_indices, list) else sorted(score_indices)
        is_m3b = bool(indices) and scored_list[indices[0]].family == "m3b_cluster_refinement"
        if is_m3b and len(indices) > 1:
            # Limit to 1 fresh score within the frontier → admitted-frontier exhaustion
            kw = dict(kwargs)
            kw["max_scores"] = 1
            result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kw)
            for sc in scored_list:
                if sc.family == "m3b_cluster_refinement" and sc.was_scored:
                    sc.proxy_cost = 1e-9  # make scored M3B appear best
            return result
        return _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)

    with patch.object(cs_mod, "_score_batch", new=_throttle_m3b_frontier):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    assert best is not None
    if diag.m3b_skipped_budget > 0:
        assert best.family != "m3b_cluster_refinement", (
            f"Partial M3B frontier won despite admitted-frontier exhaustion. "
            f"m3b_skipped={diag.m3b_skipped_budget}"
        )


# ---------------------------------------------------------------------------
# Rollover
# ---------------------------------------------------------------------------

def test_m3c_rollover_unused_m3a_to_m3b():
    """When M3A uses fewer slots than reserved, the remainder rolls over to M3B."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3a_pair_refinement=False,  # M3A disabled → uses 0 of 5 reserved slots
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=3,
        m3c_rollover_unused_budget=True,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    # M3A was disabled: all 5 reserved M3A slots roll to M3B → M3B had up to 8 slots
    assert diag.m3c_rollover_to_m3b == 5, (
        f"Expected rollover=5 (full M3A alloc) when M3A disabled, got {diag.m3c_rollover_to_m3b}"
    )


def test_m3c_no_rollover_when_disabled():
    """With rollover disabled, unused M3A budget does not go to M3B."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3a_pair_refinement=False,  # M3A disabled → 0 used
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=3,
        m3c_rollover_unused_budget=False,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)
    assert diag.m3c_rollover_to_m3b == 0, (
        f"Expected rollover=0 when rollover disabled, got {diag.m3c_rollover_to_m3b}"
    )


# ---------------------------------------------------------------------------
# Selector integrity
# ---------------------------------------------------------------------------

def test_m3c_best_is_always_valid_scored():
    """Final selected candidate must always be valid and scored when M3C is enabled."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg()
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)
    assert best is not None
    assert best.valid, f"Selected candidate {best.name!r} is invalid"
    # Scored or fallback; either way proxy_cost must be present when not fallback
    if diag.selected_due_to == "proxy_cost":
        assert best.proxy_cost is not None
        assert best.was_scored


def test_m3c_unscored_never_wins():
    """Unscored candidates must never be selected as the best candidate."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(cands, bm, plc=None,
                                           scoring_config=score_cfg,
                                           generation_config=gen_cfg)
    assert best is not None
    if diag.selected_due_to == "proxy_cost":
        assert best.was_scored, "Winner must have been scored when selected_due_to=proxy_cost"


# ---------------------------------------------------------------------------
# Profile wiring
# ---------------------------------------------------------------------------

def test_m3c_smoke_profile_registered():
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    assert "m3c-smoke" in _PROFILES, "m3c-smoke profile must be registered"


def test_m3c_default_profile_registered():
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    assert "m3c-default" in _PROFILES, "m3c-default profile must be registered"


def test_m3c_budget_stress_profile_registered():
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    assert "m3c-budget-stress" in _PROFILES, "m3c-budget-stress profile must be registered"


def test_m3c_profiles_have_budget_allocation_enabled():
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    for name in ("m3c-smoke", "m3c-default", "m3c-budget-stress"):
        p = _PROFILES[name]
        assert p.get("m3c_budget_allocation", False), (
            f"{name} must have m3c_budget_allocation=True"
        )


def test_m3c_default_profile_matches_spec():
    """m3c-default matches the required spec: max=60, pre_m3=50, m3a=5, m3b=5."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    p = _PROFILES["m3c-default"]
    assert p.get("max_official_scores") == 60
    assert p.get("m3c_pre_m3_budget") == 50
    assert p.get("m3c_m3a_reserved_budget") == 5
    assert p.get("m3c_m3b_reserved_budget") == 5
    assert p.get("m3c_rollover_unused_budget") is True


def test_m3c_budget_stress_tight_budget():
    """m3c-budget-stress must have max_official_scores ≤ 10."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    p = _PROFILES["m3c-budget-stress"]
    assert p.get("max_official_scores", 9999) <= 10, (
        "m3c-budget-stress must have max_official_scores ≤ 10"
    )


def test_m3c_smoke_small_top_k():
    """m3c-smoke must use small top-k counts for fast CI execution."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    p = _PROFILES["m3c-smoke"]
    assert p.get("m3a_top_k_pairs", 999) <= 16
    assert p.get("m3b_top_k_clusters", 999) <= 16


def test_m3c_default_enables_m3a_and_m3b():
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    p = _PROFILES["m3c-default"]
    assert p.get("m3a_pair_refinement", False), "m3c-default must enable M3A"
    assert p.get("m3b_cluster_refinement", False), "m3c-default must enable M3B"


def test_m3c_default_total_alloc_equals_max():
    """Pre-M3 + M3A + M3B must equal max_official_scores in m3c-default."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    p = _PROFILES["m3c-default"]
    total = p["m3c_pre_m3_budget"] + p["m3c_m3a_reserved_budget"] + p["m3c_m3b_reserved_budget"]
    assert total == p["max_official_scores"], (
        f"Budget slices {total} != max_official_scores {p['max_official_scores']}"
    )


def test_existing_profiles_m3c_disabled():
    """M3C must be disabled in all pre-M3C profiles."""
    from submissions.solver.scripts.run_benchmarks import _PROFILES
    frozen = ("m2b-final", "m3a-default", "m3a-smoke", "m3b-default", "m3b-smoke")
    for name in frozen:
        p = _PROFILES[name]
        assert not p.get("m3c_budget_allocation", False), (
            f"{name} must not enable M3C"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_m3c_determinism():
    """Two runs with identical config must return the same best candidate name."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg()
    score_cfg = _score_cfg(60)

    def _run():
        cands = generate_candidates(bm, gen_cfg)
        best, _ranked, _diag = score_and_select(cands, bm, plc=None,
                                                 scoring_config=score_cfg,
                                                 generation_config=gen_cfg)
        return best.name if best else None

    assert _run() == _run(), "M3C runs must be deterministic across identical invocations"


# ---------------------------------------------------------------------------
# Invariant: M3C disabled must not change existing M3B behavior
# ---------------------------------------------------------------------------

def test_m3c_disabled_behavior_unchanged():
    """With M3C disabled, score_and_select must behave identically to m3b-default config."""
    bm = _bm()
    gen_cfg_baseline = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3c_budget_allocation=False,
    )
    gen_cfg_m3c_off = CandidateGenerationConfig(
        only_original_neighborhood=True,
        m3a_pair_refinement=True,
        m3a_top_k_pairs=10,
        m3b_cluster_refinement=True,
        m3b_top_k_clusters=10,
        m3c_budget_allocation=False,
        m3c_pre_m3_budget=50,        # these fields must be ignored when m3c is off
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
    )
    score_cfg = _score_cfg(60)
    cands_a = generate_candidates(bm, gen_cfg_baseline)
    cands_b = generate_candidates(bm, gen_cfg_m3c_off)

    best_a, _r, diag_a = score_and_select(cands_a, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg_baseline)
    best_b, _r, diag_b = score_and_select(cands_b, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg_m3c_off)

    assert diag_a.m3c_enabled is False
    assert diag_b.m3c_enabled is False
    assert best_a.name == best_b.name, (
        "M3C=False with extra M3C fields must produce same result as M3C=False without them"
    )


# ---------------------------------------------------------------------------
# Regression: outside-frontier candidates must not cause family exclusion
# ---------------------------------------------------------------------------

def test_m3c_m3a_selectable_when_more_candidates_than_frontier():
    """Regression: M3A candidates outside the reserved frontier must not cause exclusion.

    When M3A generates more valid candidates than m3c_m3a_reserved_budget, only the
    top frontier candidates are admitted for scoring. The outside-frontier candidates
    must be marked m3c_not_admitted (not budget_exceeded), and M3A must remain selectable.

    This test fails under the buggy implementation where all valid candidates are passed
    to _score_batch and outside-frontier candidates are marked budget_exceeded.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3a_top_k_pairs=20,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=2,   # small frontier to ensure outside-frontier candidates
        m3c_m3b_reserved_budget=0,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _make_admitted_m3a_winner(scored_list, score_indices, benchmark, plc, **kwargs):
        result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)
        # Give the first admitted+scored M3A candidate the best proxy cost.
        for sc in scored_list:
            if sc.family == "m3a_pair_refinement" and sc.was_scored:
                sc.proxy_cost = 1e-9
                break
        return result

    with patch.object(cs_mod, "_score_batch", new=_make_admitted_m3a_winner):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    if diag.m3a_not_admitted_count == 0:
        pytest.skip("M3A did not generate enough valid candidates to exceed the frontier")

    # Outside-frontier candidates must NOT be marked budget_exceeded
    assert diag.m3a_skipped_budget == 0, (
        f"Outside-frontier M3A candidates must not be marked budget_exceeded. "
        f"m3a_skipped_budget={diag.m3a_skipped_budget}, "
        f"m3a_not_admitted={diag.m3a_not_admitted_count}"
    )
    # M3A must remain selectable when its frontier is fully scored
    assert diag.m3a_selectable, (
        f"M3A must be selectable when frontier is fully scored. "
        f"m3a_selectable={diag.m3a_selectable}, "
        f"m3a_not_admitted={diag.m3a_not_admitted_count}"
    )
    # Best admitted M3A candidate should win (mocked cost = 1e-9)
    assert best.family == "m3a_pair_refinement", (
        f"Expected M3A candidate to win (mock gave best cost) but got {best.family!r}. "
        f"m3a_skipped={diag.m3a_skipped_budget}, m3a_selectable={diag.m3a_selectable}. "
        f"This test fails under the buggy implementation."
    )


def test_m3c_m3b_selectable_when_more_candidates_than_frontier():
    """Regression: M3B candidates outside the reserved frontier must not cause exclusion.

    When M3B generates more valid candidates than m3c_m3b_reserved_budget, only the
    top frontier candidates are admitted for scoring. Outside-frontier candidates must
    be marked m3c_not_admitted (not budget_exceeded), and M3B must remain selectable.

    This test fails under the buggy implementation where all valid candidates are passed
    to _score_batch and outside-frontier candidates are marked budget_exceeded.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3b_top_k_clusters=20,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=0,
        m3c_m3b_reserved_budget=2,   # small frontier to ensure outside-frontier candidates
        m3c_rollover_unused_budget=False,
    )
    score_cfg = _score_cfg(60)
    cands = generate_candidates(bm, gen_cfg)

    _orig_score_batch = cs_mod._score_batch

    def _make_admitted_m3b_winner(scored_list, score_indices, benchmark, plc, **kwargs):
        result = _orig_score_batch(scored_list, score_indices, benchmark, plc, **kwargs)
        # Give the first admitted+scored M3B candidate the best proxy cost.
        for sc in scored_list:
            if sc.family == "m3b_cluster_refinement" and sc.was_scored:
                sc.proxy_cost = 1e-9
                break
        return result

    with patch.object(cs_mod, "_score_batch", new=_make_admitted_m3b_winner):
        best, ranked, diag = score_and_select(
            cands, bm, plc=None,
            scoring_config=score_cfg,
            generation_config=gen_cfg,
        )

    if diag.m3b_not_admitted_count == 0:
        pytest.skip("M3B did not generate enough valid candidates to exceed the frontier")

    # Outside-frontier candidates must NOT be marked budget_exceeded
    assert diag.m3b_skipped_budget == 0, (
        f"Outside-frontier M3B candidates must not be marked budget_exceeded. "
        f"m3b_skipped_budget={diag.m3b_skipped_budget}, "
        f"m3b_not_admitted={diag.m3b_not_admitted_count}"
    )
    # M3B must remain selectable when its frontier is fully scored
    assert diag.m3b_selectable > 0, (
        f"M3B must be selectable when frontier is fully scored. "
        f"m3b_selectable={diag.m3b_selectable}, "
        f"m3b_not_admitted={diag.m3b_not_admitted_count}"
    )
    # Best admitted M3B candidate should win (mocked cost = 1e-9)
    assert best.family == "m3b_cluster_refinement", (
        f"Expected M3B candidate to win (mock gave best cost) but got {best.family!r}. "
        f"m3b_skipped={diag.m3b_skipped_budget}, m3b_selectable={diag.m3b_selectable}. "
        f"This test fails under the buggy implementation."
    )


def test_m3c_config_guard_normalizes_when_m3a_m3b_exceed_max():
    """Config guard: when M3A+M3B reserved budgets exceed max_official_scores, allocation is
    normalized so total fresh scores never exceed max_official_scores.

    With late-stage-reservation-first normalization: M3A gets up to max, M3B gets the
    remainder, pre-M3 gets whatever is left. Total is always ≤ max_official_scores.
    """
    bm = _bm()
    # Configured M3A=5 + M3B=5 = 10 > max=6; allocation must be clamped
    gen_cfg = _m3c_gen_cfg(
        m3a_top_k_pairs=10,
        m3b_top_k_clusters=10,
        m3c_pre_m3_budget=50,
        m3c_m3a_reserved_budget=5,
        m3c_m3b_reserved_budget=5,
    )
    score_cfg = _score_cfg(max_scores=6)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(
        cands, bm, plc=None,
        scoring_config=score_cfg,
        generation_config=gen_cfg,
    )

    # Total fresh scores must never exceed max_official_scores
    assert diag.fresh_official_scores <= 6, (
        f"fresh_official_scores={diag.fresh_official_scores} exceeded max_official_scores=6"
    )
    # Diagnostics must report normalized allocations
    assert diag.m3c_enabled is True
    total_alloc = (
        (diag.m3c_pre_m3_budget_alloc or 0)
        + (diag.m3c_m3a_budget_alloc or 0)
        + (diag.m3c_m3b_budget_alloc or 0)
    )
    assert total_alloc <= 6, (
        f"Reported allocation total={total_alloc} exceeds max_official_scores=6. "
        f"pre_m3={diag.m3c_pre_m3_budget_alloc}, m3a={diag.m3c_m3a_budget_alloc}, "
        f"m3b={diag.m3c_m3b_budget_alloc}"
    )
    # M3A gets priority in late-stage-first normalization: min(5,6)=5
    assert diag.m3c_m3a_budget_alloc == 5, (
        f"Expected M3A alloc=5 (late-stage-first), got {diag.m3c_m3a_budget_alloc}"
    )
    # M3B gets the remainder: min(5,6-5)=1
    assert diag.m3c_m3b_budget_alloc == 1, (
        f"Expected M3B alloc=1 (6-5=1 remaining), got {diag.m3c_m3b_budget_alloc}"
    )
    # Pre-M3 gets nothing: min(50,0)=0
    assert diag.m3c_pre_m3_budget_alloc == 0, (
        f"Expected pre-M3 alloc=0 (no budget remaining), got {diag.m3c_pre_m3_budget_alloc}"
    )


# ---------------------------------------------------------------------------
# Regression: pre-M3 passes must draw from _m3c_pre_m3_alloc, not global max
# ---------------------------------------------------------------------------

def test_m3c_pre_m3_passes_respect_alloc_with_refinement_and_line_search():
    """Codex reproduction: max=10, pre=8, m3a=1, m3b=1 with refinement+line-search enabled.

    Without the fix:
      - pass1 can use up to 4 (seed budget ~32/60 of global 10 or 8)
      - pass2 remaining was computed from cfg.max_official_scores (10-4=6) not _pre_m3_alloc (8-4=4)
      - total could reach 12 > max_official_scores=10

    After the fix all pre-M3 passes draw from _m3c_pre_m3_alloc=8 exclusively.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=8,
        m3c_m3a_reserved_budget=1,
        m3c_m3b_reserved_budget=1,
        refinement_around_winners=True,
        refinement_top_k=2,
        line_search_around_winners=True,
        line_search_top_k=2,
    )
    score_cfg = _score_cfg(10)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)

    assert diag.fresh_official_scores <= 10, (
        f"Total fresh scores {diag.fresh_official_scores} exceeded max_official_scores=10"
    )
    assert diag.m3c_pre_m3_used <= 8, (
        f"Pre-M3 fresh scores {diag.m3c_pre_m3_used} exceeded _m3c_pre_m3_alloc=8"
    )
    total_post_m3 = diag.m3a_fresh_scores + diag.m3b_fresh_scores
    assert diag.m3c_pre_m3_used + total_post_m3 <= 10, (
        f"Pre-M3({diag.m3c_pre_m3_used}) + M3A/M3B({total_post_m3}) = "
        f"{diag.m3c_pre_m3_used + total_post_m3} > max=10"
    )
    assert diag.m3c_budget_invariant_holds, (
        f"m3c_budget_invariant_holds=False: pre_m3_used={diag.m3c_pre_m3_used}, "
        f"total={diag.fresh_official_scores}, max=10"
    )


def test_m3c_zero_pre_m3_alloc_fires_no_pre_m3_fresh_scores():
    """When _m3c_pre_m3_alloc=0, pre-M3 fresh scoring must be exactly 0.

    The hard minimum max(3, ...) must not fire when M3C is enabled. This test
    also verifies M3A/M3B still receive their reserved budgets.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=0,
        m3c_m3a_reserved_budget=3,
        m3c_m3b_reserved_budget=3,
        refinement_around_winners=True,
        refinement_top_k=2,
        line_search_around_winners=True,
        line_search_top_k=2,
    )
    score_cfg = _score_cfg(6)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)

    assert diag.m3c_pre_m3_budget_alloc == 0, (
        f"Expected pre_m3_alloc=0, got {diag.m3c_pre_m3_budget_alloc}"
    )
    assert diag.m3c_pre_m3_used == 0, (
        f"Pre-M3 fresh scores={diag.m3c_pre_m3_used} must be 0 when alloc=0. "
        f"Hard minimum must not fire when M3C is enabled."
    )
    assert diag.fresh_official_scores <= 6, (
        f"Total fresh scores {diag.fresh_official_scores} exceeded max_official_scores=6"
    )
    assert diag.m3c_budget_invariant_holds, (
        f"m3c_budget_invariant_holds=False: pre_m3_used={diag.m3c_pre_m3_used}, "
        f"total={diag.fresh_official_scores}, max=6"
    )


def test_m3c_arbitrary_overallocation_clamps_actual_fresh_scores():
    """When configured budgets far exceed global max, actual fresh scores stay within max.

    max=5, pre=10, m3a=10, m3b=10 → normalization clamps allocations to ≤5 total,
    and actual fresh official scores must be ≤5.
    """
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=10,
        m3c_m3a_reserved_budget=10,
        m3c_m3b_reserved_budget=10,
        refinement_around_winners=True,
        refinement_top_k=2,
        line_search_around_winners=True,
        line_search_top_k=2,
    )
    score_cfg = _score_cfg(5)
    cands = generate_candidates(bm, gen_cfg)
    _best, ranked, diag = score_and_select(cands, bm, plc=None,
                                            scoring_config=score_cfg,
                                            generation_config=gen_cfg)

    total_alloc = (
        (diag.m3c_pre_m3_budget_alloc or 0)
        + (diag.m3c_m3a_budget_alloc or 0)
        + (diag.m3c_m3b_budget_alloc or 0)
    )
    assert total_alloc <= 5, (
        f"Normalized allocation total={total_alloc} exceeds max_official_scores=5"
    )
    assert diag.fresh_official_scores <= 5, (
        f"Total fresh scores {diag.fresh_official_scores} exceeded max_official_scores=5"
    )
    assert diag.m3c_budget_invariant_holds, (
        f"m3c_budget_invariant_holds=False: pre_m3_used={diag.m3c_pre_m3_used}, "
        f"total={diag.fresh_official_scores}, max=5"
    )


def test_m3c_negative_budget_values_do_not_expand_total_budget():
    """Negative configured slices must not make later allocations exceed the global max."""
    bm = _bm()
    gen_cfg = _m3c_gen_cfg(
        m3c_pre_m3_budget=10,
        m3c_m3a_reserved_budget=-5,
        m3c_m3b_reserved_budget=0,
        refinement_around_winners=True,
        refinement_top_k=2,
        line_search_around_winners=True,
        line_search_top_k=2,
    )
    score_cfg = _score_cfg(5)
    cands = generate_candidates(bm, gen_cfg)
    _best, _ranked, diag = score_and_select(cands, bm, plc=None,
                                             scoring_config=score_cfg,
                                             generation_config=gen_cfg)

    assert diag.m3c_m3a_budget_alloc == 0
    assert diag.m3c_pre_m3_budget_alloc <= 5
    assert diag.fresh_official_scores <= 5
    assert diag.m3c_budget_invariant_holds
