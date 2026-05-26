"""test_m3d_failure_classification — M3D-slice-3: failure classification."""

import copy
from typing import Any, Dict, List, Optional

import pytest

from submissions.solver.core.m3d_failure_classification import classify_m3d_failure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_OUTPUT_FIELDS = {
    "benchmark",
    "profile",
    "classification",
    "reason",
    "late_stage_generated",
    "late_stage_valid",
    "late_stage_invalid",
    "late_stage_admitted",
    "late_stage_not_admitted",
    "late_stage_scored",
    "late_stage_selectable",
    "late_stage_best_cost",
    "late_stage_best_delta_vs_final",
    "late_stage_num_beating_final",
    "late_stage_num_near_tie",
    "recommended_next_step",
}

_VALID_CLASSIFICATIONS = {
    "late_stage_not_scored",
    "late_stage_not_selectable",
    "late_stage_valid_but_worse",
    "late_stage_good_but_missed",
    "ranking_mismatch",
    "invalidity_dominated",
    "near_local_optimum",
}

_RECOMMENDED_NEXT_STEPS = {
    "late_stage_not_scored": "inspect M3C admission/scoring budget",
    "late_stage_not_selectable": "inspect budget exhaustion/selectability guards",
    "late_stage_valid_but_worse": "design new structural move family",
    "late_stage_good_but_missed": "review selector/selectability bug",
    "ranking_mismatch": "redesign analytical prefilter/ranking",
    "invalidity_dominated": "design safer geometry generation",
    "near_local_optimum": "try larger structural search",
}


def _row(
    candidate_name: str = "c",
    family: str = "original_neighborhood",
    benchmark: str = "bm",
    profile: str = "p",
    valid: bool = True,
    duplicate: bool = False,
    admitted: bool = True,
    not_admitted: bool = False,
    scored: bool = True,
    skip_reason: str = "",
    proxy_cost: Optional[float] = 1.0,
    approx_delta: Optional[float] = None,
    is_selected: bool = False,
    scored_pool_selectable: bool = True,
    selected_via_fallback: bool = False,
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "profile": profile,
        "candidate_name": candidate_name,
        "family": family,
        "valid": valid,
        "duplicate": duplicate,
        "admitted": admitted,
        "not_admitted": not_admitted,
        "scored": scored,
        "skip_reason": skip_reason,
        "proxy_cost": proxy_cost,
        "approx_delta": approx_delta,
        "is_selected": is_selected,
        "scored_pool_selectable": scored_pool_selectable,
        "selected_via_fallback": selected_via_fallback,
    }


def _summary(
    family: str = "m3a_pair_refinement",
    benchmark: str = "bm",
    profile: str = "p",
    generated_count: int = 1,
    valid_count: int = 1,
    scored_count: int = 1,
    scored_pool_selectable_count: int = 1,
    num_beating_final: int = 0,
    num_near_tie: int = 0,
    best_official_cost: Optional[float] = 1.0,
    best_official_delta_vs_final: Optional[float] = 0.0,
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "profile": profile,
        "family": family,
        "generated_count": generated_count,
        "valid_count": valid_count,
        "scored_count": scored_count,
        "scored_pool_selectable_count": scored_pool_selectable_count,
        "num_beating_final": num_beating_final,
        "num_near_tie": num_near_tie,
        "best_official_cost": best_official_cost,
        "best_official_delta_vs_final": best_official_delta_vs_final,
    }


def _classify(rows, summaries=None, **kwargs):
    return classify_m3d_failure(rows, summaries or [], **kwargs)


def _single(rows, summaries=None, **kwargs):
    results = _classify(rows, summaries, **kwargs)
    assert len(results) == 1
    return results[0]


def _assert_fields(result: Dict[str, Any]) -> None:
    missing = _REQUIRED_OUTPUT_FIELDS - result.keys()
    assert not missing, f"Missing output fields: {missing}"
    assert result["classification"] in _VALID_CLASSIFICATIONS
    assert result["recommended_next_step"] == _RECOMMENDED_NEXT_STEPS[result["classification"]]


# ---------------------------------------------------------------------------
# Test 1: No late-stage candidates generated
# ---------------------------------------------------------------------------


def test_no_late_stage_candidates_generated():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_not_scored"
    assert result["late_stage_generated"] == 0
    assert result["late_stage_scored"] == 0
    assert "generated" in result["reason"].lower() or "no m3a" in result["reason"].lower()


def test_no_candidates_at_all_returns_empty():
    # Without any rows or summaries there is no (benchmark, profile) to classify.
    results = classify_m3d_failure([], [])
    assert results == []


# ---------------------------------------------------------------------------
# Test 2: Late-stage candidates generated but none valid
# ---------------------------------------------------------------------------


def test_generated_but_all_invalid():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=False, scored=False,
             proxy_cost=None, admitted=False, not_admitted=False),
        _row("m3a_2", family="m3a_pair_refinement", valid=False, scored=False,
             proxy_cost=None, admitted=False, not_admitted=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "invalidity_dominated"
    assert result["late_stage_generated"] == 2
    assert result["late_stage_valid"] == 0
    assert result["late_stage_invalid"] == 2
    assert "invalid" in result["reason"].lower()


def test_generated_but_all_invalid_m3b():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3b_1", family="m3b_cluster_refinement", valid=False, scored=False,
             proxy_cost=None),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "invalidity_dominated"


# ---------------------------------------------------------------------------
# Test 3: Valid/admitted but none scored
# ---------------------------------------------------------------------------


def test_valid_admitted_but_not_scored():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=False,
             proxy_cost=None, scored_pool_selectable=False),
        _row("m3a_2", family="m3a_pair_refinement", valid=True, scored=False,
             proxy_cost=None, scored_pool_selectable=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_not_scored"
    assert result["late_stage_generated"] == 2
    assert result["late_stage_valid"] == 2
    assert result["late_stage_scored"] == 0
    assert "score" in result["reason"].lower() or "generated" in result["reason"].lower()


def test_valid_but_no_official_score_means_not_scored():
    # scored=True but proxy_cost is None should not count as scored
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=None, scored_pool_selectable=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_not_scored"
    assert result["late_stage_scored"] == 0


# ---------------------------------------------------------------------------
# Test 4: Scored but none selectable
# ---------------------------------------------------------------------------


def test_scored_but_not_selectable():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.5, scored_pool_selectable=False),
        _row("m3a_2", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.3, scored_pool_selectable=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_not_selectable"
    assert result["late_stage_scored"] == 2
    assert result["late_stage_selectable"] == 0
    assert "selectable" in result["reason"].lower() or "budget" in result["reason"].lower()


def test_scored_not_selectable_with_m3b():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=2.0),
        _row("m3b_1", family="m3b_cluster_refinement", valid=True, scored=True,
             proxy_cost=2.5, scored_pool_selectable=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_not_selectable"


# ---------------------------------------------------------------------------
# Test 5: Scored late-stage beats final but was not selected (good_but_missed)
# ---------------------------------------------------------------------------


def test_good_but_missed():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_best", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.8, scored_pool_selectable=True, is_selected=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_good_but_missed"
    assert result["late_stage_num_beating_final"] == 1
    assert "beat" in result["reason"].lower() or "missed" in result["reason"].lower()


def test_good_but_missed_takes_precedence_over_invalidity():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_bad", family="m3a_pair_refinement", valid=False, scored=False,
             proxy_cost=None),
        _row("m3a_good", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.5, scored_pool_selectable=True, is_selected=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_good_but_missed"


# ---------------------------------------------------------------------------
# Test 6: Scored/selectable but worse than final
# ---------------------------------------------------------------------------


def test_scored_selectable_but_worse():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.5, scored_pool_selectable=True),
        _row("m3a_2", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=2.0, scored_pool_selectable=True),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "late_stage_valid_but_worse"
    assert result["late_stage_selectable"] == 2
    assert result["late_stage_num_beating_final"] == 0
    assert "none beat" in result["reason"].lower() or "did not beat" in result["reason"].lower()


def test_valid_but_worse_best_cost_and_delta():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.2, scored_pool_selectable=True),
    ]
    result = _single(rows)
    assert result["late_stage_best_cost"] == pytest.approx(1.2)
    assert result["late_stage_best_delta_vs_final"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Test 7: Near-tie candidates → near_local_optimum
# ---------------------------------------------------------------------------


def test_near_local_optimum():
    eps = 1e-5
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.0 + eps * 0.5, scored_pool_selectable=True),
        _row("m3a_2", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.0 - eps * 0.5, scored_pool_selectable=True),
    ]
    result = _single(rows, official_epsilon=eps)
    _assert_fields(result)
    assert result["classification"] == "near_local_optimum"
    assert result["late_stage_num_near_tie"] >= 1
    assert "tie" in result["reason"].lower() or "optimum" in result["reason"].lower()


def test_near_local_optimum_multiple_near_ties():
    eps = 1e-5
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=5.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=5.0, scored_pool_selectable=True),
        _row("m3a_2", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=5.0 + eps * 0.9, scored_pool_selectable=True),
        _row("m3a_3", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=5.0 - eps * 0.9, scored_pool_selectable=True),
    ]
    result = _single(rows, official_epsilon=eps)
    assert result["classification"] == "near_local_optimum"
    assert result["late_stage_num_near_tie"] == 3


# ---------------------------------------------------------------------------
# Test 8: Ranking mismatch (conservative)
# ---------------------------------------------------------------------------


def test_ranking_mismatch():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.1, approx_delta=-0.5, scored_pool_selectable=True),
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["classification"] == "ranking_mismatch"
    assert "ranking" in result["reason"].lower() or "prefilter" in result["reason"].lower()


def test_ranking_mismatch_not_triggered_without_approx_delta():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.1, approx_delta=None, scored_pool_selectable=True),
    ]
    result = _single(rows)
    assert result["classification"] != "ranking_mismatch"


def test_ranking_mismatch_not_triggered_when_approx_positive():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.1, approx_delta=0.1, scored_pool_selectable=True),
    ]
    result = _single(rows)
    assert result["classification"] != "ranking_mismatch"


def test_ranking_mismatch_not_triggered_when_official_also_beats():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.8, approx_delta=-0.5,
             scored_pool_selectable=True, is_selected=False),
    ]
    # proxy_cost=0.8 < 1.0, so good_but_missed fires first (not ranking_mismatch)
    result = _single(rows)
    assert result["classification"] == "late_stage_good_but_missed"


# ---------------------------------------------------------------------------
# Test 9: Aggregates M3A and M3B together
# ---------------------------------------------------------------------------


def test_aggregates_m3a_and_m3b_families():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.5, scored_pool_selectable=True),
        _row("m3a_2", family="m3a_pair_refinement", valid=False, scored=False,
             proxy_cost=None, scored_pool_selectable=False),
        _row("m3b_1", family="m3b_cluster_refinement", valid=True, scored=True,
             proxy_cost=1.3, scored_pool_selectable=True),
        _row("m3b_2", family="m3b_cluster_refinement", valid=True, scored=True,
             proxy_cost=1.4, scored_pool_selectable=True),
    ]
    result = _single(rows)
    _assert_fields(result)
    # 4 total late-stage rows (m3a_1, m3a_2, m3b_1, m3b_2)
    assert result["late_stage_generated"] == 4
    assert result["late_stage_valid"] == 3
    assert result["late_stage_invalid"] == 1
    # 3 scored (proxy_cost not None)
    assert result["late_stage_scored"] == 3
    assert result["late_stage_selectable"] == 3
    # Best cost across both families
    assert result["late_stage_best_cost"] == pytest.approx(1.3)
    assert result["classification"] == "late_stage_valid_but_worse"


def test_aggregates_counts_across_m3a_and_m3b():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=2.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=False,
             proxy_cost=None, admitted=True, not_admitted=False),
        _row("m3b_1", family="m3b_cluster_refinement", valid=True, scored=False,
             proxy_cost=None, admitted=False, not_admitted=True),
    ]
    result = _single(rows)
    assert result["late_stage_generated"] == 2
    assert result["late_stage_valid"] == 2
    assert result["late_stage_admitted"] == 1
    assert result["late_stage_not_admitted"] == 1
    assert result["classification"] == "late_stage_not_scored"


# ---------------------------------------------------------------------------
# Test 10: Multiple benchmark/profile groups — deterministic ordering
# ---------------------------------------------------------------------------


def test_multiple_groups_deterministic_ordering():
    rows = [
        _row("orig", family="original_neighborhood", benchmark="bm_b", profile="p1",
             is_selected=True, proxy_cost=1.0),
        _row("orig", family="original_neighborhood", benchmark="bm_a", profile="p1",
             is_selected=True, proxy_cost=1.0),
        _row("orig", family="original_neighborhood", benchmark="bm_a", profile="p2",
             is_selected=True, proxy_cost=1.0),
    ]
    results = _classify(rows)
    assert len(results) == 3
    keys = [(r["benchmark"], r["profile"]) for r in results]
    assert keys == sorted(keys), "Results should be sorted by (benchmark, profile)"
    assert keys[0] == ("bm_a", "p1")
    assert keys[1] == ("bm_a", "p2")
    assert keys[2] == ("bm_b", "p1")


def test_multiple_groups_independent_classifications():
    rows = [
        # Group 1: no late-stage candidates
        _row("orig_1", family="original_neighborhood", benchmark="bm1", profile="p",
             is_selected=True, proxy_cost=1.0),
        # Group 2: has late-stage, all invalid
        _row("orig_2", family="original_neighborhood", benchmark="bm2", profile="p",
             is_selected=True, proxy_cost=1.0),
        _row("m3a_bad", family="m3a_pair_refinement", benchmark="bm2", profile="p",
             valid=False, scored=False, proxy_cost=None),
    ]
    results = _classify(rows)
    assert len(results) == 2
    bm1 = next(r for r in results if r["benchmark"] == "bm1")
    bm2 = next(r for r in results if r["benchmark"] == "bm2")
    assert bm1["classification"] == "late_stage_not_scored"
    assert bm2["classification"] == "invalidity_dominated"


# ---------------------------------------------------------------------------
# Test 11: Missing fields handled safely
# ---------------------------------------------------------------------------


def test_handles_row_with_no_family_key():
    rows = [{"benchmark": "bm", "profile": "p", "is_selected": True, "proxy_cost": 1.0,
             "scored": True}]
    result = _single(rows)
    _assert_fields(result)
    assert result["late_stage_generated"] == 0


def test_handles_row_with_missing_proxy_cost():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        {"benchmark": "bm", "profile": "p", "family": "m3a_pair_refinement",
         "valid": True, "scored": True},  # missing proxy_cost
    ]
    result = _single(rows)
    _assert_fields(result)
    assert result["late_stage_scored"] == 0  # no proxy_cost → not counted as scored


def test_handles_empty_rows_and_summaries():
    results = classify_m3d_failure([], [])
    assert results == []


def test_handles_none_benchmark_profile():
    rows = [
        {"family": "m3a_pair_refinement", "valid": True, "scored": True, "proxy_cost": 1.5,
         "benchmark": None, "profile": None, "admitted": True, "not_admitted": False,
         "scored_pool_selectable": True, "is_selected": False},
    ]
    results = classify_m3d_failure(rows, [])
    assert len(results) == 1
    _assert_fields(results[0])


def test_handles_summaries_only_no_rows():
    summaries = [_summary(family="m3a_pair_refinement", benchmark="bm", profile="p")]
    results = classify_m3d_failure([], summaries)
    assert len(results) == 1
    _assert_fields(results[0])
    assert results[0]["late_stage_generated"] == 0


# ---------------------------------------------------------------------------
# Test 12: Does not mutate inputs
# ---------------------------------------------------------------------------


def test_does_not_mutate_candidate_rows():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.5, scored_pool_selectable=True),
    ]
    original_rows = copy.deepcopy(rows)
    classify_m3d_failure(rows, [])
    assert rows == original_rows


def test_does_not_mutate_family_summaries():
    summaries = [_summary(family="m3a_pair_refinement")]
    original_summaries = copy.deepcopy(summaries)
    classify_m3d_failure([], summaries)
    assert summaries == original_summaries


# ---------------------------------------------------------------------------
# Classification precedence
# ---------------------------------------------------------------------------


def test_good_but_missed_takes_precedence_over_not_selectable():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        # Not selectable but beats final — good_but_missed wins
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.5, scored_pool_selectable=False, is_selected=False),
    ]
    result = _single(rows)
    assert result["classification"] == "late_stage_good_but_missed"


def test_invalidity_dominated_takes_precedence_over_not_scored():
    # All generated, none valid → invalidity_dominated not late_stage_not_scored
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=False, scored=False,
             proxy_cost=None),
    ]
    result = _single(rows)
    assert result["classification"] == "invalidity_dominated"


def test_near_local_optimum_below_epsilon_threshold():
    # Candidates that are just outside epsilon are worse, not near-tie
    eps = 1e-5
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.0 + eps * 2, scored_pool_selectable=True),
    ]
    result = _single(rows, official_epsilon=eps)
    # proxy_cost > final + epsilon → not near tie
    assert result["classification"] == "late_stage_valid_but_worse"
    assert result["late_stage_num_near_tie"] == 0


# ---------------------------------------------------------------------------
# Test 13: Final-cost inference — exactly-one-row contract
# ---------------------------------------------------------------------------


def test_final_cost_inferred_from_exactly_one_selected_scored_row():
    rows = [
        _row("orig", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=1.5, scored_pool_selectable=True),
    ]
    result = _single(rows)
    # Exactly one selected+scored row → final_cost=1.0 inferred
    assert result["late_stage_best_delta_vs_final"] == pytest.approx(0.5)
    assert result["classification"] == "late_stage_valid_but_worse"


def test_final_cost_unavailable_with_zero_selected_scored_rows():
    # No selected row at all → final_cost cannot be inferred
    rows = [
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.8, scored_pool_selectable=True, is_selected=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    # final_cost is None → delta unavailable, good_but_missed cannot fire
    assert result["late_stage_best_delta_vs_final"] is None
    assert result["classification"] != "late_stage_good_but_missed"


def test_final_cost_unavailable_with_multiple_selected_scored_rows():
    # Two selected+scored rows → ambiguous, final_cost not inferred
    rows = [
        _row("orig1", family="original_neighborhood", is_selected=True, proxy_cost=1.0),
        _row("orig2", family="original_neighborhood", is_selected=True, proxy_cost=0.9),
        _row("m3a_1", family="m3a_pair_refinement", valid=True, scored=True,
             proxy_cost=0.5, scored_pool_selectable=True, is_selected=False),
    ]
    result = _single(rows)
    _assert_fields(result)
    # Must not pick the first row's cost and misclassify as good_but_missed
    assert result["late_stage_best_delta_vs_final"] is None
    assert result["classification"] != "late_stage_good_but_missed"
