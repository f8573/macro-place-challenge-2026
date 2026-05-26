"""test_m3d_family_summary — M3D-slice-2: family-level aggregation."""

import copy

import pytest

from submissions.solver.core.m3d_family_summary import summarize_candidate_families

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "benchmark",
    "profile",
    "family",
    "generated_count",
    "valid_count",
    "invalid_count",
    "duplicate_count",
    "admitted_count",
    "not_admitted_count",
    "scored_count",
    "skipped_budget_count",
    "scored_pool_selectable_count",
    "selected_count",
    "selected_via_fallback_count",
    "best_official_cost",
    "best_official_delta_vs_final",
    "median_official_cost",
    "median_official_delta_vs_final",
    "num_beating_final",
    "num_near_tie",
    "best_candidate_name",
    "worst_official_cost",
    "worst_candidate_name",
}


def _row(
    candidate_name="c",
    family="fam_a",
    benchmark="bm",
    profile="p",
    valid=True,
    duplicate=False,
    admitted=True,
    not_admitted=False,
    scored=True,
    skip_reason="",
    proxy_cost=1.0,
    is_selected=False,
    scored_pool_selectable=True,
    selected_via_fallback=False,
):
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
        "is_selected": is_selected,
        "scored_pool_selectable": scored_pool_selectable,
        "selected_via_fallback": selected_via_fallback,
    }


def _summarise(rows, **kwargs):
    return summarize_candidate_families(rows, **kwargs)


# ---------------------------------------------------------------------------
# All required fields present
# ---------------------------------------------------------------------------

def test_all_required_fields_present():
    rows = [_row()]
    result = _summarise(rows)
    assert len(result) == 1
    missing = _REQUIRED_FIELDS - set(result[0].keys())
    assert not missing, f"Missing fields: {missing}"


# ---------------------------------------------------------------------------
# 1. Aggregates generated/valid/invalid counts correctly
# ---------------------------------------------------------------------------

def test_generated_valid_invalid_counts():
    rows = [
        _row("c1", valid=True),
        _row("c2", valid=True),
        _row("c3", valid=False, scored=False, proxy_cost=None),
    ]
    result = _summarise(rows)
    assert len(result) == 1
    s = result[0]
    assert s["generated_count"] == 3
    assert s["valid_count"] == 2
    assert s["invalid_count"] == 1


def test_all_valid():
    rows = [_row(f"c{i}") for i in range(4)]
    s = _summarise(rows)[0]
    assert s["valid_count"] == 4
    assert s["invalid_count"] == 0


def test_all_invalid():
    rows = [
        _row(f"c{i}", valid=False, scored=False, proxy_cost=None)
        for i in range(3)
    ]
    s = _summarise(rows)[0]
    assert s["valid_count"] == 0
    assert s["invalid_count"] == 3


# ---------------------------------------------------------------------------
# 2. Aggregates duplicate/admitted/not-admitted/scored/skipped-budget counts
# ---------------------------------------------------------------------------

def test_duplicate_count():
    rows = [
        _row("c1", duplicate=False),
        _row("c2", duplicate=True),
        _row("c3", duplicate=True),
    ]
    s = _summarise(rows)[0]
    assert s["duplicate_count"] == 2


def test_admitted_not_admitted_counts():
    rows = [
        _row("c1", admitted=True, not_admitted=False),
        _row("c2", admitted=False, not_admitted=True, scored=False, proxy_cost=None),
        _row("c3", admitted=False, not_admitted=True, scored=False, proxy_cost=None),
    ]
    s = _summarise(rows)[0]
    assert s["admitted_count"] == 1
    assert s["not_admitted_count"] == 2


def test_scored_count_requires_proxy_cost():
    rows = [
        _row("c1", scored=True, proxy_cost=1.0),
        _row("c2", scored=True, proxy_cost=None),   # scored flag set but no cost
        _row("c3", scored=False, proxy_cost=None),
    ]
    s = _summarise(rows)[0]
    assert s["scored_count"] == 1


def test_skipped_budget_count():
    rows = [
        _row("c1", scored=False, proxy_cost=None, skip_reason="budget_exceeded"),
        _row("c2", scored=False, proxy_cost=None, skip_reason="budget_exceeded"),
        _row("c3", scored=True, proxy_cost=1.0, skip_reason=""),
    ]
    s = _summarise(rows)[0]
    assert s["skipped_budget_count"] == 2


def test_scored_pool_selectable_count():
    rows = [
        _row("c1", scored_pool_selectable=True),
        _row("c2", scored_pool_selectable=True),
        _row("c3", scored_pool_selectable=False),
    ]
    s = _summarise(rows)[0]
    assert s["scored_pool_selectable_count"] == 2


def test_selected_and_fallback_counts():
    rows = [
        _row("c1", is_selected=True, selected_via_fallback=False),
        _row("c2", is_selected=False, selected_via_fallback=False),
        _row("c3", is_selected=True, selected_via_fallback=True, scored=False,
             proxy_cost=None, scored_pool_selectable=False),
    ]
    s = _summarise(rows)[0]
    assert s["selected_count"] == 2
    assert s["selected_via_fallback_count"] == 1


# ---------------------------------------------------------------------------
# 3. Computes best and worst official costs correctly
# ---------------------------------------------------------------------------

def test_best_and_worst_costs():
    rows = [
        _row("c1", proxy_cost=3.0),
        _row("c2", proxy_cost=1.0),
        _row("c3", proxy_cost=2.0),
    ]
    s = _summarise(rows, final_cost=5.0)[0]
    assert s["best_official_cost"] == 1.0
    assert s["worst_official_cost"] == 3.0
    assert s["best_candidate_name"] == "c2"
    assert s["worst_candidate_name"] == "c1"


def test_single_scored_row_best_equals_worst():
    rows = [_row("only", proxy_cost=7.5)]
    s = _summarise(rows, final_cost=10.0)[0]
    assert s["best_official_cost"] == 7.5
    assert s["worst_official_cost"] == 7.5
    assert s["best_candidate_name"] == "only"
    assert s["worst_candidate_name"] == "only"


# ---------------------------------------------------------------------------
# 4. Computes median official cost correctly for odd and even counts
# ---------------------------------------------------------------------------

def test_median_odd_count():
    rows = [
        _row("c1", proxy_cost=1.0),
        _row("c2", proxy_cost=3.0),
        _row("c3", proxy_cost=2.0),
    ]
    s = _summarise(rows, final_cost=5.0)[0]
    assert s["median_official_cost"] == pytest.approx(2.0)


def test_median_even_count():
    rows = [
        _row("c1", proxy_cost=1.0),
        _row("c2", proxy_cost=3.0),
        _row("c3", proxy_cost=2.0),
        _row("c4", proxy_cost=4.0),
    ]
    s = _summarise(rows, final_cost=5.0)[0]
    assert s["median_official_cost"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 5. Computes deltas vs final cost correctly
# ---------------------------------------------------------------------------

def test_deltas_vs_final_cost():
    rows = [
        _row("c1", proxy_cost=0.8),
        _row("c2", proxy_cost=1.2),
        _row("c3", proxy_cost=1.0),
    ]
    final = 1.0
    s = _summarise(rows, final_cost=final)[0]
    assert s["best_official_delta_vs_final"] == pytest.approx(0.8 - final)
    assert s["median_official_delta_vs_final"] == pytest.approx(1.0 - final)


def test_deltas_none_when_no_final_cost_and_no_selection():
    rows = [_row("c1", proxy_cost=2.0, is_selected=False)]
    s = _summarise(rows)[0]
    assert s["best_official_delta_vs_final"] is None
    assert s["median_official_delta_vs_final"] is None


# ---------------------------------------------------------------------------
# 6. Computes beating-final and near-tie counts with epsilon
# ---------------------------------------------------------------------------

def test_beating_final_count():
    eps = 1e-5
    final = 1.0
    rows = [
        _row("c1", proxy_cost=0.5),        # beats final
        _row("c2", proxy_cost=0.9),        # beats final
        _row("c3", proxy_cost=1.0),        # near-tie, not beating
        _row("c4", proxy_cost=1.5),        # worse
    ]
    s = _summarise(rows, final_cost=final, official_epsilon=eps)[0]
    assert s["num_beating_final"] == 2


def test_near_tie_count():
    # Use eps=0.25 (exact in IEEE-754) to avoid floating-point boundary artifacts.
    eps = 0.25
    final = 1.0
    rows = [
        _row("c1", proxy_cost=1.0),    # exact tie: |0| <= 0.25 ✓
        _row("c2", proxy_cost=1.25),   # on boundary: |0.25| <= 0.25 ✓
        _row("c3", proxy_cost=0.75),   # on boundary: |0.25| <= 0.25 ✓
        _row("c4", proxy_cost=1.5),    # outside: |0.5| > 0.25 ✗
        _row("c5", proxy_cost=0.0),    # well below ✗
    ]
    s = _summarise(rows, final_cost=final, official_epsilon=eps)[0]
    assert s["num_near_tie"] == 3


def test_beating_and_near_tie_zero_when_no_final():
    rows = [_row("c1", proxy_cost=0.5, is_selected=False)]
    s = _summarise(rows)[0]
    assert s["num_beating_final"] == 0
    assert s["num_near_tie"] == 0


# ---------------------------------------------------------------------------
# 7. Handles families with no scored candidates
# ---------------------------------------------------------------------------

def test_no_scored_candidates():
    rows = [
        _row("c1", scored=False, proxy_cost=None, skip_reason="budget_exceeded"),
        _row("c2", valid=False, scored=False, proxy_cost=None),
    ]
    s = _summarise(rows, final_cost=1.0)[0]
    assert s["scored_count"] == 0
    assert s["best_official_cost"] is None
    assert s["worst_official_cost"] is None
    assert s["best_candidate_name"] is None
    assert s["worst_candidate_name"] is None
    assert s["median_official_cost"] is None
    assert s["best_official_delta_vs_final"] is None
    assert s["median_official_delta_vs_final"] is None
    assert s["num_beating_final"] == 0
    assert s["num_near_tie"] == 0


# ---------------------------------------------------------------------------
# 8. Infers final_cost from selected scored row when not provided
# ---------------------------------------------------------------------------

def test_infer_final_cost_from_selected():
    rows = [
        _row("c1", proxy_cost=1.0, is_selected=True),
        _row("c2", proxy_cost=2.0, is_selected=False),
    ]
    s = _summarise(rows)[0]
    # best_official_delta_vs_final should be 1.0 - 1.0 = 0.0
    assert s["best_official_delta_vs_final"] == pytest.approx(0.0)
    assert s["num_beating_final"] == 0
    assert s["num_near_tie"] == 1  # c1 is the exact selected cost


def test_no_inference_when_multiple_selected():
    rows = [
        _row("c1", proxy_cost=1.0, is_selected=True),
        _row("c2", proxy_cost=2.0, is_selected=True),
    ]
    s = _summarise(rows)[0]
    # Two selected scored rows — cannot infer unique final_cost
    assert s["best_official_delta_vs_final"] is None
    assert s["num_beating_final"] == 0
    assert s["num_near_tie"] == 0


# ---------------------------------------------------------------------------
# 9. Handles selected fallback row with no proxy_cost
# ---------------------------------------------------------------------------

def test_fallback_selected_no_proxy_cost():
    rows = [
        _row("fallback", scored=False, proxy_cost=None, is_selected=True,
             scored_pool_selectable=False, selected_via_fallback=True),
        _row("normal", proxy_cost=1.5, is_selected=False),
    ]
    # fallback has no proxy_cost so cannot be used to infer final_cost
    s = _summarise(rows)[0]
    assert s["selected_count"] == 1
    assert s["selected_via_fallback_count"] == 1
    assert s["scored_count"] == 1
    assert s["best_official_delta_vs_final"] is None


def test_fallback_with_explicit_final_cost():
    rows = [
        _row("fallback", scored=False, proxy_cost=None, is_selected=True,
             scored_pool_selectable=False, selected_via_fallback=True),
        _row("normal", proxy_cost=1.5, is_selected=False),
    ]
    s = _summarise(rows, final_cost=2.0)[0]
    assert s["best_official_delta_vs_final"] == pytest.approx(1.5 - 2.0)


# ---------------------------------------------------------------------------
# 10. Preserves deterministic family ordering
# ---------------------------------------------------------------------------

def test_family_ordering_alphabetical():
    rows = [
        _row("c1", family="zebra"),
        _row("c2", family="apple"),
        _row("c3", family="mango"),
    ]
    result = _summarise(rows)
    families = [s["family"] for s in result]
    assert families == sorted(families)


def test_family_ordering_stable_across_calls():
    rows = [
        _row(f"c{i}", family=f"fam_{chr(ord('z') - i)}")
        for i in range(5)
    ]
    result_a = _summarise(rows)
    result_b = _summarise(rows)
    assert [s["family"] for s in result_a] == [s["family"] for s in result_b]


def test_family_ordering_by_benchmark_profile_family():
    rows = [
        _row("c1", benchmark="b2", profile="p1", family="fam_a"),
        _row("c2", benchmark="b1", profile="p1", family="fam_b"),
        _row("c3", benchmark="b1", profile="p1", family="fam_a"),
    ]
    result = _summarise(rows)
    keys = [(s["benchmark"], s["profile"], s["family"]) for s in result]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 11. Deterministic tie-breaks for best/worst candidate names
# ---------------------------------------------------------------------------

def test_best_cost_tie_break_by_name():
    rows = [
        _row("z_cand", proxy_cost=1.0),
        _row("a_cand", proxy_cost=1.0),
    ]
    s = _summarise(rows, final_cost=2.0)[0]
    assert s["best_candidate_name"] == "a_cand"


def test_worst_cost_tie_break_by_name():
    rows = [
        _row("z_cand", proxy_cost=5.0),
        _row("a_cand", proxy_cost=5.0),
    ]
    s = _summarise(rows, final_cost=2.0)[0]
    # max with tie-break by name ascending -> "z_cand"
    assert s["worst_candidate_name"] == "z_cand"


def test_tie_break_deterministic_across_calls():
    rows = [
        _row("m_cand", proxy_cost=2.0),
        _row("b_cand", proxy_cost=2.0),
        _row("x_cand", proxy_cost=2.0),
    ]
    a = _summarise(rows, final_cost=3.0)[0]
    b = _summarise(rows, final_cost=3.0)[0]
    assert a["best_candidate_name"] == b["best_candidate_name"]
    assert a["worst_candidate_name"] == b["worst_candidate_name"]


# ---------------------------------------------------------------------------
# 12. Does not mutate input rows
# ---------------------------------------------------------------------------

def test_does_not_mutate_input_rows():
    rows = [
        _row("c1", proxy_cost=1.0, is_selected=True),
        _row("c2", proxy_cost=2.0),
        _row("c3", valid=False, scored=False, proxy_cost=None),
    ]
    originals = copy.deepcopy(rows)
    _summarise(rows, final_cost=1.5)
    for original, after in zip(originals, rows):
        assert original == after


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_rows_returns_empty():
    assert _summarise([]) == []


def test_multiple_families_produce_one_row_each():
    rows = [
        _row("c1", family="fam_a"),
        _row("c2", family="fam_b"),
        _row("c3", family="fam_a"),
        _row("c4", family="fam_c"),
    ]
    result = _summarise(rows, final_cost=5.0)
    assert len(result) == 3
    families = {s["family"] for s in result}
    assert families == {"fam_a", "fam_b", "fam_c"}
    fam_a = next(s for s in result if s["family"] == "fam_a")
    assert fam_a["generated_count"] == 2


def test_explicit_final_cost_overrides_inference():
    rows = [_row("c1", proxy_cost=1.0, is_selected=True)]
    # selected row would infer final_cost=1.0, but we override with 2.0
    s = _summarise(rows, final_cost=2.0)[0]
    assert s["best_official_delta_vs_final"] == pytest.approx(1.0 - 2.0)


# ---------------------------------------------------------------------------
# 13. Multi-family inferred final cost (regression for cross-family inference bug)
# ---------------------------------------------------------------------------

def test_multi_family_inferred_final_cost():
    """Selected row is in fam_a; fam_b should still compare against that cost."""
    final = 1.0
    rows = [
        _row("sel", family="fam_a", proxy_cost=final, is_selected=True),
        _row("other_a", family="fam_a", proxy_cost=1.5, is_selected=False),
        _row("b1", family="fam_b", proxy_cost=0.8, is_selected=False),
        _row("b2", family="fam_b", proxy_cost=1.2, is_selected=False),
    ]
    result = _summarise(rows)
    fam_b = next(s for s in result if s["family"] == "fam_b")

    assert fam_b["best_official_delta_vs_final"] == pytest.approx(0.8 - final)
    assert fam_b["median_official_delta_vs_final"] == pytest.approx(1.0 - final)
    assert fam_b["num_beating_final"] == 1   # 0.8 < 1.0 - eps
    assert fam_b["num_near_tie"] == 0        # neither 0.8 nor 1.2 is within 1e-5 of 1.0


# ---------------------------------------------------------------------------
# 14. Per benchmark/profile inference uses the correct local final cost
# ---------------------------------------------------------------------------

def test_per_benchmark_profile_inference():
    """Two (benchmark, profile) groups must each use their own inferred cost."""
    rows = [
        # group (bm1, p1): selected at cost 2.0
        _row("sel1", benchmark="bm1", profile="p1", family="fam_a",
             proxy_cost=2.0, is_selected=True),
        _row("other1", benchmark="bm1", profile="p1", family="fam_b",
             proxy_cost=3.0, is_selected=False),
        # group (bm2, p1): selected at cost 5.0
        _row("sel2", benchmark="bm2", profile="p1", family="fam_a",
             proxy_cost=5.0, is_selected=True),
        _row("other2", benchmark="bm2", profile="p1", family="fam_b",
             proxy_cost=4.0, is_selected=False),
    ]
    result = _summarise(rows)

    bm1_fam_b = next(
        s for s in result if s["benchmark"] == "bm1" and s["family"] == "fam_b"
    )
    bm2_fam_b = next(
        s for s in result if s["benchmark"] == "bm2" and s["family"] == "fam_b"
    )

    assert bm1_fam_b["best_official_delta_vs_final"] == pytest.approx(3.0 - 2.0)
    assert bm2_fam_b["best_official_delta_vs_final"] == pytest.approx(4.0 - 5.0)


# ---------------------------------------------------------------------------
# 15. Multiple selected+scored rows degrade final-cost fields safely
# ---------------------------------------------------------------------------

def test_multiple_selected_scored_rows_degrade_safely():
    """Two selected+scored rows in the same (benchmark, profile) -> no inference."""
    rows = [
        _row("s1", family="fam_a", proxy_cost=1.0, is_selected=True),
        _row("s2", family="fam_a", proxy_cost=2.0, is_selected=True),
        _row("b1", family="fam_b", proxy_cost=0.5, is_selected=False),
    ]
    result = _summarise(rows)

    fam_a = next(s for s in result if s["family"] == "fam_a")
    fam_b = next(s for s in result if s["family"] == "fam_b")

    # Both families must have no inferred cost
    for s in (fam_a, fam_b):
        assert s["best_official_delta_vs_final"] is None
        assert s["median_official_delta_vs_final"] is None
        assert s["num_beating_final"] == 0
        assert s["num_near_tie"] == 0
