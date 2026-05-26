"""test_m3d_candidate_export — M3D-slice-1: candidate metadata export."""

import copy
from typing import Optional

import torch

from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
    ScoredCandidate,
    ScoringDiagnostics,
)
from submissions.solver.core.m3d_candidate_export import export_candidate_rows

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "benchmark",
    "profile",
    "candidate_name",
    "family",
    "valid",
    "duplicate",
    "admitted",
    "not_admitted",
    "scored",
    "skip_reason",
    "proxy_cost",
    "approx_delta",
    "is_selected",
    "scored_pool_selectable",
    "selected_via_fallback",
    "placement_hash",
    "source_stage",
}


def _make_candidate(
    name: str,
    family: str = "original_neighborhood",
    valid: bool = True,
    proxy_cost: Optional[float] = 1.0,
    was_scored: bool = True,
    duplicate_of: Optional[str] = None,
    skip_reason: Optional[str] = None,
    placement_hash: Optional[str] = None,
    pass_id: Optional[int] = None,
    approx_hpwl_delta: Optional[float] = None,
) -> ScoredCandidate:
    positions = torch.zeros(2, 2)
    metadata: dict = {}
    if skip_reason is not None:
        metadata["skip_reason"] = skip_reason
    if placement_hash is not None:
        metadata["placement_hash"] = placement_hash
    if pass_id is not None:
        metadata["pass_id"] = pass_id
    if approx_hpwl_delta is not None:
        metadata["approx_hpwl_delta"] = approx_hpwl_delta
    return ScoredCandidate(
        name=name,
        family=family,
        positions=positions,
        valid=valid,
        proxy_cost=proxy_cost,
        delta_vs_original=None,
        num_overlaps=0,
        num_out_of_bounds=0,
        num_unplaced=0,
        num_moved=0,
        max_move=0.0,
        total_move=0.0,
        legalization_ms=0.0,
        scoring_ms=0.0,
        total_ms=0.0,
        was_scored=was_scored,
        duplicate_of=duplicate_of,
        metadata=metadata,
    )


def _make_diag(**kwargs) -> ScoringDiagnostics:
    defaults = dict(
        scoring_available=True,
        scoring_mode="local_proxy",
        score_is_degenerate=False,
        num_unique_scores=2,
        selected_due_to="proxy_cost",
    )
    defaults.update(kwargs)
    return ScoringDiagnostics(**defaults)


def _export(candidates, selected=None, diag=None, benchmark="b1", profile="p1"):
    if diag is None:
        diag = _make_diag()
    return export_candidate_rows(candidates, selected, diag, benchmark=benchmark, profile=profile)


# ---------------------------------------------------------------------------
# 1. All required fields are present in every row
# ---------------------------------------------------------------------------

def test_all_required_fields_present():
    cands = [
        _make_candidate("c1", skip_reason="scored", placement_hash="aabbccdd", pass_id=1),
        _make_candidate("c2", valid=False, was_scored=False, proxy_cost=None, skip_reason="invalid"),
    ]
    rows = _export(cands, selected=cands[0])
    assert len(rows) == 2
    for row in rows:
        missing = _REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Missing fields: {missing}"


# ---------------------------------------------------------------------------
# 2. Selected candidate is marked exactly once
# ---------------------------------------------------------------------------

def test_selected_marked_exactly_once():
    cands = [
        _make_candidate("winner", skip_reason="scored"),
        _make_candidate("runner_up", proxy_cost=1.5, skip_reason="scored"),
        _make_candidate("invalid_one", valid=False, was_scored=False, proxy_cost=None),
    ]
    selected = cands[0]
    rows = _export(cands, selected=selected)
    selected_rows = [r for r in rows if r["is_selected"]]
    assert len(selected_rows) == 1
    assert selected_rows[0]["candidate_name"] == "winner"


def test_no_selected_none_marked():
    cands = [_make_candidate("c1"), _make_candidate("c2", proxy_cost=0.9)]
    rows = _export(cands, selected=None)
    assert all(not r["is_selected"] for r in rows)


# ---------------------------------------------------------------------------
# 3. Scored and unscored candidates are distinguished
# ---------------------------------------------------------------------------

def test_scored_unscored_distinguished():
    scored_cand = _make_candidate("s", was_scored=True, proxy_cost=1.0, skip_reason="scored")
    unscored_cand = _make_candidate(
        "u", was_scored=False, proxy_cost=None, skip_reason="budget_exceeded"
    )
    rows = _export([scored_cand, unscored_cand])
    row_by_name = {r["candidate_name"]: r for r in rows}
    assert row_by_name["s"]["scored"] is True
    assert row_by_name["s"]["proxy_cost"] == 1.0
    assert row_by_name["u"]["scored"] is False
    assert row_by_name["u"]["proxy_cost"] is None


# ---------------------------------------------------------------------------
# 4. Invalid candidates remain present in export
# ---------------------------------------------------------------------------

def test_invalid_candidates_included():
    valid_cand = _make_candidate("v", skip_reason="scored")
    invalid_cand = _make_candidate(
        "i", valid=False, was_scored=False, proxy_cost=None, skip_reason="invalid"
    )
    rows = _export([valid_cand, invalid_cand])
    names = [r["candidate_name"] for r in rows]
    assert "i" in names
    inv_row = next(r for r in rows if r["candidate_name"] == "i")
    assert inv_row["valid"] is False
    assert inv_row["scored"] is False


# ---------------------------------------------------------------------------
# 5. Duplicate / not-admitted / skipped candidates represented correctly
# ---------------------------------------------------------------------------

def test_duplicate_flag_set():
    dup = _make_candidate(
        "dup",
        duplicate_of="original",
        was_scored=True,
        proxy_cost=1.0,
        skip_reason="duplicate",
    )
    rows = _export([dup])
    assert rows[0]["duplicate"] is True
    assert rows[0]["skip_reason"] == "duplicate"


def test_not_admitted_flag():
    na = _make_candidate(
        "na",
        family="m3a_pair_refinement",
        was_scored=False,
        proxy_cost=None,
        skip_reason="m3c_not_admitted",
    )
    rows = _export([na])
    assert rows[0]["not_admitted"] is True
    assert rows[0]["admitted"] is False


def test_admitted_for_normal_candidate():
    cand = _make_candidate("c", skip_reason="scored")
    rows = _export([cand])
    assert rows[0]["admitted"] is True
    assert rows[0]["not_admitted"] is False


def test_budget_exceeded_skip_reason():
    cand = _make_candidate(
        "b", was_scored=False, proxy_cost=None, skip_reason="budget_exceeded"
    )
    rows = _export([cand])
    assert rows[0]["skip_reason"] == "budget_exceeded"
    assert rows[0]["scored"] is False


def test_prefiltered_skip_reason():
    cand = _make_candidate(
        "pre", was_scored=False, proxy_cost=None, skip_reason="prefiltered"
    )
    rows = _export([cand])
    assert rows[0]["skip_reason"] == "prefiltered"


# ---------------------------------------------------------------------------
# 6. Family names are preserved
# ---------------------------------------------------------------------------

def test_family_names_preserved():
    families = [
        ("c1", "original_neighborhood"),
        ("c2", "spectral"),
        ("c3", "m3a_pair_refinement"),
        ("c4", "m3b_cluster_refinement"),
        ("c5", "area_degree"),
    ]
    cands = [_make_candidate(n, family=f, skip_reason="scored") for n, f in families]
    rows = _export(cands)
    row_by_name = {r["candidate_name"]: r for r in rows}
    for name, family in families:
        assert row_by_name[name]["family"] == family


# ---------------------------------------------------------------------------
# 7. Export order is deterministic
# ---------------------------------------------------------------------------

def test_export_order_deterministic():
    cands = [
        _make_candidate("c1", proxy_cost=1.0, skip_reason="scored"),
        _make_candidate("c2", proxy_cost=0.5, skip_reason="scored"),
        _make_candidate("c3", valid=False, was_scored=False, proxy_cost=None, skip_reason="invalid"),
    ]
    diag = _make_diag()
    rows_a = export_candidate_rows(cands, cands[1], diag, "b", "p")
    rows_b = export_candidate_rows(cands, cands[1], diag, "b", "p")
    assert [r["candidate_name"] for r in rows_a] == [r["candidate_name"] for r in rows_b]
    # Order must match the input list order
    assert [r["candidate_name"] for r in rows_a] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# 8. Export does not mutate candidates or diagnostics
# ---------------------------------------------------------------------------

def test_export_does_not_mutate_candidates():
    cands = [
        _make_candidate("c1", skip_reason="scored", placement_hash="aabb", pass_id=1),
        _make_candidate(
            "c2", valid=False, was_scored=False, proxy_cost=None, skip_reason="invalid"
        ),
    ]
    # Deep-copy snapshots before export
    snapshots = [copy.deepcopy(sc) for sc in cands]
    diag = _make_diag()
    export_candidate_rows(cands, cands[0], diag, "b", "p")
    for orig, snap in zip(cands, snapshots):
        assert orig.name == snap.name
        assert orig.valid == snap.valid
        assert orig.was_scored == snap.was_scored
        assert orig.proxy_cost == snap.proxy_cost
        assert orig.duplicate_of == snap.duplicate_of
        assert dict(orig.metadata) == dict(snap.metadata)


def test_export_does_not_mutate_diagnostics():
    diag = _make_diag(
        raw_original_valid=True,
        m3a_skipped_budget=0,
        m3b_skipped_budget=0,
        fresh_official_scores=5,
        candidates_officially_scored=5,
    )
    before = copy.deepcopy(diag)
    cands = [_make_candidate("c1", skip_reason="scored")]
    export_candidate_rows(cands, cands[0], diag, "b", "p")
    assert diag.fresh_official_scores == before.fresh_official_scores
    assert diag.candidates_officially_scored == before.candidates_officially_scored
    assert diag.raw_original_valid == before.raw_original_valid
    assert diag.m3a_skipped_budget == before.m3a_skipped_budget
    assert diag.m3b_skipped_budget == before.m3b_skipped_budget


# ---------------------------------------------------------------------------
# 9. Benchmark and profile metadata embedded in every row
# ---------------------------------------------------------------------------

def test_benchmark_profile_embedded():
    cands = [_make_candidate("c1", skip_reason="scored")]
    rows = _export(cands, benchmark="bm_foo", profile="pf_bar")
    assert rows[0]["benchmark"] == "bm_foo"
    assert rows[0]["profile"] == "pf_bar"


def test_benchmark_profile_defaults_to_empty():
    cands = [_make_candidate("c1", skip_reason="scored")]
    diag = _make_diag()
    rows = export_candidate_rows(cands, None, diag)
    assert rows[0]["benchmark"] == ""
    assert rows[0]["profile"] == ""


# ---------------------------------------------------------------------------
# 10. placement_hash and source_stage forwarded from metadata
# ---------------------------------------------------------------------------

def test_placement_hash_and_source_stage():
    cand = _make_candidate(
        "c1",
        skip_reason="scored",
        placement_hash="deadbeef",
        pass_id=2,
    )
    rows = _export([cand])
    assert rows[0]["placement_hash"] == "deadbeef"
    assert rows[0]["source_stage"] == 2


def test_placement_hash_none_when_absent():
    cand = _make_candidate("c1", skip_reason="scored")
    rows = _export([cand])
    assert rows[0]["placement_hash"] is None
    assert rows[0]["source_stage"] is None


# ---------------------------------------------------------------------------
# 11. approx_delta forwarded from metadata
# ---------------------------------------------------------------------------

def test_approx_delta_forwarded():
    cand = _make_candidate("c1", skip_reason="scored", approx_hpwl_delta=-0.05)
    rows = _export([cand])
    assert rows[0]["approx_delta"] == -0.05


def test_approx_delta_none_when_absent():
    cand = _make_candidate("c1", skip_reason="scored")
    rows = _export([cand])
    assert rows[0]["approx_delta"] is None


# ---------------------------------------------------------------------------
# 12. scored_pool_selectable logic
# ---------------------------------------------------------------------------

def test_selectable_valid_scored_candidate():
    cand = _make_candidate("c", family="original_neighborhood", skip_reason="scored")
    rows = _export([cand])
    assert rows[0]["scored_pool_selectable"] is True


def test_selectable_false_when_invalid():
    cand = _make_candidate(
        "c", valid=False, was_scored=False, proxy_cost=None, skip_reason="invalid"
    )
    rows = _export([cand])
    assert rows[0]["scored_pool_selectable"] is False


def test_selectable_false_when_unscored():
    cand = _make_candidate("c", was_scored=False, proxy_cost=None, skip_reason="budget_exceeded")
    rows = _export([cand])
    assert rows[0]["scored_pool_selectable"] is False


def test_selectable_m3a_excluded_when_budget_exhausted():
    cand = _make_candidate(
        "m3a_c",
        family="m3a_pair_refinement",
        was_scored=True,
        proxy_cost=0.9,
        skip_reason="scored",
    )
    diag_exhausted = _make_diag(m3a_skipped_budget=1)
    rows = export_candidate_rows([cand], cand, diag_exhausted, "b", "p")
    assert rows[0]["scored_pool_selectable"] is False


def test_selectable_m3a_not_excluded_when_budget_ok():
    cand = _make_candidate(
        "m3a_c",
        family="m3a_pair_refinement",
        was_scored=True,
        proxy_cost=0.9,
        skip_reason="scored",
    )
    diag_ok = _make_diag(m3a_skipped_budget=0)
    rows = export_candidate_rows([cand], cand, diag_ok, "b", "p")
    assert rows[0]["scored_pool_selectable"] is True


def test_selectable_m3b_excluded_when_budget_exhausted():
    cand = _make_candidate(
        "m3b_c",
        family="m3b_cluster_refinement",
        was_scored=True,
        proxy_cost=0.9,
        skip_reason="scored",
    )
    diag_exhausted = _make_diag(m3b_skipped_budget=1)
    rows = export_candidate_rows([cand], cand, diag_exhausted, "b", "p")
    assert rows[0]["scored_pool_selectable"] is False


def test_selectable_original_legalized_excluded_when_raw_original_valid():
    cand = _make_candidate(
        "original_legalized",
        family="original_neighborhood",
        was_scored=True,
        proxy_cost=1.0,
        skip_reason="scored",
    )
    diag = _make_diag(raw_original_valid=True)
    rows = export_candidate_rows([cand], None, diag, "b", "p")
    assert rows[0]["scored_pool_selectable"] is False


def test_selectable_original_legalized_ok_when_raw_original_invalid():
    cand = _make_candidate(
        "original_legalized",
        family="original_neighborhood",
        was_scored=True,
        proxy_cost=1.0,
        skip_reason="scored",
    )
    diag = _make_diag(raw_original_valid=False)
    rows = export_candidate_rows([cand], None, diag, "b", "p")
    assert rows[0]["scored_pool_selectable"] is True


# ---------------------------------------------------------------------------
# 13. Empty candidate list
# ---------------------------------------------------------------------------

def test_empty_candidate_list():
    rows = _export([], selected=None)
    assert rows == []


# ---------------------------------------------------------------------------
# 14. No behavior-change test: calling export does not alter solver outputs
# ---------------------------------------------------------------------------

def test_no_behavior_change_full_pipeline():
    """Calling export_candidate_rows must not change scoring/selection results."""
    from conftest import make_benchmark
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select

    bm = make_benchmark(
        n_hard=4,
        canvas=100.0,
        macro_size=10.0,
        net_nodes=[[0, 1], [0, 2], [1, 2], [1, 3]],
    )
    gen_cfg = CandidateGenerationConfig(only_original_neighborhood=True)
    score_cfg = CandidateScoringConfig(max_official_scores=20)

    cands = generate_candidates(bm, gen_cfg)
    best, ranked, diag = score_and_select(
        cands, bm, plc=None, scoring_config=score_cfg, generation_config=gen_cfg
    )

    # Record solver outputs before calling export
    recorded_name = best.name if best is not None else None
    recorded_family = best.family if best is not None else None
    recorded_cost = best.proxy_cost if best is not None else None
    recorded_fresh = diag.fresh_official_scores
    recorded_scored_count = diag.candidates_officially_scored

    # Call the export utility
    rows = export_candidate_rows(
        ranked, best, diag, benchmark="test_bm", profile="test_profile"
    )

    # Assert export produced output
    assert len(rows) > 0
    assert all(_REQUIRED_FIELDS <= set(r.keys()) for r in rows)

    # Assert all solver outputs are unchanged after export
    assert (best.name if best is not None else None) == recorded_name
    assert (best.family if best is not None else None) == recorded_family
    assert (best.proxy_cost if best is not None else None) == recorded_cost
    assert diag.fresh_official_scores == recorded_fresh
    assert diag.candidates_officially_scored == recorded_scored_count


# ---------------------------------------------------------------------------
# 15. Exactly one row per candidate — no duplicates or missing entries
# ---------------------------------------------------------------------------

def test_row_count_matches_candidate_count():
    cands = [_make_candidate(f"c{i}", skip_reason="scored") for i in range(10)]
    rows = _export(cands, selected=cands[0])
    assert len(rows) == len(cands)


def test_candidate_names_one_to_one():
    cands = [_make_candidate(f"c{i}", skip_reason="scored") for i in range(5)]
    rows = _export(cands)
    exported_names = [r["candidate_name"] for r in rows]
    assert exported_names == [sc.name for sc in cands]


# ---------------------------------------------------------------------------
# 16. selected_via_fallback — fallback selection path
# ---------------------------------------------------------------------------

def test_fallback_selected_is_selected_true_scored_pool_selectable_false():
    """A valid but unscored candidate chosen via fallback shows is_selected=True
    while scored_pool_selectable=False, and selected_via_fallback=True."""
    fallback = _make_candidate(
        "fallback_winner",
        valid=True,
        was_scored=False,
        proxy_cost=None,
        skip_reason="budget_exceeded",
    )
    other = _make_candidate("other", skip_reason="scored")
    rows = _export([fallback, other], selected=fallback)
    fb_row = next(r for r in rows if r["candidate_name"] == "fallback_winner")
    assert fb_row["is_selected"] is True
    assert fb_row["scored_pool_selectable"] is False
    assert fb_row["selected_via_fallback"] is True


def test_normal_selected_has_selected_via_fallback_false():
    """A candidate selected from the scored pool has selected_via_fallback=False."""
    winner = _make_candidate("winner", skip_reason="scored")
    rows = _export([winner], selected=winner)
    assert rows[0]["selected_via_fallback"] is False


# ---------------------------------------------------------------------------
# 17. Duplicate candidate names — object identity guards is_selected
# ---------------------------------------------------------------------------

def test_duplicate_names_exactly_one_selected_row():
    """Two candidates sharing the same name: only the selected object is marked."""
    cand_a = _make_candidate("shared_name", proxy_cost=1.0, skip_reason="scored")
    cand_b = _make_candidate("shared_name", proxy_cost=0.8, skip_reason="scored")
    rows = _export([cand_a, cand_b], selected=cand_b)
    selected_rows = [r for r in rows if r["is_selected"]]
    assert len(selected_rows) == 1
    # The second row (cand_b) should be selected, not the first
    assert rows[1]["is_selected"] is True
    assert rows[0]["is_selected"] is False


# ---------------------------------------------------------------------------
# 18. Selected candidate not in export list
# ---------------------------------------------------------------------------

def test_selected_not_in_export_list_no_row_marked():
    """If selected is not in the candidates list, no row is marked is_selected."""
    outside = _make_candidate("outside", skip_reason="scored")
    cands = [_make_candidate("c1", skip_reason="scored"), _make_candidate("c2", skip_reason="scored")]
    rows = _export(cands, selected=outside)
    assert all(not r["is_selected"] for r in rows)
    assert all(not r["selected_via_fallback"] for r in rows)
