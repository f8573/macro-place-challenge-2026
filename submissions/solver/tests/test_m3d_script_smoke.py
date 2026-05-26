"""test_m3d_script_smoke — M3D-slice-4: analysis script smoke tests.

Tests the pure helper functions (build_benchmark_summary_rows, render_findings_md,
write_m3d_outputs) with synthetic fixture data.  Does not run the full solver
pipeline or require official benchmarks.
"""

import copy
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from submissions.solver.scripts.analyze_m3d_effectiveness import (
    _BENCHMARK_SUMMARY_FIELDS,
    build_benchmark_summary_rows,
    render_findings_md,
    write_m3d_outputs,
)

# ---------------------------------------------------------------------------
# Helpers / synthetic fixtures
# ---------------------------------------------------------------------------

_PROFILE = "m3c-default"


def _cand_row(
    candidate_name: str = "c",
    family: str = "original_neighborhood",
    benchmark: str = "ibm01",
    profile: str = _PROFILE,
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
    placement_hash: str = "abc00001",
    source_stage: Optional[int] = 1,
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
        "placement_hash": placement_hash,
        "source_stage": source_stage,
    }


def _make_candidate_rows(
    benchmark: str = "ibm01",
    profile: str = _PROFILE,
) -> List[Dict[str, Any]]:
    return [
        _cand_row("orig", "original_neighborhood", benchmark, profile,
                  is_selected=True, proxy_cost=1.0, placement_hash="orig0001"),
        _cand_row("n1", "original_neighborhood", benchmark, profile,
                  proxy_cost=1.1, placement_hash="neigh001"),
        _cand_row("n2", "original_neighborhood", benchmark, profile,
                  proxy_cost=1.2, placement_hash="neigh002"),
        _cand_row("m3a_1", "m3a_pair_refinement", benchmark, profile,
                  proxy_cost=1.05, placement_hash="m3a00001",
                  scored_pool_selectable=True),
        _cand_row("m3a_2", "m3a_pair_refinement", benchmark, profile,
                  proxy_cost=1.08, placement_hash="m3a00002",
                  scored_pool_selectable=True),
        _cand_row("m3b_1", "m3b_cluster_refinement", benchmark, profile,
                  proxy_cost=1.03, placement_hash="m3b00001",
                  scored_pool_selectable=True),
    ]


def _make_run_info(
    benchmark: str = "ibm01",
    profile: str = _PROFILE,
    selected_candidate: str = "orig",
    selected_family: str = "original_neighborhood",
    selected_cost: float = 1.0,
    original_cost: Optional[float] = 1.1,
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "profile": profile,
        "selected_candidate": selected_candidate,
        "selected_family": selected_family,
        "selected_cost": selected_cost,
        "original_cost": original_cost,
    }


def _make_classification(
    benchmark: str = "ibm01",
    profile: str = _PROFILE,
    classification: str = "late_stage_valid_but_worse",
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "profile": profile,
        "classification": classification,
        "reason": (
            "Valid and scored late-stage candidates were available "
            "but none beat the baseline cost."
        ),
        "recommended_next_step": "design new structural move family",
        "late_stage_generated": 3,
        "late_stage_valid": 3,
        "late_stage_invalid": 0,
        "late_stage_admitted": 3,
        "late_stage_not_admitted": 0,
        "late_stage_scored": 3,
        "late_stage_selectable": 3,
        "late_stage_best_cost": 1.03,
        "late_stage_best_delta_vs_final": 0.03,
        "late_stage_num_beating_final": 0,
        "late_stage_num_near_tie": 0,
    }


def _make_family_summary(
    benchmark: str = "ibm01",
    profile: str = _PROFILE,
    family: str = "m3a_pair_refinement",
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "profile": profile,
        "family": family,
        "generated_count": 2,
        "valid_count": 2,
        "invalid_count": 0,
        "duplicate_count": 0,
        "admitted_count": 2,
        "not_admitted_count": 0,
        "scored_count": 2,
        "skipped_budget_count": 0,
        "scored_pool_selectable_count": 2,
        "selected_count": 0,
        "selected_via_fallback_count": 0,
        "best_official_cost": 1.05,
        "best_official_delta_vs_final": 0.05,
        "median_official_cost": 1.065,
        "median_official_delta_vs_final": 0.065,
        "num_beating_final": 0,
        "num_near_tie": 0,
        "best_candidate_name": "m3a_1",
        "worst_official_cost": 1.08,
        "worst_candidate_name": "m3a_2",
    }


def _make_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "profile": _PROFILE,
        "benchmarks": ["ibm01"],
        "official_epsilon": 1e-5,
        "max_official_scores": None,
        "seed_discovery_budget": None,
        "json": False,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Tests: build_benchmark_summary_rows
# ---------------------------------------------------------------------------


def test_build_summary_returns_correct_count():
    run_rows = [_make_run_info("ibm01"), _make_run_info("ibm02")]
    classifs = [_make_classification("ibm01"), _make_classification("ibm02")]
    result = build_benchmark_summary_rows(run_rows, classifs)
    assert len(result) == 2


def test_build_summary_all_required_fields_present():
    run_rows = [_make_run_info()]
    classifs = [_make_classification()]
    row = build_benchmark_summary_rows(run_rows, classifs)[0]
    missing = set(_BENCHMARK_SUMMARY_FIELDS) - set(row.keys())
    assert not missing, f"Missing benchmark summary fields: {missing}"


def test_build_summary_fields_in_correct_order():
    row = build_benchmark_summary_rows([_make_run_info()], [])[0]
    # Row keys should at least contain _BENCHMARK_SUMMARY_FIELDS in order.
    keys = list(row.keys())
    ordered = [f for f in keys if f in _BENCHMARK_SUMMARY_FIELDS]
    assert ordered == list(_BENCHMARK_SUMMARY_FIELDS)


def test_build_summary_merges_classification_fields():
    classifs = [_make_classification(classification="near_local_optimum")]
    row = build_benchmark_summary_rows([_make_run_info()], classifs)[0]
    assert row["classification"] == "near_local_optimum"
    assert row["late_stage_generated"] == 3
    assert row["late_stage_scored"] == 3


def test_build_summary_defaults_when_no_matching_classification():
    row = build_benchmark_summary_rows([_make_run_info()], [])[0]
    assert row["classification"] is None
    assert row["late_stage_generated"] == 0
    assert row["late_stage_num_beating_final"] == 0


def test_build_summary_preserves_run_info():
    run = _make_run_info(
        selected_candidate="best_cand", selected_cost=0.95, original_cost=1.0
    )
    row = build_benchmark_summary_rows([run], [])[0]
    assert row["selected_candidate"] == "best_cand"
    assert row["selected_cost"] == pytest.approx(0.95)
    assert row["original_cost"] == pytest.approx(1.0)


def test_build_summary_handles_none_original_cost():
    row = build_benchmark_summary_rows([_make_run_info(original_cost=None)], [])[0]
    assert row["original_cost"] is None


def test_build_summary_does_not_mutate_inputs():
    run_rows = [_make_run_info()]
    classifs = [_make_classification()]
    orig_runs = copy.deepcopy(run_rows)
    orig_classifs = copy.deepcopy(classifs)
    build_benchmark_summary_rows(run_rows, classifs)
    assert run_rows == orig_runs
    assert classifs == orig_classifs


def test_build_summary_handles_minimal_run_row():
    row = build_benchmark_summary_rows([{"benchmark": "bm", "profile": "p"}], [])[0]
    assert all(f in row for f in _BENCHMARK_SUMMARY_FIELDS)


# ---------------------------------------------------------------------------
# Tests: render_findings_md — required sections
# ---------------------------------------------------------------------------


def test_render_md_contains_benchmark_summary_section():
    md = render_findings_md(
        build_benchmark_summary_rows([_make_run_info()], [_make_classification()]),
        [_make_family_summary()],
        [_make_classification()],
        _make_candidate_rows(),
        _make_config(),
    )
    assert "## Benchmark Summary" in md


def test_render_md_contains_family_effectiveness_section():
    md = render_findings_md(
        build_benchmark_summary_rows([_make_run_info()], [_make_classification()]),
        [_make_family_summary()],
        [_make_classification()],
        _make_candidate_rows(),
        _make_config(),
    )
    assert "## Family Effectiveness" in md


def test_render_md_contains_failure_classification_section():
    md = render_findings_md(
        build_benchmark_summary_rows([_make_run_info()], [_make_classification()]),
        [_make_family_summary()],
        [_make_classification()],
        _make_candidate_rows(),
        _make_config(),
    )
    assert "## Failure Classification" in md


def test_render_md_contains_recommendations_section():
    md = render_findings_md(
        build_benchmark_summary_rows([_make_run_info()], [_make_classification()]),
        [_make_family_summary()],
        [_make_classification()],
        _make_candidate_rows(),
        _make_config(),
    )
    assert "## Recommendations" in md


def test_render_md_contains_run_configuration_section():
    md = render_findings_md([], [], [], [], _make_config())
    assert "## Run Configuration" in md


def test_render_md_contains_profile_name():
    md = render_findings_md([], [], [], [], _make_config(profile="m3c-smoke"))
    assert "m3c-smoke" in md


def test_render_md_contains_benchmark_name():
    bm_sums = build_benchmark_summary_rows(
        [_make_run_info("ibm99")], [_make_classification("ibm99")]
    )
    md = render_findings_md(
        bm_sums, [], [_make_classification("ibm99")], [], _make_config(benchmarks=["ibm99"])
    )
    assert "ibm99" in md


def test_render_md_top_late_stage_section_when_late_stage_scored():
    rows = _make_candidate_rows()  # includes m3a/m3b scored rows
    md = render_findings_md([], [], [], rows, _make_config())
    assert "## Top Late-Stage Candidates" in md


def test_render_md_no_top_late_stage_section_when_none():
    rows = [_cand_row("orig", "original_neighborhood")]
    md = render_findings_md([], [], [], rows, _make_config())
    assert "## Top Late-Stage Candidates" not in md


def test_render_md_is_deterministic():
    bm_sums = build_benchmark_summary_rows([_make_run_info()], [_make_classification()])
    fam_sums = [_make_family_summary()]
    classifs = [_make_classification()]
    rows = _make_candidate_rows()
    cfg = _make_config()
    md1 = render_findings_md(bm_sums, fam_sums, classifs, rows, cfg)
    md2 = render_findings_md(bm_sums, fam_sums, classifs, rows, cfg)
    assert md1 == md2


def test_render_md_handles_empty_inputs():
    md = render_findings_md([], [], [], [], _make_config())
    assert "*No benchmark results available.*" in md
    assert "*No family summaries available.*" in md
    assert "*No classifications available.*" in md


def test_render_md_handles_none_costs():
    bm_sum = _make_run_info(selected_cost=None, original_cost=None)
    bm_sums = build_benchmark_summary_rows([bm_sum], [])
    md = render_findings_md(bm_sums, [], [], [], _make_config())
    assert "N/A" in md


def test_render_md_handles_minimal_config():
    md = render_findings_md([], [], [], [], {"profile": "smoke"})
    assert "## Run Configuration" in md
    assert "smoke" in md


def test_render_md_top_late_stage_sorted_by_cost():
    rows = [
        _cand_row("m3b_cheap", "m3b_cluster_refinement", proxy_cost=0.5,
                  scored=True, scored_pool_selectable=True),
        _cand_row("m3a_expensive", "m3a_pair_refinement", proxy_cost=1.5,
                  scored=True, scored_pool_selectable=True),
    ]
    md = render_findings_md([], [], [], rows, _make_config())
    assert md.index("m3b_cheap") < md.index("m3a_expensive")


def test_render_md_recommendations_groups_by_next_step():
    classifs = [
        _make_classification("ibm01", classification="late_stage_valid_but_worse"),
        _make_classification("ibm02", classification="late_stage_valid_but_worse"),
    ]
    bm_sums = build_benchmark_summary_rows(
        [_make_run_info("ibm01"), _make_run_info("ibm02")], classifs
    )
    md = render_findings_md(bm_sums, [], classifs, [], _make_config())
    assert "design new structural move family" in md


# ---------------------------------------------------------------------------
# Tests: write_m3d_outputs — file creation
# ---------------------------------------------------------------------------


def test_write_outputs_creates_all_four_files(tmp_path: Path):
    rows = _make_candidate_rows()
    fam_sums = [_make_family_summary()]
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)

    write_m3d_outputs(tmp_path, rows, fam_sums, bm_sums, classifs, _make_config())

    assert (tmp_path / "m3d_candidate_effectiveness.csv").exists()
    assert (tmp_path / "m3d_family_summary.csv").exists()
    assert (tmp_path / "m3d_benchmark_summary.csv").exists()
    assert (tmp_path / "m3d_findings.md").exists()


def test_write_outputs_json_flag_creates_json(tmp_path: Path):
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)
    write_m3d_outputs(tmp_path, [], [], bm_sums, classifs, _make_config(json=True))
    assert (tmp_path / "m3d_findings.json").exists()


def test_write_outputs_no_json_by_default(tmp_path: Path):
    bm_sums = build_benchmark_summary_rows([_make_run_info()], [])
    write_m3d_outputs(tmp_path, [], [], bm_sums, [], _make_config(json=False))
    assert not (tmp_path / "m3d_findings.json").exists()


def test_write_outputs_creates_output_dir(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "dir"
    bm_sums = build_benchmark_summary_rows([_make_run_info()], [])
    write_m3d_outputs(nested, [], [], bm_sums, [], _make_config())
    assert nested.exists()
    assert (nested / "m3d_benchmark_summary.csv").exists()


# ---------------------------------------------------------------------------
# Tests: CSV headers
# ---------------------------------------------------------------------------


def _read_headers(path: Path) -> List[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def test_benchmark_summary_csv_headers_match_fields(tmp_path: Path):
    bm_sums = build_benchmark_summary_rows([_make_run_info()], [])
    write_m3d_outputs(tmp_path, [], [], bm_sums, [], _make_config())
    headers = _read_headers(tmp_path / "m3d_benchmark_summary.csv")
    assert headers == list(_BENCHMARK_SUMMARY_FIELDS)


def test_benchmark_summary_csv_written_even_when_empty(tmp_path: Path):
    write_m3d_outputs(tmp_path, [], [], [], [], _make_config())
    csv_path = tmp_path / "m3d_benchmark_summary.csv"
    assert csv_path.exists()
    headers = _read_headers(csv_path)
    assert headers == list(_BENCHMARK_SUMMARY_FIELDS)


def test_candidate_csv_headers_stable_across_different_benchmarks(tmp_path: Path):
    rows1 = _make_candidate_rows("ibm01")
    rows2 = _make_candidate_rows("ibm02")

    bm_sums1 = build_benchmark_summary_rows([_make_run_info("ibm01")], [])
    bm_sums2 = build_benchmark_summary_rows([_make_run_info("ibm02")], [])

    write_m3d_outputs(
        tmp_path / "run1", rows1, [_make_family_summary("ibm01")],
        bm_sums1, [], _make_config(benchmarks=["ibm01"]),
    )
    write_m3d_outputs(
        tmp_path / "run2", rows2, [_make_family_summary("ibm02")],
        bm_sums2, [], _make_config(benchmarks=["ibm02"]),
    )

    h1 = _read_headers(tmp_path / "run1" / "m3d_candidate_effectiveness.csv")
    h2 = _read_headers(tmp_path / "run2" / "m3d_candidate_effectiveness.csv")
    assert h1 == h2


def test_benchmark_summary_csv_headers_stable_across_runs(tmp_path: Path):
    bm_sums1 = build_benchmark_summary_rows([_make_run_info("ibm01")], [])
    bm_sums2 = build_benchmark_summary_rows([_make_run_info("ibm02")], [])

    write_m3d_outputs(tmp_path / "run1", [], [], bm_sums1, [], _make_config())
    write_m3d_outputs(tmp_path / "run2", [], [], bm_sums2, [], _make_config())

    h1 = _read_headers(tmp_path / "run1" / "m3d_benchmark_summary.csv")
    h2 = _read_headers(tmp_path / "run2" / "m3d_benchmark_summary.csv")
    assert h1 == h2


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(tmp_path: Path):
    rows = _make_candidate_rows()
    fam_sums = [_make_family_summary()]
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)
    cfg = _make_config()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    write_m3d_outputs(out1, rows, fam_sums, bm_sums, classifs, cfg)
    write_m3d_outputs(out2, rows, fam_sums, bm_sums, classifs, cfg)

    for fname in (
        "m3d_candidate_effectiveness.csv",
        "m3d_family_summary.csv",
        "m3d_benchmark_summary.csv",
        "m3d_findings.md",
    ):
        c1 = (out1 / fname).read_text(encoding="utf-8")
        c2 = (out2 / fname).read_text(encoding="utf-8")
        assert c1 == c2, f"{fname} output differs between two identical runs"


# ---------------------------------------------------------------------------
# Tests: mutation safety
# ---------------------------------------------------------------------------


def test_write_does_not_mutate_inputs(tmp_path: Path):
    rows = _make_candidate_rows()
    fam_sums = [_make_family_summary()]
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)

    orig_rows = copy.deepcopy(rows)
    orig_fam = copy.deepcopy(fam_sums)
    orig_cl = copy.deepcopy(classifs)
    orig_bm = copy.deepcopy(bm_sums)

    write_m3d_outputs(tmp_path, rows, fam_sums, bm_sums, classifs, _make_config())

    assert rows == orig_rows
    assert fam_sums == orig_fam
    assert classifs == orig_cl
    assert bm_sums == orig_bm


# ---------------------------------------------------------------------------
# Tests: Markdown content correctness
# ---------------------------------------------------------------------------


def test_md_benchmark_summary_table_includes_benchmark_name(tmp_path: Path):
    bm_sums = build_benchmark_summary_rows([_make_run_info("ibm77")], [])
    write_m3d_outputs(tmp_path, [], [], bm_sums, [], _make_config(benchmarks=["ibm77"]))
    md = (tmp_path / "m3d_findings.md").read_text(encoding="utf-8")
    assert "ibm77" in md


def test_md_classification_table_includes_classification_value(tmp_path: Path):
    classifs = [_make_classification(classification="near_local_optimum")]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)
    write_m3d_outputs(tmp_path, [], [], bm_sums, classifs, _make_config())
    md = (tmp_path / "m3d_findings.md").read_text(encoding="utf-8")
    assert "near_local_optimum" in md


def test_md_recommendations_includes_next_step(tmp_path: Path):
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)
    write_m3d_outputs(tmp_path, [], [], bm_sums, classifs, _make_config())
    md = (tmp_path / "m3d_findings.md").read_text(encoding="utf-8")
    assert "design new structural move family" in md


def test_md_family_table_includes_family_name(tmp_path: Path):
    fam_sums = [_make_family_summary(family="m3b_cluster_refinement")]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], [])
    write_m3d_outputs(tmp_path, [], fam_sums, bm_sums, [], _make_config())
    md = (tmp_path / "m3d_findings.md").read_text(encoding="utf-8")
    assert "m3b_cluster_refinement" in md


# ---------------------------------------------------------------------------
# Tests: script module is importable (no solver side-effects at import time)
# ---------------------------------------------------------------------------


def test_script_module_importable():
    import importlib

    mod = importlib.import_module(
        "submissions.solver.scripts.analyze_m3d_effectiveness"
    )
    assert hasattr(mod, "build_benchmark_summary_rows")
    assert hasattr(mod, "render_findings_md")
    assert hasattr(mod, "write_m3d_outputs")
    assert hasattr(mod, "main")


def test_benchmark_summary_fields_exported():
    from submissions.solver.scripts.analyze_m3d_effectiveness import (
        _BENCHMARK_SUMMARY_FIELDS,
    )

    assert isinstance(_BENCHMARK_SUMMARY_FIELDS, tuple)
    assert len(_BENCHMARK_SUMMARY_FIELDS) > 0
    assert "benchmark" in _BENCHMARK_SUMMARY_FIELDS
    assert "classification" in _BENCHMARK_SUMMARY_FIELDS
    assert "late_stage_generated" in _BENCHMARK_SUMMARY_FIELDS


# ---------------------------------------------------------------------------
# Tests: stale JSON cleanup on reuse of output directory
# ---------------------------------------------------------------------------


def test_stale_json_removed_on_non_json_run(tmp_path: Path):
    """JSON written in a first run must be removed if the second run omits --json."""
    rows = _make_candidate_rows()
    fam_sums = [_make_family_summary()]
    classifs = [_make_classification()]
    bm_sums = build_benchmark_summary_rows([_make_run_info()], classifs)

    # First run with JSON enabled — file must be created.
    write_m3d_outputs(tmp_path, rows, fam_sums, bm_sums, classifs, _make_config(json=True))
    assert (tmp_path / "m3d_findings.json").exists()

    # Second run to the same directory without JSON — stale file must be removed.
    write_m3d_outputs(tmp_path, rows, fam_sums, bm_sums, classifs, _make_config(json=False))
    assert not (tmp_path / "m3d_findings.json").exists()

    # CSV and Markdown outputs must still be present after the second run.
    assert (tmp_path / "m3d_benchmark_summary.csv").exists()
    assert (tmp_path / "m3d_candidate_effectiveness.csv").exists()
    assert (tmp_path / "m3d_family_summary.csv").exists()
    assert (tmp_path / "m3d_findings.md").exists()
