import csv
import json
from pathlib import Path

import pytest

from conftest import make_benchmark
from submissions.solver import m4a_loss_attribution as m4a
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
)
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core.m3d_candidate_export import export_candidate_rows
from submissions.solver.scripts.run_benchmarks import _PROFILES


def _profile_snapshot(name: str) -> dict:
    return dict(_PROFILES[name])


def _bm():
    return make_benchmark(
        n_hard=8,
        canvas=100.0,
        macro_size=4.0,
        positions=[
            [10.0, 10.0],
            [20.0, 10.0],
            [30.0, 30.0],
            [40.0, 30.0],
            [60.0, 60.0],
            [70.0, 60.0],
            [80.0, 80.0],
            [90.0, 80.0],
        ],
        net_nodes=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [1, 6, 7]],
        name="m4d_test",
    )


def _cfg_from_profile(profile_name: str, **overrides) -> CandidateGenerationConfig:
    profile = _PROFILES[profile_name]
    params = dict(
        candidate_budget=profile.get("candidate_budget"),
        neighborhood_macro_limit=profile.get("neighborhood_macro_limit", 20),
        neighborhood_step_profile=profile.get("neighborhood_step_profile", "medium"),
        disable_global_candidates=profile.get("disable_global_candidates", False),
        only_original_neighborhood=profile.get("only_original_neighborhood", False),
        refinement_around_winners=profile.get("refinement_around_winners", False),
        refinement_top_k=profile.get("refinement_top_k", 5),
        refinement_combo_size=profile.get("refinement_combo_size", 2),
        refinement_seed_strategy=profile.get("refinement_seed_strategy", "conservative"),
        refinement_exploration_seeds=profile.get("refinement_exploration_seeds", 1),
        line_search_around_winners=profile.get("line_search_around_winners", False),
        line_search_top_k=profile.get("line_search_top_k", 3),
        line_search_max_scale=profile.get("line_search_max_scale", 4.0),
        line_search_stop_after_worse=profile.get("line_search_stop_after_worse", 2),
        m3a_pair_refinement=profile.get("m3a_pair_refinement", False),
        m3a_top_k_pairs=profile.get("m3a_top_k_pairs", 64),
        m3a_score_budget=profile.get("m3a_score_budget", None),
        m3b_cluster_refinement=profile.get("m3b_cluster_refinement", False),
        m3b_top_k_clusters=profile.get("m3b_top_k_clusters", 32),
        m3b_score_budget=profile.get("m3b_score_budget", None),
        m3c_budget_allocation=profile.get("m3c_budget_allocation", False),
        m3c_pre_m3_budget=profile.get("m3c_pre_m3_budget", None),
        m3c_m3a_reserved_budget=profile.get("m3c_m3a_reserved_budget", None),
        m3c_m3b_reserved_budget=profile.get("m3c_m3b_reserved_budget", None),
        m3c_rollover_unused_budget=profile.get("m3c_rollover_unused_budget", True),
        m4b_region_repair=profile.get("m4b_region_repair", False),
        m4b_reserved_scores=profile.get("m4b_reserved_scores", 20),
        m4b_grid_dims=tuple(profile.get("m4b_grid_dims", (3, 3))),
        m4b_min_macros_per_region=profile.get("m4b_min_macros_per_region", 2),
        m4b_max_combos_per_region=profile.get("m4b_max_combos_per_region", 16),
        m4b_legalization_max_displacement_um=profile.get(
            "m4b_legalization_max_displacement_um", 200.0
        ),
        m4b_perturbation_fraction=profile.get("m4b_perturbation_fraction", 0.5),
        m4c_ranking=profile.get("m4c_ranking", False),
        m4c_k_ranked=profile.get("m4c_k_ranked", 16),
        m4c_exploration=profile.get("m4c_exploration", 4),
        m4c_max_per_region=profile.get("m4c_max_per_region", None),
        m4c_known_winners=profile.get("m4c_known_winners", []),
        m4d_family_normalization=profile.get("m4d_family_normalization", False),
        m4d_family_quota_floors=profile.get("m4d_family_quota_floors", None),
    )
    params.update(overrides)
    return CandidateGenerationConfig(**params)


def _score_cfg(max_scores: int = 80) -> CandidateScoringConfig:
    return CandidateScoringConfig(max_official_scores=max_scores)


def _run(profile_name: str, **overrides):
    bm = _bm()
    gen_cfg = _cfg_from_profile(profile_name, **overrides)
    best, ranked, diag = score_and_select(
        generate_candidates(bm, gen_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(_PROFILES[profile_name]["max_official_scores"]),
        generation_config=gen_cfg,
    )
    return best, ranked, diag


def _write_m4d_input_fixture(input_dir: Path, profile: str = "m4d-default") -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows = [
        {
            "benchmark": "ibm01",
            "profile": profile,
            "original_cost": "1.0",
            "selected_cost": "0.95",
            "delta_vs_original": "-0.05",
            "classification": "inconclusive",
            "classification_reason": "fixture",
            "late_stage_generated": "1",
            "late_stage_scored": "1",
        }
    ]
    with (input_dir / "m4d_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_rows[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_rows)

    family_rows = [
        {
            "benchmark": "ibm01",
            "profile": profile,
            "family": "original_refinement",
            "generated_count": "2",
            "valid_count": "2",
            "scored_count": "2",
            "selected_count": "1",
            "best_official_cost": "0.95",
            "best_official_delta_vs_final": "0.0",
            "median_official_cost": "0.955",
            "median_official_delta_vs_final": "0.005",
        }
    ]
    with (input_dir / "m4d_family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=family_rows[0].keys())
        writer.writeheader()
        writer.writerows(family_rows)

    candidate_rows = [
        {
            "benchmark": "ibm01",
            "profile": profile,
            "candidate_name": "original_refinement_m1_scale2x",
            "family": "original_refinement",
            "valid": "True",
            "duplicate": "False",
            "admitted": "True",
            "not_admitted": "False",
            "scored": "True",
            "skip_reason": "scored",
            "proxy_cost": "0.95",
            "approx_delta": "-0.20",
            "m4c_rank_score": "",
            "m4c_rank_bucket": "",
            "m4c_rank_reason": "",
            "family_rank": "",
            "family_normalized_approx_delta": "",
            "m4d_rank_score": "0.0",
            "m4d_family_normalized_approx_delta": "0.0",
            "m4d_cross_family_rank": "1",
            "m4d_rank_reason": "family=original_refinement rank=1/2 percentile=0.000000",
            "placement_hash": "abc12345",
        },
        {
            "benchmark": "ibm01",
            "profile": profile,
            "candidate_name": "original_refinement_m2_scale2x",
            "family": "original_refinement",
            "valid": "True",
            "duplicate": "False",
            "admitted": "True",
            "not_admitted": "False",
            "scored": "True",
            "skip_reason": "scored",
            "proxy_cost": "0.96",
            "approx_delta": "-0.10",
            "m4c_rank_score": "",
            "m4c_rank_bucket": "",
            "m4c_rank_reason": "",
            "family_rank": "",
            "family_normalized_approx_delta": "",
            "m4d_rank_score": "1.0",
            "m4d_family_normalized_approx_delta": "1.0",
            "m4d_cross_family_rank": "2",
            "m4d_rank_reason": "family=original_refinement rank=2/2 percentile=1.000000",
            "placement_hash": "abc12346",
        },
    ]
    with (input_dir / "m4d_candidate_effectiveness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_rows[0].keys())
        writer.writeheader()
        writer.writerows(candidate_rows)


def _artifact_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def test_profile_snapshots_unchanged_for_frozen_profiles():
    assert _profile_snapshot("m3c-default") == {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3C default: reserved M3A/M3B slices within 60-score budget, cold-run vs m2b-final",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 50,
        "m3c_m3a_reserved_budget": 5,
        "m3c_m3b_reserved_budget": 5,
        "m3c_rollover_unused_budget": True,
    }
    assert _profile_snapshot("m4b-default") == {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M4B default: M3C baseline plus reserved regional repair bucket",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 80,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 50,
        "m3c_m3a_reserved_budget": 5,
        "m3c_m3b_reserved_budget": 5,
        "m3c_rollover_unused_budget": True,
        "m4b_region_repair": True,
        "m4b_reserved_scores": 20,
        "m4b_grid_dims": (3, 3),
        "m4b_min_macros_per_region": 2,
        "m4b_max_combos_per_region": 16,
        "m4b_legalization_max_displacement_um": 200.0,
        "m4b_perturbation_fraction": 0.5,
    }
    assert _profile_snapshot("m4c-default") == {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M4C default: M4B with family-aware reserved-bucket ranking",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 80,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 50,
        "m3c_m3a_reserved_budget": 5,
        "m3c_m3b_reserved_budget": 5,
        "m3c_rollover_unused_budget": True,
        "m4b_region_repair": True,
        "m4b_reserved_scores": 20,
        "m4b_grid_dims": (3, 3),
        "m4b_min_macros_per_region": 2,
        "m4b_max_combos_per_region": 16,
        "m4b_legalization_max_displacement_um": 200.0,
        "m4b_perturbation_fraction": 0.5,
        "m4c_ranking": True,
        "m4c_k_ranked": 16,
        "m4c_exploration": 4,
        "m4c_max_per_region": None,
        "m4c_known_winners": [
            "m4b_r1_m4_m51_spread",
            "m4b_r1_m7_m43_centroid_shift",
        ],
    }


def test_m4d_default_preserves_budget_split_and_extends_m4c():
    m4c = _profile_snapshot("m4c-default")
    m4d = _profile_snapshot("m4d-default")
    for key, value in m4c.items():
        if key == "description":
            continue
        assert m4d[key] == value
    assert m4d["max_official_scores"] == 80
    assert m4d["m3c_pre_m3_budget"] == 50
    assert m4d["m3c_m3a_reserved_budget"] == 5
    assert m4d["m3c_m3b_reserved_budget"] == 5
    assert m4d["m4b_reserved_scores"] == 20
    assert m4d["m4d_family_normalization"] is True
    assert m4d["m4d_family_quota_floors"] is None


def test_disabled_flag_parity_with_m4c_default():
    best_m4c, ranked_m4c, diag_m4c = _run("m4c-default")
    best_flag_off, ranked_flag_off, diag_flag_off = _run(
        "m4d-default",
        m4d_family_normalization=False,
        m4d_family_quota_floors=None,
    )
    assert best_flag_off.name == best_m4c.name
    assert diag_flag_off.candidates_officially_scored == diag_m4c.candidates_officially_scored
    assert [
        (sc.name, sc.family, sc.was_scored, sc.metadata.get("skip_reason"))
        for sc in ranked_flag_off
        if sc.family.startswith("original")
    ] == [
        (sc.name, sc.family, sc.was_scored, sc.metadata.get("skip_reason"))
        for sc in ranked_m4c
        if sc.family.startswith("original")
    ]


def test_selected_cost_non_regression_fixture():
    best_m4c, _ranked_m4c, _diag_m4c = _run("m4c-default")
    best_m4d, _ranked_m4d, _diag_m4d = _run("m4d-default")
    assert best_m4d.proxy_cost <= best_m4c.proxy_cost + 1e-7


def test_m4c_known_winners_remain_scored():
    _best_m4c, ranked_m4c, _diag_m4c = _run("m4c-default")
    _best_m4d, ranked_m4d, _diag_m4d = _run("m4d-default")
    scored_m4c = {
        sc.name for sc in ranked_m4c if sc.family == "m4b_region_repair" and sc.was_scored
    }
    scored_m4d = {
        sc.name for sc in ranked_m4d if sc.family == "m4b_region_repair" and sc.was_scored
    }
    assert scored_m4d == scored_m4c


def test_m4d_csv_schema_includes_m4c_columns_plus_m4d_columns():
    best, ranked, diag = _run("m4d-default")
    rows = export_candidate_rows(ranked, best, diag, benchmark="m4d_test", profile="m4d-default")
    assert rows
    expected_columns = {
        "m4c_rank_score",
        "m4c_rank_bucket",
        "m4c_rank_reason",
        "family_rank",
        "family_normalized_approx_delta",
        "m4d_rank_score",
        "m4d_family_normalized_approx_delta",
        "m4d_cross_family_rank",
        "m4d_rank_reason",
    }
    assert expected_columns.issubset(rows[0])


def test_m4a_canonical_reads_m4d_artifacts(tmp_path):
    _write_m4d_input_fixture(tmp_path)
    result, _, _ = m4a.analyze(
        profile="m4d-default",
        benchmarks=["ibm01"],
        official_epsilon=1e-5,
        input_dir=tmp_path,
        input_prefix="m4d",
        runner_json=tmp_path / "missing.json",
    )
    bench = result["benchmarks"]["ibm01"]
    assert result["inputs"]["candidate_effectiveness"].endswith("m4d_candidate_effectiveness.csv")
    assert bench["prefilter_evaluator"]["rank_column"] == "approx_delta"
    assert bench["costs"]["selected_cost"] == 0.95


def test_m4a_rankscore_reads_m4d_artifacts(tmp_path):
    _write_m4d_input_fixture(tmp_path)
    result, _, _ = m4a.analyze(
        profile="m4d-default",
        benchmarks=["ibm01"],
        official_epsilon=1e-5,
        input_dir=tmp_path,
        input_prefix="m4d",
        runner_json=tmp_path / "missing.json",
        rank_column="m4d_rank_score",
    )
    bench = result["benchmarks"]["ibm01"]
    assert bench["rank_column_prefilter_evaluator"]["rank_column"] == "m4d_rank_score"
    assert bench["rank_column_prefilter_evaluator"]["usable_count"] == 2


def test_selected_cost_vs_m4c_artifact_check():
    m4d_json = _artifact_path("submissions", "solver", "artifacts", "run_m4d-default.json")
    if not m4d_json.exists():
        pytest.skip("run_m4d-default.json not generated yet")
    m4c_json = _artifact_path("submissions", "solver", "artifacts", "run_m4c-default.json")
    m4d = json.loads(m4d_json.read_text(encoding="utf-8"))
    m4c = json.loads(m4c_json.read_text(encoding="utf-8"))
    by_bench_m4d = {row["benchmark"]: row for row in m4d["results"]}
    by_bench_m4c = {row["benchmark"]: row for row in m4c["results"]}
    for benchmark in ("ibm01", "ibm02", "ibm03"):
        assert by_bench_m4d[benchmark]["proxy_cost"] <= by_bench_m4c[benchmark]["proxy_cost"] + 1e-7


def test_generated_m4d_artifacts_preserve_known_winners():
    m4d_csv = _artifact_path("analysis", "m4d", "m4d_candidate_effectiveness.csv")
    if not m4d_csv.exists():
        pytest.skip("analysis/m4d/m4d_candidate_effectiveness.csv not generated yet")
    with m4d_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for benchmark in ("ibm01", "ibm02", "ibm03"):
        benchmark_rows = [row for row in rows if row.get("benchmark") == benchmark]
        for winner in _PROFILES["m4c-default"]["m4c_known_winners"]:
            matched = [
                row
                for row in benchmark_rows
                if row.get("candidate_name") == winner and row.get("scored") == "True"
            ]
            if matched:
                assert matched[0]["family"] == "m4b_region_repair"
