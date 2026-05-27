import ast
import inspect
from pathlib import Path

from conftest import make_benchmark
from submissions.solver.core.candidate_scoring import score_and_select
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidateScoringConfig,
)
from submissions.solver.core.candidates import generate_candidates
from submissions.solver.core import m4c_ranking
from submissions.solver.core.m3d_candidate_export import export_candidate_rows
from submissions.solver.scripts.run_benchmarks import _PROFILES


def _candidate(name, delta, *, family="m4b_region_repair", fifo=0, valid=True, **extra):
    row = {
        "candidate_name": name,
        "family": family,
        "valid": valid,
        "duplicate": False,
        "post_legalization_approx_delta": delta,
        "fifo_index": fifo,
        "region_id": extra.pop("region_id", "r0"),
        "move_type": extra.pop("move_type", "spread"),
    }
    row.update(extra)
    return row


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
        name="m4c_test",
    )


def _gen_cfg(**overrides):
    params = dict(
        only_original_neighborhood=True,
        m4b_region_repair=True,
        m4b_reserved_scores=20,
        m4b_grid_dims=(2, 2),
        m4b_min_macros_per_region=2,
        m4b_max_combos_per_region=16,
        m4c_ranking=True,
        m4c_k_ranked=16,
        m4c_exploration=4,
    )
    params.update(overrides)
    return CandidateGenerationConfig(**params)


def _score_cfg(max_scores=80):
    return CandidateScoringConfig(max_official_scores=max_scores, prefilter_mode="off")


def test_m4c_rank_score_is_deterministic():
    candidates = [_candidate(f"m4b_r0_m{i}_spread", float(i), fifo=i) for i in range(8)]
    first = m4c_ranking.assign_buckets([dict(row) for row in candidates])
    second = m4c_ranking.assign_buckets([dict(row) for row in candidates])
    assert [row["m4c_rank_score"] for row in first] == [
        row["m4c_rank_score"] for row in second
    ]
    assert [row["m4c_rank_bucket"] for row in first] == [
        row["m4c_rank_bucket"] for row in second
    ]


def test_no_evaluator_or_proxy_cost_leakage_into_rank_score():
    base = [_candidate(f"m4b_r0_m{i}_spread", float(i), fifo=i) for i in range(6)]
    with_costs = [
        {**row, "proxy_cost": 1000.0 - idx, "evaluator_cost": 500.0 - idx}
        for idx, row in enumerate(base)
    ]
    assert m4c_ranking.compute_rank_scores(base) == m4c_ranking.compute_rank_scores(with_costs)


def test_no_per_benchmark_conditionals():
    source = inspect.getsource(m4c_ranking.compute_rank_scores)
    assert "benchmark" not in source
    assert "ibm01" not in source
    assert "ibm02" not in source
    assert "ibm03" not in source


def test_no_scorer_cache_legalizer_imports():
    source = Path(m4c_ranking.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    forbidden = ["scoring", "score_cache", "legalizer", "legalization"]
    assert all(
        not any(token in imported for token in forbidden) for imported in imports
    )


def test_no_swap_or_new_move_types_introduced_by_ranking():
    candidates = [
        _candidate("m4b_r0_m0_m1_centroid_shift", -2.0, fifo=0, move_type="centroid_shift"),
        _candidate("m4b_r0_m0_m1_spread", -1.0, fifo=1, move_type="spread"),
    ]
    ranked = m4c_ranking.assign_buckets(candidates, k_ranked=1, exploration=1)
    assert {row["move_type"] for row in ranked} == {"centroid_shift", "spread"}
    assert all("swap" not in row["candidate_name"] for row in ranked)


def test_ranked_and_exploration_bucket_counts():
    candidates = [_candidate(f"m4b_r0_m{i}_spread", float(i), fifo=i) for i in range(25)]
    ranked = m4c_ranking.assign_buckets(candidates, k_ranked=16, exploration=4)
    assert sum(row["m4c_rank_bucket"] == "ranked" for row in ranked) == 16
    assert sum(row["m4c_rank_bucket"] == "exploration" for row in ranked) == 4
    assert {"ranked", "exploration"}.issubset({row["m4c_rank_bucket"] for row in ranked})


def test_known_winner_force_insert_preserves_pool():
    candidates = [_candidate(f"m4b_r0_m{i}_spread", float(i), fifo=i) for i in range(25)]
    candidates[-1]["candidate_name"] = "m4b_r1_m4_m51_spread"
    ranked = m4c_ranking.assign_buckets(
        candidates,
        k_ranked=16,
        exploration=4,
        known_winners=["m4b_r1_m4_m51_spread"],
    )
    winner = next(row for row in ranked if row["candidate_name"] == "m4b_r1_m4_m51_spread")
    assert winner["m4c_rank_bucket"] == "ranked"
    assert "known_winner_force_insert" in winner["m4c_rank_reason"]
    assert sum(row["m4c_rank_bucket"] == "ranked" for row in ranked) == 16
    assert sum(row["m4c_rank_bucket"] == "exploration" for row in ranked) == 4


def test_telemetry_columns_present_in_assigned_rows():
    row = m4c_ranking.assign_buckets([_candidate("m4b_r0_m0_m1_spread", -1.0)])[0]
    assert {
        "m4c_rank_score",
        "m4c_rank_bucket",
        "m4c_rank_reason",
        "family_rank",
        "family_normalized_approx_delta",
    }.issubset(row)


def test_non_m4b_candidates_receive_null_rank_telemetry():
    rows = [
        _candidate("m4b_r0_m0_m1_spread", -1.0),
        _candidate("original_raw", None, family="original", valid=True),
    ]
    ranked = m4c_ranking.assign_buckets(rows)
    assert ranked[1]["m4c_rank_score"] is None
    assert ranked[1]["m4c_rank_bucket"] is None


def test_m4c_default_preserves_80_total_20_reserved_budget():
    profile = _PROFILES["m4c-default"]
    assert profile["max_official_scores"] == 80
    assert profile["m4b_reserved_scores"] == 20
    assert profile["m4c_ranking"] is True
    assert profile["m4c_k_ranked"] == 16
    assert profile["m4c_exploration"] == 4


def test_m4b_default_remains_fifo_profile():
    profile = _PROFILES["m4b-default"]
    assert profile["max_official_scores"] == 80
    assert profile["m4b_reserved_scores"] == 20
    assert profile["m4b_region_repair"] is True
    assert "m4c_ranking" not in profile


def test_m4c_reserved_bucket_wiring_counts_and_telemetry():
    bm = _bm()
    cfg = _gen_cfg()
    best, ranked, diag = score_and_select(
        generate_candidates(bm, cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=cfg,
    )
    rows = export_candidate_rows(ranked, best, diag, benchmark=bm.name, profile="m4c-default")
    m4b_scored = [
        row for row in rows if row["family"] == "m4b_region_repair" and row["scored"]
    ]
    assert len(m4b_scored) == min(20, diag.m4b_legalized_count)
    assert sum(row["m4c_rank_bucket"] == "ranked" for row in m4b_scored) == min(16, len(m4b_scored))
    if len(m4b_scored) >= 20:
        assert sum(row["m4c_rank_bucket"] == "exploration" for row in m4b_scored) == 4
    assert {
        "m4c_rank_score",
        "m4c_rank_bucket",
        "m4c_rank_reason",
        "family_rank",
        "family_normalized_approx_delta",
    }.issubset(rows[0])


def test_m4b_default_fifo_scored_names_unchanged_by_m4c_fields():
    bm = _bm()
    base_cfg = _gen_cfg(m4c_ranking=False)
    m4c_disabled_cfg = _gen_cfg(m4c_ranking=False, m4c_k_ranked=1, m4c_exploration=19)
    _best_a, ranked_a, _diag_a = score_and_select(
        generate_candidates(bm, base_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=base_cfg,
    )
    _best_b, ranked_b, _diag_b = score_and_select(
        generate_candidates(bm, m4c_disabled_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=m4c_disabled_cfg,
    )
    assert [
        sc.name for sc in ranked_a if sc.family == "m4b_region_repair" and sc.was_scored
    ] == [
        sc.name for sc in ranked_b if sc.family == "m4b_region_repair" and sc.was_scored
    ]


def test_non_m4b_scored_counts_identical_to_m4b_and_prefilter_unchanged():
    bm = _bm()
    m4b_cfg = _gen_cfg(m4c_ranking=False)
    m4c_cfg = _gen_cfg()
    best_b, ranked_b, diag_b = score_and_select(
        generate_candidates(bm, m4b_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=m4b_cfg,
    )
    best_c, ranked_c, diag_c = score_and_select(
        generate_candidates(bm, m4c_cfg),
        bm,
        plc=None,
        scoring_config=_score_cfg(),
        generation_config=m4c_cfg,
    )
    rows_b = export_candidate_rows(ranked_b, best_b, diag_b, benchmark=bm.name, profile="m4b-default")
    rows_c = export_candidate_rows(ranked_c, best_c, diag_c, benchmark=bm.name, profile="m4c-default")
    non_m4b_b = [row for row in rows_b if row["family"] != "m4b_region_repair"]
    non_m4b_c = [row for row in rows_c if row["family"] != "m4b_region_repair"]
    assert sum(row["scored"] for row in non_m4b_b) == sum(row["scored"] for row in non_m4b_c)
    by_name_b = {
        row["candidate_name"]: (row["admitted"], row["scored"], row["skip_reason"])
        for row in non_m4b_b
        if row["family"].startswith("original")
    }
    by_name_c = {
        row["candidate_name"]: (row["admitted"], row["scored"], row["skip_reason"])
        for row in non_m4b_c
        if row["family"].startswith("original")
    }
    assert by_name_c == by_name_b
