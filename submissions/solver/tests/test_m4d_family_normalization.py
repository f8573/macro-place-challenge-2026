import ast
import inspect
from pathlib import Path

from submissions.solver.core import m4d_family_normalization as m4d


def _candidate(
    name: str,
    family: str,
    approx_delta,
    *,
    generation_rank: int,
    extra: dict | None = None,
):
    row = {
        "candidate_name": name,
        "family": family,
        "approx_delta": approx_delta,
        "generation_rank": generation_rank,
    }
    if extra:
        row.update(extra)
    return row


def test_m4d_rank_score_deterministic():
    candidates = [
        _candidate("n2", "original_neighborhood", -1.0, generation_rank=2),
        _candidate("n0", "original_neighborhood", -3.0, generation_rank=0),
        _candidate("n1", "original_neighborhood", -2.0, generation_rank=1),
        _candidate("r0", "original_refinement", -4.0, generation_rank=3),
        _candidate("r1", "original_refinement", -1.0, generation_rank=4),
    ]
    first = m4d.build_m4d_telemetry([dict(row) for row in candidates])
    second = m4d.build_m4d_telemetry([dict(row) for row in candidates])
    assert [row["m4d_rank_score"] for row in first] == [row["m4d_rank_score"] for row in second]
    assert [row["m4d_cross_family_rank"] for row in first] == [
        row["m4d_cross_family_rank"] for row in second
    ]


def test_per_family_rank_percentile_in_unit_interval():
    candidates = [
        _candidate("n0", "original_neighborhood", -5.0, generation_rank=0),
        _candidate("n1", "original_neighborhood", -2.0, generation_rank=1),
        _candidate("r0", "original_refinement", -7.0, generation_rank=2),
        _candidate("r1", "original_refinement", 3.0, generation_rank=3),
    ]
    ranked = m4d.build_m4d_telemetry(candidates)
    values = [row["m4d_rank_score"] for row in ranked if row["m4d_rank_score"] is not None]
    assert values
    assert all(0.0 <= value <= 1.0 for value in values)


def test_lower_approx_delta_ranks_better_within_family():
    candidates = [
        _candidate("best", "original_neighborhood", -10.0, generation_rank=0),
        _candidate("mid", "original_neighborhood", -5.0, generation_rank=1),
        _candidate("worst", "original_neighborhood", 2.0, generation_rank=2),
    ]
    ranked = {row["candidate_name"]: row for row in m4d.build_m4d_telemetry(candidates)}
    assert ranked["best"]["m4d_rank_score"] < ranked["mid"]["m4d_rank_score"]
    assert ranked["mid"]["m4d_rank_score"] < ranked["worst"]["m4d_rank_score"]


def test_ties_are_deterministic_by_generation_rank_then_name():
    candidates = [
        _candidate("b", "original_refinement", -1.0, generation_rank=1),
        _candidate("a", "original_refinement", -1.0, generation_rank=0),
        _candidate("c", "original_refinement", -1.0, generation_rank=2),
    ]
    ranked = {row["candidate_name"]: row for row in m4d.build_m4d_telemetry(candidates)}
    assert ranked["a"]["m4d_rank_score"] == 0.0
    assert ranked["b"]["m4d_rank_score"] == 0.5
    assert ranked["c"]["m4d_rank_score"] == 1.0


def test_null_approx_delta_families_excluded():
    candidates = [
        _candidate("keep", "original_neighborhood", -1.0, generation_rank=0),
        _candidate("drop", "original_neighborhood", None, generation_rank=1),
    ]
    ranked = {row["candidate_name"]: row for row in m4d.build_m4d_telemetry(candidates)}
    assert ranked["keep"]["m4d_rank_score"] == 0.0
    assert ranked["drop"]["m4d_rank_score"] is None
    assert ranked["drop"]["m4d_family_normalized_approx_delta"] is None
    assert ranked["drop"]["m4d_cross_family_rank"] is None


def test_m3a_and_m3b_excluded_from_cross_family_normalization():
    candidates = [
        _candidate("n0", "original_neighborhood", -1.0, generation_rank=0),
        _candidate("m3a", "m3a_pair_refinement", -99.0, generation_rank=1),
        _candidate("m3b", "m3b_cluster_refinement", -99.0, generation_rank=2),
        _candidate("m4b", "m4b_region_repair", -99.0, generation_rank=3),
        _candidate("r0", "original_refinement", -2.0, generation_rank=4),
    ]
    ranked = {row["candidate_name"]: row for row in m4d.build_m4d_telemetry(candidates)}
    assert ranked["n0"]["m4d_cross_family_rank"] == 2
    assert ranked["r0"]["m4d_cross_family_rank"] == 1
    assert ranked["m3a"]["m4d_rank_score"] is None
    assert ranked["m3b"]["m4d_rank_score"] is None
    assert ranked["m4b"]["m4d_rank_score"] is None


def test_no_proxy_cost_evaluator_admitted_or_scored_used():
    base = [
        _candidate("n0", "original_neighborhood", -2.0, generation_rank=0),
        _candidate("n1", "original_neighborhood", 1.0, generation_rank=1),
    ]
    changed = [
        {
            **row,
            "proxy_cost": 100.0,
            "evaluator_cost": 200.0,
            "admitted": False,
            "scored": True,
        }
        for row in base
    ]
    assert m4d.compute_rank_scores(base) == m4d.compute_rank_scores(changed)
    assert m4d.compute_family_normalized_approx(base) == m4d.compute_family_normalized_approx(changed)


def test_no_m4c_rank_score_used_in_m4d_module():
    base = [
        _candidate("n0", "original_neighborhood", -2.0, generation_rank=0),
        _candidate("n1", "original_neighborhood", 1.0, generation_rank=1),
    ]
    changed = [{**row, "m4c_rank_score": 999.0} for row in base]
    assert m4d.compute_rank_scores(base) == m4d.compute_rank_scores(changed)


def test_no_benchmark_literals_or_conditionals():
    source = inspect.getsource(m4d)
    for needle in ("benchmark", "ibm01", "ibm02", "ibm03"):
        assert needle not in source


def test_no_scorer_cache_legalizer_imports():
    source = Path(m4d.__file__).read_text(encoding="utf-8")
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
    forbidden = ["scoring", "score_cache", "legalizer", "legalization", "benchmark"]
    assert all(not any(token in imported for token in forbidden) for imported in imports)
