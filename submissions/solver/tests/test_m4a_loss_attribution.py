import importlib
import json
import subprocess
import sys
from pathlib import Path

from submissions.solver import m4a_loss_attribution as m4a


REPO_ROOT = Path(__file__).resolve().parents[3]


def _classify(**overrides):
    params = {
        "delta": 0.0,
        "eps": 1e-5,
        "valid_rate_local": 0.9,
        "valid_rate_baseline": 0.9,
        "budget_saturation": 0.1,
        "prefilter_evaluator": {
            "usable_count": 0,
            "approx_coverage": 0.0,
            "spearman_rs": None,
            "top5_inversions": None,
        },
        "diversity": {
            "unique_macro_ratio": 0.9,
            "collision_ratio": 0.0,
        },
    }
    params.update(overrides)
    return m4a.classify_benchmark(**params)[0]


def _walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_keys(value)


def _walk_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_values(value)
    else:
        yield obj


def _run_cli(output_dir):
    cmd = [
        sys.executable,
        "-m",
        "submissions.solver.m4a_loss_attribution",
        "--profile",
        "m3c-default",
        "--benchmarks",
        "ibm01",
        "ibm02",
        "ibm03",
        "--official-epsilon",
        "1e-5",
        "--input-dir",
        "analysis/m3d",
        "--runner-json",
        "submissions/solver/artifacts/run_m3c-default.json",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((output_dir / "m4a_loss_attribution.json").read_text())


def test_score_banding():
    eps = 1e-5
    assert m4a.score_band(1.0, 0.9998, eps) == "meaningful_win"
    assert m4a.score_band(1.0, 0.99995, eps) == "epsilon_win"
    assert m4a.score_band(1.0, 1.0, eps) == "flat"
    assert m4a.score_band(1.0, 1.00002, eps) == "regression"


def test_parse_macro_ids():
    assert m4a.parse_macro_ids("m3a_p16_5_154_swap") == [5, 154]
    assert m4a.parse_macro_ids("original_refinement_m215_scale2x") == [215]
    assert m4a.parse_macro_ids("m3b_c30_0_52_166_centroid_shift") == [52, 166]


def test_classification_rule_a():
    assert (
        _classify(valid_rate_local=0.1, valid_rate_baseline=0.95, delta=1e-5)
        == "legality_bottleneck"
    )
    assert (
        _classify(valid_rate_local=0.25, valid_rate_baseline=0.95, delta=1e-5)
        != "legality_bottleneck"
    )


def test_classification_rule_b():
    prefilter = {
        "usable_count": 25,
        "approx_coverage": 0.8,
        "spearman_rs": 0.1,
        "top5_inversions": 3,
    }
    assert (
        _classify(delta=1.0, prefilter_evaluator=prefilter)
        == "prefilter_evaluator_disagreement"
    )
    guarded = {**prefilter, "usable_count": 19}
    assert (
        _classify(delta=1.0, prefilter_evaluator=guarded)
        != "prefilter_evaluator_disagreement"
    )


def test_classification_rule_c():
    assert (
        _classify(delta=0.0, diversity={"unique_macro_ratio": 0.2, "collision_ratio": 0.0})
        == "candidate_diversity_collapse"
    )
    assert (
        _classify(delta=0.0, diversity={"unique_macro_ratio": 0.8, "collision_ratio": 0.25})
        == "candidate_diversity_collapse"
    )


def test_classification_rule_d():
    assert (
        _classify(delta=0.0, budget_saturation=0.85)
        == "local_exhaustion_under_sampled_families"
    )


def test_classification_rule_e():
    assert _classify(delta=1.0, budget_saturation=0.1) == "inconclusive"


def test_terminology_mapping(tmp_path):
    data = _run_cli(tmp_path)
    keys = set(_walk_keys(data))
    assert "best_evaluator_cost" in keys
    assert "median_evaluator_cost" in keys
    assert "proxy_cost" not in keys
    prefilter_csv = (tmp_path / "m4a_prefilter_vs_evaluator.csv").read_text()
    assert "evaluator_cost" in prefilter_csv


def test_unsupported_diagnostics_absent(tmp_path):
    data = _run_cli(tmp_path)
    keys = set(_walk_keys(data))
    assert {"top_hotspots", "top_macros", "top_nets"}.isdisjoint(keys)


def test_no_near_local_optimum_label(tmp_path):
    data = _run_cli(tmp_path)
    string_values = [value for value in _walk_values(data) if isinstance(value, str)]
    assert "near_local_optimum" not in string_values


def test_cli_integration(tmp_path):
    data = _run_cli(tmp_path)
    assert set(data["benchmarks"]) == {"ibm01", "ibm02", "ibm03"}


def test_no_scorer_import():
    sys.modules.pop("submissions.solver.m4a_loss_attribution", None)
    importlib.import_module("submissions.solver.m4a_loss_attribution")
    assert "submissions.solver.scorer" not in sys.modules
    assert "submissions.solver.scoring" not in sys.modules


def test_output_files_exist(tmp_path):
    _run_cli(tmp_path)
    for name in [
        "m4a_loss_attribution_report.md",
        "m4a_loss_attribution.json",
        "m4a_family_effectiveness.csv",
        "m4a_prefilter_vs_evaluator.csv",
    ]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_json_structural_regression(tmp_path):
    data = _run_cli(tmp_path)
    assert {
        "profile",
        "official_epsilon",
        "inputs",
        "supported_diagnostics",
        "unsupported_diagnostics",
        "benchmarks",
        "aggregate_recommendation",
        "caveats",
    }.issubset(data)
    for benchmark in data["benchmarks"].values():
        assert {
            "costs",
            "score_band",
            "family_effectiveness",
            "skip_reasons",
            "budget",
            "prefilter_evaluator",
            "diversity",
            "classification",
            "classification_reasons",
            "m4b_recommendation",
            "caveats",
        }.issubset(benchmark)


def test_prefilter_guard():
    rows = [
        {
            "benchmark": "b1",
            "candidate_name": f"original_refinement_m{i}_scale2x",
            "family": "original_refinement",
            "scored": "True",
            "approx_delta": str(i),
            "proxy_cost": str(100 - i),
        }
        for i in range(19)
    ]
    diagnostic, _ = m4a.build_prefilter_evaluator(rows, "b1", scored_count=19)
    assert diagnostic["guard_fired"] is True
    assert diagnostic["spearman_rs"] is None

    low_coverage_rows = rows + [
        {
            "benchmark": "b1",
            "candidate_name": f"m3a_p{i}_{i}_{i + 1}_swap",
            "family": "m3a_pair_refinement",
            "scored": "True",
            "approx_delta": "",
            "proxy_cost": "1.0",
        }
        for i in range(31)
    ]
    diagnostic, _ = m4a.build_prefilter_evaluator(
        low_coverage_rows, "b1", scored_count=50
    )
    assert diagnostic["guard_fired"] is True
    assert diagnostic["spearman_rs"] is None


def test_output_contains_required_caveats(tmp_path):
    data = _run_cli(tmp_path)
    caveat_text = "\n".join(data["caveats"])
    report_text = (tmp_path / "m4a_loss_attribution_report.md").read_text()
    for caveat in m4a.REQUIRED_CAVEATS:
        assert caveat in caveat_text
        assert caveat in report_text
