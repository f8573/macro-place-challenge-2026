"""M4A reduced artifact-only score-loss attribution diagnostics.

This module intentionally reads only persisted M3D CSV/JSON artifacts. It does
not import or invoke any scorer, solver, runner, benchmark, or geometry code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_DIAGNOSTICS = [
    "score_banding",
    "family_effectiveness",
    "skip_reason_aggregation",
    "budget_use",
    "approx_prefilter_vs_evaluator",
    "candidate_name_set_diversity",
]

UNSUPPORTED_DIAGNOSTICS = [
    "region_attribution",
    "density_hotspots",
    "top_macro_contributor",
    "top_net_contributor",
    "topology_lock_proof",
    "legalization_displacement",
]

REQUIRED_CAVEATS = [
    "Geometry not persisted; region, density, per-macro displacement, and per-net HPWL attribution are unsupported by reduced M4A.",
    "proxy_cost in M3D artifacts is treated as evaluator cost in M4A, not as a separate proxy.",
    "M4A does not inherit M3D's near_local_optimum label; M4A classifications are derived independently from rules A-E.",
    "Cache decomposition is omitted if cache_hits/cache_misses are zero or inactive.",
    "Candidate name-set diversity is weak evidence and is not a substitute for geometry diversity.",
    "approx_delta is absent for m3a/m3b family candidates in current M3D artifacts; prefilter-evaluator disagreement analysis is suppressed when approx_coverage < 0.50.",
    "best_official_delta_vs_final is relative to the selected final cost, not to original_cost.",
]

M4B_RECOMMENDATIONS = {
    "legality_bottleneck": "legalization_aware_regional_repair",
    "prefilter_evaluator_disagreement": "correlation_aware_ranking_or_prefilter_repair",
    "candidate_diversity_collapse": "beam_search_elite_pool_or_multistart",
    "local_exhaustion_under_sampled_families": "regional_destroy_and_repair",
    "inconclusive": "persist_more_instrumentation_before_optimizer_changes",
}

PREFILTER_SKIP_REASONS = {
    "prefilter",
    "prefiltered",
    "prefilter_skip",
    "approx_prefilter",
    "approx_delta",
    "approx_delta_hpwl",
}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int:
    parsed = parse_float(value)
    if parsed is None:
        return 0
    return int(parsed)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_band(original_cost: float, selected_cost: float, eps: float) -> str:
    delta = original_cost - selected_cost
    if delta > 10 * eps:
        return "meaningful_win"
    if 0 < delta <= 10 * eps:
        return "epsilon_win"
    if abs(delta) <= eps:
        return "flat"
    if delta < -eps:
        return "regression"


def ranks(values: list[float]) -> list[float]:
    """Return average ranks with 1 as best/lowest."""
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for original_index, _ in indexed[i:j]:
            output[original_index] = avg_rank
        i = j
    return output


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in rx))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ry))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runner_results(path: Path, caveats: list[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        caveats.append(f"Runner JSON missing; budget diagnostics are incomplete: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        caveats.append(f"Runner JSON could not be parsed; budget diagnostics are incomplete: {exc}")
        return {}
    results = data.get("results", [])
    if not isinstance(results, list):
        caveats.append("Runner JSON has no results list; budget diagnostics are incomplete.")
        return {}
    return {
        str(row.get("benchmark")): row
        for row in results
        if isinstance(row, dict) and row.get("benchmark")
    }


def filter_rows(
    rows: Iterable[dict[str, str]], profile: str, benchmarks: set[str]
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("profile") == profile and row.get("benchmark") in benchmarks
    ]


def build_family_effectiveness(
    family_rows: list[dict[str, str]], benchmark: str, official_scored_count: int
) -> list[dict[str, Any]]:
    rows = [row for row in family_rows if row.get("benchmark") == benchmark]
    total_scored = sum(parse_int(row.get("scored_count")) for row in rows)
    denominator = total_scored if total_scored > 0 else official_scored_count
    output = []
    for row in rows:
        generated = parse_int(row.get("generated_count"))
        valid = parse_int(row.get("valid_count"))
        scored = parse_int(row.get("scored_count"))
        output.append(
            {
                "family": row.get("family", ""),
                "generated_count": generated,
                "valid_count": valid,
                "valid_rate": ratio(valid, generated),
                "scored_count": scored,
                "score_rate": ratio(scored, valid),
                "selected_count": parse_int(row.get("selected_count")),
                "best_evaluator_cost": parse_float(row.get("best_official_cost")),
                "best_evaluator_delta": parse_float(row.get("best_official_delta_vs_final")),
                "median_evaluator_cost": parse_float(row.get("median_official_cost")),
                "median_evaluator_delta": parse_float(row.get("median_official_delta_vs_final")),
                "budget_share": ratio(scored, denominator),
            }
        )
    output.sort(key=lambda item: item["family"])
    return output


def weighted_valid_rates(family_effectiveness: list[dict[str, Any]]) -> tuple[float, float]:
    local_valid = 0
    local_generated = 0
    baseline_valid = 0
    baseline_generated = 0
    for row in family_effectiveness:
        family = row["family"]
        generated = row["generated_count"]
        valid = row["valid_count"]
        if family.startswith(("m3a_", "m3b_")):
            local_generated += generated
            local_valid += valid
        if family.startswith("original"):
            baseline_generated += generated
            baseline_valid += valid
    return ratio(local_valid, local_generated), ratio(baseline_valid, baseline_generated)


def aggregate_skip_reasons(
    candidate_rows: list[dict[str, str]], benchmark: str, caveats: list[str]
) -> dict[str, Any]:
    rows = [row for row in candidate_rows if row.get("benchmark") == benchmark]
    per_family: dict[str, dict[str, Any]] = {}
    totals = {
        "invalid_count": 0,
        "duplicate_count": 0,
        "budget_skip_count": 0,
        "prefilter_skip_count": 0,
        "scored_count": 0,
    }
    distribution: Counter[str] = Counter()

    for row in rows:
        family = row.get("family", "")
        bucket = per_family.setdefault(
            family,
            {
                "family": family,
                "invalid_count": 0,
                "duplicate_count": 0,
                "budget_skip_count": 0,
                "prefilter_skip_count": 0,
                "scored_count": 0,
                "skip_reason_distribution": {},
            },
        )
        valid = parse_bool(row.get("valid"))
        duplicate = parse_bool(row.get("duplicate"))
        admitted = parse_bool(row.get("admitted"))
        not_admitted = parse_bool(row.get("not_admitted"))
        scored = parse_bool(row.get("scored"))
        skip_reason = (row.get("skip_reason") or "").strip()
        skip_reason_lc = skip_reason.lower()

        if skip_reason:
            distribution[skip_reason] += 1
            bucket["skip_reason_distribution"][skip_reason] = (
                bucket["skip_reason_distribution"].get(skip_reason, 0) + 1
            )
        if duplicate:
            bucket["duplicate_count"] += 1
            totals["duplicate_count"] += 1
        if not valid and not duplicate:
            bucket["invalid_count"] += 1
            totals["invalid_count"] += 1
        if not scored and not_admitted:
            bucket["budget_skip_count"] += 1
            totals["budget_skip_count"] += 1
        if (
            not scored
            and admitted
            and any(token in skip_reason_lc for token in PREFILTER_SKIP_REASONS)
        ):
            bucket["prefilter_skip_count"] += 1
            totals["prefilter_skip_count"] += 1
        if scored:
            bucket["scored_count"] += 1
            totals["scored_count"] += 1

    meaningful_reasons = {key for key in distribution if key.lower() != "scored"}
    if not meaningful_reasons:
        caveat = f"{benchmark}: skip reason granularity is unavailable beyond scored/blank values."
        if caveat not in caveats:
            caveats.append(caveat)

    return {
        **totals,
        "skip_reason_distribution": dict(sorted(distribution.items())),
        "per_family": sorted(per_family.values(), key=lambda item: item["family"]),
    }


def build_budget(
    runner_result: dict[str, Any] | None,
    family_effectiveness: list[dict[str, Any]],
) -> dict[str, Any]:
    runner_result = runner_result or {}
    official_scored_count = parse_int(runner_result.get("official_scored_count"))
    max_official_scores = parse_int(runner_result.get("max_official_scores"))
    denominator = official_scored_count
    per_family = {
        row["family"]: ratio(row["scored_count"], denominator)
        for row in family_effectiveness
    }
    return {
        "official_scored_count": official_scored_count,
        "max_official_scores": max_official_scores,
        "budget_saturation": ratio(official_scored_count, max_official_scores),
        "duplicate_skipped_count": parse_int(runner_result.get("duplicate_skipped_count")),
        "prefiltered_count": parse_int(runner_result.get("prefiltered_count")),
        "per_family_scored_share": per_family,
    }


def _rank_maps(usable: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    approx_values = [row["approx_delta"] for row in usable]
    evaluator_values = [row["evaluator_cost"] for row in usable]
    approx_ranks = ranks(approx_values)
    evaluator_ranks = ranks(evaluator_values)
    approx = {row["candidate_name"]: rank for row, rank in zip(usable, approx_ranks)}
    evaluator = {row["candidate_name"]: rank for row, rank in zip(usable, evaluator_ranks)}
    return approx, evaluator


def build_prefilter_evaluator(
    candidate_rows: list[dict[str, str]], benchmark: str, scored_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows = [
        row
        for row in candidate_rows
        if row.get("benchmark") == benchmark and parse_bool(row.get("scored"))
    ]
    total_scored = scored_count if scored_count > 0 else len(scored_rows)
    usable = []
    for row in scored_rows:
        approx_delta = parse_float(row.get("approx_delta"))
        evaluator_cost = parse_float(row.get("proxy_cost"))
        if approx_delta is None or evaluator_cost is None:
            continue
        usable.append(
            {
                "benchmark": benchmark,
                "candidate_name": row.get("candidate_name", ""),
                "family": row.get("family", ""),
                "approx_delta": approx_delta,
                "evaluator_cost": evaluator_cost,
            }
        )

    approx_coverage = ratio(len(usable), total_scored)
    guard_reasons = []
    if len(usable) < 20:
        guard_reasons.append("usable_count < 20")
    if approx_coverage < 0.50:
        guard_reasons.append("approx_coverage < 0.50")
    guard_fired = bool(guard_reasons)

    approx_rank_by_name: dict[str, float] = {}
    evaluator_rank_by_name: dict[str, float] = {}
    if usable:
        approx_rank_by_name, evaluator_rank_by_name = _rank_maps(usable)

    csv_rows = []
    for row in scored_rows:
        name = row.get("candidate_name", "")
        csv_rows.append(
            {
                "benchmark": benchmark,
                "candidate_name": name,
                "family": row.get("family", ""),
                "approx_delta": parse_float(row.get("approx_delta")),
                "evaluator_cost": parse_float(row.get("proxy_cost")),
                "approx_rank": approx_rank_by_name.get(name),
                "evaluator_rank": evaluator_rank_by_name.get(name),
            }
        )

    spearman_rs = None
    top5_inversions = None
    top20_inversions = None
    if not guard_fired:
        approx_values = [row["approx_delta"] for row in usable]
        evaluator_values = [row["evaluator_cost"] for row in usable]
        spearman_rs = spearman(approx_values, evaluator_values)
        top5_inversions = count_topk_inversions(usable, approx_rank_by_name, 5)
        top20_inversions = count_topk_inversions(usable, approx_rank_by_name, 20)

    return (
        {
            "usable_count": len(usable),
            "approx_coverage": approx_coverage,
            "guard_fired": guard_fired,
            "guard_reason": "; ".join(guard_reasons) if guard_reasons else None,
            "spearman_rs": spearman_rs,
            "top5_inversions": top5_inversions,
            "top20_inversions": top20_inversions,
        },
        csv_rows,
    )


def count_topk_inversions(
    usable: list[dict[str, Any]], approx_rank_by_name: dict[str, float], k: int
) -> int:
    top_by_evaluator = sorted(
        usable, key=lambda row: (row["evaluator_cost"], row["candidate_name"])
    )[:k]
    bottom_half_threshold = len(usable) / 2.0
    return sum(
        1
        for row in top_by_evaluator
        if approx_rank_by_name.get(row["candidate_name"], 0.0) > bottom_half_threshold
    )


def parse_macro_ids(candidate_name: str) -> list[int]:
    parts = candidate_name.split("_")
    if len(parts) >= 4 and parts[0] == "m3a" and parts[1].startswith("p"):
        ids = []
        for token in parts[2:4]:
            if token.isdigit():
                ids.append(int(token))
        return ids
    if len(parts) >= 5 and parts[0] == "m3b" and parts[1].startswith("c"):
        ids = []
        for token in parts[3:]:
            if not token.isdigit():
                break
            ids.append(int(token))
        return ids
    return [int(match) for match in re.findall(r"(?:^|_)m(\d+)(?=_|$)", candidate_name)]


def build_diversity(candidate_rows: list[dict[str, str]], benchmark: str, scored_count: int) -> dict[str, Any]:
    rows = [
        row
        for row in candidate_rows
        if row.get("benchmark") == benchmark and parse_bool(row.get("scored"))
    ]
    total_scored = scored_count if scored_count > 0 else len(rows)
    macro_counts: Counter[int] = Counter()
    for row in rows:
        macro_counts.update(parse_macro_ids(row.get("candidate_name", "")))
    non_null_hashes = [
        row.get("placement_hash", "").strip()
        for row in rows
        if row.get("placement_hash", "").strip()
    ]
    if non_null_hashes:
        placement_hash_collisions: int | None = len(non_null_hashes) - len(set(non_null_hashes))
        collision_ratio: float | None = ratio(placement_hash_collisions, total_scored)
    else:
        placement_hash_collisions = None
        collision_ratio = None
    return {
        "unique_macros_in_scored": len(macro_counts),
        "unique_macro_ratio": ratio(len(macro_counts), total_scored),
        "most_touched_macros": [
            {"macro": f"m{macro_id}", "count": count}
            for macro_id, count in macro_counts.most_common(5)
        ],
        "placement_hash_collisions": placement_hash_collisions,
        "collision_ratio": collision_ratio,
        "diversity_note": "weak evidence - name-set only, not geometry",
    }


def classify_benchmark(
    *,
    delta: float,
    eps: float,
    valid_rate_local: float,
    valid_rate_baseline: float,
    budget_saturation: float,
    prefilter_evaluator: dict[str, Any],
    diversity: dict[str, Any],
) -> tuple[str, list[str]]:
    threshold = 10 * eps
    reasons: list[str] = []
    if valid_rate_local < 0.20 and valid_rate_baseline > 0.80 and delta <= threshold:
        reasons = [
            f"valid_rate_local={valid_rate_local:.6g} < 0.20",
            f"valid_rate_baseline={valid_rate_baseline:.6g} > 0.80",
            f"delta={delta:.6g} <= 10*epsilon={threshold:.6g}",
        ]
        return "legality_bottleneck", reasons

    usable_count = prefilter_evaluator.get("usable_count", 0)
    approx_coverage = prefilter_evaluator.get("approx_coverage", 0.0)
    spearman_rs = prefilter_evaluator.get("spearman_rs")
    top5_inversions = prefilter_evaluator.get("top5_inversions")
    if (
        usable_count >= 20
        and approx_coverage >= 0.50
        and spearman_rs is not None
        and spearman_rs < 0.30
        and top5_inversions is not None
        and top5_inversions >= 3
    ):
        reasons = [
            f"usable_count={usable_count} >= 20",
            f"approx_coverage={approx_coverage:.6g} >= 0.50",
            f"spearman_rs={spearman_rs:.6g} < 0.30",
            f"top5_inversions={top5_inversions} >= 3",
        ]
        return "prefilter_evaluator_disagreement", reasons

    unique_macro_ratio = diversity.get("unique_macro_ratio", 0.0)
    collision_ratio = diversity.get("collision_ratio")
    unique_collapse = unique_macro_ratio < 0.40
    hash_collapse = collision_ratio is not None and collision_ratio > 0.20
    if delta <= threshold and (unique_collapse or hash_collapse):
        reasons = [f"delta={delta:.6g} <= 10*epsilon={threshold:.6g}"]
        if unique_collapse:
            reasons.append(f"unique_macro_ratio={unique_macro_ratio:.6g} < 0.40")
        if hash_collapse:
            reasons.append(f"collision_ratio={collision_ratio:.6g} > 0.20")
        return "candidate_diversity_collapse", reasons

    if delta <= threshold and budget_saturation >= 0.80:
        reasons = [
            f"delta={delta:.6g} <= 10*epsilon={threshold:.6g}",
            f"budget_saturation={budget_saturation:.6g} >= 0.80",
            "Rules A/B/C did not fire",
        ]
        return "local_exhaustion_under_sampled_families", reasons

    return "inconclusive", ["No priority rule A-D matched."]


def aggregate_recommendation(benchmarks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    classifications = {
        name: data["classification"] for name, data in sorted(benchmarks.items())
    }
    counts = Counter(classifications.values())
    if len(counts) == 1:
        classification = next(iter(counts))
        return {
            "classifications": classifications,
            "outcome": "all_benchmarks_agree",
            "m4b": M4B_RECOMMENDATIONS[classification],
            "note": f"All benchmarks classified as {classification}.",
        }

    majority, majority_count = counts.most_common(1)[0]
    if majority_count >= 2:
        dissenters = [
            bench for bench, classification in classifications.items() if classification != majority
        ]
        outcome = (
            "majority_with_inconclusive"
            if any(classifications[bench] == "inconclusive" for bench in dissenters)
            else "majority_with_scoped_dissenter"
        )
        note = f"Majority classification is {majority}; scoped dissenters: {', '.join(dissenters)}."
        return {
            "classifications": classifications,
            "outcome": outcome,
            "m4b": M4B_RECOMMENDATIONS[majority],
            "note": note,
        }

    return {
        "classifications": classifications,
        "outcome": "three_way_split",
        "m4b": "persist_more_instrumentation_before_optimizer_changes",
        "note": "Benchmarks split across classifications; choose instrumentation first.",
    }


def analyze(
    *,
    profile: str,
    benchmarks: list[str],
    official_epsilon: float,
    input_dir: Path,
    runner_json: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    caveats = list(REQUIRED_CAVEATS)
    benchmark_summary_path = input_dir / "m3d_benchmark_summary.csv"
    family_summary_path = input_dir / "m3d_family_summary.csv"
    candidate_effectiveness_path = input_dir / "m3d_candidate_effectiveness.csv"

    benchmark_set = set(benchmarks)
    benchmark_rows = filter_rows(read_csv_rows(benchmark_summary_path), profile, benchmark_set)
    family_rows = filter_rows(read_csv_rows(family_summary_path), profile, benchmark_set)
    candidate_rows = filter_rows(read_csv_rows(candidate_effectiveness_path), profile, benchmark_set)
    runner_results = load_runner_results(runner_json, caveats)

    benchmark_by_name = {row["benchmark"]: row for row in benchmark_rows}
    output_benchmarks: dict[str, dict[str, Any]] = {}
    family_csv_rows: list[dict[str, Any]] = []
    prefilter_csv_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        if benchmark not in benchmark_by_name:
            raise ValueError(f"Benchmark {benchmark!r} not found for profile {profile!r}")
        row = benchmark_by_name[benchmark]
        runner_result = runner_results.get(benchmark, {})
        original_cost = parse_float(row.get("original_cost"))
        selected_cost = parse_float(row.get("selected_cost"))
        if original_cost is None:
            original_cost = parse_float(runner_result.get("raw_original_cost"))
        if selected_cost is None:
            selected_cost = parse_float(runner_result.get("proxy_cost"))
        if original_cost is None or selected_cost is None:
            raise ValueError(f"Benchmark {benchmark!r} is missing original or selected cost")

        delta = original_cost - selected_cost
        costs = {
            "original_cost": original_cost,
            "selected_cost": selected_cost,
            "delta": delta,
            "relative_delta": ratio(delta, original_cost),
            "relative_delta_pct": 100.0 * ratio(delta, original_cost),
        }
        if original_cost < selected_cost:
            caveats.append(
                f"{benchmark}: original_cost < selected_cost; lower evaluator cost is assumed better."
            )

        family_effectiveness = build_family_effectiveness(
            family_rows, benchmark, parse_int(runner_result.get("official_scored_count"))
        )
        valid_rate_local, valid_rate_baseline = weighted_valid_rates(family_effectiveness)
        for family_row in family_effectiveness:
            family_csv_rows.append({"benchmark": benchmark, **family_row})

        skip_reasons = aggregate_skip_reasons(candidate_rows, benchmark, caveats)
        budget = build_budget(runner_result, family_effectiveness)
        prefilter_evaluator, rows_for_prefilter_csv = build_prefilter_evaluator(
            candidate_rows, benchmark, budget["official_scored_count"]
        )
        prefilter_csv_rows.extend(rows_for_prefilter_csv)
        diversity = build_diversity(candidate_rows, benchmark, budget["official_scored_count"])
        classification, classification_reasons = classify_benchmark(
            delta=delta,
            eps=official_epsilon,
            valid_rate_local=valid_rate_local,
            valid_rate_baseline=valid_rate_baseline,
            budget_saturation=budget["budget_saturation"],
            prefilter_evaluator=prefilter_evaluator,
            diversity=diversity,
        )
        spearman_rs = prefilter_evaluator.get("spearman_rs")
        top5_inversions = prefilter_evaluator.get("top5_inversions")
        if (
            classification == "legality_bottleneck"
            and not prefilter_evaluator.get("guard_fired")
            and spearman_rs is not None
            and spearman_rs < 0.30
            and top5_inversions is not None
            and top5_inversions >= 3
        ):
            caveats.append(
                f"{benchmark}: secondary signal - Rule B conditions also satisfied "
                f"(spearman_rs={spearman_rs:.6g}, top5_inversions={top5_inversions}); "
                "prefilter disagreement is suppressed by Rule A priority."
            )
        output_benchmarks[benchmark] = {
            "costs": costs,
            "score_band": score_band(original_cost, selected_cost, official_epsilon),
            "family_effectiveness": family_effectiveness,
            "skip_reasons": skip_reasons,
            "budget": budget,
            "prefilter_evaluator": prefilter_evaluator,
            "diversity": diversity,
            "classification": classification,
            "classification_reasons": classification_reasons,
            "m4b_recommendation": M4B_RECOMMENDATIONS[classification],
            "caveats": [
                caveat
                for caveat in caveats
                if caveat.startswith(f"{benchmark}:")
            ],
        }

    result = {
        "profile": profile,
        "official_epsilon": official_epsilon,
        "inputs": {
            "benchmark_summary": str(benchmark_summary_path),
            "family_summary": str(family_summary_path),
            "candidate_effectiveness": str(candidate_effectiveness_path),
            "runner_json": str(runner_json),
        },
        "supported_diagnostics": SUPPORTED_DIAGNOSTICS,
        "unsupported_diagnostics": UNSUPPORTED_DIAGNOSTICS,
        "benchmarks": output_benchmarks,
        "aggregate_recommendation": aggregate_recommendation(output_benchmarks),
        "caveats": sorted(dict.fromkeys(caveats)),
    }
    return result, family_csv_rows, prefilter_csv_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_outputs(
    *,
    output_dir: Path,
    result: dict[str, Any],
    family_rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m4a_loss_attribution.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m4a_loss_attribution_report.md").write_text(
        render_markdown_report(result), encoding="utf-8"
    )
    write_csv(
        output_dir / "m4a_family_effectiveness.csv",
        family_rows,
        [
            "benchmark",
            "family",
            "generated_count",
            "valid_count",
            "valid_rate",
            "scored_count",
            "score_rate",
            "selected_count",
            "best_evaluator_cost",
            "best_evaluator_delta",
            "median_evaluator_cost",
            "median_evaluator_delta",
            "budget_share",
        ],
    )
    write_csv(
        output_dir / "m4a_prefilter_vs_evaluator.csv",
        prefilter_rows,
        [
            "benchmark",
            "candidate_name",
            "family",
            "approx_delta",
            "evaluator_cost",
            "approx_rank",
            "evaluator_rank",
        ],
    )


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# M4A Loss Attribution Report",
        "",
        f"Profile: `{result['profile']}`",
        f"Official epsilon: `{result['official_epsilon']}`",
        "",
        "## Inputs",
        "",
    ]
    for name, path in result["inputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Supported Diagnostics",
            "",
        ]
    )
    for item in result["supported_diagnostics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Unsupported Diagnostics", ""])
    for item in result["unsupported_diagnostics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Benchmark Classifications", ""])
    lines.append(
        "| Benchmark | Score band | Delta | Classification | M4B recommendation |"
    )
    lines.append("|---|---|---:|---|---|")
    for benchmark, data in result["benchmarks"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    benchmark,
                    data["score_band"],
                    fmt(data["costs"]["delta"]),
                    data["classification"],
                    data["m4b_recommendation"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Per-Benchmark Diagnostics", ""])
    for benchmark, data in result["benchmarks"].items():
        costs = data["costs"]
        budget = data["budget"]
        prefilter = data["prefilter_evaluator"]
        diversity = data["diversity"]
        lines.extend(
            [
                f"### {benchmark}",
                "",
                f"- original_cost: `{fmt(costs['original_cost'])}`",
                f"- selected_cost: `{fmt(costs['selected_cost'])}`",
                f"- delta: `{fmt(costs['delta'])}`",
                f"- relative_delta: `{fmt(costs['relative_delta'])}`",
                f"- score_band: `{data['score_band']}`",
                f"- official_scored_count: `{budget['official_scored_count']}`",
                f"- max_official_scores: `{budget['max_official_scores']}`",
                f"- budget_saturation: `{fmt(budget['budget_saturation'])}`",
                f"- duplicate_skipped_count: `{budget['duplicate_skipped_count']}`",
                f"- prefiltered_count: `{budget['prefiltered_count']}`",
                f"- approx-prefilter vs evaluator usable_count: `{prefilter['usable_count']}`",
                f"- approx-prefilter vs evaluator approx_coverage: `{fmt(prefilter['approx_coverage'])}`",
                f"- approx-prefilter vs evaluator spearman_rs: `{fmt(prefilter['spearman_rs'])}`",
                f"- approx-prefilter vs evaluator top5_inversions: `{fmt(prefilter['top5_inversions'])}`",
                f"- unique_macros_in_scored: `{diversity['unique_macros_in_scored']}`",
                f"- unique_macro_ratio: `{fmt(diversity['unique_macro_ratio'])}`",
                f"- placement_hash_collisions: `{fmt(diversity['placement_hash_collisions'])}`",
                f"- collision_ratio: `{fmt(diversity['collision_ratio'])}`",
                f"- classification: `{data['classification']}`",
                f"- M4B recommendation: `{data['m4b_recommendation']}`",
                "",
                "Classification reasons:",
            ]
        )
        for reason in data["classification_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    aggregate = result["aggregate_recommendation"]
    lines.extend(
        [
            "## Aggregate Recommendation",
            "",
            f"- outcome: `{aggregate['outcome']}`",
            f"- M4B: `{aggregate['m4b']}`",
            f"- note: {aggregate['note']}",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in result["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--official-epsilon", type=float, default=1e-5)
    parser.add_argument("--input-dir", type=Path, default=Path("analysis/m3d"))
    parser.add_argument(
        "--runner-json",
        type=Path,
        default=Path("submissions/solver/artifacts/run_m3c-default.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("submissions/solver/reports")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result, family_rows, prefilter_rows = analyze(
        profile=args.profile,
        benchmarks=args.benchmarks,
        official_epsilon=args.official_epsilon,
        input_dir=args.input_dir,
        runner_json=args.runner_json,
    )
    write_outputs(
        output_dir=args.output_dir,
        result=result,
        family_rows=family_rows,
        prefilter_rows=prefilter_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
