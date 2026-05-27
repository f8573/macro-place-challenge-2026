"""
M3D-slice-2: Family-level aggregation over exported candidate rows.

Read-only utility that aggregates per-candidate rows (from M3D-slice-1)
into per-family summary statistics.  Does not change any solver behaviour.
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple


def summarize_candidate_families(
    rows: List[Dict[str, Any]],
    final_cost: Optional[float] = None,
    official_epsilon: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Aggregate candidate-export rows into one summary dict per family.

    Args:
        rows: List of row dicts as produced by export_candidate_rows.
        final_cost: Official cost of the final selected candidate.  If None,
            inferred from the unique selected scored row when exactly one exists.
        official_epsilon: Tolerance for near-tie comparisons.

    Returns:
        Deterministic list of family summary dicts, sorted by
        (benchmark, profile, family).
    """
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("benchmark", "") or "",
            row.get("profile", "") or "",
            row.get("family", "") or "",
        )
        groups.setdefault(key, []).append(row)

    # Infer final_cost once per (benchmark, profile) across all families.
    # Only infer when exactly one selected+scored row exists for that pair.
    bp_inferred: Dict[Tuple[str, str], Optional[float]] = {}
    if final_cost is None:
        bp_all_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in rows:
            bp_key = (
                row.get("benchmark", "") or "",
                row.get("profile", "") or "",
            )
            bp_all_rows.setdefault(bp_key, []).append(row)
        for bp_key, bp_rows in bp_all_rows.items():
            selected_scored = [
                r for r in bp_rows
                if r.get("is_selected") and r.get("scored") and r.get("proxy_cost") is not None
            ]
            bp_inferred[bp_key] = (
                selected_scored[0]["proxy_cost"] if len(selected_scored) == 1 else None
            )

    summaries: List[Dict[str, Any]] = []
    for key in sorted(groups):
        benchmark, profile, family = key
        group = groups[key]

        scored_rows = [
            r for r in group
            if r.get("scored") and r.get("proxy_cost") is not None
        ]
        scored_costs = [r["proxy_cost"] for r in scored_rows]
        legalized_count = sum(
            1 for r in group if r.get("legalization_status") == "legalized"
        )
        duplicate_after_legalization_count = sum(
            1
            for r in group
            if r.get("legalization_failure_reason") == "duplicate_after_legalization"
        )
        legalization_failed_count = sum(
            1
            for r in group
            if r.get("legalization_status") == "failed"
            or bool(r.get("legalization_failure_reason"))
        )
        adjusted_denominator = len(group) - duplicate_after_legalization_count

        # Resolve effective final cost: explicit arg > per-(benchmark,profile) inference.
        if final_cost is not None:
            effective_final_cost: Optional[float] = final_cost
        else:
            effective_final_cost = bp_inferred.get((benchmark, profile))

        # Best/worst proxy_cost with deterministic tie-break by candidate_name.
        best_cost: Optional[float] = None
        best_name: Optional[str] = None
        worst_cost: Optional[float] = None
        worst_name: Optional[str] = None
        if scored_rows:
            best_row = min(
                scored_rows,
                key=lambda r: (r["proxy_cost"], r.get("candidate_name") or ""),
            )
            worst_row = max(
                scored_rows,
                key=lambda r: (r["proxy_cost"], r.get("candidate_name") or ""),
            )
            best_cost = best_row["proxy_cost"]
            best_name = best_row.get("candidate_name")
            worst_cost = worst_row["proxy_cost"]
            worst_name = worst_row.get("candidate_name")

        median_cost: Optional[float] = (
            statistics.median(scored_costs) if scored_costs else None
        )

        best_delta: Optional[float] = (
            best_cost - effective_final_cost
            if best_cost is not None and effective_final_cost is not None
            else None
        )
        median_delta: Optional[float] = (
            median_cost - effective_final_cost
            if median_cost is not None and effective_final_cost is not None
            else None
        )

        if effective_final_cost is not None:
            num_beating = sum(
                1 for c in scored_costs if c < effective_final_cost - official_epsilon
            )
            num_near_tie = sum(
                1 for c in scored_costs
                if abs(c - effective_final_cost) <= official_epsilon
            )
        else:
            num_beating = 0
            num_near_tie = 0

        summaries.append(
            {
                "benchmark": benchmark,
                "profile": profile,
                "family": family,
                "generated_count": len(group),
                "valid_count": sum(1 for r in group if r.get("valid")),
                "invalid_count": sum(1 for r in group if not r.get("valid")),
                "duplicate_count": sum(1 for r in group if r.get("duplicate")),
                "admitted_count": sum(1 for r in group if r.get("admitted")),
                "not_admitted_count": sum(1 for r in group if r.get("not_admitted")),
                "scored_count": len(scored_rows),
                "skipped_budget_count": sum(
                    1
                    for r in group
                    if r.get("skip_reason")
                    in {"budget_exceeded", "m4b_budget_exhausted", "m4c_budget_exhausted"}
                ),
                "scored_pool_selectable_count": sum(
                    1 for r in group if r.get("scored_pool_selectable")
                ),
                "selected_count": sum(1 for r in group if r.get("is_selected")),
                "selected_via_fallback_count": sum(
                    1 for r in group if r.get("selected_via_fallback")
                ),
                "best_official_cost": best_cost,
                "best_official_delta_vs_final": best_delta,
                "median_official_cost": median_cost,
                "median_official_delta_vs_final": median_delta,
                "num_beating_final": num_beating,
                "num_near_tie": num_near_tie,
                "best_candidate_name": best_name,
                "worst_official_cost": worst_cost,
                "worst_candidate_name": worst_name,
                "legalized_count": legalized_count,
                "legalization_failed_count": legalization_failed_count,
                "duplicate_after_legalization_count": duplicate_after_legalization_count,
                "raw_legalized_rate": (
                    legalized_count / len(group) if len(group) > 0 else 0.0
                ),
                "adjusted_legalized_rate": (
                    legalized_count / adjusted_denominator
                    if adjusted_denominator > 0
                    else 0.0
                ),
            }
        )

    return summaries
