"""
M3D-slice-3: Failure classification.

Read-only diagnostic utility that classifies why M3A/M3B late-stage
refinement produced little or no official proxy-cost improvement.
Does not change any solver behaviour.

Data-source contract
--------------------
**Candidate rows are authoritative.**  All counts, costs, and flags used by
classification logic come exclusively from ``candidate_rows``.

``family_summaries`` is used only to discover (benchmark, profile) groups
that have a family summary but no candidate rows.  No summary field drives
any classification metric; summary-only groups will be classified with all
late-stage counts at zero (typically ``late_stage_not_scored``).
"""

from typing import Any, Dict, List, Optional, Tuple

_LATE_STAGE_FAMILIES = frozenset({"m3a_pair_refinement", "m3b_cluster_refinement"})

_RECOMMENDED_NEXT_STEPS: Dict[str, str] = {
    "late_stage_not_scored": "inspect M3C admission/scoring budget",
    "late_stage_not_selectable": "inspect budget exhaustion/selectability guards",
    "late_stage_valid_but_worse": "design new structural move family",
    "late_stage_good_but_missed": "review selector/selectability bug",
    "ranking_mismatch": "redesign analytical prefilter/ranking",
    "invalidity_dominated": "design safer geometry generation",
    "near_local_optimum": "try larger structural search",
}


def classify_m3d_failure(
    candidate_rows: List[Dict[str, Any]],
    family_summaries: List[Dict[str, Any]],
    official_epsilon: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Classify why late-stage M3A/M3B refinement did not improve proxy cost.

    Read-only: does not mutate candidate_rows or family_summaries.

    Args:
        candidate_rows: Per-candidate rows from export_candidate_rows.
            **Authoritative source** for all classification metrics.
        family_summaries: Per-family summaries from summarize_candidate_families.
            Used only to include (benchmark, profile) groups that have a summary
            but no candidate rows.  No summary field drives classification logic.
        official_epsilon: Tolerance for near-tie comparisons.

    Returns:
        Deterministic list of classification dicts, one per (benchmark, profile),
        sorted by (benchmark, profile).
    """
    bp_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (row.get("benchmark", "") or "", row.get("profile", "") or "")
        bp_rows.setdefault(key, []).append(row)

    bp_summaries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for summary in family_summaries:
        key = (summary.get("benchmark", "") or "", summary.get("profile", "") or "")
        bp_summaries.setdefault(key, []).append(summary)

    all_bp_keys = sorted(set(bp_rows.keys()) | set(bp_summaries.keys()))

    results: List[Dict[str, Any]] = []
    for bp_key in all_bp_keys:
        benchmark, profile = bp_key
        rows = bp_rows.get(bp_key, [])
        summaries = bp_summaries.get(bp_key, [])

        result = _classify_bp(rows, summaries, official_epsilon)
        result["benchmark"] = benchmark
        result["profile"] = profile
        results.append(result)

    return results


def _classify_bp(
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    epsilon: float,
) -> Dict[str, Any]:
    # Infer final cost only when exactly one selected+scored row exists.
    # Zero or multiple such rows leave final_cost unavailable so comparisons
    # degrade safely rather than picking an arbitrary row.
    selected_scored = [
        r for r in rows
        if r.get("is_selected") and r.get("scored") and r.get("proxy_cost") is not None
    ]
    final_cost: Optional[float] = selected_scored[0]["proxy_cost"] if len(selected_scored) == 1 else None

    # Aggregate late-stage candidate rows.
    ls_rows = [r for r in rows if r.get("family") in _LATE_STAGE_FAMILIES]

    generated = len(ls_rows)
    valid = sum(1 for r in ls_rows if r.get("valid"))
    invalid = sum(1 for r in ls_rows if not r.get("valid"))
    admitted = sum(1 for r in ls_rows if r.get("admitted"))
    not_admitted = sum(1 for r in ls_rows if r.get("not_admitted"))

    scored_ls_rows = [
        r for r in ls_rows
        if r.get("scored") and r.get("proxy_cost") is not None
    ]
    scored = len(scored_ls_rows)
    selectable = sum(1 for r in ls_rows if r.get("scored_pool_selectable"))

    best_cost: Optional[float] = (
        min(r["proxy_cost"] for r in scored_ls_rows) if scored_ls_rows else None
    )
    best_delta: Optional[float] = (
        best_cost - final_cost
        if best_cost is not None and final_cost is not None
        else None
    )

    if final_cost is not None:
        num_beating = sum(
            1 for r in scored_ls_rows if r["proxy_cost"] < final_cost - epsilon
        )
        num_near_tie = sum(
            1 for r in scored_ls_rows if abs(r["proxy_cost"] - final_cost) <= epsilon
        )
    else:
        num_beating = 0
        num_near_tie = 0

    # A scored late-stage candidate beats final but was not selected.
    good_but_missed = (
        final_cost is not None
        and any(
            r["proxy_cost"] < final_cost - epsilon and not r.get("is_selected")
            for r in scored_ls_rows
        )
    )

    # Ranking mismatch: conservative check.
    # Requires approx_delta data on scored candidates and shows predicted
    # improvement (best approx_delta < -epsilon) while official cost did not
    # improve (best_delta >= -epsilon).
    ranking_mismatch = False
    if best_delta is not None and best_delta >= -epsilon:
        approx_deltas = [
            r["approx_delta"]
            for r in scored_ls_rows
            if r.get("approx_delta") is not None
        ]
        if approx_deltas and min(approx_deltas) < -epsilon:
            ranking_mismatch = True

    # Apply classification rules in precedence order.
    if good_but_missed:
        classification = "late_stage_good_but_missed"
        reason = (
            "A scored late-stage candidate beat the final selection cost "
            "by more than epsilon but was not selected."
        )
    elif generated > 0 and valid == 0:
        classification = "invalidity_dominated"
        reason = (
            f"All {generated} generated late-stage candidate(s) were invalid; "
            "no valid candidates reached scoring."
        )
    elif generated == 0 or scored == 0:
        classification = "late_stage_not_scored"
        if generated == 0:
            reason = "No M3A/M3B candidates were generated."
        else:
            reason = (
                f"{generated} late-stage candidate(s) were generated but none "
                "received an official score."
            )
    elif selectable == 0:
        classification = "late_stage_not_selectable"
        reason = (
            f"{scored} late-stage candidate(s) were scored but none were "
            "scored-pool-selectable; budget or selectability guards prevented "
            "inclusion in the final selection pool."
        )
    elif ranking_mismatch:
        classification = "ranking_mismatch"
        reason = (
            "Approximate deltas predict improvement for late-stage candidates "
            "but official costs did not improve; prefilter ranking order does "
            "not match official scoring order."
        )
    elif num_near_tie > 0:
        classification = "near_local_optimum"
        reason = (
            f"{num_near_tie} late-stage candidate(s) tie the baseline within "
            "epsilon; the solution appears to be near a local optimum."
        )
    else:
        classification = "late_stage_valid_but_worse"
        reason = (
            "Valid and scored late-stage candidates were available but none "
            "beat the baseline cost."
        )

    return {
        "classification": classification,
        "reason": reason,
        "late_stage_generated": generated,
        "late_stage_valid": valid,
        "late_stage_invalid": invalid,
        "late_stage_admitted": admitted,
        "late_stage_not_admitted": not_admitted,
        "late_stage_scored": scored,
        "late_stage_selectable": selectable,
        "late_stage_best_cost": best_cost,
        "late_stage_best_delta_vs_final": best_delta,
        "late_stage_num_beating_final": num_beating,
        "late_stage_num_near_tie": num_near_tie,
        "recommended_next_step": _RECOMMENDED_NEXT_STEPS[classification],
    }
