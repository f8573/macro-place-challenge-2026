"""
M3D-slice-1: Candidate metadata export.

Read-only utility that converts the candidate pool and scoring diagnostics
into deterministic row dicts suitable for CSV/report generation.  Does not
change scoring, selection, validation, or any other solver behaviour.
"""

from typing import Any, Dict, List, Optional

from submissions.solver.core.candidate_types import ScoredCandidate, ScoringDiagnostics

_REQUIRED_FIELDS = (
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
)


def export_candidate_rows(
    candidates: List[ScoredCandidate],
    selected: Optional[ScoredCandidate],
    diagnostics: ScoringDiagnostics,
    benchmark: str = "",
    profile: str = "",
) -> List[Dict[str, Any]]:
    """Return one row dict per candidate, in the order supplied.

    Read-only: does not mutate candidates, selected, or diagnostics.

    Args:
        candidates: Full candidate pool as returned by score_and_select (the
            ``ranked`` list).  May include invalid, duplicate, and unscored
            entries.
        selected: The winning candidate returned by score_and_select, or None.
        diagnostics: ScoringDiagnostics returned by score_and_select.
        benchmark: Benchmark name to embed in every row (optional).
        profile: Profile name to embed in every row (optional).

    Returns:
        List of row dicts with the fields documented in _REQUIRED_FIELDS.
    """
    # Derive family-level selectability exclusions from diagnostics (read-only).
    # Mirror the exact logic in candidate_scoring.score_and_select so that
    # `scored_pool_selectable` correctly identifies which candidates would have
    # been eligible for final selection from the scored pool.  A candidate can
    # still be chosen via the fallback path even when scored_pool_selectable is
    # False; that case is captured by `selected_via_fallback`.
    m3a_family_excluded = diagnostics.m3a_skipped_budget > 0
    m3b_family_excluded = diagnostics.m3b_skipped_budget > 0
    # original_legalized is treated as diagnostic-only when the raw original
    # placement is itself valid.
    diagnostic_only_name = "original_legalized" if diagnostics.raw_original_valid else None

    rows: List[Dict[str, Any]] = []
    for sc in candidates:
        skip_reason: str = sc.metadata.get("skip_reason", "")
        not_admitted: bool = skip_reason == "m3c_not_admitted"

        scored_pool_selectable: bool = (
            sc.valid
            and sc.was_scored
            and sc.proxy_cost is not None
            and sc.name != diagnostic_only_name
            and not (m3a_family_excluded and sc.family == "m3a_pair_refinement")
            and not (m3b_family_excluded and sc.family == "m3b_cluster_refinement")
        )

        is_selected: bool = sc is selected
        selected_via_fallback: bool = is_selected and not scored_pool_selectable

        rows.append(
            {
                "benchmark": benchmark,
                "profile": profile,
                "candidate_name": sc.name,
                "family": sc.family,
                "valid": sc.valid,
                "duplicate": sc.duplicate_of is not None,
                "admitted": not not_admitted,
                "not_admitted": not_admitted,
                "scored": sc.was_scored,
                "skip_reason": skip_reason,
                "proxy_cost": sc.proxy_cost,
                "approx_delta": sc.metadata.get("approx_hpwl_delta"),
                "is_selected": is_selected,
                "scored_pool_selectable": scored_pool_selectable,
                "selected_via_fallback": selected_via_fallback,
                "placement_hash": sc.metadata.get("placement_hash"),
                "source_stage": sc.metadata.get("pass_id"),
            }
        )

    return rows
