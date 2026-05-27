"""Pure M4D family-normalization helpers.

This module computes deterministic per-family rank-percentile telemetry for
approx-bearing pre-M3 candidate metadata.  It intentionally does not import
scorer, evaluator, cache, legalizer, or loader code.
"""

from __future__ import annotations

import math
from typing import Any

_EXCLUDED_FAMILIES = {
    "m3a_pair_refinement",
    "m3b_cluster_refinement",
    "m4b_region_repair",
}
_EPS = 1e-9


def _candidate_name(candidate: dict[str, Any], fallback: int) -> str:
    return str(
        candidate.get("candidate_name", candidate.get("name", f"candidate_{fallback}"))
    )


def _generation_order(candidate: dict[str, Any], fallback: int) -> int:
    value = candidate.get("fifo_index", candidate.get("generation_rank", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _family(candidate: dict[str, Any]) -> str:
    return str(candidate.get("family", ""))


def _approx(candidate: dict[str, Any]) -> float | None:
    value = candidate.get(
        "approx_delta",
        candidate.get(
            "approx_hpwl_delta",
            candidate.get("post_legalization_approx_delta"),
        ),
    )
    if not isinstance(value, (int, float)):
        return None
    approx = float(value)
    if not math.isfinite(approx):
        return None
    return approx


def _participates(candidate: dict[str, Any]) -> bool:
    return _family(candidate) not in _EXCLUDED_FAMILIES and _approx(candidate) is not None


def compute_rank_scores(candidates: list[dict[str, Any]]) -> list[float | None]:
    """Return per-family M4D rank-percentile scores aligned with ``candidates``."""
    scores: list[float | None] = [None] * len(candidates)
    by_family: dict[str, list[tuple[float, int, str, int]]] = {}
    for idx, candidate in enumerate(candidates):
        approx = _approx(candidate)
        if _family(candidate) in _EXCLUDED_FAMILIES or approx is None:
            continue
        by_family.setdefault(_family(candidate), []).append(
            (approx, _generation_order(candidate, idx), _candidate_name(candidate, idx), idx)
        )

    for family_rows in by_family.values():
        family_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        denominator = max(len(family_rows) - 1, 1)
        for rank, (_approx_value, _gen_order, _name, idx) in enumerate(family_rows):
            scores[idx] = rank / denominator
    return scores


def compute_family_normalized_approx(
    candidates: list[dict[str, Any]],
) -> list[float | None]:
    """Return per-family min-max normalized approx deltas aligned with ``candidates``."""
    normalized: list[float | None] = [None] * len(candidates)
    by_family: dict[str, list[tuple[float, int]]] = {}
    for idx, candidate in enumerate(candidates):
        approx = _approx(candidate)
        if _family(candidate) in _EXCLUDED_FAMILIES or approx is None:
            continue
        by_family.setdefault(_family(candidate), []).append((approx, idx))

    for family_rows in by_family.values():
        values = [approx for approx, _idx in family_rows]
        family_min = min(values)
        family_range = max(values) - family_min
        for approx, idx in family_rows:
            normalized[idx] = (approx - family_min) / (family_range + _EPS)
    return normalized


def compute_cross_family_rank(
    candidates: list[dict[str, Any]],
    rank_scores: list[float | None] | None = None,
) -> list[int | None]:
    """Return 1-based cross-family ranks for participating candidates."""
    if rank_scores is None:
        rank_scores = compute_rank_scores(candidates)
    ranks: list[int | None] = [None] * len(candidates)
    ranked = []
    for idx, score in enumerate(rank_scores):
        if score is None:
            continue
        ranked.append(
            (
                float(score),
                _approx(candidates[idx]),
                _generation_order(candidates[idx], idx),
                _candidate_name(candidates[idx], idx),
                idx,
            )
        )
    ranked.sort(
        key=lambda item: (
            item[0],
            item[1] if item[1] is not None else float("inf"),
            item[2],
            item[3],
            item[4],
        )
    )
    for rank, (_score, _approx_value, _gen_order, _name, idx) in enumerate(ranked, start=1):
        ranks[idx] = rank
    return ranks


def build_m4d_telemetry(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Populate M4D telemetry fields on candidate dicts and return ``candidates``."""
    rank_scores = compute_rank_scores(candidates)
    normalized = compute_family_normalized_approx(candidates)
    cross_family = compute_cross_family_rank(candidates, rank_scores)

    family_sizes: dict[str, int] = {}
    family_ranks: dict[str, dict[int, int]] = {}
    for idx, candidate in enumerate(candidates):
        if not _participates(candidate):
            continue
        family = _family(candidate)
        family_sizes[family] = family_sizes.get(family, 0) + 1
        family_ranks.setdefault(family, {})
    for idx, candidate in enumerate(candidates):
        score = rank_scores[idx]
        if score is None:
            continue
        family = _family(candidate)
        family_ranks[family][idx] = 1 + round(score * max(family_sizes[family] - 1, 1))

    for idx, candidate in enumerate(candidates):
        approx = _approx(candidate)
        score = rank_scores[idx]
        candidate["m4d_rank_score"] = score
        candidate["m4d_family_normalized_approx_delta"] = normalized[idx]
        candidate["m4d_cross_family_rank"] = cross_family[idx]
        if score is None:
            candidate["m4d_rank_reason"] = (
                f"family={_family(candidate)} approx_null={approx is None} excluded"
            )
            continue
        family = _family(candidate)
        family_rank = family_ranks[family][idx]
        family_size = family_sizes[family]
        candidate["m4d_rank_reason"] = (
            f"family={family} rank={family_rank}/{family_size} percentile={score:.6f}"
        )
    return candidates
