"""Pure M4C reserved-bucket ranking helpers.

This module ranks already-generated M4B candidate metadata.  It intentionally
does not import scorer, evaluator, cache, legalizer, or benchmark-loading code.
"""

from __future__ import annotations

from typing import Any


_M4B_FAMILY = "m4b_region_repair"
_EPS = 1e-9


def _is_valid_m4b(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("family") == _M4B_FAMILY
        and bool(candidate.get("valid"))
        and not bool(candidate.get("duplicate"))
        and isinstance(candidate.get("post_legalization_approx_delta"), (int, float))
    )


def _fifo_index(candidate: dict[str, Any], fallback: int) -> int:
    value = candidate.get("fifo_index", candidate.get("generation_rank", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def compute_rank_scores(candidates: list[dict[str, Any]]) -> list[float | None]:
    """Return family-normalized M4C rank scores aligned with ``candidates``."""
    valid = [candidate for candidate in candidates if _is_valid_m4b(candidate)]
    if not valid:
        return [None for _candidate in candidates]

    deltas = [float(candidate["post_legalization_approx_delta"]) for candidate in valid]
    delta_min = min(deltas)
    delta_max = max(deltas)
    delta_range = delta_max - delta_min

    scores: list[float | None] = []
    for candidate in candidates:
        if not _is_valid_m4b(candidate):
            scores.append(None)
            continue
        delta = float(candidate["post_legalization_approx_delta"])
        scores.append((delta - delta_min) / (delta_range + _EPS))
    return scores


def assign_buckets(
    candidates: list[dict[str, Any]],
    k_ranked: int = 16,
    exploration: int = 4,
    max_per_region: int | None = None,
    known_winners: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Populate M4C rank telemetry and bucket assignments in candidate dicts."""
    known_winners = list(known_winners or [])
    scores = compute_rank_scores(candidates)
    valid_indices = [
        idx for idx, candidate in enumerate(candidates) if scores[idx] is not None
    ]

    for idx, candidate in enumerate(candidates):
        score = scores[idx]
        candidate["m4c_rank_score"] = score
        candidate["family_normalized_approx_delta"] = score
        candidate["family_rank"] = None
        candidate["m4c_rank_bucket"] = None
        candidate["m4c_rank_reason"] = None

    ranked_order = sorted(
        valid_indices,
        key=lambda idx: (
            scores[idx],
            _fifo_index(candidates[idx], idx),
            str(candidates[idx].get("candidate_name", candidates[idx].get("name", ""))),
        ),
    )
    for rank, idx in enumerate(ranked_order, start=1):
        candidates[idx]["family_rank"] = rank

    ranked_indices: list[int] = []
    per_region: dict[Any, int] = {}
    limit = max(0, int(k_ranked))
    for idx in ranked_order:
        if len(ranked_indices) >= limit:
            break
        region = candidates[idx].get("region_id")
        if max_per_region is not None and per_region.get(region, 0) >= max_per_region:
            continue
        ranked_indices.append(idx)
        per_region[region] = per_region.get(region, 0) + 1

    if len(ranked_indices) < limit:
        for idx in ranked_order:
            if len(ranked_indices) >= limit:
                break
            if idx not in ranked_indices:
                ranked_indices.append(idx)

    selected = set(ranked_indices)
    exploration_indices = [
        idx
        for idx in sorted(
            valid_indices,
            key=lambda item: (
                _fifo_index(candidates[item], item),
                str(candidates[item].get("candidate_name", candidates[item].get("name", ""))),
            ),
        )
        if idx not in selected
    ][: max(0, int(exploration))]
    selected.update(exploration_indices)

    for winner in known_winners:
        winner_idx = next(
            (
                idx
                for idx in valid_indices
                if candidates[idx].get("candidate_name", candidates[idx].get("name")) == winner
            ),
            None,
        )
        if winner_idx is None or winner_idx in selected:
            continue
        if ranked_indices:
            replace_idx = max(
                [idx for idx in ranked_indices if idx not in {winner_idx}]
                or ranked_indices,
                key=lambda idx: (
                    scores[idx] if scores[idx] is not None else float("-inf"),
                    _fifo_index(candidates[idx], idx),
                    str(candidates[idx].get("candidate_name", candidates[idx].get("name", ""))),
                ),
            )
            ranked_indices[ranked_indices.index(replace_idx)] = winner_idx
            selected.discard(replace_idx)
        else:
            replace_idx = None
            ranked_indices.append(winner_idx)
        selected.add(winner_idx)
        candidates[winner_idx]["m4c_rank_reason"] = (
            "ranked:known_winner_force_insert"
            + (f":replaced={candidates[replace_idx].get('candidate_name', candidates[replace_idx].get('name'))}" if replace_idx is not None else "")
        )

    ranked_set = set(ranked_indices)
    exploration_set = set(exploration_indices) - ranked_set
    for idx in valid_indices:
        candidate = candidates[idx]
        if idx in ranked_set:
            candidate["m4c_rank_bucket"] = "ranked"
            candidate.setdefault("m4c_rank_reason", None)
            if candidate["m4c_rank_reason"] is None:
                candidate["m4c_rank_reason"] = f"ranked:family_rank={candidate['family_rank']}"
        elif idx in exploration_set:
            candidate["m4c_rank_bucket"] = "exploration"
            candidate["m4c_rank_reason"] = (
                f"exploration:fifo_position={_fifo_index(candidate, idx)}"
            )
        else:
            candidate["m4c_rank_reason"] = "m4c_budget_exhausted"

    return candidates
