"""
M2B candidate scoring, validation, legalization, and selection.

Pipeline for each candidate:
  1. Legalize if needed.
  2. Validate.
  3. Skip duplicate legalized placements via placement hash cache.
  4. Prefilter obviously bad local moves using approximate HPWL delta.
  5. Score selected valid unique candidates (pass 1).
  6. If refinement_around_winners: generate + score refinement candidates (pass 2).
  7. Select the best valid scored candidate.
"""

import hashlib
import math
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from submissions.solver.core.score_cache import OfficialScoreCache

from macro_place.benchmark import Benchmark
from submissions.solver.core.candidate_types import (
    CandidateGenerationConfig,
    CandidatePlacement,
    CandidateScoringConfig,
    ScoredCandidate,
    ScoringDiagnostics,
)
from submissions.solver.core.diagnostics import check_placement
from submissions.solver.legalization.greedy_legalizer import LegalizationResult, legalize


def _detect_scoring_mode(plc, benchmark: Benchmark) -> str:
    if plc is not None:
        return "official"
    if not benchmark.net_nodes:
        return "unavailable"
    return "local_proxy"


PLACEMENT_HASH_TOLERANCE_UM = 0.001
"""Quantization tolerance for placement_hash, in µm.

Must be strictly finer than the smallest generated movement so that distinct
tiny refinement moves (the smallest is 0.05 µm — see
``original_refinement._TINY_STEPS_UM``) hash to distinct keys.

This same tolerance is reused for the persistent score-cache key, duplicate
detection, and candidate-hash diagnostics so all three are consistent.
"""

_PLACEMENT_HASH_DECIMALS = 3  # log10(1 / PLACEMENT_HASH_TOLERANCE_UM)


def placement_hash(positions: torch.Tensor) -> str:
    """Return an 8-char MD5 of positions quantized to ``PLACEMENT_HASH_TOLERANCE_UM``."""
    # Quantize in float64 to avoid float32-rounding collisions before MD5.
    arr = np.round(
        positions.detach().cpu().numpy().astype(np.float64),
        _PLACEMENT_HASH_DECIMALS,
    )
    return hashlib.md5(arr.tobytes()).hexdigest()[:8]


def connectivity_audit(benchmark: Benchmark) -> Dict:
    """Return connectivity statistics for a benchmark."""
    n_hard = benchmark.num_hard_macros
    num_net_edges = 0
    degrees = np.zeros(n_hard, dtype=np.int32)

    for nodes in benchmark.net_nodes:
        hard_pins = nodes[nodes < n_hard]
        unique_pins = torch.unique(hard_pins)
        k = unique_pins.numel()
        if k >= 2:
            num_net_edges += k * (k - 1) // 2
            for p in unique_pins.tolist():
                degrees[int(p)] += 1

    fixed_mask = benchmark.macro_fixed[:n_hard]
    num_fixed = int(fixed_mask.sum().item())
    num_ports = benchmark.port_positions.shape[0]
    num_fixed_endpoints = num_fixed + num_ports

    return {
        "num_macros": benchmark.num_macros,
        "num_nets": benchmark.num_nets,
        "num_net_edges": num_net_edges,
        "num_macros_with_degree_gt_0": int((degrees > 0).sum()),
        "num_fixed_endpoints": num_fixed_endpoints,
        "spectral_available": num_net_edges > 0,
        "terminal_anchor_available": num_fixed_endpoints > 0,
    }


def _compute_hpwl(positions: torch.Tensor, benchmark: Benchmark) -> float:
    total = 0.0
    num_macros = benchmark.num_macros
    for nodes in benchmark.net_nodes:
        valid = nodes[nodes < num_macros]
        if valid.numel() < 2:
            continue
        xs = positions[valid, 0]
        ys = positions[valid, 1]
        total += float((xs.max() - xs.min() + ys.max() - ys.min()).item())
    return total


def _hpwl_score(positions: torch.Tensor, benchmark: Benchmark) -> float:
    hpwl = _compute_hpwl(positions, benchmark)
    diag = (benchmark.canvas_width ** 2 + benchmark.canvas_height ** 2) ** 0.5
    return hpwl / max(diag, 1.0)


def _score_placement(
    positions: torch.Tensor,
    benchmark: Benchmark,
    plc,
) -> Tuple[Optional[float], float]:
    t0 = time.perf_counter()
    cost = None
    if plc is not None:
        try:
            from submissions.solver.core.scoring import score

            result = score(positions, benchmark, plc)
            if result is not None:
                cost = float(result.get("proxy_cost", None) or 0.0)
        except Exception:
            cost = None
    if cost is None:
        cost = _hpwl_score(positions, benchmark)
    return cost, (time.perf_counter() - t0) * 1000


def _prepare_candidate(
    candidate: CandidatePlacement,
    benchmark: Benchmark,
    movable_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
    legalizer_max_rings: int,
) -> ScoredCandidate:
    """Legalize if needed and validate. Scoring happens later."""
    t_total_start = time.perf_counter()

    if candidate.bypass_legalization:
        legal_positions = candidate.positions.clone().float()
        leg = LegalizationResult(
            positions=legal_positions,
            valid=True,
            num_moved=0,
            max_move=0.0,
            total_move=0.0,
            runtime_ms=0.0,
            no_op=True,
        )
    else:
        leg = legalize(
            positions=candidate.positions,
            sizes=benchmark.macro_sizes,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
            movable_mask=movable_mask,
            obstacle_mask=obstacle_mask,
            max_rings=legalizer_max_rings,
        )

    diag = check_placement(
        positions=leg.positions,
        sizes=benchmark.macro_sizes,
        canvas_w=benchmark.canvas_width,
        canvas_h=benchmark.canvas_height,
        mask=movable_mask,
        obstacle_mask=obstacle_mask,
    )
    valid = diag.valid if candidate.bypass_legalization else (leg.valid and diag.valid)
    msgs = ([] if candidate.bypass_legalization else list(leg.messages)) + list(diag.messages)

    meta = dict(candidate.metadata)
    meta["postlegal_valid"] = valid
    if not candidate.bypass_legalization:
        meta["legalization_num_moved"] = leg.num_moved
        moved_macro_id = candidate.metadata.get("moved_macro_id")
        if moved_macro_id is not None:
            mid = int(moved_macro_id)
            orig_x = float(benchmark.macro_positions[mid, 0].item())
            orig_y = float(benchmark.macro_positions[mid, 1].item())
            meta["actual_dx_after_legalization"] = float(leg.positions[mid, 0].item()) - orig_x
            meta["actual_dy_after_legalization"] = float(leg.positions[mid, 1].item()) - orig_y

    return ScoredCandidate(
        name=candidate.name,
        family=candidate.family,
        positions=leg.positions,
        valid=valid,
        proxy_cost=None,
        delta_vs_original=None,
        num_overlaps=diag.num_overlaps,
        num_out_of_bounds=diag.num_out_of_bounds,
        num_unplaced=0 if candidate.bypass_legalization else len(leg.messages),
        num_moved=leg.num_moved,
        max_move=leg.max_move,
        total_move=leg.total_move,
        legalization_ms=leg.runtime_ms,
        scoring_ms=0.0,
        total_ms=(time.perf_counter() - t_total_start) * 1000,
        no_op=leg.no_op,
        notes=candidate.notes,
        was_scored=False,
        metadata=meta,
        messages=msgs,
    )


def _mark_duplicates(
    scored: List[ScoredCandidate],
    enable_hash_cache: bool,
    existing_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str]]:
    """Mark duplicate candidates and return (dup_count, hash_map)."""
    owner_by_hash: Dict[str, str] = dict(existing_hashes) if existing_hashes else {}
    duplicate_count = 0

    for sc in scored:
        if not sc.valid:
            continue
        h = placement_hash(sc.positions)
        sc.metadata["placement_hash"] = h
        if not enable_hash_cache:
            continue
        owner = owner_by_hash.get(h)
        if owner is not None:
            sc.duplicate_of = owner
            duplicate_count += 1
        else:
            owner_by_hash[h] = sc.name

    return duplicate_count, owner_by_hash


def _prefilter_score_set(
    scored: List[ScoredCandidate],
    cfg: CandidateScoringConfig,
) -> Tuple[Set[int], int, int, Optional[float]]:
    """Return (score_indices, prefiltered_count, improving_count, best_skipped_delta).

    Candidates excluded by approx-HPWL prefilter get skip_reason="prefiltered".
    """
    score_indices: Set[int] = set()

    for idx, sc in enumerate(scored):
        if not sc.valid or sc.duplicate_of is not None:
            continue
        if sc.name in ("original_raw", "original_legalized"):
            score_indices.add(idx)

    if cfg.prefilter_mode == "off":
        for idx, sc in enumerate(scored):
            if sc.valid and sc.duplicate_of is None:
                score_indices.add(idx)
        return score_indices, 0, 0, None

    positives: List[Tuple[float, str, int]] = []
    negatives: List[Tuple[float, str, int]] = []
    prefiltered = 0
    improving_count = 0

    for idx, sc in enumerate(scored):
        if not sc.valid or sc.duplicate_of is not None or idx in score_indices:
            continue
        approx = sc.metadata.get("approx_hpwl_delta")
        if sc.family != "original_neighborhood" or not isinstance(approx, (int, float)) or not math.isfinite(approx):
            score_indices.add(idx)
            continue
        approx = float(approx)
        if approx <= 1e-9:
            score_indices.add(idx)
            improving_count += 1
            negatives.append((approx, sc.name, idx))
        else:
            positives.append((approx, sc.name, idx))

    positives.sort(key=lambda item: (item[0], item[1]))
    exploratory = {idx for _approx, _name, idx in positives[: cfg.exploratory_score_count]}
    score_indices.update(exploratory)
    prefiltered = max(0, len(positives) - len(exploratory))
    best_skipped = positives[cfg.exploratory_score_count][0] if len(positives) > cfg.exploratory_score_count else None

    # Mark prefiltered candidates so skip_reason can be populated later
    for _approx, _name, idx in positives[cfg.exploratory_score_count:]:
        scored[idx].metadata.setdefault("skip_reason", "prefiltered")

    return score_indices, prefiltered, improving_count, best_skipped


def _score_batch(
    scored: List[ScoredCandidate],
    score_indices,  # List[int] (order preserved) or Set[int] (sorted numerically)
    benchmark: Benchmark,
    plc,
    max_scores: Optional[int],
    already_scored: int = 0,
    cache: Optional[OfficialScoreCache] = None,
    benchmark_name: str = "",
    timing_records: Optional[List[float]] = None,
    timing_names: Optional[List[str]] = None,
    skipped_by_budget_acc: Optional[List[int]] = None,
    scoring_rank_counter: Optional[List[int]] = None,
) -> int:
    """Score candidates at score_indices, respecting budget. Returns new total fresh-scored count.

    When score_indices is a List[int], candidates are evaluated in that exact order so
    callers can prioritise by approx_hpwl_delta before invoking.  When it is a Set[int]
    the order falls back to sorted numerically (generation order).

    Checks persistent cache before invoking the official scorer.
    Cache hits do NOT count against max_scores (scoring budget).
    Candidates that hit the budget cap get skip_reason="budget_exceeded".
    """
    count = already_scored
    _iter = score_indices if isinstance(score_indices, list) else sorted(score_indices)
    for idx in _iter:
        sc = scored[idx]
        if not sc.valid:
            continue

        # Check persistent cache first (cache hits are free — not counted against budget)
        phash = sc.metadata.get("placement_hash")
        if cache is not None and cache.enabled and phash:
            cached_cost = cache.lookup(benchmark_name, phash)
            if cached_cost is not None:
                sc.proxy_cost = cached_cost
                sc.was_scored = True
                sc.metadata["cache_hit"] = True
                if scoring_rank_counter is not None:
                    sc.metadata["scoring_rank"] = scoring_rank_counter[0]
                    scoring_rank_counter[0] += 1
                # count += 1  intentionally omitted: cache hits don't consume budget
                continue

        if max_scores is not None and count >= max_scores:
            if skipped_by_budget_acc is not None:
                skipped_by_budget_acc[0] += 1
            sc.metadata.setdefault("skip_reason", "budget_exceeded")
            continue

        t0 = time.perf_counter()
        sc.proxy_cost, sc.scoring_ms = _score_placement(sc.positions, benchmark, plc)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sc.was_scored = sc.proxy_cost is not None

        if timing_records is not None:
            timing_records.append(elapsed_ms)
        if timing_names is not None:
            timing_names.append(sc.name)

        if sc.was_scored:
            sc.metadata["cache_hit"] = False
            sc.metadata["fresh_score_consumed"] = True
            count += 1
            if scoring_rank_counter is not None:
                sc.metadata["scoring_rank"] = scoring_rank_counter[0]
                scoring_rank_counter[0] += 1
            if cache is not None and cache.enabled and sc.proxy_cost is not None:
                phash_store = phash or placement_hash(sc.positions)
                cache.record(benchmark_name, phash_store, sc.proxy_cost, {"name": sc.name})

    return count


def _score_line_search_ordered(
    ls_scored: List[ScoredCandidate],
    benchmark: Benchmark,
    plc,
    max_scores: Optional[int],
    already_scored: int,
    stop_after_worse: int,
    raw_original_proxy_cost: Optional[float],
    cache: Optional[OfficialScoreCache],
    benchmark_name: str,
    timing_records: Optional[List[float]],
    timing_names: Optional[List[str]],
    skipped_by_budget_acc: Optional[List[int]] = None,
    scoring_rank_counter: Optional[List[int]] = None,
) -> int:
    """Score line-search candidates per macro using priority scale order.

    Scales are scored in priority order (large-then-small) rather than ascending,
    so promising far-reaching moves (e.g. 2.5x, 4.0x) are tried before
    conservative sub-step scales that tend to trigger early stopping.

    Macros are processed in seed order (best-scoring seed first).
    Early stopping per macro: after stop_after_worse consecutive official scores
    worse than the current best, stops expanding for that macro.
    Cache hits count toward the cost check but NOT against max_scores budget.
    """
    from collections import defaultdict
    from submissions.solver.core.original_line_search import _SCORING_PRIORITY_RANK

    macro_groups: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for idx, sc in enumerate(ls_scored):
        if not sc.valid or sc.duplicate_of is not None:
            continue
        macro_id = sc.metadata.get("moved_macro_id")
        scale = float(sc.metadata.get("scale_multiplier", 1.0))
        if macro_id is not None:
            macro_groups[int(macro_id)].append((scale, idx))

    # Process macros in the order seeds first appeared (best-scoring seed first)
    macro_id_order: List[int] = []
    seen_mids: Set[int] = set()
    for sc in ls_scored:
        mid = sc.metadata.get("moved_macro_id")
        if mid is not None:
            mid = int(mid)
            if mid not in seen_mids:
                macro_id_order.append(mid)
                seen_mids.add(mid)
    # Include any macro_ids present in groups but not in ls_scored order (defensive)
    for mid in sorted(macro_groups.keys()):
        if mid not in seen_mids:
            macro_id_order.append(mid)

    _priority_default = len(_SCORING_PRIORITY_RANK) + 1
    count = already_scored
    for macro_id in macro_id_order:
        if macro_id not in macro_groups:
            continue
        # Sort by priority rank (large scales first), not ascending scale
        group = sorted(
            macro_groups[macro_id],
            key=lambda x: (_SCORING_PRIORITY_RANK.get(x[0], _priority_default), x[0]),
        )
        best_cost_for_macro = raw_original_proxy_cost
        worse_streak = 0

        for scale, idx in group:
            sc = ls_scored[idx]

            # Check cache (free, no budget consumed)
            phash = sc.metadata.get("placement_hash")
            if cache is not None and cache.enabled and phash:
                cached_cost = cache.lookup(benchmark_name, phash)
                if cached_cost is not None:
                    sc.proxy_cost = cached_cost
                    sc.was_scored = True
                    sc.metadata["cache_hit"] = True
                    if scoring_rank_counter is not None:
                        sc.metadata["scoring_rank"] = scoring_rank_counter[0]
                        scoring_rank_counter[0] += 1
                    if best_cost_for_macro is None or cached_cost < best_cost_for_macro - 1e-9:
                        best_cost_for_macro = cached_cost
                        worse_streak = 0
                    else:
                        worse_streak += 1
                    if stop_after_worse > 0 and worse_streak >= stop_after_worse:
                        break
                    continue

            if max_scores is not None and count >= max_scores:
                if skipped_by_budget_acc is not None:
                    skipped_by_budget_acc[0] += 1
                sc.metadata.setdefault("skip_reason", "budget_exceeded")
                break

            if stop_after_worse > 0 and worse_streak >= stop_after_worse:
                sc.metadata.setdefault("skip_reason", "line_search_early_stop")
                break

            t0 = time.perf_counter()
            sc.proxy_cost, sc.scoring_ms = _score_placement(sc.positions, benchmark, plc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            sc.was_scored = sc.proxy_cost is not None

            if timing_records is not None:
                timing_records.append(elapsed_ms)
            if timing_names is not None:
                timing_names.append(sc.name)

            if sc.was_scored:
                sc.metadata["cache_hit"] = False
                sc.metadata["fresh_score_consumed"] = True
                count += 1
                if scoring_rank_counter is not None:
                    sc.metadata["scoring_rank"] = scoring_rank_counter[0]
                    scoring_rank_counter[0] += 1
                if cache is not None and cache.enabled and sc.proxy_cost is not None:
                    phash_store = phash or placement_hash(sc.positions)
                    cache.record(benchmark_name, phash_store, sc.proxy_cost, {"name": sc.name})

                if best_cost_for_macro is None or sc.proxy_cost < best_cost_for_macro - 1e-9:
                    best_cost_for_macro = sc.proxy_cost
                    worse_streak = 0
                else:
                    worse_streak += 1

    return count


def _macro_priority_scores(benchmark) -> Optional[Dict[int, float]]:
    """Return area*log(1+degree) priority score keyed by macro_id, or None on failure."""
    try:
        n_hard = benchmark.num_hard_macros
        degrees = np.zeros(n_hard, dtype=np.float64)
        for nodes in benchmark.net_nodes:
            hard_pins = nodes[nodes < n_hard]
            unique_pins = torch.unique(hard_pins)
            for p in unique_pins.tolist():
                degrees[int(p)] += 1
        sizes = benchmark.macro_sizes.numpy()
        areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
        priorities = areas * np.log1p(degrees)
        return {i: float(priorities[i]) for i in range(n_hard)}
    except Exception:
        return None


def _select_seeds_diverse(
    neighborhood: List[ScoredCandidate],
    name_to_placement: Dict[str, CandidatePlacement],
    top_k: int,
    ref_cost: float,
    raw_original_proxy_cost: Optional[float],
    exploration_seeds: int,
    benchmark,
) -> Tuple[List[CandidatePlacement], List[Dict]]:
    """Multi-bucket seed selection for cold-run diversity.

    Bucket A: best approx_hpwl_delta (most negative), distinct macros.
    Bucket B: best officially-scored improving candidate (distinct macro).
    Bucket C: best macro-priority (area*degree) improving candidate (distinct macro).
              Only included when top_k >= 4; falls back to approx order if no benchmark.
    Bucket D: exploratory improving candidates outside the already-selected set.
    Fill:     conservative approx-sort for any remaining slots.

    All buckets enforce distinct macro IDs (first seed per macro wins).
    Selection is deterministic: stable sort keys + stable name tiebreaker.
    """
    priority_by_macro = _macro_priority_scores(benchmark)

    def _is_improving(sc: ScoredCandidate) -> bool:
        approx = sc.metadata.get("approx_hpwl_delta")
        return (
            isinstance(approx, (int, float))
            and math.isfinite(float(approx))
            and float(approx) <= 1e-9
        )

    # Improving candidates sorted by approx delta (most negative first), then name for stability
    improving = [sc for sc in neighborhood if _is_improving(sc)]
    improving.sort(key=lambda sc: (float(sc.metadata.get("approx_hpwl_delta", 1e18)), sc.name))

    seeds: List[CandidatePlacement] = []
    seed_macros: Set[int] = set()
    seed_names: Set[str] = set()
    bucket_diag: List[Dict] = []
    skipped_macro_ids: List[int] = []

    def try_add(sc: ScoredCandidate, bucket: str) -> bool:
        if sc.name in seed_names:
            return False
        p = name_to_placement.get(sc.name)
        if p is None:
            return False
        mid = sc.metadata.get("moved_macro_id")
        mid_int = int(mid) if mid is not None else None
        if mid_int is not None and mid_int in seed_macros:
            skipped_macro_ids.append(mid_int)
            return False
        seed_names.add(sc.name)
        if mid_int is not None:
            seed_macros.add(mid_int)
        seeds.append(p)
        priority_score = (
            priority_by_macro.get(mid_int)
            if (priority_by_macro and mid_int is not None)
            else None
        )
        bucket_diag.append({
            "seed_name": sc.name,
            "macro_id": mid,
            "bucket": bucket,
            "official_proxy_cost": sc.proxy_cost if sc.was_scored else None,
            "approx_hpwl_delta": sc.metadata.get("approx_hpwl_delta"),
            "macro_priority_score": priority_score,
            "generation_rank": sc.metadata.get("generation_rank"),
            "scoring_rank": sc.metadata.get("scoring_rank"),
            "was_scored": sc.was_scored,
        })
        return True

    # Determine bucket sizes
    use_priority_bucket = top_k >= 4
    use_official_bucket = top_k >= 3
    fixed_count = (1 if use_official_bucket else 0) + (1 if use_priority_bucket else 0)
    approx_count = max(1, top_k - exploration_seeds - fixed_count)

    # Bucket A: best approx_hpwl_delta, distinct macros
    for sc in improving:
        if len(seeds) >= approx_count:
            break
        try_add(sc, "approx")

    # Bucket B: best officially-scored improving (distinct macro)
    if use_official_bucket and raw_original_proxy_cost is not None:
        official_improving = sorted(
            [
                sc for sc in neighborhood
                if sc.was_scored
                and sc.proxy_cost is not None
                and sc.proxy_cost < raw_original_proxy_cost - 1e-9
            ],
            key=lambda sc: (float(sc.proxy_cost), sc.name),
        )
        for sc in official_improving:
            if try_add(sc, "official"):
                break

    # Bucket C: best macro-priority improving (distinct macro)
    if use_priority_bucket:
        if priority_by_macro:
            priority_sorted = sorted(
                improving,
                key=lambda sc: (
                    -(
                        priority_by_macro.get(int(sc.metadata["moved_macro_id"]), 0.0)
                        if sc.metadata.get("moved_macro_id") is not None
                        else 0.0
                    ),
                    float(sc.metadata.get("approx_hpwl_delta", 1e18)),
                    sc.name,
                ),
            )
        else:
            priority_sorted = improving
        for sc in priority_sorted:
            if try_add(sc, "priority"):
                break

    # Bucket D: exploratory improving (distinct macros not yet selected)
    exploratory_added = 0
    for sc in improving:
        if len(seeds) >= top_k:
            break
        if exploratory_added >= exploration_seeds:
            break
        if try_add(sc, "exploratory"):
            exploratory_added += 1

    # Fill: conservative sort for remaining slots
    def _conservative_key(sc: ScoredCandidate) -> Tuple:
        approx = sc.metadata.get("approx_hpwl_delta")
        has_approx = isinstance(approx, (int, float)) and math.isfinite(float(approx))
        approx_f = float(approx) if has_approx else 1e18
        is_imp = has_approx and approx_f <= 1e-9
        scored_ok = sc.was_scored and sc.proxy_cost is not None
        if is_imp:
            return (0, sc.proxy_cost if scored_ok else ref_cost, approx_f, sc.name)
        if scored_ok:
            return (1, float(sc.proxy_cost), approx_f, sc.name)
        return (2, ref_cost, approx_f, sc.name)

    for sc in sorted(neighborhood, key=_conservative_key):
        if len(seeds) >= top_k:
            break
        try_add(sc, "fill")

    return seeds, bucket_diag


def _select_refinement_seeds(
    scored: List[ScoredCandidate],
    candidates: List[CandidatePlacement],
    top_k: int,
    raw_original_proxy_cost: Optional[float],
    strategy: str = "conservative",
    benchmark=None,
    exploration_seeds: int = 1,
) -> Tuple[List[CandidatePlacement], List[Dict]]:
    """Return top-K neighborhood seeds for the refinement pass.

    Budget-invariant: approx_hpwl_delta determines tier, not whether the candidate
    consumed scoring budget or was a cache hit.

    strategy="conservative": single approx-delta sort (original behavior, returns empty diag).
    strategy="diverse": multi-bucket selection — approx + official + priority + exploratory.

    Returns (seeds, bucket_diagnostics). bucket_diagnostics is empty for conservative.
    """
    name_to_placement: Dict[str, CandidatePlacement] = {c.name: c for c in candidates}
    _ref_cost = raw_original_proxy_cost if raw_original_proxy_cost is not None else 0.0

    neighborhood = [
        sc for sc in scored
        if sc.family == "original_neighborhood"
        and sc.valid
        and sc.duplicate_of is None
        and sc.name in name_to_placement
    ]

    if strategy == "diverse":
        return _select_seeds_diverse(
            neighborhood, name_to_placement, top_k, _ref_cost,
            raw_original_proxy_cost, exploration_seeds, benchmark,
        )

    # Conservative strategy (original behavior)
    def sort_key(sc: ScoredCandidate) -> Tuple:
        if sc.name not in name_to_placement:
            return (3, 1e18, 1e18, sc.name)
        approx = sc.metadata.get("approx_hpwl_delta")
        has_approx = isinstance(approx, (int, float)) and math.isfinite(float(approx))
        approx_f = float(approx) if has_approx else 1e18
        is_improving = has_approx and approx_f <= 1e-9
        scored_ok = sc.was_scored and sc.proxy_cost is not None
        if is_improving:
            # Tier 0: both scored and unscored improving candidates.
            # Scored: use proxy_cost so confirmed-better moves sort first.
            # Unscored: use approx_f so most-negative approx sorts first.
            if scored_ok:
                return (0, sc.proxy_cost, approx_f, sc.name)
            return (0, _ref_cost, approx_f, sc.name)
        if scored_ok:
            return (1, sc.proxy_cost, approx_f, sc.name)
        return (2, _ref_cost, approx_f, sc.name)

    neighborhood.sort(key=sort_key)

    seen_macros: Set[int] = set()
    seeds: List[CandidatePlacement] = []
    for sc in neighborhood:
        if len(seeds) >= top_k:
            break
        p = name_to_placement.get(sc.name)
        if p is None:
            continue
        macro_id = p.metadata.get("moved_macro_id")
        if macro_id is not None and int(macro_id) in seen_macros:
            continue
        if macro_id is not None:
            seen_macros.add(int(macro_id))
        seeds.append(p)
    return seeds, []


def _select_line_search_seeds(
    scored: List[ScoredCandidate],
    top_k: int,
) -> List[ScoredCandidate]:
    """Return top-K officially scored neighborhood candidates as line-search seeds.

    Seeds are sorted by proxy_cost ascending (best first).  Only candidates
    from original_neighborhood with moved_macro_id metadata are included.
    """
    eligible = [
        sc for sc in scored
        if sc.family == "original_neighborhood"
        and sc.valid
        and sc.was_scored
        and sc.proxy_cost is not None
        and sc.metadata.get("moved_macro_id") is not None
    ]
    eligible.sort(key=lambda sc: (float(sc.proxy_cost), sc.name))
    return eligible[:top_k]


def score_and_select(
    candidates: List[CandidatePlacement],
    benchmark: Benchmark,
    plc=None,
    scoring_config: Optional[CandidateScoringConfig] = None,
    generation_config: Optional[CandidateGenerationConfig] = None,
    score_cache: Optional[OfficialScoreCache] = None,
) -> Tuple[ScoredCandidate, List[ScoredCandidate], ScoringDiagnostics]:
    """Score candidates and return (best, ranked, diagnostics).

    Passes:
      1. original_raw + neighborhood (approx-prefiltered + exploratory)
      2. refinement around winners (if refinement_around_winners)
      3. line-search around winners (if line_search_around_winners)

    Persistent cache and runtime timing are applied across all passes.

    score_cache: pre-constructed OfficialScoreCache shared across benchmarks.
        If None, a private cache is created from scoring_config (cache is NOT
        shared across benchmark runs in a single session — use the caller-level
        cache for multi-benchmark runs).
    """
    cfg = scoring_config or CandidateScoringConfig()
    gen_cfg = generation_config
    scoring_mode = _detect_scoring_mode(plc, benchmark)

    if scoring_mode == "unavailable":
        print(
            "[M2B] WARNING: Local proxy scoring unavailable: net_nodes empty; candidate ranking is validity-only.",
            file=sys.stderr,
        )

    # --- Persistent score cache ---
    # If a shared cache is provided, use it.  Otherwise create a private one.
    if score_cache is None:
        from pathlib import Path as _Path
        _cache_path = (
            _Path(cfg.official_score_cache_path)
            if cfg.official_score_cache_path and not cfg.disable_score_cache
            else None
        )
        score_cache = OfficialScoreCache(
            cache_path=_cache_path,
            disabled=cfg.disable_score_cache or (_cache_path is None),
            clear=cfg.clear_score_cache,
        )
    # Snapshot hits/misses so diagnostics report per-benchmark deltas
    _cache_hits_before = score_cache.hits
    _cache_misses_before = score_cache.misses
    benchmark_name = getattr(benchmark, "name", "unknown")

    # --- Budget allocation for seed discovery, refinement, and line-search ---
    # Default split (for max_official_scores=60):
    #   seed discovery (neighborhood):  32  (~53%)
    #   refinement (pass 2):            10  (~17%)
    #   line-search (pass 3):           18  (~30%, uses remainder)
    # Unused budget from an earlier pass flows to downstream passes automatically
    # because pass 3 uses whatever remains after passes 1 and 2.
    _has_ls = gen_cfg is not None and gen_cfg.line_search_around_winners
    _has_ref = gen_cfg is not None and gen_cfg.refinement_around_winners

    # --- M3C deterministic budget allocation ---
    # Gated by m3c_budget_allocation flag; disabled by default so existing behavior is unchanged.
    _m3c_enabled = gen_cfg is not None and getattr(gen_cfg, "m3c_budget_allocation", False)
    _m3c_pre_m3_alloc: Optional[int] = None
    _m3c_m3a_alloc: Optional[int] = None
    _m3c_m3b_alloc: Optional[int] = None
    _m3c_rollover_enabled: bool = True
    _m3c_rollover_to_m3b: int = 0

    if _m3c_enabled and cfg.max_official_scores is not None:
        _raw_pre = getattr(gen_cfg, "m3c_pre_m3_budget", None)
        _raw_m3a = getattr(gen_cfg, "m3c_m3a_reserved_budget", None)
        _raw_m3b = getattr(gen_cfg, "m3c_m3b_reserved_budget", None)
        _m3c_rollover_enabled = bool(getattr(gen_cfg, "m3c_rollover_unused_budget", True))
        _configured_m3a = max(0, int(_raw_m3a)) if _raw_m3a is not None else 5
        _configured_m3b = max(0, int(_raw_m3b)) if _raw_m3b is not None else 5
        _configured_pre = (
            max(0, int(_raw_pre)) if _raw_pre is not None
            else max(0, cfg.max_official_scores - _configured_m3a - _configured_m3b)
        )
        # Normalize: late-stage reservation first (M3A → M3B → pre-M3).
        # M3A and M3B always receive their full reserved slices before pre-M3 is allocated.
        # This guarantees total allocated scores never exceed max_official_scores for any config.
        _remaining = cfg.max_official_scores
        _m3c_m3a_alloc = min(_configured_m3a, _remaining)
        _remaining -= _m3c_m3a_alloc
        _m3c_m3b_alloc = min(_configured_m3b, _remaining)
        _remaining -= _m3c_m3b_alloc
        _m3c_pre_m3_alloc = min(_configured_pre, _remaining)
        assert _m3c_pre_m3_alloc + _m3c_m3a_alloc + _m3c_m3b_alloc <= cfg.max_official_scores

    # Effective total for passes 1-3 (pre-M3 pool).
    # When M3C is enabled this is the reserved pre-M3 slice; otherwise it is the global cap.
    _pre_m3_total: Optional[int] = (
        _m3c_pre_m3_alloc if (_m3c_enabled and _m3c_pre_m3_alloc is not None)
        else cfg.max_official_scores
    )

    _seed_budget: Optional[int] = cfg.seed_discovery_score_budget
    _ref_budget: Optional[int] = cfg.refinement_score_budget

    if _pre_m3_total is not None and (_has_ls or _has_ref):
        total = _pre_m3_total
        if _seed_budget is None:
            # Seed discovery: ~32/60 of total.
            # When M3C is enabled, skip the hard minimum so a zero pre-M3 allocation
            # is honored exactly (no fresh scores fired from the global pool).
            if _m3c_enabled:
                _seed_budget = total * 32 // 60
            else:
                _seed_budget = max(3, total * 32 // 60)
        if _ref_budget is None and _has_ref:
            # Refinement: at least 3 slots per refinement seed (top_k seeds + 1 combo)
            # so round-robin scoring reaches the 2nd-best candidate per seed in case
            # the highest-approx candidate overshoots on dense benchmarks.
            _min_ref_by_seeds = 3 * (
                (gen_cfg.refinement_top_k + 1) if gen_cfg is not None else 6
            )
            _ref_budget = max(_min_ref_by_seeds, total * 10 // 60)

    _pass1_max: Optional[int] = _seed_budget if (_has_ls or _has_ref) else _pre_m3_total
    # M3C guard: seed_discovery_score_budget is a request, not an absolute override.
    # Clamp _pass1_max to _pre_m3_total so M3A/M3B reserved slices cannot be crowded out
    # and total fresh scores never exceed max_official_scores.
    if _m3c_enabled and _pre_m3_total is not None:
        _pass1_max = min(_pass1_max, _pre_m3_total) if _pass1_max is not None else _pre_m3_total
    _pass2_max: Optional[int] = _ref_budget

    # --- Timing and rank accumulators ---
    timing_records: List[float] = []
    timing_names: List[str] = []
    skipped_by_budget_acc = [0]  # mutable counter passed by reference
    scoring_rank_counter = [0]   # global scoring rank across all passes

    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    obstacle_mask = benchmark.macro_fixed & benchmark.get_hard_macro_mask()

    # --- Pass 1: prepare, deduplicate, prefilter, score ---
    scored: List[ScoredCandidate] = [
        _prepare_candidate(
            candidate=c,
            benchmark=benchmark,
            movable_mask=movable_mask,
            obstacle_mask=obstacle_mask,
            legalizer_max_rings=cfg.legalizer_max_rings,
        )
        for c in candidates
    ]

    # Assign generation_rank to pass-1 candidates
    for gen_rank, sc in enumerate(scored):
        sc.metadata["generation_rank"] = gen_rank
        sc.metadata["pass_id"] = 1

    pass1_dup_count, hash_map = _mark_duplicates(scored, enable_hash_cache=cfg.enable_hash_cache)
    score_indices, prefiltered_count, improving_count, best_skipped_delta = _prefilter_score_set(scored, cfg)

    # Sort pass-1 candidates: originals first, then improving neighborhood best-first
    # (most negative approx_hpwl_delta), then global families, then exploratory positives.
    # This ensures the fixed seed budget is spent on highest-value moves regardless of
    # generation order — critical for cold (empty-cache) runs where every slot costs a
    # fresh official score.
    def _pass1_sort_key(idx: int) -> tuple:
        sc = scored[idx]
        if sc.name in ("original_raw", "original_legalized"):
            return (0, 0.0, idx)
        approx = sc.metadata.get("approx_hpwl_delta")
        if sc.family != "original_neighborhood" or approx is None or not math.isfinite(float(approx)):
            return (1, 0.0, idx)  # global families: score right after originals
        approx_f = float(approx)
        if approx_f <= 1e-9:
            return (2, approx_f, idx)  # improving: best (most negative) first
        return (3, approx_f, idx)  # exploratory positives: ascending delta

    sorted_pass1_indices = sorted(score_indices, key=_pass1_sort_key)

    pass1_scored_count = _score_batch(
        scored, sorted_pass1_indices, benchmark, plc,
        max_scores=_pass1_max, already_scored=0,
        cache=score_cache, benchmark_name=benchmark_name,
        timing_records=timing_records, timing_names=timing_names,
        skipped_by_budget_acc=skipped_by_budget_acc,
        scoring_rank_counter=scoring_rank_counter,
    )

    # Assign raw_original_proxy_cost
    raw_sc = next((s for s in scored if s.name == "original_raw"), None)
    leg_sc = next((s for s in scored if s.name == "original_legalized"), None)
    raw_original_valid = bool(raw_sc is not None and raw_sc.valid)
    raw_original_proxy_cost = (
        raw_sc.proxy_cost if raw_sc is not None and raw_sc.valid and raw_sc.was_scored else None
    )

    # Propagate costs to duplicates in pass 1
    index_by_name = {sc.name: idx for idx, sc in enumerate(scored)}
    for sc in scored:
        if sc.duplicate_of is None:
            continue
        owner_idx = index_by_name.get(sc.duplicate_of)
        if owner_idx is not None:
            owner = scored[owner_idx]
            sc.proxy_cost = owner.proxy_cost
            sc.metadata["placement_hash"] = owner.metadata.get("placement_hash")
            sc.metadata["duplicate_of"] = owner.name

    for sc in scored:
        if sc.proxy_cost is not None and raw_original_proxy_cost is not None:
            sc.delta_vs_original = sc.proxy_cost - raw_original_proxy_cost
        if sc.family == "original_neighborhood":
            sc.metadata["official_proxy_cost"] = sc.proxy_cost
            sc.metadata["delta_vs_raw_original"] = sc.delta_vs_original

    # --- Pass 2: refinement ---
    refinement_candidates_generated = 0
    combo_candidates_generated = 0
    best_single_macro_move = ""
    best_single_macro_delta: Optional[float] = None
    best_combo_move = ""
    best_combo_delta: Optional[float] = None
    total_dup_count = pass1_dup_count
    pass2_scored_count = 0
    _seed_bucket_diag: List[Dict] = []

    if gen_cfg is not None and gen_cfg.refinement_around_winners:
        _seed_strategy = getattr(gen_cfg, "refinement_seed_strategy", "conservative")
        _exploration_seeds = getattr(gen_cfg, "refinement_exploration_seeds", 1)
        seeds, _seed_bucket_diag = _select_refinement_seeds(
            scored, candidates, gen_cfg.refinement_top_k, raw_original_proxy_cost,
            strategy=_seed_strategy,
            benchmark=benchmark,
            exploration_seeds=_exploration_seeds,
        )

        if seeds:
            existing_names: Set[str] = {c.name for c in candidates}
            from submissions.solver.core.original_refinement import generate_original_refinement_candidates

            refinement_placements = generate_original_refinement_candidates(
                benchmark, seeds, gen_cfg, existing_names
            )

            refinement_candidates_generated = sum(
                1 for c in refinement_placements if "combo" not in c.metadata.get("refinement_type", "")
            )
            combo_candidates_generated = sum(
                1 for c in refinement_placements if "combo" in c.metadata.get("refinement_type", "")
            )

            ref_scored: List[ScoredCandidate] = [
                _prepare_candidate(c, benchmark, movable_mask, obstacle_mask, cfg.legalizer_max_rings)
                for c in refinement_placements
            ]

            # Assign generation_rank continuing from pass-1
            _pass2_gen_base = len(scored)
            for local_rank, rsc in enumerate(ref_scored):
                rsc.metadata["generation_rank"] = _pass2_gen_base + local_rank
                rsc.metadata["pass_id"] = 2

            pass2_dup_count, hash_map = _mark_duplicates(
                ref_scored, enable_hash_cache=cfg.enable_hash_cache, existing_hashes=hash_map
            )
            total_dup_count += pass2_dup_count

            # Build per-seed queues so budget is distributed across all seed macros.
            # Without this, generation order (seed 1 first) would let the first seed
            # exhaust the entire pass-2 budget before later seeds get any scoring turns.
            #
            # Priority tiers within each seed (lower tuple = scored first):
            #   (0, approx): prelegal_valid AND improving approx — most reliable
            #   (1, approx): prelegal_bad — bypass filter but approx unreliable post-legal
            #   (2, approx): no valid approx — unknown, score after improving ones
            #   (3, approx): exploratory positive-approx (added below from ref_positives)
            _ref_positives: List[Tuple[float, int]] = []
            # macro_id -> [(sort_tier, approx_val, idx)]; None key = combo / no single macro
            _seed_q: Dict[Optional[int], List[Tuple[float, float, int]]] = {}
            for idx, rsc in enumerate(ref_scored):
                if not rsc.valid or rsc.duplicate_of is not None:
                    continue
                prelegal_bad = rsc.metadata.get("prelegal_valid") is False
                approx = rsc.metadata.get("approx_hpwl_delta")
                has_valid_approx = isinstance(approx, float) and math.isfinite(approx)
                is_improving = has_valid_approx and float(approx) <= 1e-9
                approx_f = float(approx) if has_valid_approx else 0.0
                if prelegal_bad:
                    # Prelegal overlap: include but score after valid-improving candidates.
                    # Approx is computed at pre-legalization (overlapping) position and is
                    # unreliable; tier 1 keeps it from pre-empting scale1.5x/2x candidates.
                    sort_key = (1, approx_f, idx)
                elif is_improving:
                    sort_key = (0, approx_f, idx)
                elif not has_valid_approx:
                    sort_key = (2, 0.0, idx)
                else:
                    _ref_positives.append((approx_f, idx))
                    continue
                mid = rsc.metadata.get("moved_macro_id")
                key: Optional[int] = int(mid) if mid is not None else None
                if key not in _seed_q:
                    _seed_q[key] = []
                _seed_q[key].append(sort_key)

            # Add top exploratory positive-approx candidates to their per-seed queues.
            _ref_positives.sort(key=lambda x: x[0])
            for approx_val, idx in _ref_positives[: cfg.exploratory_score_count]:
                mid = ref_scored[idx].metadata.get("moved_macro_id")
                key = int(mid) if mid is not None else None
                if key not in _seed_q:
                    _seed_q[key] = []
                _seed_q[key].append((3, approx_val, idx))
            # Mark pass-2 prefiltered candidates
            for _, idx in _ref_positives[cfg.exploratory_score_count:]:
                ref_scored[idx].metadata.setdefault("skip_reason", "prefiltered")

            # Sort within each seed queue (tier then approx), then interleave round-robin
            # across seeds so each seed gets a proportional budget share.
            _queues = [sorted(q) for q in _seed_q.values() if q]
            ref_score_order: List[int] = []
            while _queues:
                next_round = []
                for q in _queues:
                    ref_score_order.append(q[0][2])
                    if len(q) > 1:
                        next_round.append(q[1:])
                _queues = next_round

            _remaining_after_pass1 = (
                None if _pre_m3_total is None
                else max(0, _pre_m3_total - pass1_scored_count)
            )
            _pass2_budget = _remaining_after_pass1
            if _pass2_max is not None and _pass2_budget is not None:
                _pass2_budget = min(_pass2_max, _pass2_budget)
            pass2_scored_count = _score_batch(
                ref_scored, ref_score_order, benchmark, plc,
                max_scores=_pass2_budget, already_scored=0,
                cache=score_cache, benchmark_name=benchmark_name,
                timing_records=timing_records, timing_names=timing_names,
                skipped_by_budget_acc=skipped_by_budget_acc,
                scoring_rank_counter=scoring_rank_counter,
            )

            ref_by_name = {rsc.name: rsc for rsc in ref_scored}
            for rsc in ref_scored:
                if rsc.duplicate_of is not None:
                    owner = ref_by_name.get(rsc.duplicate_of) or next(
                        (s for s in scored if s.name == rsc.duplicate_of), None
                    )
                    if owner:
                        rsc.proxy_cost = owner.proxy_cost

            for rsc in ref_scored:
                if rsc.proxy_cost is not None and raw_original_proxy_cost is not None:
                    rsc.delta_vs_original = rsc.proxy_cost - raw_original_proxy_cost

            scored.extend(ref_scored)

    # --- Pass 3: line-search ---
    line_search_candidates_generated = 0
    best_line_search_move = ""
    best_line_search_delta: Optional[float] = None
    pass3_scored_count = 0

    if gen_cfg is not None and gen_cfg.line_search_around_winners:
        from submissions.solver.core.original_line_search import generate_original_line_search_candidates

        ls_seeds = _select_line_search_seeds(scored, gen_cfg.line_search_top_k)

        if ls_seeds:
            ls_existing_names: Set[str] = {c.name for c in candidates}
            ls_existing_names.update(sc.name for sc in scored)

            ls_placements = generate_original_line_search_candidates(
                benchmark, ls_seeds, gen_cfg, ls_existing_names
            )
            line_search_candidates_generated = len(ls_placements)

            if ls_placements:
                ls_scored: List[ScoredCandidate] = [
                    _prepare_candidate(c, benchmark, movable_mask, obstacle_mask, cfg.legalizer_max_rings)
                    for c in ls_placements
                ]

                # Assign generation_rank continuing from previous passes
                _pass3_gen_base = len(scored)
                for local_rank, lsc in enumerate(ls_scored):
                    lsc.metadata["generation_rank"] = _pass3_gen_base + local_rank
                    lsc.metadata["pass_id"] = 3

                pass3_dup_count, hash_map = _mark_duplicates(
                    ls_scored, enable_hash_cache=cfg.enable_hash_cache, existing_hashes=hash_map
                )
                total_dup_count += pass3_dup_count

                _remaining_for_ls = (
                    None if _pre_m3_total is None
                    else max(0, _pre_m3_total - pass1_scored_count - pass2_scored_count)
                )
                pass3_scored_count = _score_line_search_ordered(
                    ls_scored,
                    benchmark=benchmark,
                    plc=plc,
                    max_scores=_remaining_for_ls,
                    already_scored=0,
                    stop_after_worse=gen_cfg.line_search_stop_after_worse,
                    raw_original_proxy_cost=raw_original_proxy_cost,
                    cache=score_cache,
                    benchmark_name=benchmark_name,
                    timing_records=timing_records,
                    timing_names=timing_names,
                    skipped_by_budget_acc=skipped_by_budget_acc,
                    scoring_rank_counter=scoring_rank_counter,
                )

                for lsc in ls_scored:
                    if lsc.proxy_cost is not None and raw_original_proxy_cost is not None:
                        lsc.delta_vs_original = lsc.proxy_cost - raw_original_proxy_cost

                scored.extend(ls_scored)

    # --- Pass 4: M3A pair refinement ---
    # Generates coordinated 2-macro moves from the current M2B winner and adds
    # them to the candidate pool.  Only runs when m3a_pair_refinement=True in
    # generation_config.  All existing M2B invariants (original_raw, m2b winner,
    # official-score selector, budget) are preserved.
    pass4_scored_count = 0
    _m3a_pairs_considered = 0
    _m3a_top_k_pairs = 0
    _m3a_candidates_generated = 0
    _m3a_rejected_bounds = 0
    _m3a_rejected_overlap = 0
    # Fixed-hard macros are excluded before generation (ValueError in generate_pair_candidates
    # rejects any pair where either macro is fixed-hard), so this counter stays 0.
    _m3a_rejected_fixed_hard = 0
    _m3a_rejected_other = 0
    _m3a_skipped_budget = 0
    _m3a_candidates_scored = 0
    _m3a_fresh_scores = 0
    _m3a_cache_hits = 0
    _m3a_best_score: Optional[float] = None
    _m3a_best_delta: Optional[float] = None
    _m3a_valid_count = 0
    _m3a_admitted_count = 0
    _m3a_not_admitted = 0

    if gen_cfg is not None and getattr(gen_cfg, "m3a_pair_refinement", False):
        from submissions.solver.core.m3a_pair_enumeration import enumerate_net_coupled_pairs
        from submissions.solver.core.m3a_candidate_generation import generate_m3a_candidates_for_pairs

        _m3a_top_k_pairs = int(getattr(gen_cfg, "m3a_top_k_pairs", 64))

        # Determine M2B winner from passes 1–3 (before M3A extends the pool).
        # Exclude original_legalized (diagnostic only when raw is valid).
        _diag_only_set: set = {"original_legalized"} if raw_original_valid else set()
        _pre_m3a_valid_scored = [
            s for s in scored
            if s.valid and s.proxy_cost is not None and s.was_scored
            and s.name not in _diag_only_set
        ]

        if _pre_m3a_valid_scored:
            _m3a_order_tmp = {sc.name: idx for idx, sc in enumerate(scored)}
            _m2b_winner = min(
                _pre_m3a_valid_scored,
                key=lambda s: (float(s.proxy_cost), _m3a_order_tmp.get(s.name, len(scored))),
            )
            _m3a_winner_positions = _m2b_winner.positions

            # Enumerate net-coupled pairs from the M2B winner.
            pairs = enumerate_net_coupled_pairs(benchmark, _m3a_top_k_pairs)
            _m3a_pairs_considered = len(pairs)

            # Generate candidates, avoiding names already in the scored pool.
            _m3a_existing_names: set = {sc.name for sc in scored}
            m3a_placements = generate_m3a_candidates_for_pairs(
                benchmark, _m3a_winner_positions, pairs, _m3a_existing_names
            )
            _m3a_candidates_generated = len(m3a_placements)

            # Validate (bypass_legalization=True: no legalizer, validate raw coords).
            m3a_scored_list: List[ScoredCandidate] = [
                _prepare_candidate(c, benchmark, movable_mask, obstacle_mask, cfg.legalizer_max_rings)
                for c in m3a_placements
            ]

            # Classify rejections by failure type for diagnostics.
            for msc in m3a_scored_list:
                if not msc.valid:
                    if msc.num_out_of_bounds > 0 and msc.num_overlaps == 0:
                        _m3a_rejected_bounds += 1
                    elif msc.num_overlaps > 0:
                        _m3a_rejected_overlap += 1
                    else:
                        _m3a_rejected_other += 1

            # Assign generation ranks continuing from previous passes.
            _pass4_gen_base = len(scored)
            for local_rank, msc in enumerate(m3a_scored_list):
                msc.metadata["generation_rank"] = _pass4_gen_base + local_rank
                msc.metadata["pass_id"] = 4

            # Dedup against the full scored pool (including passes 1–3).
            _pass4_dup_count, hash_map = _mark_duplicates(
                m3a_scored_list,
                enable_hash_cache=cfg.enable_hash_cache,
                existing_hashes=hash_map,
            )
            total_dup_count += _pass4_dup_count

            # Compute remaining budget for M3A.
            # When M3C is enabled, use the reserved M3A slice directly (independent of
            # how much of the pre-M3 pool was consumed).  Without M3C, use the
            # existing fallback: whatever remains from the global cap.
            if _m3c_enabled and _m3c_m3a_alloc is not None:
                _m3a_budget: Optional[int] = _m3c_m3a_alloc
            else:
                _already_used = pass1_scored_count + pass2_scored_count + pass3_scored_count
                _m3a_budget_config = getattr(gen_cfg, "m3a_score_budget", None)
                _remaining_global = (
                    None if cfg.max_official_scores is None
                    else max(0, cfg.max_official_scores - _already_used)
                )
                if _m3a_budget_config is not None and _remaining_global is not None:
                    _m3a_budget = min(int(_m3a_budget_config), _remaining_global)
                elif _m3a_budget_config is not None:
                    _m3a_budget = int(_m3a_budget_config)
                else:
                    _m3a_budget = _remaining_global

            # Collect all valid non-duplicate M3A candidates in generation order.
            _m3a_all_valid_indices = [
                idx for idx, msc in enumerate(m3a_scored_list)
                if msc.valid and msc.duplicate_of is None
            ]
            _m3a_valid_count = len(_m3a_all_valid_indices)

            if _m3c_enabled and _m3c_m3a_alloc is not None:
                # Admit only the top m3c_m3a_alloc candidates to the scoring frontier.
                # Candidates outside the frontier are not budget-exhausted — they are simply
                # not admitted. Only within-frontier exhaustion triggers M3A exclusion.
                _m3a_frontier_indices = _m3a_all_valid_indices[:_m3c_m3a_alloc]
                _m3a_outside_frontier = _m3a_all_valid_indices[_m3c_m3a_alloc:]
                _m3a_admitted_count = len(_m3a_frontier_indices)
                _m3a_not_admitted = len(_m3a_outside_frontier)
                for idx in _m3a_outside_frontier:
                    m3a_scored_list[idx].metadata.setdefault("skip_reason", "m3c_not_admitted")
                pass4_scored_count = _score_batch(
                    m3a_scored_list,
                    _m3a_frontier_indices,
                    benchmark,
                    plc,
                    max_scores=_m3c_m3a_alloc,
                    already_scored=0,
                    cache=score_cache,
                    benchmark_name=benchmark_name,
                    timing_records=timing_records,
                    timing_names=timing_names,
                    skipped_by_budget_acc=skipped_by_budget_acc,
                    scoring_rank_counter=scoring_rank_counter,
                )
            else:
                _m3a_admitted_count = _m3a_valid_count
                pass4_scored_count = _score_batch(
                    m3a_scored_list,
                    _m3a_all_valid_indices,
                    benchmark,
                    plc,
                    max_scores=_m3a_budget,
                    already_scored=0,
                    cache=score_cache,
                    benchmark_name=benchmark_name,
                    timing_records=timing_records,
                    timing_names=timing_names,
                    skipped_by_budget_acc=skipped_by_budget_acc,
                    scoring_rank_counter=scoring_rank_counter,
                )

            # Propagate deltas and collect M3A-specific stats.
            for msc in m3a_scored_list:
                if msc.proxy_cost is not None and raw_original_proxy_cost is not None:
                    msc.delta_vs_original = msc.proxy_cost - raw_original_proxy_cost
                if msc.metadata.get("cache_hit"):
                    _m3a_cache_hits += 1
                if msc.was_scored:
                    _m3a_candidates_scored += 1
                if msc.metadata.get("skip_reason") == "budget_exceeded":
                    _m3a_skipped_budget += 1
                if msc.valid and msc.proxy_cost is not None and msc.was_scored:
                    if _m3a_best_score is None or msc.proxy_cost < _m3a_best_score:
                        _m3a_best_score = msc.proxy_cost
                        _m3a_best_delta = msc.delta_vs_original

            _m3a_fresh_scores = pass4_scored_count

            scored.extend(m3a_scored_list)

    # --- Pass 5: M3B cluster refinement ---
    # Generates coordinated 3-macro cluster moves from the current M2B/M3A winner and adds
    # them to the candidate pool.  Only runs when m3b_cluster_refinement=True in
    # generation_config.  All existing M2B/M3A invariants are preserved.
    pass5_scored_count = 0
    _pass5_dup_count = 0
    _m3b_clusters_considered = 0
    _m3b_candidates_generated = 0
    _m3b_valid = 0
    _m3b_invalid = 0
    _m3b_rejected_bounds = 0
    _m3b_rejected_overlap = 0
    _m3b_rejected_other = 0
    _m3b_skipped_budget = 0
    _m3b_candidates_scored = 0
    _m3b_fresh_scores = 0
    _m3b_cache_hits = 0
    _m3b_best_score: Optional[float] = None
    _m3b_best_delta: Optional[float] = None
    _m3b_best_candidate: str = ""
    _m3b_admitted_count = 0
    _m3b_not_admitted = 0

    if gen_cfg is not None and getattr(gen_cfg, "m3b_cluster_refinement", False):
        from submissions.solver.core.m3b_cluster_enumeration import enumerate_net_coupled_triples
        from submissions.solver.core.m3b_candidate_generation import generate_m3b_candidates_for_clusters

        _m3b_top_k_clusters = int(getattr(gen_cfg, "m3b_top_k_clusters", 32))

        # Determine the best winner from passes 1–4 (M2B + M3A) before M3B extends the pool.
        _diag_only_set_m3b: set = {"original_legalized"} if raw_original_valid else set()
        _pre_m3b_valid_scored = [
            s for s in scored
            if s.valid and s.proxy_cost is not None and s.was_scored
            and s.name not in _diag_only_set_m3b
            and not (_m3a_skipped_budget > 0 and s.family == "m3a_pair_refinement")
        ]

        if _pre_m3b_valid_scored:
            _m3b_order_tmp = {sc.name: idx for idx, sc in enumerate(scored)}
            _m3b_winner = min(
                _pre_m3b_valid_scored,
                key=lambda s: (float(s.proxy_cost), _m3b_order_tmp.get(s.name, len(scored))),
            )
            _m3b_winner_positions = _m3b_winner.positions

            # Enumerate net-coupled triples from the best M2B/M3A winner.
            triples = enumerate_net_coupled_triples(benchmark, _m3b_top_k_clusters)
            _m3b_clusters_considered = len(triples)

            # Generate candidates, avoiding names already in the scored pool.
            _m3b_existing_names: set = {sc.name for sc in scored}
            m3b_placements = generate_m3b_candidates_for_clusters(
                benchmark, _m3b_winner_positions, triples, _m3b_existing_names
            )
            _m3b_candidates_generated = len(m3b_placements)

            # Validate (bypass_legalization=True: no legalizer, validate raw coords).
            m3b_scored_list: List[ScoredCandidate] = [
                _prepare_candidate(c, benchmark, movable_mask, obstacle_mask, cfg.legalizer_max_rings)
                for c in m3b_placements
            ]

            # Classify rejections and valid/invalid counts for diagnostics.
            for msc in m3b_scored_list:
                if msc.valid:
                    _m3b_valid += 1
                else:
                    _m3b_invalid += 1
                    if msc.num_out_of_bounds > 0 and msc.num_overlaps == 0:
                        _m3b_rejected_bounds += 1
                    elif msc.num_overlaps > 0:
                        _m3b_rejected_overlap += 1
                    else:
                        _m3b_rejected_other += 1

            # Assign generation ranks continuing from previous passes.
            _pass5_gen_base = len(scored)
            for local_rank, msc in enumerate(m3b_scored_list):
                msc.metadata["generation_rank"] = _pass5_gen_base + local_rank
                msc.metadata["pass_id"] = 5

            # Dedup against the full scored pool (including passes 1–4).
            _pass5_dup_count, hash_map = _mark_duplicates(
                m3b_scored_list,
                enable_hash_cache=cfg.enable_hash_cache,
                existing_hashes=hash_map,
            )
            total_dup_count += _pass5_dup_count

            # Compute remaining budget for M3B.
            # When M3C is enabled, use the reserved M3B slice plus any rollover from the
            # unused M3A allocation.  Without M3C, use the existing fallback.
            if _m3c_enabled and _m3c_m3b_alloc is not None:
                if _m3c_rollover_enabled and _m3c_m3a_alloc is not None:
                    _m3c_rollover_to_m3b = max(0, _m3c_m3a_alloc - pass4_scored_count)
                _m3b_budget: Optional[int] = _m3c_m3b_alloc + _m3c_rollover_to_m3b
            else:
                _already_used_m3b = (
                    pass1_scored_count + pass2_scored_count
                    + pass3_scored_count + pass4_scored_count
                )
                _m3b_budget_config = getattr(gen_cfg, "m3b_score_budget", None)
                _remaining_global_m3b = (
                    None if cfg.max_official_scores is None
                    else max(0, cfg.max_official_scores - _already_used_m3b)
                )
                if _m3b_budget_config is not None and _remaining_global_m3b is not None:
                    _m3b_budget = min(int(_m3b_budget_config), _remaining_global_m3b)
                elif _m3b_budget_config is not None:
                    _m3b_budget = int(_m3b_budget_config)
                else:
                    _m3b_budget = _remaining_global_m3b

            # Collect all valid non-duplicate M3B candidates in generation order.
            _m3b_all_valid_indices = [
                idx for idx, msc in enumerate(m3b_scored_list)
                if msc.valid and msc.duplicate_of is None
            ]

            if _m3c_enabled and _m3c_m3b_alloc is not None and _m3b_budget is not None:
                # Admit only the top _m3b_budget candidates (alloc + rollover) to the frontier.
                # Candidates outside the frontier are not budget-exhausted — they are simply
                # not admitted. Only within-frontier exhaustion triggers M3B exclusion.
                _m3b_frontier_indices = _m3b_all_valid_indices[:_m3b_budget]
                _m3b_outside_frontier = _m3b_all_valid_indices[_m3b_budget:]
                _m3b_admitted_count = len(_m3b_frontier_indices)
                _m3b_not_admitted = len(_m3b_outside_frontier)
                for idx in _m3b_outside_frontier:
                    m3b_scored_list[idx].metadata.setdefault("skip_reason", "m3c_not_admitted")
                pass5_scored_count = _score_batch(
                    m3b_scored_list,
                    _m3b_frontier_indices,
                    benchmark,
                    plc,
                    max_scores=_m3b_budget,
                    already_scored=0,
                    cache=score_cache,
                    benchmark_name=benchmark_name,
                    timing_records=timing_records,
                    timing_names=timing_names,
                    skipped_by_budget_acc=skipped_by_budget_acc,
                    scoring_rank_counter=scoring_rank_counter,
                )
            else:
                _m3b_admitted_count = len(_m3b_all_valid_indices)
                pass5_scored_count = _score_batch(
                    m3b_scored_list,
                    _m3b_all_valid_indices,
                    benchmark,
                    plc,
                    max_scores=_m3b_budget,
                    already_scored=0,
                    cache=score_cache,
                    benchmark_name=benchmark_name,
                    timing_records=timing_records,
                    timing_names=timing_names,
                    skipped_by_budget_acc=skipped_by_budget_acc,
                    scoring_rank_counter=scoring_rank_counter,
                )

            # Propagate deltas and collect M3B-specific stats.
            for msc in m3b_scored_list:
                if msc.proxy_cost is not None and raw_original_proxy_cost is not None:
                    msc.delta_vs_original = msc.proxy_cost - raw_original_proxy_cost
                if msc.metadata.get("cache_hit"):
                    _m3b_cache_hits += 1
                if msc.was_scored:
                    _m3b_candidates_scored += 1
                if msc.metadata.get("skip_reason") == "budget_exceeded":
                    _m3b_skipped_budget += 1
                if msc.valid and msc.proxy_cost is not None and msc.was_scored:
                    if _m3b_best_score is None or msc.proxy_cost < _m3b_best_score:
                        _m3b_best_score = msc.proxy_cost
                        _m3b_best_delta = msc.delta_vs_original
                        _m3b_best_candidate = msc.name

            _m3b_fresh_scores = pass5_scored_count

            scored.extend(m3b_scored_list)

    # --- Pass 6: M4B legalization-aware regional repair ---
    pass6_scored_count = 0
    _m4b_reserved_scores = 0
    _m4b_generated_count = 0
    _m4b_legalized_count = 0
    _m4b_legalization_failed_count = 0
    _m4b_duplicate_after_legalization_count = 0
    _m4b_admitted_count = 0
    _m4b_not_admitted_count = 0
    _m4b_scored_count = 0
    _m4b_skipped_budget = 0
    _m4b_raw_legalized_rate = 0.0
    _m4b_adjusted_legalized_rate = 0.0

    if gen_cfg is not None and getattr(gen_cfg, "m4b_region_repair", False):
        from submissions.solver.core.m4b_region_repair import (
            generate_m4b_region_repair_candidates,
            summarize_m4b_audit_rows,
        )

        _m4b_reserved_scores = max(0, int(getattr(gen_cfg, "m4b_reserved_scores", 20)))

        _diag_only_set_m4b: set = {"original_legalized"} if raw_original_valid else set()
        _pre_m4b_valid_scored = [
            s for s in scored
            if s.valid and s.proxy_cost is not None and s.was_scored
            and s.name not in _diag_only_set_m4b
            and not (_m3a_skipped_budget > 0 and s.family == "m3a_pair_refinement")
            and not (_m3b_skipped_budget > 0 and s.family == "m3b_cluster_refinement")
        ]

        if _pre_m4b_valid_scored:
            _m4b_order_tmp = {sc.name: idx for idx, sc in enumerate(scored)}
            _m4b_base = min(
                _pre_m4b_valid_scored,
                key=lambda s: (float(s.proxy_cost), _m4b_order_tmp.get(s.name, len(scored))),
            )

            _m4b_grid_dims_raw = getattr(gen_cfg, "m4b_grid_dims", (3, 3))
            _m4b_grid_dims = (
                int(_m4b_grid_dims_raw[0]),
                int(_m4b_grid_dims_raw[1]),
            )
            m4b_scored_list, hash_map = generate_m4b_region_repair_candidates(
                benchmark=benchmark,
                base_positions=_m4b_base.positions,
                existing_hashes=hash_map,
                grid_dims=_m4b_grid_dims,
                min_macros_per_region=int(getattr(gen_cfg, "m4b_min_macros_per_region", 2)),
                max_combos_per_region=int(getattr(gen_cfg, "m4b_max_combos_per_region", 16)),
                legalization_max_displacement_um=float(
                    getattr(gen_cfg, "m4b_legalization_max_displacement_um", 200.0)
                ),
                perturbation_fraction=float(getattr(gen_cfg, "m4b_perturbation_fraction", 0.5)),
                legalizer_max_rings=cfg.legalizer_max_rings,
            )

            _pass6_gen_base = len(scored)
            for local_rank, msc in enumerate(m4b_scored_list):
                msc.metadata["generation_rank"] = _pass6_gen_base + local_rank
                msc.metadata["pass_id"] = 6

            m4b_summary = summarize_m4b_audit_rows(m4b_scored_list)
            _m4b_generated_count = int(m4b_summary["generated_count"])
            _m4b_legalized_count = int(m4b_summary["legalized_count"])
            _m4b_legalization_failed_count = int(m4b_summary["legalization_failed_count"])
            _m4b_duplicate_after_legalization_count = int(
                m4b_summary["duplicate_after_legalization_count"]
            )
            _m4b_raw_legalized_rate = float(m4b_summary["raw_legalized_rate"])
            _m4b_adjusted_legalized_rate = float(m4b_summary["adjusted_legalized_rate"])
            total_dup_count += _m4b_duplicate_after_legalization_count

            _m4b_all_valid_indices = [
                idx for idx, msc in enumerate(m4b_scored_list)
                if msc.valid and msc.duplicate_of is None
            ]
            if getattr(gen_cfg, "m4c_ranking", False):
                from submissions.solver.core.m4c_ranking import assign_buckets

                _m4c_rows = []
                for idx, msc in enumerate(m4b_scored_list):
                    _m4c_rows.append(
                        {
                            "candidate_name": msc.name,
                            "family": msc.family,
                            "valid": msc.valid,
                            "duplicate": msc.duplicate_of is not None,
                            "post_legalization_approx_delta": msc.metadata.get(
                                "post_legalization_approx_delta"
                            ),
                            "fifo_index": idx,
                            "generation_rank": msc.metadata.get("generation_rank", idx),
                            "region_id": msc.metadata.get("region_id"),
                            "move_type": msc.metadata.get("move_type"),
                        }
                    )
                assign_buckets(
                    _m4c_rows,
                    k_ranked=int(getattr(gen_cfg, "m4c_k_ranked", 16)),
                    exploration=int(getattr(gen_cfg, "m4c_exploration", 4)),
                    max_per_region=getattr(gen_cfg, "m4c_max_per_region", None),
                    known_winners=list(getattr(gen_cfg, "m4c_known_winners", []) or []),
                )
                for idx, row in enumerate(_m4c_rows):
                    for key in (
                        "m4c_rank_score",
                        "m4c_rank_bucket",
                        "m4c_rank_reason",
                        "family_rank",
                        "family_normalized_approx_delta",
                    ):
                        m4b_scored_list[idx].metadata[key] = row.get(key)

                _m4b_ranked_indices = [
                    idx
                    for idx in _m4b_all_valid_indices
                    if m4b_scored_list[idx].metadata.get("m4c_rank_bucket") == "ranked"
                ]
                _m4b_exploration_indices = [
                    idx
                    for idx in _m4b_all_valid_indices
                    if m4b_scored_list[idx].metadata.get("m4c_rank_bucket") == "exploration"
                ]
                _m4b_ranked_indices.sort(
                    key=lambda idx: (
                        m4b_scored_list[idx].metadata.get("family_rank") or 10**9,
                        m4b_scored_list[idx].metadata.get("generation_rank", idx),
                        m4b_scored_list[idx].name,
                    )
                )
                _m4b_exploration_indices.sort(
                    key=lambda idx: (
                        m4b_scored_list[idx].metadata.get("generation_rank", idx),
                        m4b_scored_list[idx].name,
                    )
                )
                _m4b_frontier_indices = _m4b_ranked_indices + _m4b_exploration_indices
                _m4b_outside_frontier = [
                    idx
                    for idx in _m4b_all_valid_indices
                    if idx not in set(_m4b_frontier_indices)
                ]
                for idx in _m4b_outside_frontier:
                    m4b_scored_list[idx].metadata.setdefault(
                        "skip_reason", "m4c_budget_exhausted"
                    )
            else:
                _m4b_frontier_indices = _m4b_all_valid_indices[:_m4b_reserved_scores]
                _m4b_outside_frontier = _m4b_all_valid_indices[_m4b_reserved_scores:]
                for idx in _m4b_outside_frontier:
                    m4b_scored_list[idx].metadata.setdefault("skip_reason", "m4b_budget_exhausted")
            _m4b_admitted_count = len(_m4b_frontier_indices)
            _m4b_not_admitted_count = len(_m4b_outside_frontier)

            pass6_scored_count = _score_batch(
                m4b_scored_list,
                _m4b_frontier_indices,
                benchmark,
                plc,
                max_scores=_m4b_reserved_scores,
                already_scored=0,
                cache=score_cache,
                benchmark_name=benchmark_name,
                timing_records=timing_records,
                timing_names=timing_names,
                skipped_by_budget_acc=skipped_by_budget_acc,
                scoring_rank_counter=scoring_rank_counter,
            )

            for msc in m4b_scored_list:
                if msc.proxy_cost is not None and raw_original_proxy_cost is not None:
                    msc.delta_vs_original = msc.proxy_cost - raw_original_proxy_cost
                if msc.was_scored:
                    _m4b_scored_count += 1
                if msc.metadata.get("skip_reason") in {
                    "budget_exceeded",
                    "m4b_budget_exhausted",
                    "m4c_budget_exhausted",
                }:
                    _m4b_skipped_budget += 1

            scored.extend(m4b_scored_list)

    candidates_officially_scored = (
        pass1_scored_count + pass2_scored_count + pass3_scored_count
        + pass4_scored_count + pass5_scored_count + pass6_scored_count
    )
    fresh_official_scores = candidates_officially_scored  # fresh only (cache hits excluded)

    # --- M3C budget invariant ---
    _m3c_pre_m3_fresh_total = pass1_scored_count + pass2_scored_count + pass3_scored_count
    if not _m3c_enabled or cfg.max_official_scores is None:
        _m3c_budget_invariant_holds = True
    else:
        _m3c_budget_invariant_holds = (
            fresh_official_scores <= cfg.max_official_scores
            and (
                _m3c_pre_m3_alloc is None
                or _m3c_pre_m3_fresh_total <= _m3c_pre_m3_alloc
            )
        )

    # --- Post-process: ensure all unscored candidates have a skip_reason ---
    for sc in scored:
        if sc.was_scored:
            sc.metadata.setdefault("skip_reason", "scored")
            continue
        if sc.metadata.get("skip_reason") is not None:
            continue
        if not sc.valid:
            sc.metadata["skip_reason"] = "invalid"
        elif sc.duplicate_of is not None:
            sc.metadata["skip_reason"] = "duplicate"
        else:
            sc.metadata["skip_reason"] = "not_scored"

    # --- Compute family-level best stats ---
    for sc in scored:
        if not sc.valid or sc.proxy_cost is None or not sc.was_scored:
            continue
        delta = sc.delta_vs_original
        if sc.family == "original_refinement":
            rtype = sc.metadata.get("refinement_type", "")
            if "combo" in rtype:
                if best_combo_move == "" or (delta is not None and (best_combo_delta is None or delta < best_combo_delta)):
                    best_combo_move = sc.name
                    best_combo_delta = delta
            else:
                if best_single_macro_move == "" or (delta is not None and (best_single_macro_delta is None or delta < best_single_macro_delta)):
                    best_single_macro_move = sc.name
                    best_single_macro_delta = delta
        elif sc.family == "original_line_search":
            if best_line_search_move == "" or (delta is not None and (best_line_search_delta is None or delta < best_line_search_delta)):
                best_line_search_move = sc.name
                best_line_search_delta = delta

    # --- Compute timing stats ---
    _times = np.array(timing_records) if timing_records else np.array([], dtype=np.float64)
    scorer_time_total = float(_times.sum()) if _times.size else 0.0
    scorer_time_avg = float(_times.mean()) if _times.size else 0.0
    scorer_time_p50 = float(np.percentile(_times, 50)) if _times.size else 0.0
    scorer_time_p95 = float(np.percentile(_times, 95)) if _times.size else 0.0
    scorer_time_max = float(_times.max()) if _times.size else 0.0
    slowest_candidate = timing_names[int(_times.argmax())] if _times.size else ""

    # --- Selection ---
    # If any M3A candidate was skipped due to budget exhaustion the M3A pass is
    # incomplete: a partially-scored slice is not a fair representative of the
    # full M3A pool.  Exclude every m3a_pair_refinement candidate from selection
    # and fall back to the pre-M3A (M2B) pool, which is always complete.
    _m3a_budget_exhausted = _m3a_skipped_budget > 0
    _m3b_budget_exhausted = _m3b_skipped_budget > 0
    diagnostic_only = {"original_legalized"} if raw_original_valid else set()
    valid_scored = [
        s for s in scored
        if s.valid and s.proxy_cost is not None and s.was_scored and s.name not in diagnostic_only
        and not (_m3a_budget_exhausted and s.family == "m3a_pair_refinement")
        and not (_m3b_budget_exhausted and s.family == "m3b_cluster_refinement")
    ]

    unique_costs = {round(float(s.proxy_cost), 9) for s in valid_scored}
    num_unique_scores = len(unique_costs)
    score_is_degenerate = num_unique_scores <= 1
    scoring_available = scoring_mode != "unavailable"

    order = {sc.name: idx for idx, sc in enumerate(scored)}
    valid_sorted = sorted(valid_scored, key=lambda s: (float(s.proxy_cost), order[s.name]))
    diagnostic_scored = [s for s in scored if s.name in diagnostic_only]
    ranked_names = {s.name for s in valid_sorted}.union(diagnostic_only)
    non_diagnostic_invalid = [s for s in scored if s.name not in ranked_names]
    non_diagnostic_invalid.sort(key=lambda s: s.name)
    ranked = valid_sorted + diagnostic_scored + non_diagnostic_invalid

    if not valid_scored:
        # No valid scored candidate.  Critical invariant: NEVER select an invalid
        # candidate (in particular, never select an invalid original_raw — that would
        # emit a placement that overlaps fixed obstacles or violates bounds).
        # Preference order: legalized → raw → any other valid candidate → sentinel.
        fallback_best = None
        fallback_reason = None
        if leg_sc is not None and leg_sc.valid:
            fallback_best = leg_sc
            fallback_reason = "fallback_legalized_original"
        elif raw_sc is not None and raw_sc.valid:
            fallback_best = raw_sc
            fallback_reason = "fallback_original"
        else:
            other_valid = sorted(
                (s for s in scored if s.valid and s.name not in {"original_raw", "original_legalized"}),
                key=lambda s: s.name,
            )
            if other_valid:
                fallback_best = other_valid[0]
                fallback_reason = "fallback_other_valid"
        if fallback_best is not None:
            best = fallback_best
            selected_due_to = fallback_reason
        else:
            # No valid candidate at all.  Return raw_sc (or first scored) as a sentinel
            # marked invalid via best.valid=False; placer/caller must refuse to emit it.
            best = raw_sc if raw_sc is not None else (leg_sc if leg_sc is not None else (scored[0] if scored else None))
            selected_due_to = "no_valid_scored_candidate"
    elif score_is_degenerate:
        if raw_sc is not None and raw_sc.valid and raw_sc.was_scored and raw_sc.name not in diagnostic_only:
            best = raw_sc
        else:
            best = valid_sorted[0]
        selected_due_to = "validity_only" if scoring_mode == "unavailable" else "tie_break"
    else:
        best = valid_sorted[0]
        selected_due_to = "proxy_cost"

    best_cost = best.proxy_cost if best is not None else None
    delta_vs_raw = (
        best_cost - raw_original_proxy_cost
        if best_cost is not None and raw_original_proxy_cost is not None
        else None
    )
    invariant_holds = (
        best_cost is not None
        and raw_original_proxy_cost is not None
        and best_cost <= raw_original_proxy_cost + 1e-9
    )
    # No-valid-candidate sentinel case must NEVER be reported as success.
    if selected_due_to == "no_valid_scored_candidate":
        invariant_holds = False

    # --- Candidate-admission audit ---
    _admission_prelegal_overlap = sum(
        1 for sc in scored if sc.metadata.get("prelegal_valid") is False
    )
    _admission_leg_success = sum(
        1 for sc in scored
        if sc.valid and sc.metadata.get("legalization_num_moved", 0) > 0
    )
    _admission_leg_failed = sum(
        1 for sc in scored
        if not sc.valid and sc.metadata.get("postlegal_valid") is False
        and sc.metadata.get("legalization_num_moved") is not None
    )

    diagnostics = ScoringDiagnostics(
        scoring_available=scoring_available,
        scoring_mode=scoring_mode,
        score_is_degenerate=score_is_degenerate,
        num_unique_scores=num_unique_scores,
        selected_due_to=selected_due_to,
        raw_original_valid=raw_original_valid,
        raw_original_proxy_cost=raw_original_proxy_cost,
        delta_vs_raw_original=delta_vs_raw,
        best_proxy_cost=best_cost,
        winning_candidate=best.name if best is not None else "",
        winning_family=best.family if best is not None else "",
        invariant_holds=invariant_holds,
        candidates_generated=len(candidates),
        candidates_prefiltered=prefiltered_count,
        candidates_officially_scored=candidates_officially_scored,
        duplicate_count=total_dup_count,
        prefilter_mode=cfg.prefilter_mode,
        refinement_candidates_generated=refinement_candidates_generated,
        combo_candidates_generated=combo_candidates_generated,
        best_single_macro_move=best_single_macro_move,
        best_single_macro_delta=best_single_macro_delta,
        best_combo_move=best_combo_move,
        best_combo_delta=best_combo_delta,
        prefilter_improving_count=improving_count,
        prefilter_best_skipped_hpwl_delta=best_skipped_delta,
        exploratory_count=min(cfg.exploratory_score_count, len(
            [sc for sc in scored if sc.family == "original_neighborhood" and sc.was_scored and
             sc.metadata.get("approx_hpwl_delta") is not None and
             isinstance(sc.metadata.get("approx_hpwl_delta"), float) and
             sc.metadata["approx_hpwl_delta"] > 1e-9]
        )),
        line_search_candidates_generated=line_search_candidates_generated,
        best_line_search_move=best_line_search_move,
        best_line_search_delta=best_line_search_delta,
        cache_hits=score_cache.hits - _cache_hits_before,
        cache_misses=score_cache.misses - _cache_misses_before,
        official_scorer_time_ms_total=scorer_time_total,
        official_scorer_time_ms_avg=scorer_time_avg,
        official_scorer_time_ms_p50=scorer_time_p50,
        official_scorer_time_ms_p95=scorer_time_p95,
        official_scorer_time_ms_max=scorer_time_max,
        slowest_candidate=slowest_candidate,
        candidates_skipped_by_budget=skipped_by_budget_acc[0],
        fresh_official_scores=fresh_official_scores,
        admission_prelegal_overlap_candidates=_admission_prelegal_overlap,
        admission_legalized_successfully=_admission_leg_success,
        admission_legalization_failed=_admission_leg_failed,
        refinement_seed_bucket_diagnostics=_seed_bucket_diag,
        m3a_pairs_considered=_m3a_pairs_considered,
        m3a_top_k_pairs=_m3a_top_k_pairs,
        m3a_candidates_generated=_m3a_candidates_generated,
        m3a_rejected_bounds=_m3a_rejected_bounds,
        m3a_rejected_overlap=_m3a_rejected_overlap,
        m3a_rejected_fixed_hard=_m3a_rejected_fixed_hard,
        m3a_rejected_other=_m3a_rejected_other,
        m3a_skipped_budget=_m3a_skipped_budget,
        m3a_candidates_scored=_m3a_candidates_scored,
        m3a_fresh_scores=_m3a_fresh_scores,
        m3a_cache_hits=_m3a_cache_hits,
        m3a_best_score=_m3a_best_score,
        m3a_best_delta=_m3a_best_delta,
        m3a_winner_source=(
            "m3a_pair_refinement"
            if best is not None and best.family == "m3a_pair_refinement"
            else ("original_raw" if best is not None and best.name == "original_raw" else "m2b_final")
        ) if best is not None else "",
        m3a_valid_count=_m3a_valid_count,
        m3a_admitted_count=_m3a_admitted_count,
        m3a_not_admitted_count=_m3a_not_admitted,
        m3a_selectable=(
            not _m3a_budget_exhausted and any(
                s.family == "m3a_pair_refinement" and s.valid and s.was_scored and s.proxy_cost is not None
                for s in scored
            )
        ),
        m3b_clusters_considered=_m3b_clusters_considered,
        m3b_candidates_generated=_m3b_candidates_generated,
        m3b_valid=_m3b_valid,
        m3b_invalid=_m3b_invalid,
        m3b_duplicates=_pass5_dup_count if gen_cfg is not None and getattr(gen_cfg, "m3b_cluster_refinement", False) else 0,
        m3b_scored=_m3b_candidates_scored,
        m3b_skipped_budget=_m3b_skipped_budget,
        m3b_budget_exhausted=_m3b_budget_exhausted,
        m3b_selectable=(
            sum(
                1 for s in scored
                if s.family == "m3b_cluster_refinement"
                and s.valid and s.was_scored and s.proxy_cost is not None
            ) if not _m3b_budget_exhausted else 0
        ),
        m3b_best_candidate=_m3b_best_candidate,
        m3b_best_delta=_m3b_best_delta,
        m3b_rejected_bounds=_m3b_rejected_bounds,
        m3b_rejected_overlap=_m3b_rejected_overlap,
        m3b_rejected_other=_m3b_rejected_other,
        m3b_fresh_scores=_m3b_fresh_scores,
        m3b_cache_hits=_m3b_cache_hits,
        m3b_best_score=_m3b_best_score,
        m3b_admitted_count=_m3b_admitted_count,
        m3b_not_admitted_count=_m3b_not_admitted,
        m3c_enabled=_m3c_enabled,
        m3c_pre_m3_budget_alloc=_m3c_pre_m3_alloc,
        m3c_m3a_budget_alloc=_m3c_m3a_alloc,
        m3c_m3b_budget_alloc=_m3c_m3b_alloc,
        m3c_pre_m3_used=_m3c_pre_m3_fresh_total,
        m3c_m3a_used=pass4_scored_count,
        m3c_m3b_used=pass5_scored_count,
        m3c_rollover_to_m3b=_m3c_rollover_to_m3b,
        m3c_budget_invariant_holds=_m3c_budget_invariant_holds,
        m4b_reserved_scores=_m4b_reserved_scores,
        m4b_generated_count=_m4b_generated_count,
        m4b_legalized_count=_m4b_legalized_count,
        m4b_legalization_failed_count=_m4b_legalization_failed_count,
        m4b_duplicate_after_legalization_count=_m4b_duplicate_after_legalization_count,
        m4b_admitted_count=_m4b_admitted_count,
        m4b_not_admitted_count=_m4b_not_admitted_count,
        m4b_scored_count=_m4b_scored_count,
        m4b_skipped_budget=_m4b_skipped_budget,
        m4b_raw_legalized_rate=_m4b_raw_legalized_rate,
        m4b_adjusted_legalized_rate=_m4b_adjusted_legalized_rate,
    )
    return best, ranked, diagnostics
