"""
Benchmark runner with profiles for M2B.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig  # noqa: E402
from submissions.solver.core.io import save_csv, save_json  # noqa: E402


_PROFILES: Dict[str, Dict] = {
    "smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console"],
        "description": "Quick sanity check - 3 small benchmarks",
    },
    "standard": {
        "benchmarks": None,
        "repeat": 1,
        "output": ["console", "csv"],
        "description": "All public benchmarks, one run",
    },
    "heavy": {
        "benchmarks": None,
        "repeat": 3,
        "output": ["console", "csv", "json"],
        "description": "All benchmarks x3 repeats",
    },
    "m2b-prep": {
        "benchmarks": None,
        "repeat": 1,
        "output": ["console", "csv", "json"],
        "description": "Full M2B candidate pipeline, ranked output",
        "show_candidates": True,
    },
    "audit": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console"],
        "description": "Candidate diversity + connectivity diagnostics for ibm01/02/03",
        "show_candidates": True,
        "show_audit": True,
    },
    "official-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "Basic official scoring sanity check - not the final submission config",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
    },
    "m2b-final": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M2B final submission profile: diverse seeds, bounded budget, cold-run validated",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
    },
    "m2b-final-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M2B final cold profile: priority line-search + refinement, budget-aware",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "candidate_budget": 100,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 80,
    },
    "tuning": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "Tuning profile with refinement + line-search",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "candidate_budget": 200,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 10,
        "refinement_combo_size": 2,
        "line_search_around_winners": True,
        "line_search_top_k": 5,
        "line_search_max_scale": 6.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 150,
    },
    "stress": {
        "benchmarks": None,
        "repeat": 5,
        "output": ["json"],
        "description": "Stress test - all benchmarks x5",
    },
    # --- M3A profiles ---
    "m3a-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3A smoke: tiny pair count, fast CI validation",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 16,
        "m3a_score_budget": None,
    },
    "m3a-default": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3A default: standard pair refinement, cold-run comparison vs m2b-final",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3a_score_budget": None,
    },
    "m3a-budget-stress": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3A budget-stress: reduced budget forces graceful fallback to m2b-final",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 5,   # very tight: most M3A candidates will be skipped
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3a_score_budget": None,
    },
    # --- M3B profiles ---
    "m3b-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3B smoke: small cluster count, fast CI sanity validation",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 16,
        "m3a_score_budget": None,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 8,
        "m3b_score_budget": None,
    },
    "m3b-default": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3B default: standard cluster refinement, cold-run comparison vs m3a-default / m2b-final",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3a_score_budget": None,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3b_score_budget": None,
    },
    "m3b-budget-stress": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3B budget-stress: intentionally reduced budget, verifies no partial M3B candidate wins",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 5,   # very tight: most M3B (and M3A) candidates will be skipped
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3a_score_budget": None,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3b_score_budget": None,
    },
    # --- M3C profiles ---
    "m3c-smoke": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3C smoke: deterministic budget allocation, fast CI sanity validation",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 8,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 4,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 50,
        "m3c_m3a_reserved_budget": 5,
        "m3c_m3b_reserved_budget": 5,
        "m3c_rollover_unused_budget": True,
    },
    "m3c-default": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3C default: reserved M3A/M3B slices within 60-score budget, cold-run vs m2b-final",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 50,
        "m3c_m3a_reserved_budget": 5,
        "m3c_m3b_reserved_budget": 5,
        "m3c_rollover_unused_budget": True,
    },
    "m3c-budget-stress": {
        "benchmarks": ["ibm01", "ibm02", "ibm03"],
        "repeat": 1,
        "output": ["console", "json"],
        "description": "M3C budget-stress: tiny budget forces graceful fallback, verifies partial-slice exclusion",
        "show_candidates": True,
        "show_audit": True,
        "require_official": True,
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_combo_size": 2,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 10,   # very tight: pre_m3=8, m3a=1, m3b=1
        "m3a_pair_refinement": True,
        "m3a_top_k_pairs": 64,
        "m3b_cluster_refinement": True,
        "m3b_top_k_clusters": 32,
        "m3c_budget_allocation": True,
        "m3c_pre_m3_budget": 8,
        "m3c_m3a_reserved_budget": 1,
        "m3c_m3b_reserved_budget": 1,
        "m3c_rollover_unused_budget": True,
    },
}


def _discover_benchmarks(names: Optional[List[str]]) -> List[Path]:
    if names is not None:
        paths = []
        for name in names:
            p = BENCHMARKS_PT_DIR / f"{name}.pt"
            if p.exists():
                paths.append(p)
            else:
                print(f"  Warning: benchmark '{name}' not found at {p}")
        return paths

    if not BENCHMARKS_PT_DIR.exists():
        print(f"  Warning: benchmark dir not found: {BENCHMARKS_PT_DIR}")
        return []
    return sorted(BENCHMARKS_PT_DIR.glob("*.pt"))


def _get_benchmark_class():
    try:
        from macro_place.benchmark import Benchmark
        return Benchmark
    except ImportError:
        pass

    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "macro_place.benchmark",
        Path(__file__).resolve().parent.parent.parent.parent / "macro_place" / "benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    if "macro_place" not in sys.modules:
        sys.modules["macro_place"] = types.ModuleType("macro_place")
    sys.modules["macro_place.benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod.Benchmark


def _load_benchmark(pt_path: Path, require_official: bool = False):
    name = pt_path.stem
    if require_official:
        ibm_dir = IBM_TESTCASES_DIR / name
        if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
            try:
                from macro_place.loader import load_benchmark_from_dir

                print(f"  Loading from {ibm_dir} (official plc)")
                return load_benchmark_from_dir(ibm_dir.as_posix())
            except Exception as exc:
                print(f"  Failed to load official plc for {name}: {exc}")
        else:
            print(f"  IBM testcases not found for '{name}' - cannot use official scoring")

    try:
        Benchmark = _get_benchmark_class()
        return Benchmark.load(str(pt_path)), None
    except Exception as exc:
        print(f"  Error loading {pt_path.name}: {exc}")
        return None, None


def _candidate_diversity(ranked, benchmark) -> Dict:
    import torch
    from submissions.solver.core.candidate_scoring import placement_hash

    ref_pos = benchmark.macro_positions.float()
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    all_hashes = []
    families = set()
    displacements = []

    for sc in ranked:
        h = placement_hash(sc.positions)
        all_hashes.append((sc.name, h))
        families.add(sc.family)
        if movable_mask.any():
            disp = torch.norm(sc.positions - ref_pos, dim=1)[movable_mask]
            displacements.append((sc.name, float(disp.mean()), float(disp.max())))

    unique_hashes = {h for _, h in all_hashes}
    avg_disp_mean = sum(d[1] for d in displacements) / max(len(displacements), 1)
    avg_disp_max = max((d[2] for d in displacements), default=0.0)
    return {
        "num_candidates": len(ranked),
        "num_valid": sum(1 for s in ranked if s.valid),
        "num_invalid": sum(1 for s in ranked if not s.valid),
        "families": sorted(families),
        "num_distinct_placements": len(unique_hashes),
        "all_candidates_collapse_to_same": len(unique_hashes) == 1,
        "avg_displacement_mean_um": round(avg_disp_mean, 2),
        "max_displacement_max_um": round(avg_disp_max, 2),
        "candidate_hashes": {name: h for name, h in all_hashes},
    }


def _run_one(
    benchmark,
    plc,
    generation_config: CandidateGenerationConfig,
    scoring_config: CandidateScoringConfig,
    show_candidates: bool = False,
    show_audit: bool = False,
    show_admission_audit: bool = False,
    score_cache=None,
) -> Dict:
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import connectivity_audit, score_and_select

    t0 = time.perf_counter()
    candidates = generate_candidates(benchmark, config=generation_config)
    best, ranked, diag = score_and_select(
        candidates, benchmark, plc=plc,
        scoring_config=scoring_config,
        generation_config=generation_config,
        score_cache=score_cache,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000

    family_counts: Dict[str, int] = {}
    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1

    row = {
        "benchmark": benchmark.name,
        "valid": best.valid if best else False,
        "best_candidate": best.name if best else "none",
        "best_family": best.family if best else "none",
        "proxy_cost": best.proxy_cost if best else None,
        "raw_original_cost": diag.raw_original_proxy_cost,
        "raw_original_valid": diag.raw_original_valid,
        "delta_vs_raw_original": diag.delta_vs_raw_original,
        "num_candidates": len(ranked),
        "num_valid": sum(1 for s in ranked if s.valid),
        "runtime_ms": round(runtime_ms, 1),
        "overlaps": best.num_overlaps if best else -1,
        "oob": best.num_out_of_bounds if best else -1,
        "scoring_available": diag.scoring_available,
        "scoring_mode": diag.scoring_mode,
        "score_is_degenerate": diag.score_is_degenerate,
        "num_unique_scores": diag.num_unique_scores,
        "selected_due_to": diag.selected_due_to,
        "max_official_scores": scoring_config.max_official_scores,
        "official_scored_count": diag.candidates_officially_scored,
        "fresh_official_scores": diag.fresh_official_scores,
        "duplicate_skipped_count": diag.duplicate_count,
        "prefiltered_count": diag.candidates_prefiltered,
        "prefilter_mode": diag.prefilter_mode,
        "invariant_holds": diag.invariant_holds,
        "family_counts": family_counts,
        "refinement_candidates_generated": diag.refinement_candidates_generated,
        "combo_candidates_generated": diag.combo_candidates_generated,
        "best_single_macro_move": diag.best_single_macro_move,
        "best_single_macro_delta": diag.best_single_macro_delta,
        "best_combo_move": diag.best_combo_move,
        "best_combo_delta": diag.best_combo_delta,
        "prefilter_improving_count": diag.prefilter_improving_count,
        "prefilter_best_skipped_hpwl_delta": diag.prefilter_best_skipped_hpwl_delta,
        "line_search_candidates_generated": diag.line_search_candidates_generated,
        "best_line_search_move": diag.best_line_search_move,
        "best_line_search_delta": diag.best_line_search_delta,
        "cache_hits": diag.cache_hits,
        "cache_misses": diag.cache_misses,
        "official_scorer_time_ms_total": diag.official_scorer_time_ms_total,
        "official_scorer_time_ms_avg": diag.official_scorer_time_ms_avg,
        "official_scorer_time_ms_p95": diag.official_scorer_time_ms_p95,
        "official_scorer_time_ms_max": diag.official_scorer_time_ms_max,
        "slowest_candidate": diag.slowest_candidate,
        "candidates_skipped_by_budget": diag.candidates_skipped_by_budget,
        "admission_prelegal_overlap_candidates": diag.admission_prelegal_overlap_candidates,
        "admission_legalized_successfully": diag.admission_legalized_successfully,
        "admission_legalization_failed": diag.admission_legalization_failed,
        "m3a_valid_count": diag.m3a_valid_count,
        "m3a_admitted_count": diag.m3a_admitted_count,
        "m3a_not_admitted_count": diag.m3a_not_admitted_count,
        "m3a_candidates_scored": diag.m3a_candidates_scored,
        "m3a_skipped_budget": diag.m3a_skipped_budget,
        "m3a_selectable": diag.m3a_selectable,
        "m3b_valid": diag.m3b_valid,
        "m3b_invalid": diag.m3b_invalid,
        "m3b_duplicates": diag.m3b_duplicates,
        "m3b_scored": diag.m3b_scored,
        "m3b_skipped_budget": diag.m3b_skipped_budget,
        "m3b_budget_exhausted": diag.m3b_budget_exhausted,
        "m3b_admitted_count": diag.m3b_admitted_count,
        "m3b_not_admitted_count": diag.m3b_not_admitted_count,
        "m3b_selectable": diag.m3b_selectable,
        "m3c_enabled": diag.m3c_enabled,
        "m3c_pre_m3_budget_alloc": diag.m3c_pre_m3_budget_alloc,
        "m3c_m3a_budget_alloc": diag.m3c_m3a_budget_alloc,
        "m3c_m3b_budget_alloc": diag.m3c_m3b_budget_alloc,
        "m3c_pre_m3_used": diag.m3c_pre_m3_used,
        "m3c_m3a_used": diag.m3c_m3a_used,
        "m3c_m3b_used": diag.m3c_m3b_used,
        "m3c_rollover_to_m3b": diag.m3c_rollover_to_m3b,
        "m3c_budget_invariant_holds": diag.m3c_budget_invariant_holds,
    }

    if show_candidates:
        row["candidates"] = [
            {
                "rank": k + 1,
                "name": s.name,
                "family": s.family,
                "valid": s.valid,
                "no_op": s.no_op,
                "proxy_cost": round(s.proxy_cost, 6) if s.proxy_cost is not None else None,
                "delta_vs_raw_original": round(s.delta_vs_original, 6) if s.delta_vs_original is not None else None,
                "num_moved": s.num_moved,
                "runtime_ms": round(s.total_ms, 1),
                "was_scored": s.was_scored,
                "duplicate_of": s.duplicate_of,
                "moved_macro_id": s.metadata.get("moved_macro_id"),
                "dx": round(float(s.metadata.get("dx", 0.0)), 4) if "dx" in s.metadata else None,
                "dy": round(float(s.metadata.get("dy", 0.0)), 4) if "dy" in s.metadata else None,
                "approx_hpwl_delta": (
                    round(float(s.metadata.get("approx_hpwl_delta")), 6)
                    if s.metadata.get("approx_hpwl_delta") is not None else None
                ),
            }
            for k, s in enumerate(ranked)
        ]

    if show_audit:
        row["connectivity"] = connectivity_audit(benchmark)
        row["diversity"] = _candidate_diversity(ranked, benchmark)

    if show_admission_audit:
        hood_gen = sum(1 for s in ranked if s.family == "original_neighborhood")
        ref_gen = sum(1 for s in ranked if s.family == "original_refinement")
        ls_gen = sum(1 for s in ranked if s.family == "original_line_search")
        row["admission_audit"] = {
            "generated_neighborhood": hood_gen,
            "generated_refinement": ref_gen,
            "generated_line_search": ls_gen,
            "prelegal_overlap_allowed": diag.admission_prelegal_overlap_candidates,
            "legalized_successfully": diag.admission_legalized_successfully,
            "legalization_failed": diag.admission_legalization_failed,
            "duplicate_hash_skipped": diag.duplicate_count,
            "skipped_due_to_budget": diag.candidates_skipped_by_budget,
        }

    return row


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        print("  (no results)")
        return
    header = (
        f"{'Benchmark':<25} {'Valid':>5} {'Best':>28} {'Family':>20} {'Cost':>8} "
        f"{'Orig':>8} {'Delta':>8} {'Score#':>7} {'Dup':>5} {'Pref':>5} {'ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        cost = f"{row['proxy_cost']:.4f}" if row["proxy_cost"] is not None else "N/A"
        orig = f"{row['raw_original_cost']:.4f}" if row.get("raw_original_cost") is not None else "N/A"
        delta = f"{row['delta_vs_raw_original']:+.4f}" if row.get("delta_vs_raw_original") is not None else "N/A"
        print(
            f"{row['benchmark']:<25} {str(row['valid']):>5} {row['best_candidate']:>28} "
            f"{row['best_family']:>20} {cost:>8} {orig:>8} {delta:>8} "
            f"{row['official_scored_count']:>7} {row['duplicate_skipped_count']:>5} "
            f"{row['prefiltered_count']:>5} {row['runtime_ms']:>7.1f}"
        )


def _print_audit(row: Dict) -> None:
    print(f"\n  --- Audit: {row['benchmark']} ---")
    conn = row.get("connectivity")
    if conn:
        print("  Connectivity:")
        print(f"    num_macros={conn['num_macros']}  num_nets={conn['num_nets']}")
        print(f"    num_net_edges={conn['num_net_edges']}  macros_with_degree>0={conn['num_macros_with_degree_gt_0']}")
        print(f"    num_fixed_endpoints={conn['num_fixed_endpoints']}")
        print(f"    spectral_available={conn['spectral_available']}  terminal_anchor_available={conn['terminal_anchor_available']}")

    div = row.get("diversity")
    if div:
        print("  Candidate Diversity:")
        print(f"    total={div['num_candidates']}  valid={div['num_valid']}  invalid={div['num_invalid']}")
        print(f"    families={div['families']}")
        print(f"    family_counts={row.get('family_counts', {})}")
        print(f"    distinct_placements={div['num_distinct_placements']}  all_collapse={div['all_candidates_collapse_to_same']}")
        print(f"    avg_displacement={div['avg_displacement_mean_um']} um  max_displacement={div['max_displacement_max_um']} um")

    print(
        f"  Scoring: mode={row['scoring_mode']}  selected_due_to={row['selected_due_to']}  "
        f"scores={row['official_scored_count']}  duplicates={row['duplicate_skipped_count']}  "
        f"prefiltered={row['prefiltered_count']}  invariant={row['invariant_holds']}"
    )
    if row.get("m3c_enabled"):
        print(
            "  M3C Budget:"
            f" pre_alloc={row.get('m3c_pre_m3_budget_alloc')} pre_used={row.get('m3c_pre_m3_used')}"
            f" m3a_alloc={row.get('m3c_m3a_budget_alloc')} m3a_used={row.get('m3c_m3a_used')}"
            f" m3b_alloc={row.get('m3c_m3b_budget_alloc')} m3b_used={row.get('m3c_m3b_used')}"
            f" rollover_to_m3b={row.get('m3c_rollover_to_m3b')}"
            f" invariant={row.get('m3c_budget_invariant_holds')}"
        )
        print(
            "  M3C Admission:"
            f" m3a_valid={row.get('m3a_valid_count')}"
            f" m3a_admitted={row.get('m3a_admitted_count')}"
            f" m3a_not_admitted={row.get('m3a_not_admitted_count')}"
            f" m3a_scored={row.get('m3a_candidates_scored')}"
            f" m3a_skipped={row.get('m3a_skipped_budget')}"
            f" m3a_selectable={row.get('m3a_selectable')}"
            f" m3b_valid={row.get('m3b_valid')}"
            f" m3b_invalid={row.get('m3b_invalid')}"
            f" m3b_duplicates={row.get('m3b_duplicates')}"
            f" m3b_admitted={row.get('m3b_admitted_count')}"
            f" m3b_not_admitted={row.get('m3b_not_admitted_count')}"
            f" m3b_scored={row.get('m3b_scored')}"
            f" m3b_skipped={row.get('m3b_skipped_budget')}"
            f" m3b_selectable={row.get('m3b_selectable')}"
        )
    refinement_gen = row.get("refinement_candidates_generated", 0)
    combo_gen = row.get("combo_candidates_generated", 0)
    if refinement_gen or combo_gen:
        print(
            f"  Refinement: single={refinement_gen}  combo={combo_gen}  "
            f"best_single={row.get('best_single_macro_move', '-')}  "
            f"best_single_delta={row.get('best_single_macro_delta')}  "
            f"best_combo={row.get('best_combo_move', '-')}  "
            f"best_combo_delta={row.get('best_combo_delta')}"
        )
    ls_gen = row.get("line_search_candidates_generated", 0)
    if ls_gen:
        print(
            f"  LineSearch: candidates={ls_gen}  "
            f"best_ls={row.get('best_line_search_move', '-')}  "
            f"best_ls_delta={row.get('best_line_search_delta')}"
        )
    prefilter_imp = row.get("prefilter_improving_count", 0)
    prefilter_skip = row.get("prefilter_best_skipped_hpwl_delta")
    if prefilter_imp or prefilter_skip is not None:
        print(
            f"  Prefilter: improving_scored={prefilter_imp}  "
            f"best_skipped_approx_delta={prefilter_skip}"
        )
    cache_hits = row.get("cache_hits", 0)
    cache_misses = row.get("cache_misses", 0)
    skipped_budget = row.get("candidates_skipped_by_budget", 0)
    scorer_total = row.get("official_scorer_time_ms_total", 0.0)
    scorer_avg = row.get("official_scorer_time_ms_avg", 0.0)
    scorer_p95 = row.get("official_scorer_time_ms_p95", 0.0)
    scorer_max = row.get("official_scorer_time_ms_max", 0.0)
    slowest = row.get("slowest_candidate", "-")
    if scorer_total > 0 or cache_hits or cache_misses:
        print(
            f"  ScorerTime: total={scorer_total:.0f}ms  avg={scorer_avg:.0f}ms  "
            f"p95={scorer_p95:.0f}ms  max={scorer_max:.0f}ms  slowest={slowest}"
        )
        fresh_scores = row.get("fresh_official_scores", row.get("official_scored_count", 0))
        total_effective = fresh_scores + cache_hits
        print(
            f"  Cache: fresh={fresh_scores}  hits={cache_hits}  "
            f"effective_total={total_effective}  misses={cache_misses}  "
            f"skipped_by_budget={skipped_budget}"
        )
    admission = row.get("admission_audit")
    if admission:
        print("  AdmissionAudit:")
        print(f"    generated: neighborhood={admission['generated_neighborhood']}  "
              f"refinement={admission['generated_refinement']}  "
              f"line_search={admission['generated_line_search']}")
        print(f"    prelegal_overlap_allowed={admission['prelegal_overlap_allowed']}  "
              f"(rejected_prelegal_overlap=0 after fix)")
        print(f"    legalized_successfully={admission['legalized_successfully']}  "
              f"legalization_failed={admission['legalization_failed']}")
        print(f"    duplicate_hash_skipped={admission['duplicate_hash_skipped']}  "
              f"skipped_due_to_budget={admission['skipped_due_to_budget']}")


def run_profile(
    profile_name: str,
    benchmark_names: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    generation_config: Optional[CandidateGenerationConfig] = None,
    scoring_config: Optional[CandidateScoringConfig] = None,
    no_official_score: bool = False,
    show_admission_audit: bool = False,
) -> List[Dict]:
    if profile_name not in _PROFILES:
        print(f"Unknown profile '{profile_name}'. Available: {list(_PROFILES.keys())}")
        return []

    profile = _PROFILES[profile_name]
    names = benchmark_names or profile["benchmarks"]
    show_candidates = profile.get("show_candidates", False)
    show_audit = profile.get("show_audit", False)
    require_official = profile.get("require_official", False)
    pt_paths = _discover_benchmarks(names)
    gen_cfg = generation_config or CandidateGenerationConfig(
        candidate_budget=profile.get("candidate_budget"),
        neighborhood_macro_limit=profile.get("neighborhood_macro_limit", 20),
        neighborhood_step_profile=profile.get("neighborhood_step_profile", "medium"),
        disable_global_candidates=profile.get("disable_global_candidates", False),
        only_original_neighborhood=profile.get("only_original_neighborhood", False),
        refinement_around_winners=profile.get("refinement_around_winners", False),
        refinement_top_k=profile.get("refinement_top_k", 5),
        refinement_combo_size=profile.get("refinement_combo_size", 2),
        refinement_seed_strategy=profile.get("refinement_seed_strategy", "conservative"),
        refinement_exploration_seeds=profile.get("refinement_exploration_seeds", 1),
        line_search_around_winners=profile.get("line_search_around_winners", False),
        line_search_top_k=profile.get("line_search_top_k", 3),
        line_search_max_scale=profile.get("line_search_max_scale", 4.0),
        line_search_stop_after_worse=profile.get("line_search_stop_after_worse", 2),
        m3a_pair_refinement=profile.get("m3a_pair_refinement", False),
        m3a_top_k_pairs=profile.get("m3a_top_k_pairs", 64),
        m3a_score_budget=profile.get("m3a_score_budget", None),
        m3b_cluster_refinement=profile.get("m3b_cluster_refinement", False),
        m3b_top_k_clusters=profile.get("m3b_top_k_clusters", 32),
        m3b_score_budget=profile.get("m3b_score_budget", None),
        m3c_budget_allocation=profile.get("m3c_budget_allocation", False),
        m3c_pre_m3_budget=profile.get("m3c_pre_m3_budget", None),
        m3c_m3a_reserved_budget=profile.get("m3c_m3a_reserved_budget", None),
        m3c_m3b_reserved_budget=profile.get("m3c_m3b_reserved_budget", None),
        m3c_rollover_unused_budget=profile.get("m3c_rollover_unused_budget", True),
    )
    score_cfg = scoring_config or CandidateScoringConfig(
        max_official_scores=profile.get("max_official_scores"),
    )

    print(f"\n{'=' * 65}")
    print(f"Profile: {profile_name} - {profile['description']}")
    print(f"Benchmarks: {len(pt_paths)}  Repeats: {profile['repeat']}")
    print(f"{'=' * 65}")

    if not pt_paths:
        print("  No benchmarks found. Check BENCHMARKS_PT_DIR in config.py.")
        return []

    # --- Create persistent cache once for the whole session ---
    shared_cache = None
    if score_cfg.official_score_cache_path and not score_cfg.disable_score_cache:
        from pathlib import Path as _Path
        from submissions.solver.core.score_cache import OfficialScoreCache
        shared_cache = OfficialScoreCache(
            cache_path=_Path(score_cfg.official_score_cache_path),
            disabled=False,
            clear=score_cfg.clear_score_cache,
        )
        print(
            f"  Score cache: {score_cfg.official_score_cache_path}  "
            f"entries={shared_cache.size}  clear={score_cfg.clear_score_cache}"
        )

    all_rows: List[Dict] = []
    for pt_path in pt_paths:
        benchmark, plc = _load_benchmark(pt_path, require_official=require_official)
        if benchmark is None:
            continue
        if require_official and plc is None:
            print(
                f"  SKIP {pt_path.stem}: official-smoke requires plc_client_os. "
                "Run 'git submodule update --init external/MacroPlacement' then retry."
            )
            continue

        # --no-official-score: force local-proxy mode regardless of whether plc loaded
        effective_plc = None if no_official_score else plc

        for rep in range(profile["repeat"]):
            try:
                row = _run_one(
                    benchmark=benchmark,
                    plc=effective_plc,
                    generation_config=gen_cfg,
                    scoring_config=score_cfg,
                    show_candidates=show_candidates,
                    show_audit=show_audit,
                    show_admission_audit=show_admission_audit,
                    score_cache=shared_cache,
                )
                if profile["repeat"] > 1:
                    row["repeat"] = rep
                all_rows.append(row)
                if "console" in profile["output"]:
                    cost_str = f"{row['proxy_cost']:.4f}" if row["proxy_cost"] is not None else "N/A"
                    print(
                        f"  [{row['benchmark']:20s}] valid={row['valid']}  "
                        f"best={row['best_candidate']:35s}  cost={cost_str:>8}  "
                        f"fresh={row.get('fresh_official_scores', row['official_scored_count']):>3}"
                        f"  cache_hits={row.get('cache_hits', 0):>3}"
                        f"  mode={row['scoring_mode']:12s}  "
                        f"{row['runtime_ms']:.0f}ms"
                    )
                    if show_audit or show_admission_audit:
                        _print_audit(row)
            except Exception as exc:
                print(f"  ERROR {pt_path.name} rep{rep}: {exc}")

    if "console" in profile["output"]:
        print()
        _print_table(all_rows)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if "csv" in profile["output"]:
            save_csv(
                [{k: v for k, v in row.items() if k not in ("candidates", "connectivity", "diversity")} for row in all_rows],
                out_dir / f"run_{profile_name}.csv",
            )
        if "json" in profile["output"]:
            save_json({"profile": profile_name, "results": all_rows}, out_dir / f"run_{profile_name}.json")

    return all_rows


def main():
    parser = argparse.ArgumentParser(prog="run_benchmarks", description="M2B benchmark runner.")
    parser.add_argument("--profile", default="smoke", choices=list(_PROFILES.keys()),
                        help="Run profile. Use 'm2b-final-smoke' for the recommended cold-run profile.")
    parser.add_argument("-b", "--benchmark", action="append", dest="benchmarks", metavar="NAME", help="Benchmark name override.")
    parser.add_argument("--out", metavar="DIR", default=None, help="Output directory for CSV/JSON.")
    parser.add_argument("--candidate-budget", type=int, default=None, help="Max generated candidates.")
    parser.add_argument("--neighborhood-macro-limit", type=int, default=None, help="Max perturbed macros.")
    parser.add_argument(
        "--neighborhood-step-profile",
        choices=["small", "medium", "large"],
        default=None,
        help="Neighborhood move step profile.",
    )
    parser.add_argument("--disable-global-candidates", action="store_true", help="Disable global families.")
    parser.add_argument("--only-original-neighborhood", action="store_true", help="Use only original + original_neighborhood.")
    parser.add_argument("--refinement-around-winners", action="store_true", help="Enable second-pass refinement around winning moves.")
    parser.add_argument("--refinement-top-k", type=int, default=None, help="Seeds for refinement pass (default 5).")
    parser.add_argument("--refinement-combo-size", type=int, default=None, choices=[2, 3], help="Max combo size (2=combo2, 3=also combo3).")
    parser.add_argument("--refinement-seed-strategy", choices=["conservative", "diverse"], default=None, help="Seed selection strategy for refinement (default conservative).")
    parser.add_argument("--refinement-exploration-seeds", type=int, default=None, help="Exploratory seeds in diverse strategy (default 1).")
    parser.add_argument("--line-search-around-winners", action="store_true", help="Enable directional line-search pass after neighborhood scoring.")
    parser.add_argument("--line-search-top-k", type=int, default=None, help="Seeds for line-search pass (default 3).")
    parser.add_argument("--line-search-max-scale", type=float, default=None, help="Max scale multiplier for line-search (default 4.0).")
    parser.add_argument("--line-search-stop-after-worse", type=int, default=None, help="Stop per-macro line-search after N consecutive worse official scores (default 2).")
    parser.add_argument("--max-official-scores", type=int, default=None, help="Cap total official proxy scoring calls.")
    parser.add_argument("--exploration-count", type=int, default=None, help="Exploratory candidates scored despite bad approx delta.")
    parser.add_argument("--official-score-cache", metavar="PATH", default=None, help="Enable persistent score cache at PATH.")
    parser.add_argument("--disable-score-cache", action="store_true", help="Disable persistent score cache even if configured.")
    parser.add_argument("--clear-score-cache", action="store_true", help="Clear score cache file before run.")
    parser.add_argument("--no-official-score", action="store_true",
                        help="Generate candidates and legalize but skip official scorer (local-proxy only). "
                             "Useful for --audit-candidate-admission without burning score budget.")
    parser.add_argument("--audit-candidate-admission", action="store_true",
                        help="Print per-benchmark candidate admission audit (prelegal overlaps, legalization counts, etc.).")
    parser.add_argument("--seed-discovery-budget", type=int, default=None,
                        help="Official score budget for neighborhood seed-discovery pass (default: ~32/60 of max).")
    parser.add_argument("--refinement-budget", type=int, default=None,
                        help="Official score budget for refinement pass (default: ~10/60 of max).")
    # Note: line-search budget is the remainder of max_official_scores after
    # the seed-discovery and refinement passes (see candidate_scoring) and is
    # not configurable separately.
    args = parser.parse_args()

    profile = _PROFILES[args.profile]
    out_dir = Path(args.out) if args.out else (_SOLVER_DIR / "artifacts")
    gen_cfg = CandidateGenerationConfig(
        candidate_budget=args.candidate_budget if args.candidate_budget is not None else profile.get("candidate_budget"),
        neighborhood_macro_limit=(
            args.neighborhood_macro_limit
            if args.neighborhood_macro_limit is not None else profile.get("neighborhood_macro_limit", 20)
        ),
        neighborhood_step_profile=(
            args.neighborhood_step_profile
            if args.neighborhood_step_profile is not None else profile.get("neighborhood_step_profile", "medium")
        ),
        disable_global_candidates=args.disable_global_candidates or profile.get("disable_global_candidates", False),
        only_original_neighborhood=args.only_original_neighborhood or profile.get("only_original_neighborhood", False),
        refinement_around_winners=args.refinement_around_winners or profile.get("refinement_around_winners", False),
        refinement_top_k=args.refinement_top_k if args.refinement_top_k is not None else profile.get("refinement_top_k", 5),
        refinement_combo_size=args.refinement_combo_size if args.refinement_combo_size is not None else profile.get("refinement_combo_size", 2),
        refinement_seed_strategy=(
            args.refinement_seed_strategy
            if args.refinement_seed_strategy is not None
            else profile.get("refinement_seed_strategy", "conservative")
        ),
        refinement_exploration_seeds=(
            args.refinement_exploration_seeds
            if args.refinement_exploration_seeds is not None
            else profile.get("refinement_exploration_seeds", 1)
        ),
        line_search_around_winners=args.line_search_around_winners or profile.get("line_search_around_winners", False),
        line_search_top_k=args.line_search_top_k if args.line_search_top_k is not None else profile.get("line_search_top_k", 3),
        line_search_max_scale=args.line_search_max_scale if args.line_search_max_scale is not None else profile.get("line_search_max_scale", 4.0),
        line_search_stop_after_worse=args.line_search_stop_after_worse if args.line_search_stop_after_worse is not None else profile.get("line_search_stop_after_worse", 2),
        m3a_pair_refinement=profile.get("m3a_pair_refinement", False),
        m3a_top_k_pairs=profile.get("m3a_top_k_pairs", 64),
        m3a_score_budget=profile.get("m3a_score_budget", None),
        m3b_cluster_refinement=profile.get("m3b_cluster_refinement", False),
        m3b_top_k_clusters=profile.get("m3b_top_k_clusters", 32),
        m3b_score_budget=profile.get("m3b_score_budget", None),
        m3c_budget_allocation=profile.get("m3c_budget_allocation", False),
        m3c_pre_m3_budget=profile.get("m3c_pre_m3_budget", None),
        m3c_m3a_reserved_budget=profile.get("m3c_m3a_reserved_budget", None),
        m3c_m3b_reserved_budget=profile.get("m3c_m3b_reserved_budget", None),
        m3c_rollover_unused_budget=profile.get("m3c_rollover_unused_budget", True),
    )
    score_cfg = CandidateScoringConfig(
        max_official_scores=args.max_official_scores if args.max_official_scores is not None else profile.get("max_official_scores"),
        exploratory_score_count=args.exploration_count if args.exploration_count is not None else 8,
        official_score_cache_path=args.official_score_cache,
        disable_score_cache=args.disable_score_cache,
        clear_score_cache=args.clear_score_cache,
        seed_discovery_score_budget=args.seed_discovery_budget,
        refinement_score_budget=args.refinement_budget,
    )
    run_profile(
        args.profile,
        benchmark_names=args.benchmarks,
        out_dir=out_dir,
        generation_config=gen_cfg,
        scoring_config=score_cfg,
        no_official_score=args.no_official_score,
        show_admission_audit=args.audit_candidate_admission,
    )


if __name__ == "__main__":
    main()
