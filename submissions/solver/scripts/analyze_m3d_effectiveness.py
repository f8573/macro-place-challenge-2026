"""
M3D-slice-4: Effectiveness analysis script.

Wires M3D-slice-1/2/3 into a deterministic analysis pipeline and writes:
  m3d_candidate_effectiveness.csv — all candidate rows (one per candidate)
  m3d_family_summary.csv          — per-family aggregated statistics
  m3d_benchmark_summary.csv       — per-benchmark summary merged with classification
  m3d_findings.md                 — Markdown report

Does not change any solver behaviour.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.core.m3d_candidate_export import export_candidate_rows  # noqa: E402
from submissions.solver.core.m3d_family_summary import summarize_candidate_families  # noqa: E402
from submissions.solver.core.m3d_failure_classification import classify_m3d_failure  # noqa: E402
from submissions.solver.core.io import save_csv  # noqa: E402

# Stable, explicit column ordering for m3d_benchmark_summary.csv.
_BENCHMARK_SUMMARY_FIELDS: Tuple[str, ...] = (
    "benchmark",
    "profile",
    "selected_candidate",
    "selected_family",
    "selected_cost",
    "original_cost",
    "classification",
    "reason",
    "recommended_next_step",
    "late_stage_generated",
    "late_stage_valid",
    "late_stage_scored",
    "late_stage_num_beating_final",
    "late_stage_num_near_tie",
)

_LATE_STAGE_FAMILIES = frozenset(
    {"m3a_pair_refinement", "m3b_cluster_refinement", "m4b_region_repair"}
)


# ---------------------------------------------------------------------------
# Pure helper functions — testable without running the solver
# ---------------------------------------------------------------------------


def build_benchmark_summary_rows(
    run_rows: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge per-benchmark run info with failure classifications.

    Returns one ordered row per entry in run_rows with every field in
    _BENCHMARK_SUMMARY_FIELDS populated.  Missing classification fields
    default to None / 0.  Read-only: does not mutate inputs.
    """
    classif_by_bp: Dict[Tuple[str, str], Dict[str, Any]] = {
        (c.get("benchmark", "") or "", c.get("profile", "") or ""): c
        for c in classifications
    }

    result: List[Dict[str, Any]] = []
    for run in run_rows:
        bm = run.get("benchmark", "") or ""
        pf = run.get("profile", "") or ""
        cl = classif_by_bp.get((bm, pf), {})

        row: Dict[str, Any] = {f: None for f in _BENCHMARK_SUMMARY_FIELDS}
        row.update(
            {
                "benchmark": bm,
                "profile": pf,
                "selected_candidate": run.get("selected_candidate"),
                "selected_family": run.get("selected_family"),
                "selected_cost": run.get("selected_cost"),
                "original_cost": run.get("original_cost"),
                "classification": cl.get("classification"),
                "reason": cl.get("reason"),
                "recommended_next_step": cl.get("recommended_next_step"),
                "late_stage_generated": cl.get("late_stage_generated", 0),
                "late_stage_valid": cl.get("late_stage_valid", 0),
                "late_stage_scored": cl.get("late_stage_scored", 0),
                "late_stage_num_beating_final": cl.get("late_stage_num_beating_final", 0),
                "late_stage_num_near_tie": cl.get("late_stage_num_near_tie", 0),
            }
        )
        result.append(row)

    return result


def render_findings_md(
    benchmark_summaries: List[Dict[str, Any]],
    family_summaries: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Render a deterministic Markdown findings report.

    Read-only: does not mutate any input.
    """
    lines: List[str] = []

    # --- Title ---
    lines += ["# M3D Effectiveness Analysis", ""]

    # --- Run configuration ---
    lines += ["## Run Configuration", ""]
    lines += ["| Parameter | Value |", "|-----------|-------|"]
    lines.append(f"| Profile | `{config.get('profile', '')}` |")
    bms = config.get("benchmarks") or []
    lines.append(f"| Benchmarks | {', '.join(bms) if bms else '(all)'} |")
    if config.get("official_epsilon") is not None:
        lines.append(f"| Official epsilon | `{config['official_epsilon']}` |")
    if config.get("max_official_scores") is not None:
        lines.append(f"| Max official scores | `{config['max_official_scores']}` |")
    if config.get("seed_discovery_budget") is not None:
        lines.append(f"| Seed discovery budget | `{config['seed_discovery_budget']}` |")
    if config.get("timestamp"):
        lines.append(f"| Timestamp | `{config['timestamp']}` |")
    lines.append("")

    # --- Benchmark summary table ---
    lines += ["## Benchmark Summary", ""]
    if benchmark_summaries:
        lines.append(
            "| Benchmark | Selected | Family | Cost | Orig Cost | Classification |"
        )
        lines.append(
            "|-----------|----------|--------|------|-----------|----------------|"
        )
        for r in benchmark_summaries:
            cost = (
                f"{r['selected_cost']:.6f}"
                if r.get("selected_cost") is not None
                else "N/A"
            )
            orig = (
                f"{r['original_cost']:.6f}"
                if r.get("original_cost") is not None
                else "N/A"
            )
            lines.append(
                f"| {r.get('benchmark', '')} "
                f"| {r.get('selected_candidate') or ''} "
                f"| {r.get('selected_family') or ''} "
                f"| {cost} "
                f"| {orig} "
                f"| {r.get('classification') or ''} |"
            )
    else:
        lines.append("*No benchmark results available.*")
    lines.append("")

    # --- Family effectiveness table ---
    lines += ["## Family Effectiveness", ""]
    if family_summaries:
        lines.append(
            "| Benchmark | Family | Generated | Valid | Scored"
            " | Beating Final | Near Tie | Best Cost |"
        )
        lines.append(
            "|-----------|--------|-----------|-------|--------"
            "|---------------|----------|-----------|"
        )
        for s in family_summaries:
            best = (
                f"{s['best_official_cost']:.6f}"
                if s.get("best_official_cost") is not None
                else "N/A"
            )
            lines.append(
                f"| {s.get('benchmark', '')} "
                f"| {s.get('family', '')} "
                f"| {s.get('generated_count', 0)} "
                f"| {s.get('valid_count', 0)} "
                f"| {s.get('scored_count', 0)} "
                f"| {s.get('num_beating_final', 0)} "
                f"| {s.get('num_near_tie', 0)} "
                f"| {best} |"
            )
    else:
        lines.append("*No family summaries available.*")
    lines.append("")

    # --- Failure classification table ---
    lines += ["## Failure Classification", ""]
    if classifications:
        lines.append(
            "| Benchmark | Classification | LS Generated | LS Scored"
            " | Beating Final | Next Step |"
        )
        lines.append(
            "|-----------|----------------|--------------|----------"
            "|---------------|-----------|"
        )
        for c in classifications:
            lines.append(
                f"| {c.get('benchmark', '')} "
                f"| {c.get('classification', '')} "
                f"| {c.get('late_stage_generated', 0)} "
                f"| {c.get('late_stage_scored', 0)} "
                f"| {c.get('late_stage_num_beating_final', 0)} "
                f"| {c.get('recommended_next_step', '')} |"
            )
    else:
        lines.append("*No classifications available.*")
    lines.append("")

    # --- Top late-stage candidates (if any) ---
    late_stage = [
        r
        for r in candidate_rows
        if r.get("family") in _LATE_STAGE_FAMILIES
        and r.get("scored")
        and r.get("proxy_cost") is not None
    ]
    if late_stage:
        top_ls = sorted(
            late_stage,
            key=lambda r: (r["proxy_cost"], r.get("candidate_name") or ""),
        )[:10]
        lines += ["## Top Late-Stage Candidates", ""]
        lines.append(
            "| Benchmark | Candidate | Family | Cost | Selectable | Selected |"
        )
        lines.append(
            "|-----------|-----------|--------|------|------------|----------|"
        )
        for r in top_ls:
            lines.append(
                f"| {r.get('benchmark', '')} "
                f"| {r.get('candidate_name', '')} "
                f"| {r.get('family', '')} "
                f"| {r['proxy_cost']:.6f} "
                f"| {r.get('scored_pool_selectable', False)} "
                f"| {r.get('is_selected', False)} |"
            )
        lines.append("")

    # --- Recommendations ---
    lines += ["## Recommendations", ""]
    if classifications:
        step_to_bms: Dict[str, List[str]] = {}
        for c in classifications:
            step = c.get("recommended_next_step", "") or ""
            bm = c.get("benchmark", "") or ""
            if step:
                step_to_bms.setdefault(step, []).append(bm)
        if step_to_bms:
            for step in sorted(step_to_bms):
                bms_str = ", ".join(sorted(step_to_bms[step]))
                lines.append(f"- **{step}** ({bms_str})")
        else:
            lines.append("*No actionable recommendations.*")
    else:
        lines.append("*No classifications available for recommendations.*")
    lines.append("")

    return "\n".join(lines)


def _write_csv_with_fields(
    rows: List[Dict[str, Any]],
    fields: Tuple[str, ...],
    path: Path,
) -> None:
    """Write rows to CSV with explicit field ordering.

    Always creates the file (with header row) even when rows is empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_m3d_outputs(
    output_dir: Path,
    candidate_rows: List[Dict[str, Any]],
    family_summaries: List[Dict[str, Any]],
    benchmark_summaries: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_prefix: str = "m3d",
) -> None:
    """Write all M3D output artifacts to output_dir.

    Always writes m3d_benchmark_summary.csv (with headers) even when empty.
    m3d_candidate_effectiveness.csv and m3d_family_summary.csv are skipped
    when the respective lists are empty (matching save_csv behaviour).
    Read-only: does not mutate any input.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(candidate_rows, output_dir / f"{output_prefix}_candidate_effectiveness.csv")
    save_csv(family_summaries, output_dir / f"{output_prefix}_family_summary.csv")
    _write_csv_with_fields(
        benchmark_summaries,
        _BENCHMARK_SUMMARY_FIELDS,
        output_dir / f"{output_prefix}_benchmark_summary.csv",
    )

    md = render_findings_md(
        benchmark_summaries, family_summaries, classifications, candidate_rows, config
    )
    (output_dir / f"{output_prefix}_findings.md").write_text(md, encoding="utf-8")

    json_path = output_dir / f"{output_prefix}_findings.json"
    if config.get("json"):
        from submissions.solver.core.io import save_json  # noqa: PLC0415

        save_json(
            {
                "config": {k: v for k, v in config.items() if k != "json"},
                "benchmark_summaries": benchmark_summaries,
                "family_summaries": family_summaries,
                "classifications": classifications,
            },
            json_path,
        )
    elif json_path.exists():
        json_path.unlink()


# ---------------------------------------------------------------------------
# Solver pipeline integration (not called in tests)
# ---------------------------------------------------------------------------


def _run_benchmark_m3d(
    benchmark: Any,
    plc: Any,
    gen_cfg: Any,
    score_cfg: Any,
    profile_name: str,
    score_cache: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run the solver pipeline for one benchmark and collect M3D candidate rows.

    Returns (run_info, candidate_rows).  Read-only: does not mutate solver state.
    """
    from submissions.solver.core.candidates import generate_candidates  # noqa: PLC0415
    from submissions.solver.core.candidate_scoring import score_and_select  # noqa: PLC0415

    candidates = generate_candidates(benchmark, config=gen_cfg)
    best, ranked, diag = score_and_select(
        candidates,
        benchmark,
        plc=plc,
        scoring_config=score_cfg,
        generation_config=gen_cfg,
        score_cache=score_cache,
    )

    cand_rows = export_candidate_rows(
        ranked, best, diag,
        benchmark=benchmark.name,
        profile=profile_name,
    )

    run_info: Dict[str, Any] = {
        "benchmark": benchmark.name,
        "profile": profile_name,
        "selected_candidate": best.name if best else None,
        "selected_family": best.family if best else None,
        "selected_cost": best.proxy_cost if best else None,
        "original_cost": diag.raw_original_proxy_cost,
    }
    return run_info, cand_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_m3d_effectiveness",
        description=(
            "M3D effectiveness analysis: run solver pipeline, export candidates, "
            "summarize families, classify failures, and write analysis artifacts."
        ),
    )
    parser.add_argument(
        "--profile",
        default="m3c-default",
        help="Benchmark profile name (see run_benchmarks.py).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        help="Benchmark names to analyze (default: profile default).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/m3d",
        metavar="DIR",
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--output-prefix",
        default="m3d",
        help="Prefix for CSV/Markdown artifact filenames (default: m3d).",
    )
    parser.add_argument(
        "--clear-score-cache",
        action="store_true",
        help="Clear the score cache file before running.",
    )
    parser.add_argument(
        "--official-epsilon",
        type=float,
        default=1e-5,
        help="Tolerance for near-tie/beating comparisons (default: 1e-5).",
    )
    parser.add_argument(
        "--seed-discovery-budget",
        type=int,
        default=None,
        help="Official score budget for the seed-discovery pass.",
    )
    parser.add_argument(
        "--max-official-scores",
        type=int,
        default=None,
        help="Cap total official scoring calls per benchmark.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write m3d_findings.json alongside the CSV/Markdown outputs.",
    )
    args = parser.parse_args()

    from submissions.solver.scripts.run_benchmarks import (  # noqa: PLC0415
        _PROFILES,
        _discover_benchmarks,
        _load_benchmark,
    )
    from submissions.solver.core.candidate_types import (  # noqa: PLC0415
        CandidateGenerationConfig,
        CandidateScoringConfig,
    )

    profile_name = args.profile
    if profile_name not in _PROFILES:
        print(f"Unknown profile '{profile_name}'. Available: {sorted(_PROFILES)}")
        sys.exit(1)

    profile = _PROFILES[profile_name]
    names = args.benchmarks or profile.get("benchmarks")
    pt_paths = _discover_benchmarks(names)

    if not pt_paths:
        print("No benchmarks found. Check BENCHMARKS_PT_DIR in config.py.")
        sys.exit(1)

    require_official = profile.get("require_official", False)

    gen_cfg = CandidateGenerationConfig(
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
        m4b_region_repair=profile.get("m4b_region_repair", False),
        m4b_reserved_scores=profile.get("m4b_reserved_scores", 20),
        m4b_grid_dims=tuple(profile.get("m4b_grid_dims", (3, 3))),
        m4b_min_macros_per_region=profile.get("m4b_min_macros_per_region", 2),
        m4b_max_combos_per_region=profile.get("m4b_max_combos_per_region", 16),
        m4b_legalization_max_displacement_um=profile.get("m4b_legalization_max_displacement_um", 200.0),
        m4b_perturbation_fraction=profile.get("m4b_perturbation_fraction", 0.5),
        m4c_ranking=profile.get("m4c_ranking", False),
        m4c_k_ranked=profile.get("m4c_k_ranked", 16),
        m4c_exploration=profile.get("m4c_exploration", 4),
        m4c_max_per_region=profile.get("m4c_max_per_region", None),
        m4c_known_winners=profile.get("m4c_known_winners", []),
        m4d_family_normalization=profile.get("m4d_family_normalization", False),
        m4d_family_quota_floors=profile.get("m4d_family_quota_floors", None),
    )
    max_scores = (
        args.max_official_scores
        if args.max_official_scores is not None
        else profile.get("max_official_scores")
    )
    score_cfg = CandidateScoringConfig(
        max_official_scores=max_scores,
        clear_score_cache=args.clear_score_cache,
        seed_discovery_score_budget=args.seed_discovery_budget,
    )

    shared_cache = None
    if score_cfg.official_score_cache_path and not score_cfg.disable_score_cache:
        from submissions.solver.core.score_cache import OfficialScoreCache  # noqa: PLC0415

        shared_cache = OfficialScoreCache(
            cache_path=Path(score_cfg.official_score_cache_path),
            disabled=False,
            clear=score_cfg.clear_score_cache,
        )

    all_candidate_rows: List[Dict[str, Any]] = []
    all_run_infos: List[Dict[str, Any]] = []

    print(f"\nM3D Analysis — profile: {profile_name}  benchmarks: {len(pt_paths)}")
    print("-" * 60)

    for pt_path in pt_paths:
        benchmark, plc = _load_benchmark(pt_path, require_official=require_official)
        if benchmark is None:
            continue
        if require_official and plc is None:
            print(
                f"  SKIP {pt_path.stem}: official scoring requires plc_client_os. "
                "Run 'git submodule update --init external/MacroPlacement'."
            )
            continue

        try:
            run_info, cand_rows = _run_benchmark_m3d(
                benchmark=benchmark,
                plc=plc,
                gen_cfg=gen_cfg,
                score_cfg=score_cfg,
                profile_name=profile_name,
                score_cache=shared_cache,
            )
            all_run_infos.append(run_info)
            all_candidate_rows.extend(cand_rows)
            cost_str = (
                f"{run_info['selected_cost']:.6f}"
                if run_info["selected_cost"] is not None
                else "N/A"
            )
            print(
                f"  [{benchmark.name}]  "
                f"selected={run_info['selected_candidate']}  "
                f"cost={cost_str}  "
                f"candidates={len(cand_rows)}"
            )
        except Exception as exc:
            print(f"  ERROR {pt_path.stem}: {exc}")

    if not all_run_infos:
        print("No benchmarks ran successfully.")
        sys.exit(1)

    family_summaries = summarize_candidate_families(
        all_candidate_rows, official_epsilon=args.official_epsilon
    )
    classifications = classify_m3d_failure(
        all_candidate_rows, family_summaries, official_epsilon=args.official_epsilon
    )
    benchmark_summaries = build_benchmark_summary_rows(all_run_infos, classifications)

    config: Dict[str, Any] = {
        "profile": profile_name,
        "benchmarks": [p.stem for p in pt_paths],
        "official_epsilon": args.official_epsilon,
        "max_official_scores": max_scores,
        "seed_discovery_budget": args.seed_discovery_budget,
        "json": args.json,
    }

    output_dir = Path(args.output_dir)
    write_m3d_outputs(
        output_dir=output_dir,
        candidate_rows=all_candidate_rows,
        family_summaries=family_summaries,
        benchmark_summaries=benchmark_summaries,
        classifications=classifications,
        config=config,
        output_prefix=args.output_prefix,
    )

    print(f"\nM3D artifacts written to: {output_dir.resolve()}")
    for fname in (
        f"{args.output_prefix}_candidate_effectiveness.csv",
        f"{args.output_prefix}_family_summary.csv",
        f"{args.output_prefix}_benchmark_summary.csv",
        f"{args.output_prefix}_findings.md",
    ):
        print(f"  {output_dir / fname}")


if __name__ == "__main__":
    main()
