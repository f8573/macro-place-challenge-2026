"""
Official scoring smoke test for M2B.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig  # noqa: E402


def _load_official(name: str):
    ibm_dir = IBM_TESTCASES_DIR / name
    if not ibm_dir.exists():
        raise FileNotFoundError(
            f"IBM testcase not found: {ibm_dir}\nRun: git submodule update --init external/MacroPlacement"
        )
    if not (ibm_dir / "netlist.pb.txt").exists():
        raise FileNotFoundError(f"netlist.pb.txt missing in {ibm_dir}")

    try:
        from macro_place.loader import load_benchmark_from_dir
    except ImportError as exc:
        raise ImportError("plc_client_os is required for official scoring.") from exc

    benchmark, plc = load_benchmark_from_dir(ibm_dir.as_posix())
    if plc is None:
        raise RuntimeError("load_benchmark_from_dir returned plc=None; official scoring unavailable.")
    return benchmark, plc


def _check_connectivity(benchmark) -> None:
    if not benchmark.net_nodes:
        print("FAIL: net_nodes is empty - official benchmark should have connectivity data.")
        sys.exit(1)
    total_pins = sum(n.numel() for n in benchmark.net_nodes)
    print(f"  net_nodes: {len(benchmark.net_nodes)} nets, {total_pins} total pin references - OK")


def _family_counts(candidates) -> dict:
    counts = {}
    for candidate in candidates:
        counts[candidate.family] = counts.get(candidate.family, 0) + 1
    return counts


def _run_official_smoke(
    name: str,
    generation_config: CandidateGenerationConfig,
    scoring_config: Optional[CandidateScoringConfig] = None,
    score_cache=None,
) -> dict:
    print(f"\n{'=' * 65}")
    print(f"Official scoring smoke: {name}")
    print(f"{'=' * 65}")

    benchmark, plc = _load_official(name)
    print(f"  Loaded: {benchmark.num_hard_macros} hard macros, canvas {benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f} um")
    _check_connectivity(benchmark)

    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import connectivity_audit, score_and_select

    cfg = scoring_config or CandidateScoringConfig()
    t0 = time.perf_counter()
    candidates = generate_candidates(benchmark, config=generation_config)
    best, ranked, diag = score_and_select(
        candidates, benchmark, plc=plc,
        scoring_config=cfg,
        generation_config=generation_config,
        score_cache=score_cache,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000

    raw_cost = diag.raw_original_proxy_cost
    raw_valid = diag.raw_original_valid
    family_counts = _family_counts(candidates)
    print(f"\n  Generated {len(candidates)} candidates, scored in {runtime_ms:.0f} ms")
    print(f"  Families: {family_counts}")
    print(f"  original_raw valid={raw_valid}  proxy_cost={f'{raw_cost:.6f}' if raw_cost is not None else 'N/A'}")
    print(f"  Winner: {best.name if best else 'none'}  family={best.family if best else 'none'}")
    print(
        f"  Score fresh={diag.fresh_official_scores}  cache_hits={diag.cache_hits}"
        f"  effective_total={diag.fresh_official_scores + diag.cache_hits}"
        f"  duplicates={diag.duplicate_count}  prefiltered={diag.candidates_prefiltered}"
        f"  skipped_by_budget={diag.candidates_skipped_by_budget}"
    )
    print(f"  Selected due to: {diag.selected_due_to}  invariant={diag.invariant_holds}")

    selectable_ranked = [
        s for s in ranked
        if s.valid and s.proxy_cost is not None and s.was_scored and s.name != "original_legalized"
    ]
    print(f"\n  Ranked proxy costs (top {min(12, len(selectable_ranked))}):")
    print(f"  {'Rank':>4}  {'Name':<44}  {'Family':<22}  {'Cost':>10}  {'Delta':>10}  {'Move':<18}")
    print(f"  {'-' * 120}")
    for idx, sc in enumerate(selectable_ranked[:12], start=1):
        delta = f"{sc.delta_vs_original:+.6f}" if sc.delta_vs_original is not None else "N/A"
        moved_macro = sc.metadata.get("moved_macro_id")
        move = ""
        if moved_macro is not None:
            move = f"m{moved_macro} ({sc.metadata.get('dx', 0.0):+.1f},{sc.metadata.get('dy', 0.0):+.1f})"
        marker = " <-- BEST" if best is not None and sc.name == best.name else ""
        print(f"  {idx:>4}  {sc.name:<44}  {sc.family:<22}  {sc.proxy_cost:>10.6f}  {delta:>10}  {move:<18}{marker}")

    if any(s.family == "original_neighborhood" for s in ranked):
        print("\n  original_neighborhood details:")
        print(f"  {'Name':<44}  {'Macro':>5}  {'dx':>8}  {'dy':>8}  {'Approx':>10}  {'Official':>10}  {'Delta':>10}")
        print(f"  {'-' * 112}")
        for sc in ranked:
            if sc.family != "original_neighborhood":
                continue
            approx_str = (
                f"{float(sc.metadata.get('approx_hpwl_delta')):.6f}"
                if sc.metadata.get("approx_hpwl_delta") is not None else "N/A"
            )
            official_str = f"{sc.proxy_cost:.6f}" if sc.proxy_cost is not None else "N/A"
            delta_str = f"{sc.delta_vs_original:+.6f}" if sc.delta_vs_original is not None else "N/A"
            print(
                f"  {sc.name:<44}  {str(sc.metadata.get('moved_macro_id', '')):>5}  "
                f"{float(sc.metadata.get('dx', 0.0)):>8.2f}  {float(sc.metadata.get('dy', 0.0)):>8.2f}  "
                f"{approx_str:>10}  {official_str:>10}  {delta_str:>10}"
            )

    if diag.refinement_seed_bucket_diagnostics:
        print("\n  Refinement seed buckets (diverse strategy):")
        hdr = f"  {'#':>3}  {'Name':<46}  {'Macro':>5}  {'Bucket':<12}  {'Official':>10}  {'Approx':>10}  {'Priority':>10}  {'GenRk':>6}  {'ScRk':>6}"
        print(hdr)
        print(f"  {'-' * 114}")
        for i, d in enumerate(diag.refinement_seed_bucket_diagnostics, start=1):
            official = f"{d['official_proxy_cost']:.6f}" if d.get("official_proxy_cost") is not None else "N/A"
            approx = f"{d['approx_hpwl_delta']:.6f}" if d.get("approx_hpwl_delta") is not None else "N/A"
            priority = f"{d['macro_priority_score']:.1f}" if d.get("macro_priority_score") is not None else "N/A"
            gen_rk = str(d.get("generation_rank", "N/A"))
            sc_rk = str(d.get("scoring_rank", "N/A"))
            print(
                f"  {i:>3}  {d['seed_name']:<46}  {str(d.get('macro_id', '')):>5}  "
                f"{d.get('bucket', ''):.<12}  {official:>10}  {approx:>10}  "
                f"{priority:>10}  {gen_rk:>6}  {sc_rk:>6}"
            )

        # Summary: improving candidates not selected as seeds
        all_improving_approx = sorted(
            [
                (sc.metadata.get("approx_hpwl_delta"), sc.name, sc.metadata.get("moved_macro_id"))
                for sc in ranked
                if sc.family == "original_neighborhood"
                and isinstance(sc.metadata.get("approx_hpwl_delta"), float)
                and sc.metadata["approx_hpwl_delta"] <= 1e-9
            ],
            key=lambda x: (x[0], x[1]),
        )
        selected_names = {d["seed_name"] for d in diag.refinement_seed_bucket_diagnostics}
        unselected = [(a, n, m) for a, n, m in all_improving_approx if n not in selected_names]
        selected_macros = [d.get("macro_id") for d in diag.refinement_seed_bucket_diagnostics]
        skipped_dupes = [
            m for (a, n, m) in all_improving_approx
            if n not in selected_names and m in {d.get("macro_id") for d in diag.refinement_seed_bucket_diagnostics}
        ]
        print(f"\n  Seed summary: total_improving={len(all_improving_approx)}  selected_macros={selected_macros}")
        if skipped_dupes:
            print(f"    skipped_duplicate_macro_ids={sorted(set(skipped_dupes))}")
        if unselected:
            best_u = unselected[0]
            print(f"    best_unselected_improving: approx={best_u[0]:.6f}  name={best_u[1]}  macro={best_u[2]}")

    conn = connectivity_audit(benchmark)
    print("\n  Connectivity audit:")
    print(f"    num_net_edges={conn['num_net_edges']}  macros_with_degree>0={conn['num_macros_with_degree_gt_0']}")
    print(f"    spectral_available={conn['spectral_available']}  terminal_anchor_available={conn['terminal_anchor_available']}")

    best_cost = best.proxy_cost if best and best.proxy_cost is not None else None
    improved = diag.delta_vs_raw_original is not None and diag.delta_vs_raw_original < -1e-6
    # Winner-detail reporting
    winner_macro_id = None
    winner_intended_dx = None
    winner_intended_dy = None
    winner_postlegal_dx = None
    winner_postlegal_dy = None
    winner_legalizer_moved_extra = None
    if best is not None:
        winner_macro_id = best.metadata.get("moved_macro_id")
        winner_intended_dx = best.metadata.get("intended_dx", best.metadata.get("dx"))
        winner_intended_dy = best.metadata.get("intended_dy", best.metadata.get("dy"))
        # Postlegal displacement = actual positions minus original positions for the moved macro
        if winner_macro_id is not None:
            mid = int(winner_macro_id)
            orig_pos = benchmark.macro_positions.float()
            actual_pos = best.positions.float()
            winner_postlegal_dx = float((actual_pos[mid, 0] - orig_pos[mid, 0]).item())
            winner_postlegal_dy = float((actual_pos[mid, 1] - orig_pos[mid, 1]).item())
            # Extra macros moved by legalizer beyond the intended macro
            import torch as _torch
            disp = _torch.norm(actual_pos - orig_pos, dim=1)
            extra_moved = [
                i for i in range(benchmark.num_hard_macros)
                if i != mid and float(disp[i].item()) > 1e-4
            ]
            winner_legalizer_moved_extra = extra_moved

    print("\n  Summary:")
    print(f"    raw_original_proxy_cost={f'{raw_cost:.6f}' if raw_cost is not None else 'N/A'}")
    print(f"    best_proxy_cost={f'{best_cost:.6f}' if best_cost is not None else 'N/A'}")
    print(f"    delta_vs_raw_original={f'{diag.delta_vs_raw_original:+.6f}' if diag.delta_vs_raw_original is not None else 'N/A'}")
    print(f"    winning_candidate={best.name if best else 'none'}")
    print(f"    winning_family={best.family if best else 'none'}")
    print(f"    moved_macro_id={winner_macro_id}")
    if winner_intended_dx is not None:
        print(f"    intended_dx={winner_intended_dx:.4f}  intended_dy={winner_intended_dy:.4f}")
    if winner_postlegal_dx is not None:
        print(f"    actual_postlegal_dx={winner_postlegal_dx:.4f}  actual_postlegal_dy={winner_postlegal_dy:.4f}")
    if winner_legalizer_moved_extra is not None:
        print(f"    legalizer_moved_extra_macros={winner_legalizer_moved_extra}  "
              f"(count={len(winner_legalizer_moved_extra)})")
    print(f"    official_scores_used={diag.candidates_officially_scored}  "
          f"skipped_by_budget={diag.candidates_skipped_by_budget}")
    print(f"    runtime_ms={runtime_ms:.0f}")
    print(f"    invariant={'OK' if diag.invariant_holds else 'VIOLATED'}")
    print(f"    improved={improved}")

    return {
        "benchmark": name,
        "runtime_ms": runtime_ms,
        "family_counts": family_counts,
        "raw_original_cost": raw_cost,
        "best_cost": best_cost,
        "delta_vs_raw_original": diag.delta_vs_raw_original,
        "best_candidate": best.name if best else "none",
        "best_family": best.family if best else "none",
        "moved_macro_id": winner_macro_id,
        "intended_dx": winner_intended_dx,
        "intended_dy": winner_intended_dy,
        "actual_postlegal_dx": winner_postlegal_dx,
        "actual_postlegal_dy": winner_postlegal_dy,
        "legalizer_moved_extra_macros": winner_legalizer_moved_extra,
        "improved": improved,
        "official_scored_count": diag.candidates_officially_scored,
        "fresh_official_scores": diag.fresh_official_scores,
        "skipped_by_budget": diag.candidates_skipped_by_budget,
        "cache_hits": diag.cache_hits,
        "duplicate_count": diag.duplicate_count,
        "prefiltered_count": diag.candidates_prefiltered,
        "invariant_holds": diag.invariant_holds,
        "selected_due_to": diag.selected_due_to,
        "winning_moves": [
            {
                "name": sc.name,
                "macro_id": sc.metadata.get("moved_macro_id"),
                "dx": sc.metadata.get("dx"),
                "dy": sc.metadata.get("dy"),
                "delta_vs_raw_original": sc.delta_vs_original,
            }
            for sc in ranked
            if sc.family == "original_neighborhood" and sc.delta_vs_original is not None and sc.delta_vs_original < -1e-9
        ],
    }


_SMOKE_PROFILES = {
    "official-smoke": {
        "description": "Basic official scoring sanity check - not the final submission config",
        "only_original_neighborhood": False,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_seed_strategy": "conservative",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
    },
    "m2b-final": {
        "description": "M2B final submission profile: diverse seeds, bounded budget, cold-run validated",
        "only_original_neighborhood": True,
        "candidate_budget": 80,
        "neighborhood_macro_limit": 20,
        "neighborhood_step_profile": "medium",
        "refinement_around_winners": True,
        "refinement_top_k": 5,
        "refinement_seed_strategy": "diverse",
        "refinement_exploration_seeds": 1,
        "line_search_around_winners": True,
        "line_search_top_k": 3,
        "line_search_max_scale": 4.0,
        "line_search_stop_after_worse": 2,
        "max_official_scores": 60,
    },
}


def main():
    parser = argparse.ArgumentParser(prog="run_official_scoring_smoke", description="Official M2B scoring smoke test.")
    parser.add_argument("-b", "--benchmark", action="append", dest="benchmarks", default=None, metavar="NAME")
    parser.add_argument(
        "--profile",
        choices=list(_SMOKE_PROFILES.keys()),
        default=None,
        help="Load a named profile as defaults (overridden by explicit flags).",
    )
    parser.add_argument("--candidate-budget", type=int, default=None)
    parser.add_argument("--neighborhood-macro-limit", type=int, default=None)
    parser.add_argument("--neighborhood-step-profile", choices=["small", "medium", "large"], default=None)
    parser.add_argument("--disable-global-candidates", action="store_true")
    parser.add_argument("--only-original-neighborhood", action="store_true")
    parser.add_argument("--refinement-around-winners", action="store_true")
    parser.add_argument("--refinement-top-k", type=int, default=None)
    parser.add_argument("--refinement-seed-strategy", choices=["conservative", "diverse"], default=None)
    parser.add_argument("--refinement-exploration-seeds", type=int, default=None)
    parser.add_argument("--line-search-around-winners", action="store_true")
    parser.add_argument("--line-search-top-k", type=int, default=None)
    parser.add_argument("--line-search-max-scale", type=float, default=None)
    parser.add_argument("--line-search-stop-after-worse", type=int, default=None)
    parser.add_argument("--max-official-scores", type=int, default=None)
    parser.add_argument("--official-score-cache", metavar="PATH", default=None)
    parser.add_argument("--clear-score-cache", action="store_true")
    args = parser.parse_args()

    # Load profile defaults, then let explicit CLI args override
    p = _SMOKE_PROFILES.get(args.profile, {}) if args.profile else {}
    if args.profile:
        print(f"\n  Profile: {args.profile} - {p.get('description', '')}")

    names = args.benchmarks or ["ibm01", "ibm02", "ibm03"]
    gen_cfg = CandidateGenerationConfig(
        candidate_budget=args.candidate_budget if args.candidate_budget is not None else p.get("candidate_budget", 80),
        neighborhood_macro_limit=args.neighborhood_macro_limit if args.neighborhood_macro_limit is not None else p.get("neighborhood_macro_limit", 20),
        neighborhood_step_profile=args.neighborhood_step_profile if args.neighborhood_step_profile is not None else p.get("neighborhood_step_profile", "medium"),
        disable_global_candidates=args.disable_global_candidates,
        only_original_neighborhood=args.only_original_neighborhood or p.get("only_original_neighborhood", False),
        refinement_around_winners=args.refinement_around_winners or p.get("refinement_around_winners", False),
        refinement_top_k=args.refinement_top_k if args.refinement_top_k is not None else p.get("refinement_top_k", 5),
        refinement_seed_strategy=args.refinement_seed_strategy if args.refinement_seed_strategy is not None else p.get("refinement_seed_strategy", "conservative"),
        refinement_exploration_seeds=args.refinement_exploration_seeds if args.refinement_exploration_seeds is not None else p.get("refinement_exploration_seeds", 1),
        line_search_around_winners=args.line_search_around_winners or p.get("line_search_around_winners", False),
        line_search_top_k=args.line_search_top_k if args.line_search_top_k is not None else p.get("line_search_top_k", 3),
        line_search_max_scale=args.line_search_max_scale if args.line_search_max_scale is not None else p.get("line_search_max_scale", 4.0),
        line_search_stop_after_worse=args.line_search_stop_after_worse if args.line_search_stop_after_worse is not None else p.get("line_search_stop_after_worse", 2),
    )
    score_cfg = CandidateScoringConfig(
        max_official_scores=args.max_official_scores if args.max_official_scores is not None else p.get("max_official_scores"),
        official_score_cache_path=args.official_score_cache,
        clear_score_cache=args.clear_score_cache,
        disable_score_cache=(args.official_score_cache is None),
    )

    # Build shared cache once for the session (so ibm01 hits populate ibm02 run, etc.)
    shared_cache = None
    if args.official_score_cache:
        from pathlib import Path as _Path
        from submissions.solver.core.score_cache import OfficialScoreCache
        shared_cache = OfficialScoreCache(
            cache_path=_Path(args.official_score_cache),
            disabled=False,
            clear=args.clear_score_cache,
        )
        print(f"  Score cache: {args.official_score_cache}  entries={shared_cache.size}  clear={args.clear_score_cache}")

    results = []
    failures = []
    for name in names:
        try:
            results.append(_run_official_smoke(
                name,
                generation_config=gen_cfg,
                scoring_config=score_cfg,
                score_cache=shared_cache,
            ))
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(f"\nFAIL [{name}]: {exc}")
            failures.append(name)

    if failures:
        print(f"\nFAILED benchmarks: {failures}")
        sys.exit(1)

    print(f"\n{'=' * 65}")
    print(f"Official scoring smoke PASSED: {len(results)} benchmarks")
    for row in results:
        delta = row["delta_vs_raw_original"]
        print(
            f"  {row['benchmark']:20s}  raw={row['raw_original_cost']:.4f}  best={row['best_cost']:.4f}  "
            f"delta={delta:+.6f}  winner={row['best_candidate']}  family={row['best_family']}  "
            f"fresh={row.get('fresh_official_scores', row['official_scored_count'])}"
            f"  cache_hits={row.get('cache_hits', 0)}"
            f"  inv={'OK' if row['invariant_holds'] else 'VIOLATED'}"
        )
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
