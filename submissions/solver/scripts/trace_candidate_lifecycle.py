"""
Trace the full pipeline lifecycle of a named candidate on a benchmark.

Usage:
    python -m submissions.solver.scripts.trace_candidate_lifecycle \
        --benchmark ibm02 \
        --candidate original_refinement_m256_scale0p5x \
        [--refinement-around-winners] \
        [--line-search-around-winners] \
        [--official-score-cache PATH] \
        [--clear-score-cache]

Prints:
    generation_rank, pass_id, approx_hpwl_delta, prelegal_valid,
    postlegal_valid, placement_hash, cache_hit, scoring_rank,
    fresh_score_consumed, official_cost, skip_reason

Also reports which pass-2 seeds were selected (to explain if the
candidate was never generated at all).
"""

import argparse
import sys
from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.candidate_types import (  # noqa: E402
    CandidateGenerationConfig,
    CandidateScoringConfig,
)


def _load_official(name: str):
    ibm_dir = IBM_TESTCASES_DIR / name
    if not ibm_dir.exists() or not (ibm_dir / "netlist.pb.txt").exists():
        raise FileNotFoundError(f"IBM testcase not found: {ibm_dir}")
    from macro_place.loader import load_benchmark_from_dir
    benchmark, plc = load_benchmark_from_dir(ibm_dir.as_posix())
    if plc is None:
        raise RuntimeError("load_benchmark_from_dir returned plc=None")
    return benchmark, plc


def _na(v, fmt=None):
    if v is None:
        return "N/A"
    if fmt:
        return f"{v:{fmt}}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(prog="trace_candidate_lifecycle")
    parser.add_argument("-b", "--benchmark", default="ibm02")
    parser.add_argument("-c", "--candidate", default="original_refinement_m256_scale0p5x")
    parser.add_argument("--refinement-around-winners", action="store_true", default=True)
    parser.add_argument("--no-refinement-around-winners", dest="refinement_around_winners", action="store_false")
    parser.add_argument("--line-search-around-winners", action="store_true", default=False)
    parser.add_argument("--refinement-top-k", type=int, default=5)
    parser.add_argument("--neighborhood-macro-limit", type=int, default=20)
    parser.add_argument("--neighborhood-step-profile", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--candidate-budget", type=int, default=80)
    parser.add_argument("--max-official-scores", type=int, default=None)
    parser.add_argument("--official-score-cache", metavar="PATH", default=None)
    parser.add_argument("--clear-score-cache", action="store_true")
    args = parser.parse_args()

    target = args.candidate
    print(f"\nLifecycle trace: '{target}' on {args.benchmark}")
    print(f"cache={'COLD (no cache)' if not args.official_score_cache else args.official_score_cache}")
    print("=" * 80)

    benchmark, plc = _load_official(args.benchmark)

    gen_cfg = CandidateGenerationConfig(
        candidate_budget=args.candidate_budget,
        neighborhood_macro_limit=args.neighborhood_macro_limit,
        neighborhood_step_profile=args.neighborhood_step_profile,
        refinement_around_winners=args.refinement_around_winners,
        refinement_top_k=args.refinement_top_k,
        line_search_around_winners=args.line_search_around_winners,
    )
    score_cfg = CandidateScoringConfig(
        max_official_scores=args.max_official_scores,
        official_score_cache_path=args.official_score_cache,
        disable_score_cache=(args.official_score_cache is None),
        clear_score_cache=args.clear_score_cache,
    )

    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select

    # Intercept pass-2 seed selection by patching generate_original_refinement_candidates
    _seeds_used = []
    _ref_placements_generated = []

    import submissions.solver.core.original_refinement as _ref_mod
    _orig_gen = _ref_mod.generate_original_refinement_candidates

    def _patched_gen(benchmark_, seed_candidates, config, existing_names):
        _seeds_used.extend(seed_candidates)
        result = _orig_gen(benchmark_, seed_candidates, config, existing_names)
        _ref_placements_generated.extend(result)
        return result

    _ref_mod.generate_original_refinement_candidates = _patched_gen

    candidates = generate_candidates(benchmark, config=gen_cfg)
    best, ranked, diag = score_and_select(
        candidates, benchmark, plc=plc,
        scoring_config=score_cfg,
        generation_config=gen_cfg,
    )

    _ref_mod.generate_original_refinement_candidates = _orig_gen  # restore

    # --- Check pass 1: was macro 256 a neighborhood seed candidate? ---
    print("\n[PASS 1] original_neighborhood candidates for macro 256:")
    m256_p1 = [sc for sc in ranked if sc.family == "original_neighborhood"
               and sc.metadata.get("moved_macro_id") == 256]
    if not m256_p1:
        print("  (none generated)")
    else:
        print(f"  {'Name':<44}  {'approx_delta':>12}  {'valid':>6}  {'was_scored':>10}  {'skip_reason':<20}  {'scoring_rank':>12}")
        for sc in m256_p1:
            print(
                f"  {sc.name:<44}  "
                f"{_na(sc.metadata.get('approx_hpwl_delta'), '.6f'):>12}  "
                f"{str(sc.valid):>6}  "
                f"{str(sc.was_scored):>10}  "
                f"{_na(sc.metadata.get('skip_reason')):<20}  "
                f"{_na(sc.metadata.get('scoring_rank')):>12}"
            )

    # --- Check pass-2 seeds selected ---
    print("\n[PASS 2] Seeds selected for refinement:")
    if not _seeds_used:
        print("  (refinement pass not triggered or no seeds)")
    else:
        print(f"  {'Name':<44}  {'macro_id':>8}  {'dx':>8}  {'dy':>8}")
        for s in _seeds_used:
            mid = s.metadata.get("moved_macro_id", "?")
            dx = s.metadata.get("dx", 0.0)
            dy = s.metadata.get("dy", 0.0)
            print(f"  {s.name:<44}  {str(mid):>8}  {float(dx):>8.2f}  {float(dy):>8.2f}")

    m256_in_seeds = any(s.metadata.get("moved_macro_id") == 256 for s in _seeds_used)
    print(f"\n  macro 256 selected as refinement seed: {m256_in_seeds}")

    # --- Check if target was generated in pass 2 ---
    target_generated = any(p.name == target for p in _ref_placements_generated)
    print(f"\n[GENERATION] '{target}' generated by refinement pass: {target_generated}")
    if not target_generated and m256_in_seeds:
        # Search for scale0.5x variants
        scale05 = [p for p in _ref_placements_generated if "m256" in p.name and "scale" in p.name]
        print(f"  scale variants for m256 that WERE generated: {[p.name for p in scale05]}")

    # --- Find target in ranked output ---
    print(f"\n[RANKED OUTPUT] Searching for '{target}':")
    target_sc = next((sc for sc in ranked if sc.name == target), None)

    if target_sc is None:
        print(f"  NOT FOUND in ranked output.")
        print(f"  Total candidates in ranked: {len(ranked)}")
        # Look for close matches
        close = [sc.name for sc in ranked if "m256" in sc.name and "scale" in sc.name]
        print(f"  m256 scale variants in ranked: {close}")
    else:
        meta = target_sc.metadata
        print(f"\n  {'Field':<28}  {'Value'}")
        print(f"  {'-'*60}")
        fields = [
            ("generation_rank",     meta.get("generation_rank")),
            ("pass_id",             meta.get("pass_id")),
            ("approx_hpwl_delta",   meta.get("approx_hpwl_delta")),
            ("prelegal_valid",      meta.get("prelegal_valid")),
            ("postlegal_valid",     meta.get("postlegal_valid")),
            ("placement_hash",      meta.get("placement_hash")),
            ("cache_hit",           meta.get("cache_hit")),
            ("scoring_rank",        meta.get("scoring_rank")),
            ("fresh_score_consumed",meta.get("fresh_score_consumed")),
            ("official_cost",       target_sc.proxy_cost),
            ("skip_reason",         meta.get("skip_reason")),
            ("was_scored",          target_sc.was_scored),
            ("valid",               target_sc.valid),
            ("duplicate_of",        target_sc.duplicate_of),
        ]
        for name_, val in fields:
            if isinstance(val, float):
                print(f"  {name_:<28}  {val:.8f}")
            else:
                print(f"  {name_:<28}  {val}")

    # --- Overall run summary ---
    print(f"\n[RUN SUMMARY]")
    print(f"  winner:              {best.name if best else 'none'}")
    print(f"  best_cost:           {_na(diag.best_proxy_cost, '.8f')}")
    print(f"  delta_vs_original:   {_na(diag.delta_vs_raw_original, '+.6f')}")
    print(f"  raw_original_cost:   {_na(diag.raw_original_proxy_cost, '.8f')}")
    print(f"  fresh_scores:        {diag.fresh_official_scores}")
    print(f"  cache_hits:          {diag.cache_hits}")
    print(f"  prefiltered:         {diag.candidates_prefiltered}")
    print(f"  skipped_by_budget:   {diag.candidates_skipped_by_budget}")
    print(f"  refinement_generated:{diag.refinement_candidates_generated}")
    print()


if __name__ == "__main__":
    main()
