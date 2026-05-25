"""
Candidate inspection script for M2B.
"""

import argparse
import sys
from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
for _p in [str(_SOLVER_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submissions.solver.config import BENCHMARKS_PT_DIR, IBM_TESTCASES_DIR  # noqa: E402
from submissions.solver.core.candidate_types import CandidateGenerationConfig, CandidateScoringConfig  # noqa: E402


def _load(name: str):
    ibm_dir = IBM_TESTCASES_DIR / name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        from submissions.solver._env import require_submodule

        require_submodule()
        from macro_place.loader import load_benchmark_from_dir

        return load_benchmark_from_dir(ibm_dir.as_posix())

    pt_path = BENCHMARKS_PT_DIR / f"{name}.pt"
    if pt_path.exists():
        from macro_place.benchmark import Benchmark

        return Benchmark.load(str(pt_path)), None
    raise FileNotFoundError(f"Benchmark '{name}' not found.")


def _load_pt(pt_path: str):
    pt = Path(pt_path)
    if not pt.exists():
        raise FileNotFoundError(f".pt file not found: {pt}")
    from macro_place.benchmark import Benchmark

    return Benchmark.load(str(pt)), None


def _fmt(value, spec=".4f"):
    return f"{value:{spec}}" if value is not None else "N/A"


def run(benchmark, plc, generation_config: CandidateGenerationConfig) -> None:
    from submissions.solver.core.benchmark_adapter import inspect as inspect_bm
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select

    stats = inspect_bm(benchmark)
    print(f"\n{'=' * 100}")
    print(f"inspect_candidates - {benchmark.name}")
    print(f"  Hard macros: {stats['num_hard_macros']}  Movable: {stats['num_movable_hard']}  Nets: {stats['num_nets']}")
    print(f"  Canvas: {stats['canvas_width']:.1f}x{stats['canvas_height']:.1f} um  Util: {stats['utilization']:.1%}")
    print(f"{'=' * 100}")

    candidates = generate_candidates(benchmark, config=generation_config)
    best, ranked, diag = score_and_select(candidates, benchmark, plc=plc, scoring_config=CandidateScoringConfig())
    print(f"\nGenerated {len(candidates)} candidates")
    print(f"raw_original_proxy_cost={_fmt(diag.raw_original_proxy_cost)}  best_proxy_cost={_fmt(diag.best_proxy_cost)}  delta={_fmt(diag.delta_vs_raw_original, '+.4f')}")
    print(f"winner={diag.winning_candidate}  family={diag.winning_family}  scores={diag.candidates_officially_scored}  duplicates={diag.duplicate_count}  prefiltered={diag.candidates_prefiltered}")

    headers = ["rank", "candidate", "family", "valid", "scored", "cost", "delta", "move"]
    widths = [4, 42, 22, 6, 6, 10, 10, 20]
    row_fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print()
    print(row_fmt.format(*headers))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for idx, sc in enumerate(ranked, start=1):
        move = ""
        if sc.metadata.get("moved_macro_id") is not None:
            move = f"m{sc.metadata['moved_macro_id']} ({float(sc.metadata.get('dx', 0.0)):+.1f},{float(sc.metadata.get('dy', 0.0)):+.1f})"
        print(
            row_fmt.format(
                str(idx),
                sc.name[:42],
                sc.family[:22],
                str(sc.valid),
                str(sc.was_scored),
                _fmt(sc.proxy_cost),
                _fmt(sc.delta_vs_original, "+.4f"),
                move[:20],
            )
        )

    print()
    if best is not None:
        print(f"Best: {best.name}  cost={_fmt(best.proxy_cost)}  valid={best.valid}")
    print(f"{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser(prog="inspect_candidates", description="Inspect candidate placements and ranking.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", default="ibm01")
    group.add_argument("--pt", metavar="PATH")
    parser.add_argument("--candidate-budget", type=int, default=None)
    parser.add_argument("--neighborhood-macro-limit", type=int, default=20)
    parser.add_argument("--neighborhood-step-profile", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--disable-global-candidates", action="store_true")
    parser.add_argument("--only-original-neighborhood", action="store_true")
    args = parser.parse_args()

    benchmark, plc = _load_pt(args.pt) if args.pt else _load(args.benchmark)
    run(
        benchmark,
        plc,
        generation_config=CandidateGenerationConfig(
            candidate_budget=args.candidate_budget,
            neighborhood_macro_limit=args.neighborhood_macro_limit,
            neighborhood_step_profile=args.neighborhood_step_profile,
            disable_global_candidates=args.disable_global_candidates,
            only_original_neighborhood=args.only_original_neighborhood,
        ),
    )


if __name__ == "__main__":
    main()
