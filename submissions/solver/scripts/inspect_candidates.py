"""
Candidate inspection script — M2B.

Prints a ranked candidate table for a benchmark:
  rank | candidate | family | valid | proxy_cost | delta_vs_original | runtime_ms | notes

Usage (from repo root):
    python -m submissions.solver.scripts.inspect_candidates -b ibm01
    python -m submissions.solver.scripts.inspect_candidates --pt benchmarks/processed/public/ibm01.pt
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


def _load(name: str):
    """Load by name: IBM testcase first, then .pt fallback."""
    ibm_dir = IBM_TESTCASES_DIR / name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        from submissions.solver._env import require_submodule
        require_submodule()
        from macro_place.loader import load_benchmark_from_dir
        return load_benchmark_from_dir(ibm_dir.as_posix())

    pt_path = BENCHMARKS_PT_DIR / f"{name}.pt"
    if pt_path.exists():
        print(f"  (submodule absent — loading from {pt_path})")
        from macro_place.benchmark import Benchmark
        return Benchmark.load(str(pt_path)), None

    raise FileNotFoundError(
        f"Benchmark '{name}' not found.\n  Tried: {ibm_dir}\n  Tried: {pt_path}"
    )


def _load_pt(pt_path: str):
    pt = Path(pt_path)
    if not pt.exists():
        raise FileNotFoundError(f".pt file not found: {pt}")
    from macro_place.benchmark import Benchmark
    return Benchmark.load(str(pt)), None


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "N/A"


def run(benchmark, plc) -> None:
    from submissions.solver.core.candidates import generate_candidates
    from submissions.solver.core.candidate_scoring import score_and_select
    from submissions.solver.core.benchmark_adapter import inspect as inspect_bm

    stats = inspect_bm(benchmark)
    print(f"\n{'='*80}")
    print(f"inspect_candidates — {benchmark.name}")
    print(f"  Hard macros: {stats['num_hard_macros']}  Movable: {stats['num_movable_hard']}  Nets: {stats['num_nets']}")
    print(f"  Canvas: {stats['canvas_width']:.1f}x{stats['canvas_height']:.1f}µm  Util: {stats['utilization']:.1%}")
    print(f"{'='*80}")

    candidates = generate_candidates(benchmark)
    print(f"\nGenerated {len(candidates)} candidates. Legalizing + scoring...")

    best, ranked = score_and_select(candidates, benchmark, plc=plc)

    # Print ranked table
    col_w = [4, 40, 15, 6, 11, 12, 10, 30]
    headers = ["rank", "candidate", "family", "valid", "cost", "delta", "ms", "notes"]
    row_fmt = "  ".join(f"{{:{w}}}" for w in col_w)

    print()
    print(row_fmt.format(*headers))
    print("  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1)))

    for k, sc in enumerate(ranked):
        delta_str = _fmt(sc.delta_vs_original, "+.4f") if sc.delta_vs_original is not None else "N/A"
        best_marker = " ◀ BEST" if sc is best else ""
        notes = (sc.notes or "") + best_marker
        print(
            row_fmt.format(
                str(k + 1),
                sc.name[:38],
                sc.family[:13],
                str(sc.valid),
                _fmt(sc.proxy_cost),
                delta_str,
                f"{sc.total_ms:.0f}",
                notes[:28],
            )
        )

    print()
    if best is not None:
        print(f"  Best: {best.name}  cost={_fmt(best.proxy_cost)}  valid={best.valid}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="inspect_candidates",
        description="M2B: inspect candidate placements and ranking.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", default="ibm01", help="Benchmark name (default: ibm01)")
    group.add_argument("--pt", metavar="PATH", help="Load from explicit .pt file")
    args = parser.parse_args()

    try:
        if args.pt:
            benchmark, plc = _load_pt(args.pt)
        else:
            benchmark, plc = _load(args.benchmark)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error loading benchmark: {exc}")
        sys.exit(1)

    run(benchmark, plc)


if __name__ == "__main__":
    main()
